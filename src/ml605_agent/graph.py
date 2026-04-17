"""LangGraph pipeline graph assembly for the ml605 agent.

This module defines the full StateGraph topology connecting all worker nodes
with conditional routing for drift detection, HITL interrupt/resume, and error handling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import base64
import io
import json
import os
import tempfile
import time
from contextlib import nullcontext as _nullctx
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from jinja2 import Environment, FileSystemLoader
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ml605_agent.state import PipelineState
from ml605_agent.workers import (
    drift_worker,
    feature_worker,
    fetch_worker,
    retrain_worker,
    test_worker,
)
from ml605_pipeline.registry import load_production_model

TEMPLATE_DIR = Path(__file__).parent / "templates"
REPORTS_DIR = Path("reports")


# ---------------------------------------------------------------------------
# Report generation helpers
# ---------------------------------------------------------------------------

TARGET_COL = "actual_intensity"


def _compute_shap(model, X: pd.DataFrame, max_display: int = 10):
    """Compute SHAP values using shap.Explainer.

    Returns (shap_values Explanation, top_features list) or (None, []) if model
    type is not supported by TreeExplainer (e.g., Ridge from AutoML fallback).
    """
    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:max_display]
        top_features = [X.columns[i] for i in top_idx]
        return shap_values, top_features
    except Exception as exc:  # noqa: BLE001
        mlflow.log_param("shap_warning", str(exc)[:200])
        return None, []


def _fig_to_base64(fig) -> str:
    """Encode a matplotlib Figure as a base64 PNG string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    plt.close(fig)
    return encoded


def _shap_bar_to_base64(shap_values, max_display: int = 10) -> str:
    """Render SHAP bar plot and return as base64 PNG string."""
    shap.plots.bar(shap_values, max_display=max_display, show=False)
    fig = plt.gcf()
    return _fig_to_base64(fig)


def _forecast_chart_to_base64(df, feature_cols: list[str], model) -> str:
    """Render forecast vs. actual chart and return as base64 PNG string.

    Uses last 50 rows of df for readability. Falls back to empty string on error.
    """
    try:
        X = df[feature_cols].tail(50)
        y_true = df[TARGET_COL].tail(50).values
        y_pred = model.predict(X)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(y_true)), y_true, label="Actual", color="#2c5f8f", linewidth=1.5)
        ax.plot(range(len(y_pred)), y_pred, label="Forecast", color="#e8963e",
                linestyle="--", linewidth=1.5)
        ax.set_xlabel("Time step (last 50)")
        ax.set_ylabel("Carbon Intensity (gCO2/kWh)")
        ax.set_title("Forecast vs. Actual")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return _fig_to_base64(fig)
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Report generation helpers (Jinja2 / Groq / MLflow)
# ---------------------------------------------------------------------------


def _fallback_summary(ctx: dict) -> str:
    """Structured fallback when LLM is unavailable."""
    drift_str = "detected" if ctx.get("overall_drift") else "not detected"
    return (
        f"<p>Pipeline completed at {ctx.get('run_timestamp', 'N/A')}. "
        f"Drift was {drift_str}.</p>"
        f"<p>Current RMSE: {ctx.get('rmse', 'N/A')}. "
        f"Retrain triggered: {ctx.get('retrain_triggered', False)}.</p>"
        f"<p>Top features: {', '.join(ctx.get('shap_top5_features', [])) or 'N/A'}.</p>"
    )


def _generate_llm_summary(context_payload: dict) -> str:
    """Generate 2-3 paragraph plain-English summary via Groq. Falls back to template on failure."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return _fallback_summary(context_payload)
    try:
        import groq as groq_sdk  # noqa: PLC0415
        client = groq_sdk.Groq(api_key=api_key)
        prompt = (
            "You are an MLOps analyst. Given the following pipeline run data as JSON, "
            "write exactly 2-3 paragraphs covering: (1) drift diagnosis, "
            "(2) top drifted/important features, (3) retrain recommendation.\n\n"
            f"{json.dumps(context_payload, indent=2)}"
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=512,
        )
        return response.choices[0].message.content
    except Exception as exc:  # noqa: BLE001
        mlflow.log_param("llm_error", str(exc)[:200])
        return _fallback_summary(context_payload)


def _build_model_comparison(eval_result, new_model_version=None):
    """Fetch previous Production model metrics from MLflow for comparison table.

    Returns dict with current_rmse, current_mae, production_rmse, production_mae,
    production_version — or None if no Production model exists.
    """
    from mlflow.tracking import MlflowClient  # noqa: PLC0415
    from ml605_pipeline.registry import MODEL_NAME  # noqa: PLC0415
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return {
            "current_rmse": eval_result.rmse,
            "current_mae": eval_result.mae,
            "production_rmse": run.data.metrics.get("rmse"),
            "production_mae": run.data.metrics.get("mae"),
            "production_version": versions[0].version,
        }
    except Exception:  # noqa: BLE001
        return None


def _render_report(context: dict) -> str:
    """Render the Jinja2 HTML template with the given context dict."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=False,  # base64 data URIs must not be escaped
    )
    template = env.get_template("report.html.j2")
    return template.render(**context)


def _save_and_log_report(html_content: str) -> str:
    """Save HTML to reports/ with timestamp and log to MLflow. Returns local file path."""
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"pipeline_report_{timestamp}.html"
    report_path.write_text(html_content, encoding="utf-8")
    mlflow.log_artifact(str(report_path), artifact_path="reports")
    return str(report_path)


# ---------------------------------------------------------------------------
# Stub workers (Phase 2 stubs — full implementation in later phases)
# ---------------------------------------------------------------------------


def report_worker(state: PipelineState) -> dict:
    """Generate SHAP explainability, HTML report, LLM summary, and log to MLflow.

    Returns {"report_path": str} with path to generated HTML,
    or {"status": "error", "error": str} on fatal failure.
    """
    try:
        # -- 1. Load model --
        model = load_production_model()
        if model is None:
            return {"status": "error", "error": "No Production model available for report_worker"}

        df = state.get("df_featured")
        feature_cols = state.get("feature_cols", [])
        eval_result = state.get("eval_result")
        drift_report = state.get("drift_report")

        if df is None or not feature_cols or eval_result is None:
            return {
                "status": "error",
                "error": "Required state keys missing: df_featured, feature_cols, or eval_result",
            }

        X = df[feature_cols]
        run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ctx = (
            mlflow.start_run(run_name="report_worker", nested=True)
            if state.get("mlflow_run_id")
            else _nullctx()
        )
        with ctx:
            # -- 2. SHAP computation --
            shap_values, top_features = _compute_shap(model, X)
            shap_available = shap_values is not None
            shap_chart_b64 = ""

            if shap_available:
                shap_chart_b64 = _shap_bar_to_base64(shap_values)
                # Log SHAP top-features JSON artifact
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_shap_top_features.json", delete=False
                ) as f:
                    json.dump({"top_features": top_features}, f, indent=2)
                    shap_artifact_tmp = f.name
                mlflow.log_artifact(shap_artifact_tmp, artifact_path="shap")
                os.unlink(shap_artifact_tmp)

            # -- 3. Forecast vs. actual chart --
            forecast_chart_b64 = _forecast_chart_to_base64(df, feature_cols, model)

            # -- 4. Model comparison table data --
            model_comparison = _build_model_comparison(
                eval_result, state.get("new_model_version")
            )

            # -- 5. LLM summary --
            shap_top5 = top_features[:5] if top_features else []
            drifted_5 = (
                drift_report.drifted_features[:5]
                if drift_report and drift_report.drifted_features
                else []
            )
            llm_payload = {
                "rmse": round(eval_result.rmse, 4),
                "mae": round(eval_result.mae, 4),
                "r2": round(eval_result.r2, 4),
                "overall_drift": state.get("overall_drift", False),
                "drifted_features": drifted_5,
                "shap_top5_features": shap_top5,
                "retrain_triggered": state.get("retrain_done", False),
                "new_model_version": state.get("new_model_version"),
                "run_timestamp": run_timestamp,
            }
            llm_summary = _generate_llm_summary(llm_payload)

            # -- 6. Drift feature table for template --
            drift_features_ctx = []
            if drift_report and drift_report.feature_results:
                drift_features_ctx = [
                    {
                        "feature": r.feature,
                        "psi": r.psi,
                        "ks_p_value": r.ks_p_value,
                        "drift_detected": r.drift_detected,
                    }
                    for r in drift_report.feature_results
                ]

            # -- 7. Render Jinja2 template --
            template_context = {
                "run_timestamp": run_timestamp,
                "mlflow_run_id": state.get("mlflow_run_id", "N/A"),
                "status": state.get("status", "complete"),
                "metrics": {
                    "rmse": eval_result.rmse,
                    "mae": eval_result.mae,
                    "r2": eval_result.r2,
                    "mape": eval_result.mape,
                },
                "forecast_chart_b64": forecast_chart_b64,
                "shap_available": shap_available,
                "shap_chart_b64": shap_chart_b64,
                "overall_drift": state.get("overall_drift", False),
                "rmse_degradation_pct": state.get("rmse_degradation_pct"),
                "drift_features": drift_features_ctx,
                "model_comparison": model_comparison,
                "llm_summary": llm_summary,
            }
            html_content = _render_report(template_context)

            # -- 8. Save HTML + log MLflow artifact --
            report_path = _save_and_log_report(html_content)
            mlflow.log_param("report_path", report_path)

        return {"report_path": report_path, "shap_top_features": top_features}

    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def alert_worker(state: PipelineState) -> dict:
    """Post pipeline results to Slack (per D-06, D-07, D-10).

    Reads SLACK_BOT_TOKEN and SLACK_CHANNEL_ID from env.
    Posts drift alert or no-drift summary Block Kit message.
    Uploads HTML report via files_upload_v2.
    Returns {"alert_sent": True} on success, {"alert_sent": False} if no token.
    """
    from ml605_slack.blocks import build_drift_alert_blocks, build_no_drift_blocks  # noqa: PLC0415

    slack_token = os.getenv("SLACK_BOT_TOKEN")
    channel_id = os.getenv("SLACK_CHANNEL_ID", "")

    if not slack_token or not channel_id:
        # Graceful degradation: no Slack posting if tokens not configured
        if state.get("mlflow_run_id"):
            with mlflow.start_run(run_name="alert_worker", nested=True):
                mlflow.log_param("alert_status", "skipped_no_slack_token")
        return {"alert_sent": False}

    try:
        from slack_sdk import WebClient  # noqa: PLC0415
        client = WebClient(token=slack_token)

        eval_result = state.get("eval_result")
        eval_dict = eval_result.__dict__ if eval_result is not None else {}

        if state.get("overall_drift"):
            drift_report = state.get("drift_report")
            drifted = []
            if drift_report is not None and hasattr(drift_report, "feature_results"):
                drifted = [
                    {"feature": r.feature, "psi": r.psi}
                    for r in drift_report.feature_results[:3]
                    if r.drift_detected
                ]
            blocks = build_drift_alert_blocks(
                overall_drift=True,
                drifted_features=drifted,
                shap_top_features=state.get("shap_top_features", []),
                eval_result_dict=eval_dict,
                thread_id=f"pipeline-{state.get('mlflow_run_id', 'unknown')}",
            )
        else:
            # Get production model version for no-drift summary
            from mlflow.tracking import MlflowClient as _MlflowClient  # noqa: PLC0415
            from ml605_pipeline.registry import MODEL_NAME as _MODEL_NAME  # noqa: PLC0415
            try:
                _client = _MlflowClient()
                versions = _client.get_latest_versions(_MODEL_NAME, stages=["Production"])
                prod_version = versions[0].version if versions else "unknown"
            except Exception:  # noqa: BLE001
                prod_version = "unknown"

            blocks = build_no_drift_blocks(
                eval_result_dict=eval_dict,
                production_version=prod_version,
            )

        client.chat_postMessage(
            channel=channel_id,
            blocks=blocks,
            text="Pipeline alert",
        )

        # Upload HTML report (per D-04)
        report_path = state.get("report_path")
        if report_path and Path(report_path).exists():
            client.files_upload_v2(
                channel=channel_id,
                file=report_path,
                title="Pipeline Report",
                filename=Path(report_path).name,
            )

        if state.get("mlflow_run_id"):
            with mlflow.start_run(run_name="alert_worker", nested=True):
                mlflow.log_param("alert_status", "sent")
                mlflow.log_param("alert_channel", channel_id)

        return {"alert_sent": True}

    except Exception as exc:  # noqa: BLE001
        # Per D-10: post error to Slack channel if possible
        try:
            from slack_sdk import WebClient as _WC  # noqa: PLC0415
            err_client = _WC(token=slack_token)
            err_client.chat_postMessage(
                channel=channel_id,
                text=f":x: Alert worker error: {exc}",
            )
        except Exception:  # noqa: BLE001
            pass
        return {"alert_sent": False}


def error_handler(state: PipelineState) -> dict:
    """Logs error to MLflow nested run and marks pipeline complete."""
    if state.get("mlflow_run_id"):
        with mlflow.start_run(run_name="error_handler", nested=True):
            mlflow.log_param("error", state.get("error", "unknown"))
    return {"status": "error"}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_fetch(state: PipelineState) -> str:
    """Route after fetch_worker: error or continue to feature_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "feature_worker"


def route_after_feature(state: PipelineState) -> str:
    """Route after feature_worker: error or continue to test_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "test_worker"


def route_after_test(state: PipelineState) -> str:
    """Route after test_worker: error or continue to drift_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "drift_worker"


def route_after_drift(state: PipelineState) -> str:
    """Route after drift_worker.

    - status=error → error_handler
    - all other cases → report_worker (HITL decides retrain; report runs before HITL)
    """
    if state.get("status") == "error":
        return "error_handler"
    return "report_worker"


def hitl_decision_node(state: PipelineState) -> dict:
    """Pause for human approval when drift is detected and retrain not yet done (per D-08).

    If no drift OR retrain already done, returns immediately with hitl_decision='no_drift'.
    If drift detected and retrain_done=False, calls interrupt() to pause the graph.
    The caller (bot.py) posts Block Kit with Approve/Reject buttons.
    When resumed via Command(resume='approve'|'reject'|'reject_timeout'),
    records the decision and logs to MLflow (per HITL-03).
    """
    if not state.get("overall_drift") or state.get("retrain_done", False):
        return {"hitl_decision": "no_drift"}

    # Record time when HITL decision was requested for MTTA calculation
    hitl_start = time.time()

    # interrupt() pauses the graph and returns control to the caller.
    # The payload is informational — the bot uses it to build the Block Kit message.
    drift_report = state.get("drift_report")
    drifted_features = []
    if drift_report and hasattr(drift_report, "drifted_features"):
        drifted_features = drift_report.drifted_features[:3]
    elif drift_report and hasattr(drift_report, "feature_results"):
        drifted_features = [
            r.feature for r in drift_report.feature_results[:3] if r.drift_detected
        ]

    decision = interrupt({
        "type": "retrain_approval",
        "drifted_features": drifted_features,
        "shap_top": state.get("shap_top_features", [])[:3],
    })

    # Execution resumes here after Command(resume=decision) is called.
    hitl_end = time.time()
    mtta_seconds = hitl_end - hitl_start

    # Log HITL decision to MLflow (HITL-03)
    if state.get("mlflow_run_id"):
        with mlflow.start_run(run_name="hitl_decision", nested=True):
            mlflow.log_param("hitl_decision", decision)
            mlflow.log_metric("mtta_seconds", mtta_seconds)

    return {"hitl_decision": decision}


def route_after_hitl(state: PipelineState) -> str:
    """Route after hitl_decision_node.

    - approve: proceed to retrain_worker
    - reject / reject_timeout / no_drift: proceed to alert_worker
    - error: go to error_handler
    """
    if state.get("status") == "error":
        return "error_handler"
    decision = state.get("hitl_decision", "no_drift")
    if decision == "approve":
        return "retrain_worker"
    return "alert_worker"


def route_after_retrain(state: PipelineState) -> str:
    """Route after retrain_worker: back-edge to test_worker to re-verify new model.

    - status=error → error_handler
    - otherwise → test_worker (back-edge)
    """
    if state.get("status") == "error":
        return "error_handler"
    return "test_worker"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph(checkpointer=None):
    """Assemble and compile the full ml605 pipeline StateGraph.

    Topology:
        START → fetch_worker → feature_worker → test_worker → drift_worker
        drift_worker → report_worker → hitl_decision_node
        hitl_decision_node → [approve: retrain_worker → test_worker loop; reject/no_drift: alert_worker]
        alert_worker → END
        Any error → error_handler → END

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g. MemorySaver) for HITL interrupt/resume.
            If None, graph cannot use interrupt() — for backward-compatible non-Slack runs.

    Returns:
        Compiled LangGraph StateGraph.
    """
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("fetch_worker", fetch_worker)
    builder.add_node("feature_worker", feature_worker)
    builder.add_node("test_worker", test_worker)
    builder.add_node("drift_worker", drift_worker)
    builder.add_node("retrain_worker", retrain_worker)
    builder.add_node("report_worker", report_worker)
    builder.add_node("hitl_decision_node", hitl_decision_node)
    builder.add_node("alert_worker", alert_worker)
    builder.add_node("error_handler", error_handler)

    # Entry: START → fetch_worker (with error routing)
    builder.add_conditional_edges(
        START,
        lambda s: "fetch_worker",
        {"fetch_worker": "fetch_worker"},
    )

    # After fetch: check for error, then go to feature_worker
    builder.add_conditional_edges(
        "fetch_worker",
        route_after_fetch,
        {"feature_worker": "feature_worker", "error_handler": "error_handler"},
    )

    # After feature: check for error, then go to test_worker
    builder.add_conditional_edges(
        "feature_worker",
        route_after_feature,
        {"test_worker": "test_worker", "error_handler": "error_handler"},
    )

    # After test: check for error, then go to drift_worker
    builder.add_conditional_edges(
        "test_worker",
        route_after_test,
        {"drift_worker": "drift_worker", "error_handler": "error_handler"},
    )

    # After drift: always go to report_worker (HITL decides retrain after seeing report)
    builder.add_conditional_edges(
        "drift_worker",
        route_after_drift,
        {
            "report_worker": "report_worker",
            "error_handler": "error_handler",
        },
    )

    # After retrain: back-edge to test_worker (re-verify new model)
    builder.add_conditional_edges(
        "retrain_worker",
        route_after_retrain,
        {"test_worker": "test_worker", "error_handler": "error_handler"},
    )

    # After report: HITL decision node (may interrupt for human approval)
    builder.add_edge("report_worker", "hitl_decision_node")

    # After HITL: route to retrain (approve) or alert (reject/no_drift)
    builder.add_conditional_edges(
        "hitl_decision_node",
        route_after_hitl,
        {
            "retrain_worker": "retrain_worker",
            "alert_worker": "alert_worker",
            "error_handler": "error_handler",
        },
    )

    # Linear tail: alert → END
    builder.add_edge("alert_worker", END)
    builder.add_edge("error_handler", END)

    return builder.compile(checkpointer=checkpointer)
