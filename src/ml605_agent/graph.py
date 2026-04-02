"""LangGraph pipeline graph assembly for the ml605 agent.

This module defines the full StateGraph topology connecting all worker nodes
with conditional routing for drift detection and error handling.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # headless backend — must be set before pyplot import

import base64
import io
import json
import os
import tempfile
from contextlib import nullcontext as _nullctx
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from jinja2 import Environment, FileSystemLoader
from langgraph.graph import END, START, StateGraph

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

TARGET_COL = "intensity.actual"


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
    """Generate SHAP explainability, charts, HTML report, and LLM summary.

    Phase 3 implementation. Replaces Phase 2 stub.
    Returns dict with report_path (None until Plan 04), shap_top_features, and
    intermediate chart data. Returns {"status": "error", "error": str} on failure.
    """
    try:
        # -- 1. Load model for SHAP --
        model = load_production_model()
        if model is None:
            return {"status": "error", "error": "No Production model available for report_worker"}

        df = state.get("df_featured")
        feature_cols = state.get("feature_cols", [])
        if df is None or not feature_cols:
            return {"status": "error", "error": "df_featured or feature_cols missing from state"}

        X = df[feature_cols]

        # -- 2. SHAP computation (with fallback for non-tree models) --
        shap_available = False
        shap_chart_b64 = ""
        top_features = []

        ctx = mlflow.start_run(run_name="report_worker", nested=True) if state.get("mlflow_run_id") else _nullctx()
        with ctx:
            shap_values, top_features = _compute_shap(model, X)
            shap_available = shap_values is not None

            if shap_available:
                shap_chart_b64 = _shap_bar_to_base64(shap_values)
                # Log SHAP top features as JSON artifact
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix="_shap_top_features.json", delete=False
                ) as f:
                    json.dump({"top_features": top_features}, f, indent=2)
                    shap_artifact_path = f.name
                mlflow.log_artifact(shap_artifact_path, artifact_path="shap")
                import os
                os.unlink(shap_artifact_path)

            # -- 3. Forecast vs. actual chart --
            forecast_chart_b64 = _forecast_chart_to_base64(df, feature_cols, model)

            # Plan 04 will complete: Jinja2 rendering + Groq + file save + log_artifact(html)
            # Intermediate return — report_path set to None until Plan 04
            return {
                "report_path": None,
                "shap_top_features": top_features,
                "_shap_chart_b64": shap_chart_b64,
                "_forecast_chart_b64": forecast_chart_b64,
                "_shap_available": shap_available,
            }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def alert_worker(state: PipelineState) -> dict:
    """Phase 4 stub: log placeholder, return alert_sent=False."""
    if state.get("mlflow_run_id"):
        with mlflow.start_run(run_name="alert_worker", nested=True):
            mlflow.log_param("alert_status", "stub_phase4")
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
    - overall_drift=True and retrain_done=False → retrain_worker
    - overall_drift=False OR retrain_done=True → report_worker (prevents second retrain)
    """
    if state.get("status") == "error":
        return "error_handler"
    if state.get("overall_drift") and not state.get("retrain_done", False):
        return "retrain_worker"
    return "report_worker"


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


def build_graph():
    """Assemble and compile the full ml605 pipeline StateGraph.

    Topology:
        START → fetch_worker → feature_worker → test_worker → drift_worker
        drift_worker → [drift? yes: retrain_worker → test_worker loop; no: report_worker]
        report_worker → alert_worker → END
        Any error → error_handler → END

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("fetch_worker", fetch_worker)
    builder.add_node("feature_worker", feature_worker)
    builder.add_node("test_worker", test_worker)
    builder.add_node("drift_worker", drift_worker)
    builder.add_node("retrain_worker", retrain_worker)
    builder.add_node("report_worker", report_worker)
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

    # After drift: conditional routing (retrain, report, error)
    builder.add_conditional_edges(
        "drift_worker",
        route_after_drift,
        {
            "retrain_worker": "retrain_worker",
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

    # Linear tail: report → alert → END
    builder.add_edge("report_worker", "alert_worker")
    builder.add_edge("alert_worker", END)
    builder.add_edge("error_handler", END)

    return builder.compile()
