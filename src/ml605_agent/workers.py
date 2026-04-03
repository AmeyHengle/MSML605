"""Worker node functions for the ml605 LangGraph pipeline.

Each worker is a pure function: (PipelineState) -> dict.
Workers return partial dicts — only the keys they own are updated in state.
All workers log metrics/artifacts to MLflow nested runs when mlflow_run_id is set.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import nullcontext
from pathlib import Path

import mlflow
import pandas as pd

from langchain_mcp_adapters.client import MultiServerMCPClient

from ml605_pipeline.automl import run_automl
from ml605_pipeline.drift import detect_drift
from ml605_pipeline.evaluate import compute_metrics
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
)
from ml605_pipeline.registry import load_production_model, register_model, transition_model_stage
from ml605_agent.state import PipelineState

# Module-level constant so tests can monkeypatch it
FEATURES_FILE = Path("features_used.txt")

# Target column in the pipeline DataFrame (used by test_worker and retrain_worker)
TARGET_COL = "actual_intensity"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


async def _async_fetch(window_hours: int) -> dict:
    """Async inner implementation for fetch_worker."""
    client = MultiServerMCPClient(
        {
            "carbon_intensity": {
                "transport": "streamable_http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )
    tools = await client.get_tools()

    fetch_tool = next((t for t in tools if t.name == "fetch_intensity"), None)
    if fetch_tool is None:
        raise RuntimeError("MCP tool 'fetch_intensity' not found on server")

    raw = await fetch_tool.ainvoke({"hours_back": window_hours})

    # langchain-mcp-adapters >= 0.2 returns a list of LangChain content blocks
    # (response_format="content_and_artifact"), not the raw dict.  Extract the
    # JSON text from the first TextContent block and parse it.
    if isinstance(raw, list):
        text = next(
            (b["text"] for b in raw if isinstance(b, dict) and b.get("type") == "text"),
            "{}",
        )
        result = json.loads(text)
    else:
        result = raw

    readings = result.get("readings", [])
    factors = result.get("factors", {})

    df = pd.DataFrame(readings)
    return {"df": df, "factors": factors, "rows_fetched": len(df)}


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------


def fetch_worker(state: PipelineState) -> dict:
    """Fetch carbon intensity readings via the MCP server.

    Returns:
        {"df": DataFrame, "factors": dict, "rows_fetched": int}
        or {"status": "error", "error": str} on failure.
    """
    try:
        result = asyncio.run(_async_fetch(state.get("window_hours", 12)))

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_name="fetch_worker", nested=True):
                mlflow.log_metric("rows_fetched", result["rows_fetched"])

        return result
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def feature_worker(state: PipelineState) -> dict:
    """Apply feature engineering pipeline to the raw DataFrame.

    Returns:
        {"df_featured": DataFrame, "feature_cols": list[str]}
        or {"status": "error", "error": str} on failure.
    """
    try:
        try:
            feature_cols = load_feature_list(FEATURES_FILE)
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "features_used.txt missing or empty — run fetch_historical_data.py first",
            }

        if not feature_cols:
            return {
                "status": "error",
                "error": "features_used.txt missing or empty — run fetch_historical_data.py first",
            }

        df = state["df"].copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = add_time_features(df)
        df = apply_factor_columns(df, state.get("factors", {}))
        df = ensure_feature_columns(df, feature_cols)

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_name="feature_worker", nested=True):
                mlflow.log_param("feature_count", len(feature_cols))

        return {"df_featured": df, "feature_cols": feature_cols}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def test_worker(state: PipelineState) -> dict:
    """Evaluate the production model against current window data.

    Returns:
        {"eval_result": EvalResult}
        or {"status": "error", "error": str} when no model or on failure.
    """
    try:
        model = load_production_model()
        if model is None:
            return {
                "status": "error",
                "error": "No Production model registered in MLflow — run train_with_mlflow.py first",
            }

        df = state["df_featured"].dropna(subset=[TARGET_COL])
        feature_cols = state["feature_cols"]
        X = df[feature_cols]
        y_true = df[TARGET_COL].values
        y_pred = model.predict(X)
        eval_result = compute_metrics(pd.Series(y_true), y_pred)

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_name="test_worker", nested=True):
                mlflow.log_metric("rmse", eval_result.rmse)
                mlflow.log_metric("mae", eval_result.mae)
                mlflow.log_metric("r2", eval_result.r2)
                mlflow.log_metric("mape", eval_result.mape)

        return {"eval_result": eval_result}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def _get_production_rmse() -> float:
    """Fetch RMSE of current Production model from MLflow. Raises RuntimeError if not found."""
    from mlflow.tracking import MlflowClient
    from ml605_pipeline.registry import MODEL_NAME

    client = MlflowClient()
    versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    if not versions:
        raise RuntimeError(
            "No Production model in MLflow — cannot compute RMSE degradation. "
            "Run train_with_mlflow.py and promote a model to Production first."
        )
    run_id = versions[0].run_id
    run = client.get_run(run_id)
    rmse = run.data.metrics.get("rmse")
    if rmse is None:
        raise RuntimeError(
            f"Production model run {run_id} has no 'rmse' metric logged. "
            "Cannot compute RMSE degradation signal."
        )
    return float(rmse)


def drift_worker(state: PipelineState) -> dict:
    """Detect distribution drift between historical reference and current window.

    Returns:
        {"drift_report": DriftReport, "overall_drift": bool,
         "rmse_degradation_pct": float, "rmse_degradation_fired": bool,
         "production_rmse": float}
        or {"status": "error", "error": str} on failure.
    """
    try:
        try:
            reference_df = pd.read_csv(Path("historical_data.csv"))
        except FileNotFoundError:
            return {
                "status": "error",
                "error": "historical_data.csv missing — run fetch_historical_data.py first",
            }

        feature_cols = state["feature_cols"]
        factors = state.get("factors", {})

        # Align reference data with the same feature pipeline
        # Only call add_time_features if the timestamp column is present
        if "timestamp" in reference_df.columns:
            reference_df["timestamp"] = pd.to_datetime(reference_df["timestamp"], utc=True, errors="coerce")
            reference_df = add_time_features(reference_df)
        reference_df = apply_factor_columns(reference_df, factors)
        reference_df = ensure_feature_columns(reference_df, feature_cols)

        drift_report = detect_drift(reference_df, state["df_featured"], feature_cols)

        # RMSE degradation signal (third drift signal alongside PSI and KS)
        try:
            production_rmse = _get_production_rmse()
        except RuntimeError as exc:
            return {"status": "error", "error": str(exc)}

        current_rmse = state["eval_result"].rmse
        rmse_degradation_pct = (current_rmse - production_rmse) / production_rmse * 100.0
        rmse_degradation_fired = current_rmse > production_rmse * 1.20

        # overall_drift fires when ANY signal fires
        overall_drift = drift_report.overall_drift or rmse_degradation_fired

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_name="drift_worker", nested=True):
                mlflow.log_metric("rmse_degradation_pct", rmse_degradation_pct)
                mlflow.log_metric("rmse_degradation_fired", int(rmse_degradation_fired))
                mlflow.log_metric("overall_drift", int(overall_drift))
                mlflow.log_text(
                    json.dumps(
                        {
                            "overall_drift": overall_drift,
                            "drifted_features": drift_report.drifted_features,
                            "rmse_degradation_fired": rmse_degradation_fired,
                            "rmse_degradation_pct": rmse_degradation_pct,
                        }
                    ),
                    "drift_report.json",
                )

        return {
            "drift_report": drift_report,
            "overall_drift": overall_drift,
            "rmse_degradation_pct": rmse_degradation_pct,
            "rmse_degradation_fired": rmse_degradation_fired,
            "production_rmse": production_rmse,
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def retrain_worker(state: PipelineState) -> dict:
    """Retrain the model using AutoML and register new version in Staging.

    Returns:
        {"new_model_version": str, "retrain_done": True}
        or {"status": "error", "error": str} on failure.
    """
    try:
        df = state["df_featured"].dropna(subset=[TARGET_COL])
        feature_cols = state["feature_cols"]

        # time_split expects target_col; our DataFrame uses actual_intensity
        split_idx = int(len(df) * 0.8)
        X = df[feature_cols].fillna(0.0)
        y = df[TARGET_COL]
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        mlflow_run_id = state.get("mlflow_run_id")
        ctx = mlflow.start_run(run_name="retrain_worker", nested=True) if mlflow_run_id else nullcontext()

        with ctx:
            automl_result = run_automl(X_train, y_train, X_test, y_test)
            version = register_model(automl_result.best.run_id)
            transition_model_stage(version, "Staging")

        return {"new_model_version": version, "retrain_done": True}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}
