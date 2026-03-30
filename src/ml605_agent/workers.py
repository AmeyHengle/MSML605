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
TARGET_COL = "intensity.actual"


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

    result = await fetch_tool.ainvoke({"hours_back": window_hours})

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
            with mlflow.start_run(run_id=mlflow_run_id):
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
        df = add_time_features(df)
        df = apply_factor_columns(df, state.get("factors", {}))
        df = ensure_feature_columns(df, feature_cols)

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
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

        df = state["df_featured"]
        feature_cols = state["feature_cols"]
        X = df[feature_cols]
        y_true = df[TARGET_COL].values
        y_pred = model.predict(X)
        eval_result = compute_metrics(pd.Series(y_true), y_pred)

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                with mlflow.start_run(run_name="test_worker", nested=True):
                    mlflow.log_metric("rmse", eval_result.rmse)
                    mlflow.log_metric("mae", eval_result.mae)
                    mlflow.log_metric("r2", eval_result.r2)
                    mlflow.log_metric("mape", eval_result.mape)

        return {"eval_result": eval_result}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def drift_worker(state: PipelineState) -> dict:
    """Detect distribution drift between historical reference and current window.

    Returns:
        {"drift_report": DriftReport, "overall_drift": bool}
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
            reference_df = add_time_features(reference_df)
        reference_df = apply_factor_columns(reference_df, factors)
        reference_df = ensure_feature_columns(reference_df, feature_cols)

        drift_report = detect_drift(reference_df, state["df_featured"], feature_cols)

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with mlflow.start_run(run_id=mlflow_run_id):
                with mlflow.start_run(run_name="drift_worker", nested=True):
                    mlflow.log_metric("overall_drift", int(drift_report.overall_drift))
                    mlflow.log_text(
                        json.dumps(
                            {
                                "overall_drift": drift_report.overall_drift,
                                "drifted_features": drift_report.drifted_features,
                            }
                        ),
                        "drift_report.json",
                    )

        return {"drift_report": drift_report, "overall_drift": drift_report.overall_drift}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}


def retrain_worker(state: PipelineState) -> dict:
    """Retrain the model using AutoML and register new version in Staging.

    Returns:
        {"new_model_version": str, "retrain_done": True}
        or {"status": "error", "error": str} on failure.
    """
    try:
        df = state["df_featured"]
        feature_cols = state["feature_cols"]

        # time_split expects target_col; our DataFrame uses intensity.actual
        split_idx = int(len(df) * 0.8)
        X = df[feature_cols].fillna(0.0)
        y = df[TARGET_COL]
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        mlflow_run_id = state.get("mlflow_run_id")
        ctx = mlflow.start_run(run_id=mlflow_run_id) if mlflow_run_id else nullcontext()

        with ctx:
            with mlflow.start_run(run_name="retrain_worker", nested=bool(mlflow_run_id)):
                automl_result = run_automl(X_train, y_train, X_test, y_test)
                version = register_model(automl_result.best.run_id)
                transition_model_stage(version, "Staging")

        return {"new_model_version": version, "retrain_done": True}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "error": str(exc)}
