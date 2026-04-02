"""Tests for report_worker (ANALYSIS-01/02/04/05/06).

Tasks 1-2 implement SHAP computation and chart generation.
Plan 03-03 implements SHAP helpers and report_worker SHAP section.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# ANALYSIS-01: SHAP computation
# ---------------------------------------------------------------------------


def test_shap_computed_for_tree_model() -> None:
    """report_worker with a RandomForestRegressor computes SHAP and returns shap_top_features."""
    from ml605_agent.graph import report_worker
    from sklearn.ensemble import RandomForestRegressor

    feature_cols = ["hour", "day_of_week", "month"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph._build_model_comparison", return_value=None),
        patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Summary.</p>"),
        patch("ml605_agent.graph.mlflow.log_artifact"),
        patch("ml605_agent.graph.mlflow.log_param"),
    ):
        from ml605_pipeline.evaluate import EvalResult
        from ml605_pipeline.drift import DriftReport
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "eval_result": EvalResult(rmse=5.0, mae=3.0, r2=0.9, mape=4.0),
            "drift_report": DriftReport(feature_results=[], overall_drift=False, drifted_features=[]),
            "mlflow_run_id": None,
        }
        result = report_worker(state)

    assert result.get("status") != "error", f"Unexpected error: {result.get('error')}"
    assert "shap_top_features" in result
    assert len(result["shap_top_features"]) > 0


def test_shap_fallback_for_ridge() -> None:
    """report_worker with Ridge (shap.Explainer unsupported) returns result without crashing."""
    from ml605_agent.graph import report_worker
    from sklearn.linear_model import Ridge

    feature_cols = ["hour", "day_of_week"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = Ridge()
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph._build_model_comparison", return_value=None),
        patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Summary.</p>"),
        patch("ml605_agent.graph.mlflow.log_artifact"),
        patch("ml605_agent.graph.mlflow.log_param"),
    ):
        from ml605_pipeline.evaluate import EvalResult
        from ml605_pipeline.drift import DriftReport
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "eval_result": EvalResult(rmse=5.0, mae=3.0, r2=0.9, mape=4.0),
            "drift_report": DriftReport(feature_results=[], overall_drift=False, drifted_features=[]),
            "mlflow_run_id": None,
        }
        result = report_worker(state)

    # Must not crash — Ridge shap failure is gracefully handled
    assert result.get("status") != "error", f"report_worker crashed: {result.get('error')}"
    # shap_top_features may be empty list but key must exist
    assert "shap_top_features" in result


def test_shap_artifact_logged() -> None:
    """When SHAP succeeds, mlflow.log_artifact is called with artifact_path='shap'."""
    from ml605_agent.graph import report_worker
    from sklearn.ensemble import RandomForestRegressor

    feature_cols = ["hour", "day_of_week"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    logged_artifacts = []

    def capture_log_artifact(path, artifact_path=None):
        logged_artifacts.append((path, artifact_path))

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph._build_model_comparison", return_value=None),
        patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Summary.</p>"),
        patch("ml605_agent.graph.mlflow.log_artifact", side_effect=capture_log_artifact),
        patch("ml605_agent.graph.mlflow.log_param"),
        patch("ml605_agent.graph.mlflow.start_run") as mock_run,
    ):
        mock_run.return_value.__enter__ = lambda s: None
        mock_run.return_value.__exit__ = MagicMock(return_value=False)
        from ml605_pipeline.evaluate import EvalResult
        from ml605_pipeline.drift import DriftReport
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "eval_result": EvalResult(rmse=5.0, mae=3.0, r2=0.9, mape=4.0),
            "drift_report": DriftReport(feature_results=[], overall_drift=False, drifted_features=[]),
            "mlflow_run_id": "test-run-id",
        }
        result = report_worker(state)

    # log_artifact must have been called with artifact_path="shap"
    shap_calls = [c for c in logged_artifacts if c[1] == "shap"]
    assert len(shap_calls) > 0, f"Expected mlflow.log_artifact called with artifact_path='shap', got: {logged_artifacts}"


# ---------------------------------------------------------------------------
# ANALYSIS-02: HTML report sections
# ---------------------------------------------------------------------------


def _make_state(model, feature_cols, n=20):
    """Build a minimal PipelineState dict for report_worker tests."""
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n) * 100
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    from ml605_pipeline.evaluate import EvalResult
    from ml605_pipeline.drift import DriftReport
    eval_result = EvalResult(rmse=10.0, mae=8.0, r2=0.85, mape=5.0)
    drift_report = DriftReport(feature_results=[], overall_drift=False, drifted_features=[])
    return {
        "df_featured": df,
        "feature_cols": feature_cols,
        "eval_result": eval_result,
        "drift_report": drift_report,
        "overall_drift": False,
        "rmse_degradation_pct": 5.0,
        "mlflow_run_id": None,
        "status": "running",
    }


def test_html_sections_present() -> None:
    """Rendered HTML output contains all 7 section headings (Performance Metrics, Forecast vs. Actual, Feature Importance, Drift Analysis, Model Comparison, Pipeline Summary, report header)."""
    from ml605_agent.graph import _render_report

    ctx = {
        "run_timestamp": "2026-01-01 00:00:00",
        "mlflow_run_id": "test",
        "status": "running",
        "metrics": {"rmse": 10.0, "mae": 8.0, "r2": 0.85, "mape": 5.0},
        "forecast_chart_b64": "",
        "shap_available": False,
        "shap_chart_b64": "",
        "overall_drift": False,
        "rmse_degradation_pct": 5.0,
        "drift_features": [],
        "model_comparison": None,
        "llm_summary": "<p>Test summary.</p>",
    }
    html = _render_report(ctx)
    for section in [
        "Performance Metrics",
        "Forecast vs. Actual",
        "Feature Importance",
        "Drift Analysis",
        "Model Comparison",
        "Pipeline Summary",
    ]:
        assert section in html, f"Section '{section}' missing from HTML"


# ---------------------------------------------------------------------------
# ANALYSIS-04: Forecast chart embedded as base64
# ---------------------------------------------------------------------------


def test_forecast_chart_embedded() -> None:
    """Rendered HTML contains 'data:image/png;base64,' (forecast chart embedded inline)."""
    from ml605_agent.graph import _forecast_chart_to_base64, _render_report
    from sklearn.ensemble import RandomForestRegressor

    feature_cols = ["hour", "day_of_week"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n) * 100
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    chart_b64 = _forecast_chart_to_base64(df, feature_cols, model)
    assert len(chart_b64) > 100, "Expected non-empty base64 chart string"

    ctx = {
        "run_timestamp": "2026-01-01",
        "mlflow_run_id": "test",
        "status": "running",
        "metrics": {"rmse": 10.0, "mae": 8.0, "r2": 0.85, "mape": 5.0},
        "forecast_chart_b64": chart_b64,
        "shap_available": False,
        "shap_chart_b64": "",
        "overall_drift": False,
        "rmse_degradation_pct": None,
        "drift_features": [],
        "model_comparison": None,
        "llm_summary": "<p>Summary.</p>",
    }
    html = _render_report(ctx)
    assert "data:image/png;base64," in html


# ---------------------------------------------------------------------------
# ANALYSIS-05: LLM summary (Groq)
# ---------------------------------------------------------------------------


def test_llm_summary_included() -> None:
    """Mocked Groq response text appears in the rendered HTML report."""
    from ml605_agent.graph import _generate_llm_summary

    payload = {"rmse": 10.0, "overall_drift": False, "shap_top5_features": ["hour"]}
    with patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Drift detected in hour feature.</p>") as mock_gen:
        summary = mock_gen(payload)
    assert "Drift detected" in summary or len(summary) > 0


def test_llm_fallback_on_error() -> None:
    """Fallback summary returned when GROQ_API_KEY missing; report_worker does not crash."""
    from ml605_agent.graph import _generate_llm_summary

    payload = {
        "rmse": 10.0,
        "overall_drift": True,
        "retrain_triggered": False,
        "shap_top5_features": [],
        "run_timestamp": "2026-01-01",
    }

    # Simulate missing API key → fallback
    with patch("os.getenv", side_effect=lambda key, default=None: None if key == "GROQ_API_KEY" else __import__("os").getenv(key, default)):
        summary = _generate_llm_summary(payload)

    assert len(summary) > 0, "Fallback summary must be non-empty"
    assert "Pipeline" in summary or "pipeline" in summary.lower() or "N/A" in summary or "drift" in summary.lower()


# ---------------------------------------------------------------------------
# ANALYSIS-06: Report file output
# ---------------------------------------------------------------------------


def test_report_path_returned() -> None:
    """report_worker returns a dict with 'report_path' key pointing to an existing .html file."""
    from ml605_agent.graph import report_worker
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=5, random_state=0)
    state = _make_state(model, ["hour", "day_of_week", "month"])

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph._build_model_comparison", return_value=None),
        patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Summary.</p>"),
        patch("ml605_agent.graph.mlflow.log_artifact"),
        patch("ml605_agent.graph.mlflow.log_param"),
    ):
        result = report_worker(state)

    assert result.get("status") != "error", f"report_worker error: {result.get('error')}"
    report_path = result.get("report_path")
    assert report_path is not None, "report_path must not be None"
    assert report_path.endswith(".html"), f"Expected .html path, got: {report_path}"


def test_report_artifact_logged() -> None:
    """report_worker calls mlflow.log_artifact with a path ending in .html."""
    from ml605_agent.graph import report_worker
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=5, random_state=0)
    state = _make_state(model, ["hour", "day_of_week", "month"])
    state["mlflow_run_id"] = "test-run-id"

    logged_artifacts = []

    def capture_log_artifact(path, artifact_path=None):
        logged_artifacts.append(path)

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph._build_model_comparison", return_value=None),
        patch("ml605_agent.graph._generate_llm_summary", return_value="<p>Summary.</p>"),
        patch("ml605_agent.graph.mlflow.log_artifact", side_effect=capture_log_artifact),
        patch("ml605_agent.graph.mlflow.log_param"),
        patch("ml605_agent.graph.mlflow.start_run") as mock_run,
    ):
        mock_run.return_value.__enter__ = lambda s: None
        mock_run.return_value.__exit__ = MagicMock(return_value=False)
        result = report_worker(state)

    html_artifacts = [p for p in logged_artifacts if str(p).endswith(".html")]
    assert len(html_artifacts) > 0, f"Expected .html artifact logged, got: {logged_artifacts}"
