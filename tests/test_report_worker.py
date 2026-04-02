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

    with patch("ml605_agent.graph.load_production_model", return_value=model):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
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

    with patch("ml605_agent.graph.load_production_model", return_value=model):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
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

    with (
        patch("ml605_agent.graph.load_production_model", return_value=model),
        patch("ml605_agent.graph.mlflow") as mock_mlflow,
    ):
        mock_mlflow.start_run.return_value.__enter__ = lambda s: None
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "mlflow_run_id": "test-run-id",
        }
        result = report_worker(state)

    # log_artifact must have been called with artifact_path="shap"
    calls = mock_mlflow.log_artifact.call_args_list
    shap_calls = [c for c in calls if "shap" in str(c)]
    assert len(shap_calls) > 0, f"Expected mlflow.log_artifact called with shap, got: {calls}"


# ---------------------------------------------------------------------------
# ANALYSIS-02: HTML report sections
# ---------------------------------------------------------------------------


def test_html_sections_present() -> None:
    """Rendered HTML output contains all 7 section headings (Performance Metrics, Forecast vs. Actual, Feature Importance, Drift Analysis, Model Comparison, Pipeline Summary, report header)."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-04: Forecast chart embedded as base64
# ---------------------------------------------------------------------------


def test_forecast_chart_embedded() -> None:
    """Rendered HTML contains 'data:image/png;base64,' (forecast chart embedded inline)."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-05: LLM summary (Groq)
# ---------------------------------------------------------------------------


def test_llm_summary_included() -> None:
    """Mocked Groq client response text appears in the rendered HTML report."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_llm_fallback_on_error() -> None:
    """When Groq raises groq.APIConnectionError, fallback text appears in HTML and pipeline does not crash."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


# ---------------------------------------------------------------------------
# ANALYSIS-06: Report file output
# ---------------------------------------------------------------------------


def test_report_path_returned() -> None:
    """report_worker returns a dict with 'report_path' key pointing to an existing .html file."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")


def test_report_artifact_logged() -> None:
    """report_worker calls mlflow.log_artifact with a path ending in .html."""
    pytest.fail("not implemented — Wave 1/2 plan implements this")
