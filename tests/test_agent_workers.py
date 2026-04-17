"""Unit tests for ml605_agent worker functions.

Tests are organized per worker. Mocking is done with unittest.mock.patch.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from ml605_pipeline.drift import DriftReport, FeatureDriftResult
from ml605_pipeline.evaluate import EvalResult


# ---------------------------------------------------------------------------
# fetch_worker tests
# ---------------------------------------------------------------------------


def test_fetch_worker() -> None:
    """fetch_worker returns a dict with df, factors, and rows_fetched keys."""
    from ml605_agent.workers import fetch_worker

    fake_tool = MagicMock()
    fake_tool.ainvoke = AsyncMock(
        return_value={
            "readings": [{"from": "2024-01-01T00:00Z", "actual_intensity": 150}],
            "factors": {"gas": 0.394},
        }
    )

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[fake_tool])

    # Make fake_tool findable by name attribute
    fake_tool.name = "fetch_intensity"

    with patch("ml605_agent.workers.MultiServerMCPClient", return_value=mock_client):
        state = {"window_hours": 6, "status": "running"}
        result = fetch_worker(state)

    assert "df" in result, f"Expected 'df' key, got keys: {list(result.keys())}"
    assert "factors" in result
    assert "rows_fetched" in result
    assert result["rows_fetched"] == 1
    assert isinstance(result["df"], pd.DataFrame)


# ---------------------------------------------------------------------------
# feature_worker tests
# ---------------------------------------------------------------------------


def _minimal_intensity_df() -> pd.DataFrame:
    """Build a minimal DataFrame with columns feature_worker needs."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T10:00:00Z"], utc=True),
            "actual_intensity": [150],
            "intensity_index": ["moderate"],
        }
    )


def test_feature_worker(tmp_path: Path) -> None:
    """feature_worker enriches df with time features."""
    from ml605_agent.workers import feature_worker

    features_file = tmp_path / "features_used.txt"
    features_file.write_text("hour\nday_of_week\n")

    df = _minimal_intensity_df()
    state = {"df": df, "factors": {}, "feature_cols": []}

    with patch("ml605_agent.workers.FEATURES_FILE", features_file):
        result = feature_worker(state)

    assert "df_featured" in result, f"Expected 'df_featured', got: {list(result.keys())}"
    df_out = result["df_featured"]
    assert isinstance(df_out, pd.DataFrame)
    assert df_out.shape[1] > df.shape[1], "df_featured should have more columns than input"
    assert "hour" in df_out.columns


def test_feature_worker_empty_features(tmp_path: Path) -> None:
    """feature_worker returns error when features_used.txt is missing."""
    from ml605_agent.workers import feature_worker

    missing = tmp_path / "nonexistent_features.txt"
    df = _minimal_intensity_df()
    state = {"df": df, "factors": {}, "feature_cols": []}

    with patch("ml605_agent.workers.FEATURES_FILE", missing):
        result = feature_worker(state)

    assert result.get("status") == "error"
    assert "features_used.txt" in result.get("error", ""), (
        f"Expected 'features_used.txt' in error message, got: {result.get('error')}"
    )


# ---------------------------------------------------------------------------
# test_worker tests
# ---------------------------------------------------------------------------


def _featured_df(feature_cols: list[str]) -> pd.DataFrame:
    """Build a minimal df_featured for test_worker."""
    data = {col: [0.5] for col in feature_cols}
    data["actual_intensity"] = [150.0]
    return pd.DataFrame(data)


def test_test_worker() -> None:
    """test_worker returns eval_result when production model exists."""
    from ml605_agent.workers import test_worker

    feature_cols = ["hour", "day_of_week"]
    df = _featured_df(feature_cols)

    fake_model = MagicMock()
    fake_model.predict = MagicMock(return_value=np.array([145.0]))

    with patch("ml605_agent.workers.load_production_model", return_value=fake_model):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "status": "running",
        }
        result = test_worker(state)

    assert "eval_result" in result, f"Expected 'eval_result', got: {list(result.keys())}"
    assert isinstance(result["eval_result"].rmse, float)


def test_test_worker_no_model() -> None:
    """test_worker returns error status when load_production_model() returns None."""
    from ml605_agent.workers import test_worker

    with patch("ml605_agent.workers.load_production_model", return_value=None):
        state = {
            "df_featured": _featured_df(["hour"]),
            "feature_cols": ["hour"],
            "status": "running",
        }
        result = test_worker(state)

    assert result.get("status") == "error"
    assert isinstance(result.get("error"), str)
    assert result["error"], "Error message should be non-empty"


# ---------------------------------------------------------------------------
# drift_worker tests
# ---------------------------------------------------------------------------


def test_drift_worker() -> None:
    """drift_worker returns drift_report and overall_drift from detect_drift."""
    from ml605_agent.workers import drift_worker

    feature_cols = ["hour", "day_of_week"]
    df = _featured_df(feature_cols)

    fake_report = DriftReport(
        feature_results=[],
        overall_drift=True,
        drifted_features=["hour"],
    )

    with (
        patch("ml605_agent.workers.pd.read_csv", return_value=df),
        patch("ml605_agent.workers.detect_drift", return_value=fake_report),
        patch("ml605_agent.workers._get_production_rmse", return_value=12.0),
    ):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "factors": {},
            "eval_result": EvalResult(rmse=10.0, mae=8.0, r2=0.9, mape=5.0),
            "mlflow_run_id": None,
            "status": "running",
        }
        result = drift_worker(state)

    assert "drift_report" in result, f"Expected 'drift_report', got: {list(result.keys())}"
    assert "overall_drift" in result
    assert result["overall_drift"] is True


# ---------------------------------------------------------------------------
# retrain_worker tests
# ---------------------------------------------------------------------------


def test_retrain_worker() -> None:
    """retrain_worker returns retrain_done=True and new_model_version after automl."""
    from ml605_agent.workers import retrain_worker

    feature_cols = ["hour", "day_of_week"]
    # Need enough rows for time_split (which splits 80/20)
    n = 10
    data = {col: list(range(n)) for col in feature_cols}
    data["actual_intensity"] = [100.0 + i for i in range(n)]
    df = pd.DataFrame(data)

    fake_candidate = MagicMock()
    fake_candidate.run_id = "abc123"

    fake_automl_result = MagicMock()
    fake_automl_result.best = fake_candidate

    with (
        patch("ml605_agent.workers.run_automl", return_value=fake_automl_result),
        patch("ml605_agent.workers.register_model", return_value="1"),
        patch("ml605_agent.workers.transition_model_stage"),
        patch("ml605_agent.workers.mlflow"),
    ):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "mlflow_run_id": None,
            "status": "running",
        }
        result = retrain_worker(state)

    assert result.get("retrain_done") is True
    assert result.get("new_model_version") == "1"


# ---------------------------------------------------------------------------
# Stub worker tests (report_worker, alert_worker) — kept from original plan 01
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="report_worker stub not yet implemented in workers.py — defined in graph.py", strict=False)
def test_report_worker_stub() -> None:
    """report_worker returns {'report_path': None} as a stub."""
    from ml605_agent.workers import report_worker

    state = {"status": "running"}
    result = report_worker(state)
    assert result == {"report_path": None}


@pytest.mark.xfail(reason="alert_worker stub not yet implemented in workers.py — defined in graph.py", strict=False)
def test_alert_worker_stub() -> None:
    """alert_worker returns {'alert_sent': False} as a stub."""
    from ml605_agent.workers import alert_worker

    state = {"status": "running"}
    result = alert_worker(state)
    assert result == {"alert_sent": False}


# ---------------------------------------------------------------------------
# Integration test — requires live MCP server
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fetch_worker_integration() -> None:
    """fetch_worker returns rows from the live MCP server (integration).

    Requires MCP server running on localhost:8000.
    Skip if server is not reachable.
    """
    import subprocess
    import time

    import httpx

    # Check if MCP server is reachable; skip if not
    try:
        r = httpx.get("http://localhost:8000/health", timeout=2.0)
        if r.status_code != 200:
            pytest.skip("MCP server not running on localhost:8000")
    except Exception:
        pytest.skip("MCP server not running on localhost:8000")

    from ml605_agent.workers import fetch_worker

    state = {"window_hours": 1, "status": "running"}
    result = fetch_worker(state)

    assert "df" in result or result.get("status") == "error", (
        f"Unexpected result keys: {list(result.keys())}"
    )
    if result.get("status") != "error":
        assert result.get("rows_fetched", 0) > 0, "Expected at least 1 row fetched"


# ANALYSIS-03: RMSE degradation signal


def test_drift_worker_rmse_degradation() -> None:
    """drift_worker sets rmse_degradation_fired=True when current RMSE exceeds production RMSE by > 20%."""
    from ml605_agent.workers import drift_worker

    feature_cols = ["hour", "day_of_week"]
    df = _featured_df(feature_cols)
    fake_report = DriftReport(feature_results=[], overall_drift=False, drifted_features=[])

    with (
        patch("ml605_agent.workers.pd.read_csv", return_value=df),
        patch("ml605_agent.workers.detect_drift", return_value=fake_report),
        patch("ml605_agent.workers._get_production_rmse", return_value=10.0),
    ):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "factors": {},
            "eval_result": EvalResult(rmse=13.0, mae=9.0, r2=0.8, mape=6.0),  # 30% above 10.0
            "mlflow_run_id": None,
            "status": "running",
        }
        result = drift_worker(state)

    assert result.get("rmse_degradation_fired") is True
    assert result.get("overall_drift") is True  # even though PSI/KS are clean


def test_drift_worker_no_production_rmse() -> None:
    """drift_worker returns status='error' when no Production model RMSE is available in MLflow."""
    from ml605_agent.workers import drift_worker

    feature_cols = ["hour", "day_of_week"]
    df = _featured_df(feature_cols)
    fake_report = DriftReport(feature_results=[], overall_drift=False, drifted_features=[])

    with (
        patch("ml605_agent.workers.pd.read_csv", return_value=df),
        patch("ml605_agent.workers.detect_drift", return_value=fake_report),
        patch(
            "ml605_agent.workers._get_production_rmse",
            side_effect=RuntimeError("No Production model in MLflow"),
        ),
    ):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "factors": {},
            "eval_result": EvalResult(rmse=10.0, mae=8.0, r2=0.9, mape=5.0),
            "mlflow_run_id": None,
            "status": "running",
        }
        result = drift_worker(state)

    assert result.get("status") == "error"
    assert "Production" in result.get("error", "")


def test_drift_worker_three_signals() -> None:
    """All three signals evaluated independently; overall_drift=True if any fires."""
    from ml605_agent.workers import drift_worker

    feature_cols = ["hour"]
    df = _featured_df(feature_cols)
    # PSI/KS are clean (detect_drift returns False), but RMSE is 25% degraded
    fake_report = DriftReport(feature_results=[], overall_drift=False, drifted_features=[])

    with (
        patch("ml605_agent.workers.pd.read_csv", return_value=df),
        patch("ml605_agent.workers.detect_drift", return_value=fake_report),
        patch("ml605_agent.workers._get_production_rmse", return_value=10.0),
    ):
        state = {
            "df_featured": df,
            "feature_cols": feature_cols,
            "factors": {},
            "eval_result": EvalResult(rmse=12.5, mae=9.0, r2=0.85, mape=5.5),  # 25% above 10.0
            "mlflow_run_id": None,
            "status": "running",
        }
        result = drift_worker(state)

    assert result.get("rmse_degradation_fired") is True, "RMSE signal must fire at 25% degradation"
    assert result.get("overall_drift") is True, "overall_drift must be True when any signal fires"
    # PSI/KS were not fired (detect_drift returned False for feature results)
    assert result["drift_report"].overall_drift is False, "detect_drift's own verdict was False"
