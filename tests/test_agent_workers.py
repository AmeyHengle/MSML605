"""Stub tests for ml605_agent worker functions.

Each test is marked xfail until workers.py is implemented in plan 02-02.
When workers.py exists, remove the xfail markers.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_fetch_worker() -> None:
    """fetch_worker returns a dict with df, factors, and rows_fetched keys."""
    from ml605_agent.workers import fetch_worker

    state = {"window_hours": 24, "status": "running"}
    result = fetch_worker(state)
    assert "df" in result
    assert "factors" in result
    assert "rows_fetched" in result


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_test_worker() -> None:
    """test_worker returns a dict with eval_result key when df_featured is present."""
    from ml605_agent.workers import test_worker

    state = {
        "df_featured": None,  # filled by feature_worker in real run
        "feature_cols": ["hour", "day_of_week"],
        "status": "running",
    }
    result = test_worker(state)
    assert "eval_result" in result


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_test_worker_no_model() -> None:
    """test_worker returns error status when load_production_model() returns None."""
    from ml605_agent.workers import test_worker

    state = {
        "df_featured": None,
        "feature_cols": [],
        "status": "running",
    }
    result = test_worker(state)
    # When no production model exists, worker sets status=error
    assert result.get("status") == "error"
    assert isinstance(result.get("error"), str)


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_drift_worker() -> None:
    """drift_worker returns a dict with drift_report and overall_drift keys."""
    from ml605_agent.workers import drift_worker

    state = {"eval_result": None, "df_featured": None, "feature_cols": [], "status": "running"}
    result = drift_worker(state)
    assert "drift_report" in result
    assert "overall_drift" in result


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_retrain_worker() -> None:
    """retrain_worker returns a dict with new_model_version and retrain_done keys."""
    from ml605_agent.workers import retrain_worker

    state = {"overall_drift": True, "df_featured": None, "feature_cols": [], "status": "running"}
    result = retrain_worker(state)
    assert "new_model_version" in result
    assert "retrain_done" in result


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_report_worker_stub() -> None:
    """report_worker returns {'report_path': None} as a stub."""
    from ml605_agent.workers import report_worker

    state = {"status": "running"}
    result = report_worker(state)
    assert result == {"report_path": None}


@pytest.mark.xfail(reason="workers.py not yet implemented", strict=False)
def test_alert_worker_stub() -> None:
    """alert_worker returns {'alert_sent': False} as a stub."""
    from ml605_agent.workers import alert_worker

    state = {"status": "running"}
    result = alert_worker(state)
    assert result == {"alert_sent": False}
