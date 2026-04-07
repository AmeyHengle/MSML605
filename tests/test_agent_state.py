"""Tests for ml605_agent.state.PipelineState TypedDict contract."""
from __future__ import annotations


def test_state_schema() -> None:
    """PipelineState is a TypedDict with all 21 required fields."""
    from ml605_agent.state import PipelineState

    # TypedDict instances are plain dicts at runtime
    assert issubclass(PipelineState, dict)

    expected_fields = {
        "window_hours",
        "df",
        "factors",
        "rows_fetched",
        "df_featured",
        "feature_cols",
        "eval_result",
        "drift_report",
        "overall_drift",
        # RMSE degradation signal fields (added in Plan 03-02)
        "rmse_degradation_pct",
        "rmse_degradation_fired",
        "production_rmse",
        "retrain_done",
        "new_model_version",
        "report_path",
        # Slack/HITL fields (added in Plan 04-01)
        "shap_top_features",
        "alert_sent",
        "hitl_decision",
        "mlflow_run_id",
        "status",
        "error",
    }
    assert expected_fields == set(PipelineState.__annotations__), (
        f"Field mismatch. Missing: {expected_fields - set(PipelineState.__annotations__)}, "
        f"Extra: {set(PipelineState.__annotations__) - expected_fields}"
    )


def test_partial_state_updates() -> None:
    """PipelineState can be constructed with a subset of fields (total=False)."""
    from ml605_agent.state import PipelineState

    s: PipelineState = {"window_hours": 24, "status": "running"}
    assert s["window_hours"] == 24
    assert s["status"] == "running"
    assert "df" not in s
    assert "drift_report" not in s
