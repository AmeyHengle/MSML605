"""Shared PipelineState TypedDict contract for the ml605 LangGraph agent.

This is a pure type definition file — no implementation logic here.
Every worker and the graph node depend on these field names.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from typing_extensions import TypedDict

from ml605_pipeline.drift import DriftReport
from ml605_pipeline.evaluate import EvalResult


class PipelineState(TypedDict, total=False):
    # Input
    window_hours: int

    # fetch_worker output
    df: Optional[pd.DataFrame]
    factors: Optional[dict]
    rows_fetched: int

    # feature_worker output
    df_featured: Optional[pd.DataFrame]
    feature_cols: list[str]

    # test_worker output
    eval_result: Optional[EvalResult]

    # drift_worker output
    drift_report: Optional[DriftReport]
    overall_drift: bool
    rmse_degradation_pct: Optional[float]
    rmse_degradation_fired: Optional[bool]
    production_rmse: Optional[float]

    # retrain_worker output
    # retrain_done: bool — set True by retrain_worker to prevent second retrain cycle in drift routing
    retrain_done: bool
    new_model_version: Optional[str]
    # MLflow registry stage the new version was promoted to (currently "Production"
    # — see docs/SLACK_HITL_ROADMAP.md for why HITL-gated Staging was removed).
    model_stage: Optional[str]

    # report_worker output (Phase 3)
    report_path: Optional[str]
    shap_top_features: Optional[list[str]]

    # alert_worker output
    alert_sent: Optional[bool]

    # HITL output (Phase 4)
    hitl_decision: Optional[str]

    # cross-cutting
    mlflow_run_id: Optional[str]
    status: str  # "running | complete | error"
    error: Optional[str]
