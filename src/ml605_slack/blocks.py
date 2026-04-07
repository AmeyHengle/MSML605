"""Block Kit message builder functions (pure, testable).

All functions return list[dict] suitable for Slack chat_postMessage blocks param.
"""
from __future__ import annotations


def build_drift_alert_blocks(
    overall_drift: bool,
    drifted_features: list[dict],
    shap_top_features: list[str],
    eval_result_dict: dict,
    thread_id: str,
) -> list[dict]:
    """Build Block Kit blocks for a drift-detected alert (per D-06)."""
    raise NotImplementedError("Plan 04-02 implements this")


def build_no_drift_blocks(
    eval_result_dict: dict,
    production_version: str,
) -> list[dict]:
    """Build Block Kit blocks for a no-drift pipeline summary (per D-07)."""
    raise NotImplementedError("Plan 04-02 implements this")
