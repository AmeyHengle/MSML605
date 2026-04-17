"""Block Kit message builder functions (pure, testable).

All functions return list[dict] suitable for Slack chat_postMessage blocks param.
Implements D-06 (drift alert) and D-07 (no-drift summary) from CONTEXT.md.
"""
from __future__ import annotations


def build_drift_alert_blocks(
    overall_drift: bool,
    drifted_features: list[dict],  # each dict: {"feature": str, "psi": float}
    shap_top_features: list[str],
    eval_result_dict: dict,  # {"rmse": float, "mae": float, "r2": float, "mape": float}
    thread_id: str,
) -> list[dict]:
    """Build Block Kit blocks for a drift-detected alert (per D-06).

    Returns a list of Block Kit block dicts with:
    1. Header block with drift verdict
    2. Divider
    3. Top drifted features section
    4. SHAP top features section
    5. Current metrics section
    6. Divider
    7. Actions block with Approve Retrain / Reject buttons
    """
    n_drifted = len(drifted_features)

    # 1. Header block
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f":warning: DRIFT DETECTED \u2014 {n_drifted} features drifted",
        },
    }

    # 2. Divider
    divider = {"type": "divider"}

    # 3. Top Drifted Features section (up to 3)
    feature_lines = "\n".join(
        f"- `{f['feature']}` (PSI: {f['psi']:.4f})"
        for f in drifted_features[:3]
    )
    top_drifted_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Top Drifted Features:*\n{feature_lines}",
        },
    }

    # 4. SHAP Top Features section (up to 3)
    shap_lines = "\n".join(
        f"- `{feat}`"
        for feat in shap_top_features[:3]
    )
    shap_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*SHAP Top Features:*\n{shap_lines}",
        },
    }

    # 5. Current metrics section
    rmse = eval_result_dict.get("rmse", "N/A")
    mae = eval_result_dict.get("mae", "N/A")
    rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
    mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
    metrics_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Current Metrics:*\nRMSE: {rmse_str} | MAE: {mae_str}",
        },
    }

    # 6. Divider
    divider2 = {"type": "divider"}

    # 7. Actions block with Approve / Reject buttons
    actions = {
        "type": "actions",
        "block_id": f"hitl_{thread_id}",
        "elements": [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Approve Retrain"},
                "style": "primary",
                "action_id": "approve_retrain",
                "value": thread_id,
            },
            {
                "type": "button",
                "text": {"type": "plain_text", "text": "Reject"},
                "style": "danger",
                "action_id": "reject_retrain",
                "value": thread_id,
            },
        ],
    }

    return [header, divider, top_drifted_section, shap_section, metrics_section, divider2, actions]


def build_no_drift_blocks(
    eval_result_dict: dict,
    production_version: str,
) -> list[dict]:
    """Build Block Kit blocks for a no-drift pipeline summary (per D-07).

    Returns a list of Block Kit block dicts with:
    1. Header block
    2. Metrics section (RMSE + MAE)
    3. Threshold confirmation section
    4. Production version section
    """
    # 1. Header block
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": ":white_check_mark: Pipeline Complete \u2014 No Drift",
        },
    }

    # 2. Metrics section
    rmse = eval_result_dict.get("rmse", "N/A")
    mae = eval_result_dict.get("mae", "N/A")
    rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
    mae_str = f"{mae:.4f}" if isinstance(mae, (int, float)) else str(mae)
    metrics_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Metrics:*\nRMSE: {rmse_str} | MAE: {mae_str}",
        },
    }

    # 3. Threshold confirmation section
    threshold_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": "All 3 drift signals below threshold",
        },
    }

    # 4. Production version section
    prod_section = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"*Production Model:* v{production_version}",
        },
    }

    return [header, metrics_section, threshold_section, prod_section]
