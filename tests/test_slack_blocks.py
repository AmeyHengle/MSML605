"""Unit tests for Block Kit message builders (blocks.py)."""
import pytest

from ml605_slack.blocks import build_drift_alert_blocks, build_no_drift_blocks


SAMPLE_DRIFTED = [
    {"feature": "hour", "psi": 0.2500},
    {"feature": "day_of_week", "psi": 0.1800},
    {"feature": "is_weekend", "psi": 0.1500},
]
SAMPLE_SHAP = ["hour", "day_of_week", "month"]
SAMPLE_EVAL = {"rmse": 15.2345, "mae": 10.1234, "r2": 0.85, "mape": 12.5}
SAMPLE_THREAD_ID = "pipeline-abc123"


class TestDriftAlertBlocks:
    """SLACK-01: Drift alert Block Kit message."""

    def test_drift_alert_has_header_with_verdict(self):
        """Header section contains 'DRIFT DETECTED' text."""
        blocks = build_drift_alert_blocks(
            overall_drift=True,
            drifted_features=SAMPLE_DRIFTED,
            shap_top_features=SAMPLE_SHAP,
            eval_result_dict=SAMPLE_EVAL,
            thread_id=SAMPLE_THREAD_ID,
        )
        assert blocks[0]["type"] == "header"
        assert "DRIFT DETECTED" in blocks[0]["text"]["text"]

    def test_drift_alert_has_top_drifted_features(self):
        """Section lists top 3 drifted features with PSI scores."""
        blocks = build_drift_alert_blocks(
            overall_drift=True,
            drifted_features=SAMPLE_DRIFTED,
            shap_top_features=SAMPLE_SHAP,
            eval_result_dict=SAMPLE_EVAL,
            thread_id=SAMPLE_THREAD_ID,
        )
        # Find a section block containing "Top Drifted Features"
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section"
        ]
        combined = "\n".join(section_texts)
        assert "Top Drifted Features" in combined
        assert "hour" in combined
        assert "0.2500" in combined

    def test_drift_alert_has_shap_top_features(self):
        """Section lists SHAP top-3 feature names."""
        blocks = build_drift_alert_blocks(
            overall_drift=True,
            drifted_features=SAMPLE_DRIFTED,
            shap_top_features=SAMPLE_SHAP,
            eval_result_dict=SAMPLE_EVAL,
            thread_id=SAMPLE_THREAD_ID,
        )
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section"
        ]
        combined = "\n".join(section_texts)
        assert "SHAP Top Features" in combined
        assert "hour" in combined
        assert "day_of_week" in combined

    def test_drift_alert_has_action_buttons(self):
        """HITL-01: Actions block has Approve Retrain and Reject buttons."""
        blocks = build_drift_alert_blocks(
            overall_drift=True,
            drifted_features=SAMPLE_DRIFTED,
            shap_top_features=SAMPLE_SHAP,
            eval_result_dict=SAMPLE_EVAL,
            thread_id=SAMPLE_THREAD_ID,
        )
        action_blocks = [b for b in blocks if b["type"] == "actions"]
        assert len(action_blocks) >= 1
        elements = action_blocks[0]["elements"]
        action_ids = [e["action_id"] for e in elements]
        assert "approve_retrain" in action_ids
        assert "reject_retrain" in action_ids

    def test_drift_alert_buttons_have_thread_id(self):
        """Button value fields contain the thread_id for graph resume."""
        blocks = build_drift_alert_blocks(
            overall_drift=True,
            drifted_features=SAMPLE_DRIFTED,
            shap_top_features=SAMPLE_SHAP,
            eval_result_dict=SAMPLE_EVAL,
            thread_id=SAMPLE_THREAD_ID,
        )
        action_blocks = [b for b in blocks if b["type"] == "actions"]
        elements = action_blocks[0]["elements"]
        for elem in elements:
            assert elem["value"] == SAMPLE_THREAD_ID


class TestNoDriftBlocks:
    """SLACK-02: No-drift pipeline summary."""

    def test_no_drift_has_metrics(self):
        """RMSE and MAE metrics appear in message."""
        blocks = build_no_drift_blocks(
            eval_result_dict=SAMPLE_EVAL,
            production_version="7",
        )
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section"
        ]
        combined = "\n".join(section_texts)
        assert "RMSE" in combined
        assert "MAE" in combined
        assert "15.2345" in combined

    def test_no_drift_has_threshold_confirmation(self):
        """Message contains 'All 3 drift signals below threshold'."""
        blocks = build_no_drift_blocks(
            eval_result_dict=SAMPLE_EVAL,
            production_version="7",
        )
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section"
        ]
        combined = "\n".join(section_texts)
        assert "All 3 drift signals below threshold" in combined

    def test_no_drift_has_production_version(self):
        """Message contains Production model version string."""
        blocks = build_no_drift_blocks(
            eval_result_dict=SAMPLE_EVAL,
            production_version="7",
        )
        section_texts = [
            b["text"]["text"]
            for b in blocks
            if b["type"] == "section"
        ]
        combined = "\n".join(section_texts)
        assert "7" in combined
