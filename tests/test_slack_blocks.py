"""Unit tests for Block Kit message builders (blocks.py)."""
import pytest


class TestDriftAlertBlocks:
    """SLACK-01: Drift alert Block Kit message."""

    def test_drift_alert_has_header_with_verdict(self):
        """Header section contains 'DRIFT DETECTED' text."""
        pytest.fail("RED: not implemented")

    def test_drift_alert_has_top_drifted_features(self):
        """Section lists top 3 drifted features with PSI scores."""
        pytest.fail("RED: not implemented")

    def test_drift_alert_has_shap_top_features(self):
        """Section lists SHAP top-3 feature names."""
        pytest.fail("RED: not implemented")

    def test_drift_alert_has_action_buttons(self):
        """HITL-01: Actions block has Approve Retrain and Reject buttons."""
        pytest.fail("RED: not implemented")

    def test_drift_alert_buttons_have_thread_id(self):
        """Button value fields contain the thread_id for graph resume."""
        pytest.fail("RED: not implemented")


class TestNoDriftBlocks:
    """SLACK-02: No-drift pipeline summary."""

    def test_no_drift_has_metrics(self):
        """RMSE and MAE metrics appear in message."""
        pytest.fail("RED: not implemented")

    def test_no_drift_has_threshold_confirmation(self):
        """Message contains 'All 3 drift signals below threshold'."""
        pytest.fail("RED: not implemented")

    def test_no_drift_has_production_version(self):
        """Message contains Production model version string."""
        pytest.fail("RED: not implemented")
