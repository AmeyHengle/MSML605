"""Integration tests for HITL interrupt/resume flow."""
import pytest


class TestHITLInterrupt:
    """HITL-02: LangGraph interrupt() pauses graph."""

    def test_graph_pauses_at_hitl_node_when_drift_detected(self):
        """Graph invoke returns __interrupt__ when overall_drift=True."""
        pytest.fail("RED: not implemented")

    def test_graph_skips_hitl_when_no_drift(self):
        """Graph proceeds to alert_worker without pausing when overall_drift=False."""
        pytest.fail("RED: not implemented")

    def test_approve_resumes_graph_to_retrain(self):
        """Command(resume='approve') continues graph through retrain_worker."""
        pytest.fail("RED: not implemented")

    def test_reject_resumes_graph_to_alert(self):
        """Command(resume='reject') skips retrain and goes to alert_worker."""
        pytest.fail("RED: not implemented")


class TestHITLTimeout:
    """HITL timeout auto-reject."""

    def test_timeout_auto_rejects(self):
        """After timeout, graph resumes with reject_timeout decision."""
        pytest.fail("RED: not implemented")


class TestHITLLogging:
    """HITL-03: Approval/rejection logged to MLflow."""

    def test_hitl_decision_logged_to_mlflow(self):
        """mlflow.log_param('hitl_decision', ...) called with decision value."""
        pytest.fail("RED: not implemented")

    def test_hitl_mtta_logged_to_mlflow(self):
        """mlflow.log_metric('mtta_seconds', ...) called with elapsed time."""
        pytest.fail("RED: not implemented")
