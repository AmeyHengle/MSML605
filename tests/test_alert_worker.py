"""Tests for alert_worker Slack posting.

Tests that alert_worker posts drift alerts and no-drift summaries to Slack via WebClient.
Uses mock WebClient to avoid actual Slack API calls.

NOTE: These tests are skipped until Slack alerting is re-wired into the agent
graph. The ``alert_worker`` function is still defined in ``ml605_agent.graph``
for that future work. See ``docs/SLACK_HITL_ROADMAP.md`` for restoration steps.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "Slack alert_worker is currently unwired from the graph. "
        "See docs/SLACK_HITL_ROADMAP.md to re-enable and unskip."
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_drift_state(tmp_path, report_file=True):
    """Build a PipelineState-like dict for drift scenario."""
    report_path = None
    if report_file:
        report_path = str(tmp_path / "report.html")
        Path(report_path).write_text("<html>report</html>", encoding="utf-8")

    return {
        "overall_drift": True,
        "drift_report": None,  # No real DriftReport object needed
        "shap_top_features": ["hour", "day_of_week", "month"],
        "eval_result": None,
        "report_path": report_path,
        "mlflow_run_id": None,
        "status": "running",
    }


def _make_no_drift_state(tmp_path, report_file=True):
    """Build a PipelineState-like dict for no-drift scenario."""
    report_path = None
    if report_file:
        report_path = str(tmp_path / "report_no_drift.html")
        Path(report_path).write_text("<html>no drift report</html>", encoding="utf-8")

    return {
        "overall_drift": False,
        "drift_report": None,
        "shap_top_features": [],
        "eval_result": None,
        "report_path": report_path,
        "mlflow_run_id": None,
        "status": "running",
    }


# ---------------------------------------------------------------------------
# Drift alert tests
# ---------------------------------------------------------------------------


class TestAlertWorkerDrift:
    """alert_worker posts drift alert to Slack."""

    def test_drift_alert_posted_to_slack(self, tmp_path):
        """chat_postMessage called with drift alert blocks when overall_drift=True."""
        from ml605_agent.graph import alert_worker

        state = _make_drift_state(tmp_path)
        mock_client = MagicMock()

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL_ID": "C123"}), \
             patch("slack_sdk.WebClient", return_value=mock_client):
            result = alert_worker(state)

        assert result.get("alert_sent") is True
        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args
        # channel should be C123
        channel = call_kwargs.kwargs.get("channel") or call_kwargs[1].get("channel")
        assert channel == "C123"
        # Blocks should be a non-empty list
        blocks_arg = call_kwargs.kwargs.get("blocks") or call_kwargs[1].get("blocks")
        assert isinstance(blocks_arg, list) and len(blocks_arg) > 0

    def test_drift_alert_uploads_report(self, tmp_path):
        """files_upload_v2 called with report_path when drift detected."""
        from ml605_agent.graph import alert_worker

        state = _make_drift_state(tmp_path)
        mock_client = MagicMock()

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL_ID": "C123"}), \
             patch("slack_sdk.WebClient", return_value=mock_client):
            result = alert_worker(state)

        assert result.get("alert_sent") is True
        mock_client.files_upload_v2.assert_called_once()
        upload_call = mock_client.files_upload_v2.call_args
        # Should include the report file path
        file_arg = upload_call.kwargs.get("file") or upload_call[1].get("file")
        assert file_arg == state["report_path"]


# ---------------------------------------------------------------------------
# No-drift summary tests
# ---------------------------------------------------------------------------


class TestAlertWorkerNoDrift:
    """alert_worker posts no-drift summary."""

    def test_no_drift_summary_posted(self, tmp_path):
        """chat_postMessage called with no-drift blocks when overall_drift=False."""
        from ml605_agent.graph import alert_worker

        state = _make_no_drift_state(tmp_path)
        mock_client = MagicMock()

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL_ID": "C456"}), \
             patch("slack_sdk.WebClient", return_value=mock_client), \
             patch("mlflow.tracking.MlflowClient") as mock_mlflow_client:
            mock_mlflow_client.return_value.get_latest_versions.return_value = []
            result = alert_worker(state)

        assert result.get("alert_sent") is True
        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args
        blocks_arg = call_kwargs.kwargs.get("blocks") or call_kwargs[1].get("blocks")
        assert isinstance(blocks_arg, list) and len(blocks_arg) > 0

    def test_no_drift_uploads_report(self, tmp_path):
        """files_upload_v2 called with report_path for no-drift run."""
        from ml605_agent.graph import alert_worker

        state = _make_no_drift_state(tmp_path)
        mock_client = MagicMock()

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL_ID": "C456"}), \
             patch("slack_sdk.WebClient", return_value=mock_client), \
             patch("mlflow.tracking.MlflowClient") as mock_mlflow_client:
            mock_mlflow_client.return_value.get_latest_versions.return_value = []
            result = alert_worker(state)

        assert result.get("alert_sent") is True
        mock_client.files_upload_v2.assert_called_once()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestAlertWorkerError:
    """alert_worker error handling."""

    def test_alert_worker_returns_alert_sent_true(self, tmp_path):
        """alert_worker returns {'alert_sent': True} on success."""
        from ml605_agent.graph import alert_worker

        state = _make_drift_state(tmp_path)
        mock_client = MagicMock()

        with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test", "SLACK_CHANNEL_ID": "C123"}), \
             patch("slack_sdk.WebClient", return_value=mock_client):
            result = alert_worker(state)

        assert result == {"alert_sent": True}

    def test_alert_worker_handles_missing_slack_token(self, tmp_path):
        """alert_worker returns {'alert_sent': False} gracefully when no SLACK_BOT_TOKEN."""
        from ml605_agent.graph import alert_worker

        state = _make_drift_state(tmp_path)

        # Clear Slack-related env vars
        env_without_token = {
            k: v for k, v in os.environ.items()
            if k not in ("SLACK_BOT_TOKEN", "SLACK_CHANNEL_ID")
        }

        with patch.dict(os.environ, env_without_token, clear=True):
            result = alert_worker(state)

        assert result.get("alert_sent") is False
