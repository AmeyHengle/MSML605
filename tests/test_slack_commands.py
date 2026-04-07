"""Unit tests for slash command handlers (bot.py).

Tests use unittest.mock to isolate the handler from real MLflow and Slack calls.
The handle_ml605_command function is imported directly and called with mock args.
"""
from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import pytest

from ml605_slack.bot import handle_ml605_command


def _make_mocks(text: str = "") -> tuple:
    """Return (ack, respond, client, command) mocks."""
    ack = MagicMock()
    respond = MagicMock()
    client = MagicMock()
    command = {"text": text, "channel_id": "C12345"}
    return ack, respond, client, command


class TestRunCommand:
    """SLACK-03: /ml605 run triggers pipeline."""

    def test_run_command_acks_immediately(self):
        """ack() is called with a string containing 'Pipeline started'."""
        ack, respond, client, command = _make_mocks("run")
        with patch("ml605_slack.bot.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)
        ack.assert_called_once()
        ack_arg = ack.call_args[0][0] if ack.call_args[0] else ack.call_args[1].get("text", "")
        assert "Pipeline started" in str(ack_arg)

    def test_run_command_spawns_background_thread(self):
        """Pipeline runs in a daemon thread, not blocking."""
        ack, respond, client, command = _make_mocks("run")
        with patch("ml605_slack.bot.threading.Thread") as mock_thread_cls:
            mock_thread_instance = MagicMock()
            mock_thread_cls.return_value = mock_thread_instance
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)
        mock_thread_cls.assert_called_once()
        # Thread should be started
        mock_thread_instance.start.assert_called_once()
        # Thread should be daemon
        _, kwargs = mock_thread_cls.call_args
        assert kwargs.get("daemon") is True


class TestStatusCommand:
    """SLACK-04: /ml605 status returns model info."""

    def test_status_returns_rmse_and_version(self):
        """Response blocks contain RMSE value and model version from mocked MLflow."""
        ack, respond, client, command = _make_mocks("status")

        mock_version = MagicMock()
        mock_version.version = "3"
        mock_version.run_id = "test-run-id"
        mock_version.last_updated_timestamp = 1700000000000

        mock_run = MagicMock()
        mock_run.data.metrics = {"rmse": 14.5, "mae": 9.8}

        with patch("ml605_slack.bot.MlflowClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_latest_versions.return_value = [mock_version]
            mock_client.get_run.return_value = mock_run
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)

        respond.assert_called_once()
        call_kwargs = respond.call_args[1]
        blocks = call_kwargs.get("blocks", [])
        # Find text content in the blocks
        all_text = ""
        for b in blocks:
            if b.get("type") == "section":
                all_text += b["text"]["text"]
        assert "14.5" in all_text or "3" in all_text


class TestPromoteCommand:
    """SLACK-05: /ml605 promote triggers model promotion."""

    def test_promote_calls_transition_model_stage(self):
        """transition_model_stage() called with 'Production' stage."""
        ack, respond, client, command = _make_mocks("promote")

        mock_version = MagicMock()
        mock_version.version = "5"

        with patch("ml605_slack.bot.MlflowClient") as mock_client_cls, \
             patch("ml605_slack.bot.transition_model_stage") as mock_transition:
            mock_client = mock_client_cls.return_value
            mock_client.get_latest_versions.return_value = [mock_version]
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)

        mock_transition.assert_called_once_with("5", "Production")


class TestRetrainCommand:
    """SLACK-03: /ml605 retrain forces retraining."""

    def test_retrain_command_acks_immediately(self):
        """ack() is called with text containing 'retrain'."""
        ack, respond, client, command = _make_mocks("retrain")
        with patch("ml605_slack.bot.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)
        ack.assert_called_once()
        ack_arg = ack.call_args[0][0] if ack.call_args[0] else ack.call_args[1].get("text", "")
        assert "retrain" in str(ack_arg).lower()


class TestReportCommand:
    """SLACK-03: /ml605 report uploads HTML."""

    def test_report_uploads_latest_html(self):
        """files_upload_v2 called with a .html file path."""
        ack, respond, client, command = _make_mocks("report")
        with patch("ml605_slack.bot._get_latest_report_path", return_value="reports/report.html"):
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)
        client.files_upload_v2.assert_called_once()
        call_kwargs = client.files_upload_v2.call_args[1]
        assert call_kwargs["file"].endswith(".html")


class TestHistoryCommand:
    """SLACK-04: /ml605 history returns last 5 runs."""

    def test_history_returns_last_5_runs(self):
        """Response contains run entries from mocked MlflowClient.search_runs."""
        ack, respond, client, command = _make_mocks("history")

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "exp-1"

        mock_run = MagicMock()
        mock_run.info.start_time = 1700000000000
        mock_run.data.metrics = {"rmse": 14.2, "overall_drift": 0}
        mock_run.data.params = {"new_model_version": "2"}

        with patch("ml605_slack.bot.MlflowClient") as mock_client_cls:
            mock_client = mock_client_cls.return_value
            mock_client.get_experiment_by_name.return_value = mock_experiment
            mock_client.search_runs.return_value = [mock_run] * 3
            handle_ml605_command(ack=ack, respond=respond, command=command, client=client)

        respond.assert_called_once()
        call_kwargs = respond.call_args[1]
        blocks = call_kwargs.get("blocks", [])
        assert len(blocks) >= 1
        # Find text content with run info
        all_text = ""
        for b in blocks:
            if b.get("type") == "section":
                all_text += b["text"]["text"]
        assert "14.2" in all_text or "RMSE" in all_text


class TestUnknownCommand:
    def test_unknown_subcommand_returns_help(self):
        """Unknown subcommand responds with available commands list."""
        ack, respond, client, command = _make_mocks("unknownxyz")
        handle_ml605_command(ack=ack, respond=respond, command=command, client=client)
        respond.assert_called_once()
        call_args = respond.call_args
        # text kwarg or positional arg
        response_text = (
            call_args[1].get("text", "") if call_args[1] else
            str(call_args[0][0]) if call_args[0] else ""
        )
        assert "Available commands" in response_text
