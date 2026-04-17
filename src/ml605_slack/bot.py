"""Slack Bolt App with Socket Mode, slash commands, and action handlers.

Per D-01: Socket Mode via slack-bolt.
Per D-02: Library is slack-bolt (not raw slack-sdk).
Per D-03: /ml605 run uses ack + background thread pattern.
Per D-08: HITL button handlers resume paused LangGraph via Command(resume=...).
Per D-09: Auto-reject timeout via threading.Timer.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ml605_slack.blocks import build_drift_alert_blocks, build_no_drift_blocks
from ml605_pipeline.registry import MODEL_NAME, transition_model_stage


# ---------------------------------------------------------------------------
# Module-level shared graph instance (with MemorySaver for HITL)
# ---------------------------------------------------------------------------

_graph_instance = None
_graph_checkpointer = None
_graph_lock = threading.Lock()


def _get_or_create_graph():
    """Get or create the shared graph instance with MemorySaver checkpointer.

    Thread-safe: uses lock to prevent race condition on initialization.
    """
    global _graph_instance, _graph_checkpointer
    with _graph_lock:
        if _graph_instance is None:
            from ml605_agent.graph import build_graph  # noqa: PLC0415
            _graph_checkpointer = MemorySaver()
            _graph_instance = build_graph(checkpointer=_graph_checkpointer)
    return _graph_instance


def start_hitl_timeout(thread_id: str, channel_id: str, timeout_minutes: int, client) -> dict:
    """Start a timer that auto-rejects if no human response within timeout_minutes (per D-09).

    Returns the pending_hitl dict entry for tracking.
    """
    def on_timeout():
        try:
            graph = _get_or_create_graph()
            config = {"configurable": {"thread_id": thread_id}}
            graph.invoke(Command(resume="reject_timeout"), config=config)
            client.chat_postMessage(
                channel=channel_id,
                text=(
                    f":warning: No response received within {timeout_minutes} minutes "
                    "— retrain rejected automatically."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            try:
                client.chat_postMessage(
                    channel=channel_id,
                    text=f":x: HITL timeout error: {exc}",
                )
            except Exception:  # noqa: BLE001
                pass

    timer = threading.Timer(timeout_minutes * 60, on_timeout)
    timer.daemon = True
    timer.start()
    return {"timer": timer, "channel": channel_id}


def _get_latest_report_path() -> str | None:
    """Find the most recent HTML report in reports/."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return None
    html_files = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(html_files[0]) if html_files else None


def _get_recent_runs(experiment_name: str = "agentic-pipeline", max_results: int = 5) -> list[dict]:
    """Query MLflow for last N runs from the given experiment (per D-11)."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=max_results,
    )
    results = []
    for run in runs:
        results.append({
            "timestamp": run.info.start_time,
            "rmse": run.data.metrics.get("rmse"),
            "overall_drift": run.data.metrics.get("overall_drift"),
            "model_version": run.data.params.get("new_model_version"),
        })
    return results


def _build_status_blocks() -> list[dict]:
    """Build Block Kit blocks for /ml605 status (per D-05)."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return [{"type": "section", "text": {"type": "mrkdwn", "text": "No Production model registered."}}]
        v = versions[0]
        run = client.get_run(v.run_id)
        rmse = run.data.metrics.get("rmse", "N/A")
        mae = run.data.metrics.get("mae", "N/A")
        return [
            {"type": "header", "text": {"type": "plain_text", "text": "Model Status"}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Model:* `{MODEL_NAME}` v{v.version}\n"
                        f"*Stage:* Production\n"
                        f"*RMSE:* {rmse}\n"
                        f"*MAE:* {mae}\n"
                        f"*Last Updated:* {v.last_updated_timestamp}"
                    ),
                },
            },
        ]
    except Exception as exc:
        return [{"type": "section", "text": {"type": "mrkdwn", "text": f"Error querying model status: {exc}"}}]


def _build_history_blocks() -> list[dict]:
    """Build Block Kit blocks for /ml605 history (per D-11)."""
    runs = _get_recent_runs()
    if not runs:
        return [{"type": "section", "text": {"type": "mrkdwn", "text": "No runs found in `agentic-pipeline` experiment."}}]
    lines = []
    for r in runs:
        drift_str = "Yes" if r.get("overall_drift") else "No"
        rmse_val = r.get("rmse")
        rmse_str = f"{rmse_val:.4f}" if rmse_val is not None else "N/A"
        version_str = r.get("model_version") or "-"
        lines.append(f"- *{r['timestamp']}* | RMSE: {rmse_str} | Drift: {drift_str} | Version: {version_str}")
    return [
        {"type": "header", "text": {"type": "plain_text", "text": "Last 5 Pipeline Runs"}},
        {"type": "section", "text": {"type": "mrkdwn", "text": "\n".join(lines)}},
    ]


def _run_pipeline_background(client, channel_id: str) -> None:
    """Run the full LangGraph pipeline in a background thread (per D-03).

    Posts results or error to the Slack channel when complete.
    Per D-10: all errors post to channel.
    """
    try:
        from ml605_agent.__main__ import ensure_mcp_server
        from ml605_agent.graph import build_graph
        from ml605_pipeline.config import load_config_from_env

        ensure_mcp_server()
        config = load_config_from_env()
        graph = build_graph()
        mlflow.set_experiment("agentic-pipeline")

        with mlflow.start_run(run_name="slack-triggered-run") as parent_run:
            initial_state = {
                "window_hours": config.window_hours,
                "mlflow_run_id": parent_run.info.run_id,
                "status": "running",
                "retrain_done": False,
            }
            result = graph.invoke(initial_state, config={"recursion_limit": 25})

        # Post results
        if result.get("overall_drift"):
            eval_dict = result["eval_result"].__dict__ if result.get("eval_result") else {}
            drift_report = result.get("drift_report")
            drifted = [
                {"feature": f.feature, "psi": f.psi}
                for f in (drift_report.feature_results[:3] if drift_report else [])
                if f.drift_detected
            ]
            blocks = build_drift_alert_blocks(
                overall_drift=True,
                drifted_features=drifted,
                shap_top_features=result.get("shap_top_features", []),
                eval_result_dict=eval_dict,
                thread_id=f"pipeline-{parent_run.info.run_id}",
            )
        else:
            eval_dict = result["eval_result"].__dict__ if result.get("eval_result") else {}
            blocks = build_no_drift_blocks(
                eval_result_dict=eval_dict,
                production_version=result.get("new_model_version", "current"),
            )
        client.chat_postMessage(channel=channel_id, blocks=blocks, text="Pipeline complete")

        # Upload report if available
        report_path = result.get("report_path")
        if report_path and Path(report_path).exists():
            client.files_upload_v2(
                channel=channel_id,
                file=report_path,
                title="Pipeline Report",
                initial_comment="Full HTML report attached.",
            )
    except Exception as exc:
        client.chat_postMessage(
            channel=channel_id,
            text=f":x: Pipeline failed: {exc}",
        )


def _handle_promote(respond) -> None:
    """Handle /ml605 promote (per D-05, SLACK-05)."""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(MODEL_NAME, stages=["Staging"])
        if not versions:
            respond(text="No model in Staging to promote.")
            return
        version = versions[0].version
        transition_model_stage(version, "Production")
        respond(text=f":rocket: Model v{version} promoted to Production.")
    except Exception as exc:
        respond(text=f":x: Promote failed: {exc}")


def _handle_retrain_background(client, channel_id: str) -> None:
    """Force retrain in background thread (per D-05)."""
    try:
        from ml605_agent.__main__ import ensure_mcp_server
        from ml605_agent.graph import build_graph
        from ml605_pipeline.config import load_config_from_env

        ensure_mcp_server()
        config = load_config_from_env()
        graph = build_graph()
        mlflow.set_experiment("agentic-pipeline")

        with mlflow.start_run(run_name="slack-forced-retrain") as parent_run:
            initial_state = {
                "window_hours": config.window_hours,
                "mlflow_run_id": parent_run.info.run_id,
                "status": "running",
                "retrain_done": False,
                "overall_drift": True,  # Force drift to trigger retrain path
            }
            result = graph.invoke(initial_state, config={"recursion_limit": 25})

        version = result.get("new_model_version", "unknown")
        client.chat_postMessage(
            channel=channel_id,
            text=f":white_check_mark: Forced retrain complete. New model version: {version}",
        )
    except Exception as exc:
        client.chat_postMessage(channel=channel_id, text=f":x: Retrain failed: {exc}")


def _handle_report(client, channel_id: str) -> None:
    """Handle /ml605 report — upload latest HTML report (per D-04)."""
    report_path = _get_latest_report_path()
    if not report_path:
        client.chat_postMessage(channel=channel_id, text="No reports found in reports/ directory.")
        return
    client.files_upload_v2(
        channel=channel_id,
        file=report_path,
        title="Latest Pipeline Report",
        filename=Path(report_path).name,
    )


def handle_ml605_command(ack, respond, command, client) -> None:
    """Single slash command dispatcher for /ml605.

    This is a module-level function so it can be imported and tested directly.
    Per Pitfall 7: register as a single /ml605 handler; dispatch on first word of text.
    Per D-03: ack() is always called before any I/O.
    """
    text = (command.get("text") or "").strip()
    parts = text.split(maxsplit=1)
    subcommand = parts[0].lower() if parts else "help"
    channel_id = command["channel_id"]

    if subcommand == "run":
        ack(":hourglass_flowing_sand: Pipeline started... I'll post results when done.")
        threading.Thread(
            target=_run_pipeline_background,
            args=(client, channel_id),
            daemon=True,
        ).start()
    elif subcommand == "status":
        ack()
        respond(blocks=_build_status_blocks())
    elif subcommand == "promote":
        ack()
        _handle_promote(respond)
    elif subcommand == "retrain":
        ack(":gear: Forced retrain started... I'll post results when done.")
        threading.Thread(
            target=_handle_retrain_background,
            args=(client, channel_id),
            daemon=True,
        ).start()
    elif subcommand == "report":
        ack()
        _handle_report(client, channel_id)
    elif subcommand == "history":
        ack()
        respond(blocks=_build_history_blocks())
    else:
        ack()
        respond(text="Available commands: `run`, `status`, `promote`, `retrain`, `report`, `history`")


def create_app() -> App:
    """Create and configure the Slack Bolt App with all handlers.

    Requires env vars: SLACK_BOT_TOKEN.
    """
    app = App(token=os.environ.get("SLACK_BOT_TOKEN", "xoxb-placeholder"))
    app.command("/ml605")(handle_ml605_command)

    # --- HITL Button Handlers (per D-08, D-09) ---

    # Pending HITL decisions: thread_id -> {"timer": Timer, "channel": str}
    _pending_hitl: dict[str, dict] = {}

    @app.action("approve_retrain")
    def handle_approve(ack, body, client):
        """Resume graph with 'approve' decision (per HITL-01)."""
        ack()
        try:
            thread_id = body["actions"][0]["value"]
            channel_id = body["channel"]["id"]

            # Cancel timeout timer if pending
            if thread_id in _pending_hitl:
                _pending_hitl[thread_id]["timer"].cancel()
                del _pending_hitl[thread_id]

            # Resume graph with approve decision
            graph = _get_or_create_graph()
            config = {"configurable": {"thread_id": thread_id}}
            graph.invoke(Command(resume="approve"), config=config)

            client.chat_postMessage(
                channel=channel_id,
                text=":white_check_mark: Retrain approved. Pipeline continuing...",
            )
        except Exception as exc:  # noqa: BLE001
            try:
                client.chat_postMessage(
                    channel=body.get("channel", {}).get("id", ""),
                    text=f":x: Error resuming pipeline: {exc}",
                )
            except Exception:  # noqa: BLE001
                pass

    @app.action("reject_retrain")
    def handle_reject(ack, body, client):
        """Resume graph with 'reject' decision (per HITL-01)."""
        ack()
        try:
            thread_id = body["actions"][0]["value"]
            channel_id = body["channel"]["id"]

            # Cancel timeout timer if pending
            if thread_id in _pending_hitl:
                _pending_hitl[thread_id]["timer"].cancel()
                del _pending_hitl[thread_id]

            # Resume graph with reject decision
            graph = _get_or_create_graph()
            config = {"configurable": {"thread_id": thread_id}}
            graph.invoke(Command(resume="reject"), config=config)

            client.chat_postMessage(
                channel=channel_id,
                text=":no_entry_sign: Retrain rejected. Pipeline completing without retraining.",
            )
        except Exception as exc:  # noqa: BLE001
            try:
                client.chat_postMessage(
                    channel=body.get("channel", {}).get("id", ""),
                    text=f":x: Error resuming pipeline: {exc}",
                )
            except Exception:  # noqa: BLE001
                pass

    return app
