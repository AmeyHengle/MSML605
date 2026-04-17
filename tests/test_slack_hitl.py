"""Integration tests for HITL interrupt/resume flow.

Tests the HITL decision node that pauses LangGraph for human approval when drift detected.
Uses a minimal test graph (not the full pipeline) to isolate HITL behavior.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from ml605_agent.graph import hitl_decision_node, route_after_hitl
from ml605_agent.state import PipelineState


# ---------------------------------------------------------------------------
# Minimal test graph for HITL isolation
# ---------------------------------------------------------------------------


def _build_test_graph():
    """Minimal graph: START -> report -> hitl_decision -> [retrain|alert] -> END.

    Uses MemorySaver to support interrupt/resume.
    Does NOT use the full pipeline graph.
    """
    checkpointer = MemorySaver()

    def mock_report(state):
        return {"report_path": "/tmp/test.html", "shap_top_features": ["hour", "day_of_week"]}

    def mock_retrain(state):
        return {"retrain_done": True, "new_model_version": "99"}

    def mock_alert(state):
        return {"alert_sent": True}

    builder = StateGraph(PipelineState)
    builder.add_node("report_worker", mock_report)
    builder.add_node("hitl_decision_node", hitl_decision_node)
    builder.add_node("retrain_worker", mock_retrain)
    builder.add_node("alert_worker", mock_alert)

    builder.add_edge(START, "report_worker")
    builder.add_edge("report_worker", "hitl_decision_node")
    builder.add_conditional_edges(
        "hitl_decision_node",
        route_after_hitl,
        {
            "retrain_worker": "retrain_worker",
            "alert_worker": "alert_worker",
            "error_handler": END,
        },
    )
    builder.add_edge("retrain_worker", "alert_worker")
    builder.add_edge("alert_worker", END)

    return builder.compile(checkpointer=checkpointer), checkpointer


# ---------------------------------------------------------------------------
# HITL Interrupt tests (HITL-02)
# ---------------------------------------------------------------------------


class TestHITLInterrupt:
    """HITL-02: LangGraph interrupt() pauses graph."""

    def test_graph_pauses_at_hitl_node_when_drift_detected(self):
        """Graph invoke returns state with __interrupt__ key when overall_drift=True."""
        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-pause-drift"}}

        initial_state = {
            "overall_drift": True,
            "retrain_done": False,
            "status": "running",
        }

        # graph.invoke() will return the state at the interrupt point
        result = graph.invoke(initial_state, config=config)

        # Check that graph is actually paused - get_state shows pending tasks
        graph_state = graph.get_state(config)
        assert graph_state.tasks, "Graph should be paused (tasks non-empty means interrupt pending)"

    def test_graph_skips_hitl_when_no_drift(self):
        """Graph proceeds to alert_worker without pausing when overall_drift=False."""
        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-no-drift-skip"}}

        initial_state = {
            "overall_drift": False,
            "retrain_done": False,
            "status": "running",
        }

        result = graph.invoke(initial_state, config=config)

        # Should complete without interruption
        graph_state = graph.get_state(config)
        assert not graph_state.tasks, "Graph should be complete (no pending tasks)"
        assert result.get("alert_sent") is True

    def test_approve_resumes_graph_to_retrain(self):
        """Command(resume='approve') continues graph through retrain_worker."""
        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-approve-resume"}}

        # Start graph - will pause at HITL
        graph.invoke(
            {"overall_drift": True, "retrain_done": False, "status": "running"},
            config=config,
        )

        # Verify paused
        assert graph.get_state(config).tasks, "Graph must be paused before resume"

        # Resume with approve
        result = graph.invoke(Command(resume="approve"), config=config)

        # Should have gone through retrain_worker
        assert result.get("retrain_done") is True
        assert result.get("new_model_version") is not None

    def test_reject_resumes_graph_to_alert(self):
        """Command(resume='reject') skips retrain and routes to alert_worker."""
        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-reject-resume"}}

        # Start graph - will pause at HITL
        graph.invoke(
            {"overall_drift": True, "retrain_done": False, "status": "running"},
            config=config,
        )

        # Verify paused
        assert graph.get_state(config).tasks, "Graph must be paused before resume"

        # Resume with reject
        result = graph.invoke(Command(resume="reject"), config=config)

        # Should have gone to alert_worker (not retrain)
        assert result.get("alert_sent") is True
        assert not result.get("retrain_done"), "retrain_done should be falsy when rejected"


# ---------------------------------------------------------------------------
# HITL Timeout test
# ---------------------------------------------------------------------------


class TestHITLTimeout:
    """HITL timeout auto-reject."""

    def test_timeout_auto_rejects(self):
        """After timeout, graph resumes with reject_timeout - same as reject path."""
        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-timeout-rejects"}}

        # Start graph - will pause at HITL
        graph.invoke(
            {"overall_drift": True, "retrain_done": False, "status": "running"},
            config=config,
        )

        # Simulate timeout by resuming with reject_timeout
        result = graph.invoke(Command(resume="reject_timeout"), config=config)

        # reject_timeout should go to alert_worker (same as reject)
        assert result.get("alert_sent") is True
        assert not result.get("retrain_done"), "No retrain on timeout"


# ---------------------------------------------------------------------------
# HITL MLflow logging tests (HITL-03)
# ---------------------------------------------------------------------------


class TestHITLLogging:
    """HITL-03: Approval/rejection logged to MLflow."""

    def test_hitl_decision_logged_to_mlflow(self):
        """mlflow.log_param('hitl_decision', ...) called with decision value."""
        import mlflow

        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-logging-param"}}

        # Start and pause
        graph.invoke(
            {
                "overall_drift": True,
                "retrain_done": False,
                "status": "running",
                "mlflow_run_id": "test-run-123",
            },
            config=config,
        )

        # Resume and check MLflow logging
        with patch.object(mlflow, "log_param") as mock_log_param, \
             patch.object(mlflow, "log_metric") as mock_log_metric, \
             patch.object(mlflow, "start_run") as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            graph.invoke(Command(resume="approve"), config=config)

            # Verify mlflow.log_param was called with hitl_decision
            param_calls_str = str(mock_log_param.call_args_list)
            assert "hitl_decision" in param_calls_str, (
                f"mlflow.log_param('hitl_decision', ...) not called. Calls: {param_calls_str}"
            )

    def test_hitl_mtta_logged_to_mlflow(self):
        """mlflow.log_metric('mtta_seconds', ...) called with elapsed time."""
        import mlflow

        graph, _ = _build_test_graph()
        config = {"configurable": {"thread_id": "test-logging-metric"}}

        # Start and pause
        graph.invoke(
            {
                "overall_drift": True,
                "retrain_done": False,
                "status": "running",
                "mlflow_run_id": "test-run-456",
            },
            config=config,
        )

        # Resume and check MLflow metric logging
        with patch.object(mlflow, "log_param") as mock_log_param, \
             patch.object(mlflow, "log_metric") as mock_log_metric, \
             patch.object(mlflow, "start_run") as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
            mock_start_run.return_value.__exit__ = MagicMock(return_value=False)

            graph.invoke(Command(resume="reject"), config=config)

            # Verify mlflow.log_metric was called with mtta_seconds
            metric_calls_str = str(mock_log_metric.call_args_list)
            assert "mtta_seconds" in metric_calls_str, (
                f"mlflow.log_metric('mtta_seconds', ...) not called. Calls: {metric_calls_str}"
            )
