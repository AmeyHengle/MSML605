"""Tests for ml605_agent graph topology and routing.

Graph topology tests, routing function unit tests, and mocked full-pipeline runs.

Note: Slack alerting and HITL approval have been removed from the active graph
(see docs/SLACK_HITL_ROADMAP.md). Tests for those behaviours live in
tests/test_alert_worker.py and tests/test_slack_hitl.py and are currently
skipped at module level until Slack + HITL are re-wired.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Graph compile test
# ---------------------------------------------------------------------------


def test_graph_compiles() -> None:
    """build_graph() returns a compiled graph with the active node set."""
    from ml605_agent.graph import build_graph

    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")
    assert hasattr(graph, "ainvoke")
    nodes = graph.get_graph().nodes
    expected = {
        "fetch_worker",
        "feature_worker",
        "test_worker",
        "drift_worker",
        "retrain_worker",
        "report_worker",
        "error_handler",
    }
    for name in expected:
        assert name in nodes, f"Node '{name}' missing from graph. Got: {list(nodes.keys())}"

    # HITL + alert nodes must NOT be wired into the active graph right now.
    for removed in ("hitl_decision_node", "alert_worker"):
        assert removed not in nodes, (
            f"Node '{removed}' should be unwired while Slack/HITL is disabled"
        )


# ---------------------------------------------------------------------------
# Routing function unit tests (pure Python, no LangGraph invocation)
# ---------------------------------------------------------------------------


def test_routing_no_drift() -> None:
    """route_after_drift returns 'report_worker' when overall_drift=False."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "running", "overall_drift": False})
    assert result == "report_worker"


def test_routing_with_drift() -> None:
    """route_after_drift returns 'report_worker' even when drift=True.

    The retrain decision now happens in route_after_report, not here.
    """
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "running", "overall_drift": True, "retrain_done": False})
    assert result == "report_worker"


def test_routing_error() -> None:
    """route_after_drift returns 'error_handler' when status='error'."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "error"})
    assert result == "error_handler"


def test_route_after_report_drift_triggers_retrain() -> None:
    """drift + not retrain_done → retrain_worker."""
    from ml605_agent.graph import route_after_report

    assert route_after_report(
        {"status": "running", "overall_drift": True, "retrain_done": False}
    ) == "retrain_worker"


def test_route_after_report_no_drift_ends() -> None:
    """No drift → end (no HITL, no alert, no retrain)."""
    from ml605_agent.graph import route_after_report

    assert route_after_report(
        {"status": "running", "overall_drift": False, "retrain_done": False}
    ) == "end"


def test_route_after_report_retrain_done_ends() -> None:
    """Drift still fires but retrain already ran this invocation → end."""
    from ml605_agent.graph import route_after_report

    assert route_after_report(
        {"status": "running", "overall_drift": True, "retrain_done": True}
    ) == "end"


def test_route_after_report_error() -> None:
    """status=error → error_handler."""
    from ml605_agent.graph import route_after_report

    assert route_after_report({"status": "error"}) == "error_handler"


def test_routing_after_retrain() -> None:
    """route_after_retrain returns 'test_worker' (back-edge confirmed)."""
    from ml605_agent.graph import route_after_retrain

    result = route_after_retrain({"status": "running"})
    assert result == "test_worker"


def test_routing_after_retrain_error() -> None:
    """route_after_retrain returns 'error_handler' when status='error'."""
    from ml605_agent.graph import route_after_retrain

    result = route_after_retrain({"status": "error"})
    assert result == "error_handler"


# ---------------------------------------------------------------------------
# Full pipeline mocked invocation tests
# ---------------------------------------------------------------------------


def test_full_pipeline_no_drift(monkeypatch) -> None:
    """No drift: pipeline runs end-to-end without retraining and terminates."""
    from ml605_agent import graph as graph_module
    from ml605_agent.graph import build_graph

    monkeypatch.setattr(
        graph_module,
        "fetch_worker",
        lambda s: {"factors": {}, "rows_fetched": 1},
    )
    monkeypatch.setattr(
        graph_module,
        "feature_worker",
        lambda s: {"feature_cols": ["hour"]},
    )
    monkeypatch.setattr(graph_module, "test_worker", lambda s: {"eval_result": None})
    monkeypatch.setattr(
        graph_module,
        "drift_worker",
        lambda s: {"drift_report": None, "overall_drift": False},
    )
    monkeypatch.setattr(graph_module, "report_worker", lambda s: {"report_path": None})

    g = build_graph()
    result = g.invoke(
        {"window_hours": 6, "status": "running", "retrain_done": False},
        config={"configurable": {"thread_id": "test-no-drift"}, "recursion_limit": 25},
    )
    assert result.get("retrain_done") is False
    assert result.get("overall_drift") is False


def test_full_pipeline_with_drift(monkeypatch) -> None:
    """Drift: pipeline auto-retrains (no HITL), sets retrain_done, then ends.

    The back-edge retrain → test_worker must run exactly once: retrain_done=True
    causes route_after_report to route to END on the second pass.
    """
    from ml605_agent import graph as graph_module
    from ml605_agent.graph import build_graph

    call_count = {"test": 0}

    def fake_test_worker(s):
        call_count["test"] += 1
        return {"eval_result": None}

    def fake_drift_worker(s):
        # Always reports drift; retrain_done is what stops the loop.
        return {"drift_report": None, "overall_drift": True}

    def fake_retrain_worker(s):
        return {
            "new_model_version": "2",
            "retrain_done": True,
            "model_stage": "Production",
        }

    monkeypatch.setattr(
        graph_module,
        "fetch_worker",
        lambda s: {"factors": {}, "rows_fetched": 1},
    )
    monkeypatch.setattr(
        graph_module,
        "feature_worker",
        lambda s: {"feature_cols": ["hour"]},
    )
    monkeypatch.setattr(graph_module, "test_worker", fake_test_worker)
    monkeypatch.setattr(graph_module, "drift_worker", fake_drift_worker)
    monkeypatch.setattr(graph_module, "retrain_worker", fake_retrain_worker)
    monkeypatch.setattr(graph_module, "report_worker", lambda s: {"report_path": None})

    g = build_graph()
    result = g.invoke(
        {"window_hours": 6, "status": "running", "retrain_done": False},
        config={"configurable": {"thread_id": "test-with-drift"}, "recursion_limit": 25},
    )
    assert result.get("retrain_done") is True
    assert result.get("model_stage") == "Production"
    # test_worker should have run twice: once before drift, once on the retrain back-edge.
    assert call_count["test"] == 2


def test_error_handler_routing(monkeypatch) -> None:
    """Graph routes to error_handler when fetch_worker returns status='error'."""
    from ml605_agent import graph as graph_module
    from ml605_agent.graph import build_graph

    monkeypatch.setattr(
        graph_module,
        "fetch_worker",
        lambda s: {"status": "error", "error": "simulated fetch failure"},
    )

    g = build_graph()
    initial_state = {
        "window_hours": 6,
        "status": "running",
    }
    result = g.invoke(
        initial_state,
        config={"configurable": {"thread_id": "test-error-routing"}, "recursion_limit": 25},
    )
    assert result.get("status") == "error"
    assert result.get("error") is not None
