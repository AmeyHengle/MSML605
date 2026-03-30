"""Tests for ml605_agent graph topology and routing.

Graph topology tests, routing function unit tests, and mocked full-pipeline runs.
"""
from __future__ import annotations

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Graph compile test
# ---------------------------------------------------------------------------


def test_graph_compiles() -> None:
    """build_graph() returns a compiled graph object with invoke and ainvoke methods."""
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
        "alert_worker",
        "error_handler",
    }
    for name in expected:
        assert name in nodes, f"Node '{name}' missing from graph. Got: {list(nodes.keys())}"


# ---------------------------------------------------------------------------
# Routing function unit tests (pure Python, no LangGraph invocation)
# ---------------------------------------------------------------------------


def test_routing_no_drift() -> None:
    """route_after_drift returns 'report_worker' when overall_drift=False."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "running", "overall_drift": False})
    assert result == "report_worker"


def test_routing_with_drift() -> None:
    """route_after_drift returns 'retrain_worker' when overall_drift=True and retrain_done=False."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "running", "overall_drift": True, "retrain_done": False})
    assert result == "retrain_worker"


def test_routing_retrain_done() -> None:
    """route_after_drift returns 'report_worker' when retrain_done=True (prevents second retrain)."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "running", "overall_drift": True, "retrain_done": True})
    assert result == "report_worker"


def test_routing_error() -> None:
    """route_after_drift returns 'error_handler' when status='error'."""
    from ml605_agent.graph import route_after_drift

    result = route_after_drift({"status": "error"})
    assert result == "error_handler"


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
    """Graph run with overall_drift=False reaches alert_worker (alert_sent=False).

    Note: Workers are mocked to return only scalar/primitive state values to avoid
    DataFrame serialization issues with the MemorySaver checkpointer.
    """
    from ml605_agent import graph as graph_module
    from ml605_agent.graph import build_graph

    # Return only serializable primitives — MemorySaver cannot serialize DataFrames
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
    monkeypatch.setattr(
        graph_module,
        "alert_worker",
        lambda s: {"alert_sent": False, "status": "complete"},
    )

    g = build_graph()
    result = g.invoke(
        {"window_hours": 6, "status": "running"},
        config={"configurable": {"thread_id": "test-no-drift"}, "recursion_limit": 25},
    )
    assert result["alert_sent"] is False


def test_full_pipeline_with_drift(monkeypatch) -> None:
    """Graph run with overall_drift=True triggers retrain path; retrain_done=True in final state.

    Note: Workers are mocked to return only scalar/primitive state values to avoid
    DataFrame serialization issues with the MemorySaver checkpointer.
    """
    from ml605_agent import graph as graph_module
    from ml605_agent.graph import build_graph

    call_count = {"test": 0}

    def fake_test_worker(s):
        call_count["test"] += 1
        return {"eval_result": None}

    def fake_drift_worker(s):
        # First pass: drift detected; after retrain retrain_done is True so route to report
        return {"drift_report": None, "overall_drift": True}

    def fake_retrain_worker(s):
        return {"new_model_version": "2", "retrain_done": True}

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
    monkeypatch.setattr(
        graph_module,
        "alert_worker",
        lambda s: {"alert_sent": False, "status": "complete"},
    )

    g = build_graph()
    result = g.invoke(
        {"window_hours": 6, "status": "running", "retrain_done": False},
        config={"configurable": {"thread_id": "test-with-drift"}, "recursion_limit": 25},
    )
    assert result.get("retrain_done") is True


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
    # error_handler sets status=error and preserves error field
    assert result.get("status") == "error"
    assert result.get("error") is not None
