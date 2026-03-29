"""Stub tests for ml605_agent graph topology and routing.

Each test is marked xfail until graph.py is implemented in plan 02-02.
When graph.py exists, remove the xfail markers.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="graph.py not yet implemented", strict=False)
def test_graph_compiles() -> None:
    """build_graph() returns a compiled graph object with an invoke method."""
    from ml605_agent.graph import build_graph

    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


@pytest.mark.xfail(reason="graph.py not yet implemented", strict=False)
def test_full_pipeline_no_drift() -> None:
    """Graph run with overall_drift=False produces status='complete' in final state."""
    from ml605_agent.graph import build_graph

    graph = build_graph()
    initial_state = {
        "window_hours": 24,
        "overall_drift": False,
        "status": "running",
    }
    final_state = graph.invoke(initial_state)
    assert final_state.get("status") == "complete"


@pytest.mark.xfail(reason="graph.py not yet implemented", strict=False)
def test_full_pipeline_with_drift() -> None:
    """Graph run that triggers the retrain path sets retrain_done=True in final state."""
    from ml605_agent.graph import build_graph

    graph = build_graph()
    initial_state = {
        "window_hours": 24,
        "overall_drift": True,
        "retrain_done": False,
        "status": "running",
    }
    final_state = graph.invoke(initial_state)
    assert final_state.get("retrain_done") is True


@pytest.mark.xfail(reason="graph.py not yet implemented", strict=False)
def test_error_handler_routing() -> None:
    """Graph routes to error handler when status='error' and populates error field."""
    from ml605_agent.graph import build_graph

    graph = build_graph()
    initial_state = {
        "window_hours": 24,
        "status": "error",
        "error": "simulated fetch failure",
    }
    final_state = graph.invoke(initial_state)
    assert final_state.get("error") is not None
    assert isinstance(final_state.get("error"), str)
