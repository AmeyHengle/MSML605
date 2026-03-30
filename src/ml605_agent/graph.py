"""LangGraph pipeline graph assembly for the ml605 agent.

This module defines the full StateGraph topology connecting all worker nodes
with conditional routing for drift detection and error handling.
"""
from __future__ import annotations

import mlflow
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ml605_agent.state import PipelineState
from ml605_agent.workers import (
    drift_worker,
    feature_worker,
    fetch_worker,
    retrain_worker,
    test_worker,
)


# ---------------------------------------------------------------------------
# Stub workers (Phase 2 stubs — full implementation in later phases)
# ---------------------------------------------------------------------------


def report_worker(state: PipelineState) -> dict:
    """Phase 3 stub: log placeholder, return report_path=None."""
    mlflow_run_id = state.get("mlflow_run_id")
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            with mlflow.start_run(run_name="report_worker", nested=True):
                mlflow.log_param("report_status", "stub_phase2")
    return {"report_path": None}


def alert_worker(state: PipelineState) -> dict:
    """Phase 4 stub: log placeholder, return alert_sent=False."""
    mlflow_run_id = state.get("mlflow_run_id")
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            with mlflow.start_run(run_name="alert_worker", nested=True):
                mlflow.log_param("alert_status", "stub_phase4")
    return {"alert_sent": False}


def error_handler(state: PipelineState) -> dict:
    """Logs error to MLflow nested run and marks pipeline complete."""
    mlflow_run_id = state.get("mlflow_run_id")
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id):
            with mlflow.start_run(run_name="error_handler", nested=True):
                mlflow.log_param("error", state.get("error", "unknown"))
    return {"status": "error"}


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def route_after_fetch(state: PipelineState) -> str:
    """Route after fetch_worker: error or continue to feature_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "feature_worker"


def route_after_feature(state: PipelineState) -> str:
    """Route after feature_worker: error or continue to test_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "test_worker"


def route_after_test(state: PipelineState) -> str:
    """Route after test_worker: error or continue to drift_worker."""
    if state.get("status") == "error":
        return "error_handler"
    return "drift_worker"


def route_after_drift(state: PipelineState) -> str:
    """Route after drift_worker.

    - status=error → error_handler
    - overall_drift=True and retrain_done=False → retrain_worker
    - overall_drift=False OR retrain_done=True → report_worker (prevents second retrain)
    """
    if state.get("status") == "error":
        return "error_handler"
    if state.get("overall_drift") and not state.get("retrain_done", False):
        return "retrain_worker"
    return "report_worker"


def route_after_retrain(state: PipelineState) -> str:
    """Route after retrain_worker: back-edge to test_worker to re-verify new model.

    - status=error → error_handler
    - otherwise → test_worker (back-edge)
    """
    if state.get("status") == "error":
        return "error_handler"
    return "test_worker"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_graph():
    """Assemble and compile the full ml605 pipeline StateGraph.

    Topology:
        START → fetch_worker → feature_worker → test_worker → drift_worker
        drift_worker → [drift? yes: retrain_worker → test_worker loop; no: report_worker]
        report_worker → alert_worker → END
        Any error → error_handler → END

    Returns:
        Compiled LangGraph StateGraph with MemorySaver checkpointer.
    """
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("fetch_worker", fetch_worker)
    builder.add_node("feature_worker", feature_worker)
    builder.add_node("test_worker", test_worker)
    builder.add_node("drift_worker", drift_worker)
    builder.add_node("retrain_worker", retrain_worker)
    builder.add_node("report_worker", report_worker)
    builder.add_node("alert_worker", alert_worker)
    builder.add_node("error_handler", error_handler)

    # Entry: START → fetch_worker (with error routing)
    builder.add_conditional_edges(
        START,
        lambda s: "fetch_worker",
        {"fetch_worker": "fetch_worker"},
    )

    # After fetch: check for error, then go to feature_worker
    builder.add_conditional_edges(
        "fetch_worker",
        route_after_fetch,
        {"feature_worker": "feature_worker", "error_handler": "error_handler"},
    )

    # After feature: check for error, then go to test_worker
    builder.add_conditional_edges(
        "feature_worker",
        route_after_feature,
        {"test_worker": "test_worker", "error_handler": "error_handler"},
    )

    # After test: check for error, then go to drift_worker
    builder.add_conditional_edges(
        "test_worker",
        route_after_test,
        {"drift_worker": "drift_worker", "error_handler": "error_handler"},
    )

    # After drift: conditional routing (retrain, report, error)
    builder.add_conditional_edges(
        "drift_worker",
        route_after_drift,
        {
            "retrain_worker": "retrain_worker",
            "report_worker": "report_worker",
            "error_handler": "error_handler",
        },
    )

    # After retrain: back-edge to test_worker (re-verify new model)
    builder.add_conditional_edges(
        "retrain_worker",
        route_after_retrain,
        {"test_worker": "test_worker", "error_handler": "error_handler"},
    )

    # Linear tail: report → alert → END
    builder.add_edge("report_worker", "alert_worker")
    builder.add_edge("alert_worker", END)
    builder.add_edge("error_handler", END)

    return builder.compile(checkpointer=MemorySaver())
