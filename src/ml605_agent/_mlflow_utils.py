"""MLflow logging helpers for agent workers.

Every worker in the LangGraph pipeline opens a nested MLflow run so the parent
``agent-pipeline-run`` has a child per agent/worker. This module provides a
single context manager, ``worker_run``, that standardizes three things across
every worker:

1. Run name is zero-padded with a step number (e.g. ``"01_fetch_worker"``) so
   child runs sort alphabetically in the MLflow UI in execution order — the
   default MLflow UI sort is by name, not start_time.
2. ``step_order``, ``worker_name``, and ``routed_to_on_success`` tags are set
   so you can filter/search in the UI and quickly see what each worker hands
   off to in the happy path.
3. A ``duration_seconds`` metric is logged on exit so slow workers are easy to
   spot without digging into per-step timings elsewhere.

The helper assumes the caller has already verified a parent MLflow run exists
(i.e. ``state["mlflow_run_id"]`` is truthy) — keeping the ``if mlflow_run_id:``
guard at the call site mirrors the existing worker pattern and keeps this
helper free of branching logic.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import mlflow


@contextmanager
def worker_run(
    step: int,
    name: str,
    routed_to: str | None = None,
) -> Iterator[None]:
    """Open a standardized nested MLflow run for an agent worker.

    Must be called *inside* an already-active parent run. Callers should guard
    on ``state["mlflow_run_id"]`` themselves:

        mlflow_run_id = state.get("mlflow_run_id")
        if mlflow_run_id:
            with worker_run(step=1, name="fetch_worker", routed_to="feature_worker"):
                mlflow.log_metric("rows_fetched", 42)

    Args:
        step: 1-based execution order of this worker in the graph. Used as a
            zero-padded prefix on the run name so the UI sorts children in the
            order they ran.
        name: Worker function name (``"fetch_worker"``, ``"drift_worker"``...).
        routed_to: Default downstream node on a successful run. Optional — used
            only as a tag so the graph edges are visible from MLflow alone.
    """
    run_name = f"{step:02d}_{name}"
    started = time.time()
    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.set_tag("step_order", run_name)
        mlflow.set_tag("worker_name", name)
        if routed_to:
            mlflow.set_tag("routed_to_on_success", routed_to)
        try:
            yield
        finally:
            mlflow.log_metric("duration_seconds", time.time() - started)
