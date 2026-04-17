---
status: resolved
phase: 02-langgraph-agent-pipeline
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md]
started: 2026-03-30T02:30:00Z
updated: 2026-03-31T22:05:00Z
resolved_by: 02-04-PLAN.md
---

## Current Test

[testing complete]

## Tests

### 1. Full Test Suite Passes
expected: Run `uv run pytest` — should report 55 passed, 2 xfailed, 0 failures.
result: pass

### 2. Graph Compiles with 8 Nodes
expected: Run `PYTHONPATH=src uv run python -c "from ml605_agent.graph import build_graph; g = build_graph(); print(list(g.nodes))"` — should print a list containing all 8 nodes: fetch_worker, feature_worker, test_worker, drift_worker, retrain_worker, report_worker, alert_worker, error_handler (plus __start__ and __end__).
result: pass

### 3. Workers Import Cleanly
expected: Run `PYTHONPATH=src uv run python -c "from ml605_agent.workers import fetch_worker, feature_worker, test_worker, drift_worker, retrain_worker; print('OK')"` — should print "OK" with no errors.
result: pass

### 4. PipelineState Has 16 Fields
expected: Run `PYTHONPATH=src uv run python -c "from ml605_agent.state import PipelineState; import typing; print(len(typing.get_type_hints(PipelineState)), 'fields')"` — should print "16 fields".
result: pass

### 5. Pipeline Entry Point Starts
expected: Run `PYTHONPATH=src uv run python -m ml605_agent` — should attempt to start the MCP server subprocess. If the MCP server isn't already running it will either launch it or fail with a timeout/connection error message (not an ImportError or Python crash). The program should exit cleanly with a meaningful status, not a raw traceback.
result: issue
reported: "MCP server started fine, but pipeline crashed with: Exception: Run with UUID ... is already active. To start a new run, first end the current run with mlflow.end_run(). To start a nested run, call start_run with nested=True — in error_handler at graph.py line 51"
severity: major

## Summary

total: 5
passed: 4
issues: 1
pending: 0
skipped: 0

## Gaps

- truth: "Pipeline runs end-to-end without crashing when error_handler, report_worker, or alert_worker are invoked"
  status: resolved
  resolved_by: 02-04-PLAN.md
  reason: "All 8 worker functions now use mlflow.start_run(run_name=..., nested=True) directly — outer start_run(run_id=...) wrapper removed from all workers; MemorySaver dropped from graph.py"
  severity: major
  test: 5
  root_cause: "error_handler, report_worker, and alert_worker in graph.py each call `with mlflow.start_run(run_id=mlflow_run_id):` around their nested run, but main() already has that run active when graph.invoke() is called — MLflow rejects re-opening an already-active run. Fix: remove the outer start_run wrapper; use `mlflow.start_run(run_name=..., nested=True)` directly."
  artifacts:
    - path: "src/ml605_agent/graph.py"
      issue: "Lines 31, 41, 51 — outer `with mlflow.start_run(run_id=mlflow_run_id)` wrapping nested run inside already-active parent context"
  fix:
    - "Removed outer start_run(run_id=...) wrapper from all workers in graph.py and workers.py"
    - "MemorySaver removed from graph.py; thread_id key removed from __main__.py config"
    - "commit: 8641379 (fix(02-04): remove double mlflow.start_run wrappers from all workers and drop MemorySaver)"
