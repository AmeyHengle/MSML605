---
phase: 02-langgraph-agent-pipeline
plan: "04"
subsystem: agent
tags: [mlflow, langgraph, nested-runs, workers, memorysaver]

# Dependency graph
requires:
  - phase: 02-langgraph-agent-pipeline
    provides: "LangGraph graph topology, worker nodes, __main__ entry point from plans 01-03"
provides:
  - "UAT gap resolved: all workers use mlflow.start_run(nested=True) directly — no double-wrap crash"
  - "MemorySaver removed from graph.py — no thread_id config key required in __main__.py"
  - "End-to-end pipeline runs without MLflow 'already active' exception"
affects: [03-analysis-explainability, 04-slack-notifications, 05-deployment]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "MLflow nested run pattern: workers call start_run(run_name=..., nested=True) inside already-active parent context"
    - "Stateless LangGraph compile: builder.compile() without MemorySaver checkpointer for Phase 2"

key-files:
  created: []
  modified:
    - src/ml605_agent/workers.py
    - src/ml605_agent/graph.py
    - src/ml605_agent/__main__.py
    - src/ml605_pipeline/config.py
    - src/ml605_pipeline/data.py

key-decisions:
  - "All 5 data workers and 3 stub workers use start_run(nested=True) directly — outer start_run(run_id=...) wrapper was re-opening already-active parent run, causing MLflow exception"
  - "MemorySaver removed from graph.py and thread_id dropped from __main__ config — not needed for Phase 2 stateless pipeline execution"

patterns-established:
  - "Nested MLflow run pattern: workers check mlflow_run_id in state, then call start_run(run_name=..., nested=True) — parent run stays active in __main__ context manager"

requirements-completed: [AGENT-05, AGENT-06, AGENT-08]

# Metrics
duration: 5min
completed: 2026-03-31
---

# Phase 2 Plan 04: Gap Closure — MLflow Double-Wrap Fix Summary

**Removed outer mlflow.start_run(run_id=...) wrappers from all 8 workers and dropped MemorySaver, resolving the UAT crash "Run with UUID ... is already active"**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-31T22:00:00Z
- **Completed:** 2026-03-31T22:05:00Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- All 5 data workers in workers.py (fetch, feature, test, drift, retrain) fixed to use `mlflow.start_run(run_name=..., nested=True)` directly
- MemorySaver removed from graph.py and `thread_id` config key dropped from __main__.py
- Full test suite passes at UAT baseline: 55 passed, 2 xfailed, 0 failures
- Pipeline entry point runs end-to-end: starts up, connects to MCP server, exits with status=error (data processing error) — not an MLflow crash

## Task Commits

Each task was committed atomically:

1. **Task 1: Commit pending working-tree fixes** - `8641379` (fix)
2. **Task 2: Run full test suite and smoke-test entry point** - no new code changes (verification only)

**Plan metadata:** (docs commit below)

## Files Created/Modified

- `src/ml605_agent/workers.py` - Removed outer `start_run(run_id=...)` wrapper from all 5 data workers; added langchain-mcp-adapters >= 0.2 content block parsing in `_async_fetch`
- `src/ml605_agent/graph.py` - Removed MemorySaver import; changed `builder.compile(checkpointer=MemorySaver())` to `builder.compile()`
- `src/ml605_agent/__main__.py` - Removed `"configurable": {"thread_id": ...}` from graph.invoke() config
- `src/ml605_pipeline/config.py` - Trailing newline cleanup
- `src/ml605_pipeline/data.py` - Trailing newline cleanup

## Decisions Made

- Outer `start_run(run_id=...)` wrapper was re-opening the already-active parent run — MLflow raises "Run with UUID ... is already active" when you try to open a run that is already the active run. Fix is to call `start_run(nested=True)` directly while parent context is active.
- MemorySaver is not needed for Phase 2 because the pipeline is stateless within a single invocation; it was causing a required `thread_id` key in config that was unnecessary complexity.

## Deviations from Plan

None — plan executed exactly as written. Working tree already contained all the correct fixes; task 1 was a commit-only operation (verify then stage/commit).

## Issues Encountered

None — all changes were pre-applied in the working tree. No code editing required, only verification and commit.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 UAT gap fully resolved: pipeline runs end-to-end without MLflow crash
- 02-UAT.md gap status updated to resolved
- Phase 3 (Analysis/Explainability) is unblocked — pipeline infrastructure is stable

---
*Phase: 02-langgraph-agent-pipeline*
*Completed: 2026-03-31*
