---
phase: 02-langgraph-agent-pipeline
plan: "02"
subsystem: agent
tags: [langgraph, mlflow, mcp, asyncio, drift-detection, automl, sklearn]

# Dependency graph
requires:
  - phase: 02-01
    provides: PipelineState TypedDict, xfail stubs in test_agent_workers.py, langgraph installed
  - phase: 01-02
    provides: FastMCP server with fetch_intensity tool on streamable_http transport
provides:
  - src/ml605_agent/workers.py with 5 worker functions: fetch_worker, feature_worker, test_worker, drift_worker, retrain_worker
  - All 7 worker unit tests GREEN
affects: [02-03-graph-assembly, 02-04-entrypoint]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Workers return partial dicts (only owned keys) to be merged into PipelineState by LangGraph"
    - "fetch_worker wraps async MCP call via asyncio.run(_async_fetch()) for sync compatibility"
    - "MLflow two-level run context: parent run_id (if set) then nested=True child run per worker"
    - "FEATURES_FILE module constant allows test monkeypatching with patch()"
    - "drift_worker guards timestamp presence before calling add_time_features on reference CSV"
    - "retrain_worker uses nullcontext() pattern for optional parent MLflow run"

key-files:
  created:
    - src/ml605_agent/workers.py
  modified:
    - tests/test_agent_workers.py

key-decisions:
  - "TARGET_COL='intensity.actual' matches MCP server response field name; time_split is inlined with that column rather than using modeling.time_split (which defaults to actual_intensity)"
  - "drift_worker guards timestamp column presence before add_time_features — historical_data.csv may not have datetime-typed timestamp column"
  - "retrain_worker does inline time split instead of calling modeling.time_split to avoid target_col mismatch"
  - "FEATURES_FILE constant at module level enables test monkeypatching without patching Path directly"

patterns-established:
  - "Worker pattern: try/except wrapping entire body; on exception return {status: error, error: str(e)}"
  - "Async MCP: asyncio.run(async_fn()) at top of sync worker — safe for graph.invoke() sync context"
  - "MLflow nesting: with mlflow.start_run(run_id=parent_id) then with mlflow.start_run(run_name=worker, nested=True)"
  - "Test mocking: patch module-level references (ml605_agent.workers.pd.read_csv) not source modules"

requirements-completed: [AGENT-02, AGENT-03, AGENT-04, AGENT-07]

# Metrics
duration: 6min
completed: "2026-03-29"
---

# Phase 02 Plan 02: Worker Node Implementations Summary

**Five LangGraph worker functions implemented with MLflow nested run logging, MCP fetch via streamable_http, and full unit test suite (7 tests GREEN)**

## Performance

- **Duration:** 6 min
- **Started:** 2026-03-29T22:02:39Z
- **Completed:** 2026-03-29T22:08:39Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `fetch_worker` connects to MCP server via `MultiServerMCPClient(transport="streamable_http")`, builds DataFrame from readings, returns `df/factors/rows_fetched`
- `feature_worker` applies full feature pipeline (add_time_features, apply_factor_columns, ensure_feature_columns), guards on missing `features_used.txt`
- `test_worker` loads production model, runs inference, returns `EvalResult`; returns `status=error` when no Production model exists
- `drift_worker` loads `historical_data.csv` as reference, applies feature alignment, calls `detect_drift`, returns full `DriftReport` and `overall_drift` bool
- `retrain_worker` splits data, calls `run_automl()` inside parent+nested MLflow context, registers model in Staging, returns `retrain_done=True`
- Full test suite: 44 pass, 6 xfail, 0 failures (7 new worker tests + 0 regressions)

## Task Commits

Each task was committed atomically:

1. **RED phase (all tests)** - `c83d117` (test: add failing tests for all 5 workers)
2. **Task 1: fetch_worker + feature_worker** - `83498e5` (feat: implement fetch_worker and feature_worker)
3. **Task 2: test_worker + drift_worker + retrain_worker** - `14ceae4` (feat: implement test_worker, drift_worker, and retrain_worker)

_Note: TDD tasks have RED commit first, then GREEN implementation commit_

## Files Created/Modified
- `src/ml605_agent/workers.py` - 5 worker node functions for the LangGraph pipeline
- `tests/test_agent_workers.py` - 7 worker unit tests (replaced 7 xfail stubs with real assertions)

## Decisions Made
- Used `TARGET_COL = "intensity.actual"` matching the MCP server response field name, not `actual_intensity` from `data.py` (MCP server response preserves the original API key names)
- Inlined time_split in `retrain_worker` instead of calling `modeling.time_split()` to avoid the `target_col="actual_intensity"` default mismatch
- `FEATURES_FILE` module constant (not hardcoded `Path("features_used.txt")`) enables test monkeypatching via `patch("ml605_agent.workers.FEATURES_FILE", tmp_path / "...")`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] drift_worker guards timestamp presence before add_time_features**
- **Found during:** Task 2 (test_drift_worker)
- **Issue:** `add_time_features()` always accesses `df["timestamp"].dt.hour` — KeyError when reference_df loaded from CSV lacks datetime-typed timestamp column
- **Fix:** Added `if "timestamp" in reference_df.columns:` guard before calling `add_time_features(reference_df)`
- **Files modified:** src/ml605_agent/workers.py
- **Verification:** test_drift_worker passes; drift_worker handles CSV without timestamp column
- **Committed in:** 14ceae4 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Auto-fix necessary for correctness. No scope creep.

## Issues Encountered
- `test_drift_worker` failed on first run — drift_worker called `add_time_features(reference_df)` on a DataFrame that didn't have `timestamp` column (mocked via `pd.read_csv`). Fixed by guarding on column presence.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 5 workers implemented and tested — ready for graph assembly in plan 02-03
- Workers use `PipelineState` partial dict returns — compatible with LangGraph `StateGraph`
- MLflow nested run pattern established — graph can pass `mlflow_run_id` to all workers

---
*Phase: 02-langgraph-agent-pipeline*
*Completed: 2026-03-29*
