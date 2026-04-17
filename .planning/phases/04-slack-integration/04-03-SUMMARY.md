---
phase: 04-slack-integration
plan: 03
subsystem: hitl
tags: [langgraph, hitl, interrupt, command, memorysaver, slack-sdk, webclient, threading]

# Dependency graph
requires:
  - phase: 04-slack-integration/04-02
    provides: build_drift_alert_blocks, build_no_drift_blocks, bot.py create_app skeleton
  - phase: 03-analysis-explainability
    provides: report_worker producing shap_top_features and report_path in PipelineState

provides:
  - hitl_decision_node in graph.py (interrupt/resume for human approval of retrain)
  - route_after_hitl routing function (approve -> retrain_worker, reject/timeout -> alert_worker)
  - Modified build_graph with checkpointer=None default parameter for HITL support
  - Real alert_worker posting drift alerts and no-drift summaries to Slack via WebClient
  - approve_retrain and reject_retrain @app.action button handlers in bot.py
  - HITL timeout auto-reject via threading.Timer in bot.py
  - _get_or_create_graph() shared graph instance with MemorySaver in bot.py

affects: [04-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "HITL interrupt/resume: interrupt() pauses graph, Command(resume=...) resumes from checkpoint"
    - "MemorySaver checkpointer: build_graph(checkpointer=None) backward-compatible, bot passes MemorySaver"
    - "alert_worker lazy import pattern: slack_sdk.WebClient and ml605_slack.blocks imported inside function for testability"
    - "HITL timeout: threading.Timer on daemon thread auto-rejects after SLACK_HITL_TIMEOUT_MINUTES"
    - "Graph topology: drift_worker always routes to report_worker; HITL decides retrain after seeing report"

key-files:
  created:
    - tests/test_slack_hitl.py
    - tests/test_alert_worker.py
  modified:
    - src/ml605_agent/graph.py
    - src/ml605_slack/bot.py
    - tests/test_agent_graph.py

key-decisions:
  - "hitl_decision_node checks retrain_done: if retrain already done, returns no_drift (prevents double-interrupt on back-edge loop)"
  - "route_after_drift simplified: always returns report_worker (HITL decides retrain, not drift routing)"
  - "test_agent_graph.py updated: route_after_drift no longer routes to retrain_worker directly; tests updated to mock hitl_decision_node"
  - "alert_worker lazy imports slack_sdk.WebClient and ml605_slack.blocks inside try block for graceful degradation"
  - "bot.py _get_or_create_graph: thread-safe with Lock, creates single shared MemorySaver graph instance"

patterns-established:
  - "HITL node pattern: interrupt(payload) -> Command(resume=decision) -> log to MLflow"
  - "Timeout pattern: threading.Timer(minutes*60, on_timeout) with daemon=True and cancel() on human action"
  - "Alert worker pattern: check token -> build blocks -> chat_postMessage -> files_upload_v2 if report exists"

requirements-completed: [SLACK-01, SLACK-02, HITL-01, HITL-02, HITL-03]

# Metrics
duration: 11min
completed: 2026-04-07
---

# Phase 4 Plan 3: HITL Interrupt/Resume + Real Slack Alert Posting Summary

**LangGraph HITL interrupt/resume node for human retrain approval, real Slack WebClient alert_worker posting drift/no-drift Block Kit messages with report upload, and button action handlers with auto-reject timeout — 13 tests GREEN**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-04-07T19:34:36Z
- **Completed:** 2026-04-07T19:45:58Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Implemented `hitl_decision_node` in graph.py: calls `interrupt()` when `overall_drift=True` and `retrain_done=False`, skips when no drift or after retrain, logs `hitl_decision` param and `mtta_seconds` metric to MLflow nested run (HITL-03)
- Added `route_after_hitl`: routes `approve` to retrain_worker, `reject`/`reject_timeout`/`no_drift` to alert_worker
- Modified `build_graph` to accept `checkpointer=None` parameter and insert `hitl_decision_node` between `report_worker` and `alert_worker`
- Replaced `alert_worker` stub with real Slack posting via `WebClient`: drift path posts `build_drift_alert_blocks`, no-drift path posts `build_no_drift_blocks`, uploads HTML report via `files_upload_v2` (SLACK-01, SLACK-02)
- Added `@app.action("approve_retrain")` and `@app.action("reject_retrain")` handlers in bot.py — both cancel HITL timeout timer and resume graph via `Command(resume=...)` (HITL-01)
- Added `start_hitl_timeout()` with `threading.Timer` for auto-reject after configurable timeout (D-09)
- Added `_get_or_create_graph()` thread-safe module-level shared graph instance with MemorySaver

## Task Commits

1. **Task 1: Add hitl_decision_node to graph, modify topology, compile with MemorySaver** - `85e618d` (feat)
2. **Task 2: Replace alert_worker stub with real Slack posting, add button handlers and timeout** - `13b73f6` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `src/ml605_agent/graph.py` — `hitl_decision_node`, `route_after_hitl`, real `alert_worker`, modified `route_after_drift` and `build_graph(checkpointer=None)`
- `src/ml605_slack/bot.py` — `_get_or_create_graph()`, `start_hitl_timeout()`, `@app.action("approve_retrain")`, `@app.action("reject_retrain")`
- `tests/test_slack_hitl.py` — 7 HITL interrupt/resume/timeout/logging tests (replaces RED stubs)
- `tests/test_alert_worker.py` — 6 alert_worker drift/no-drift/error tests (replaces RED stubs)
- `tests/test_agent_graph.py` — Updated routing tests + mocked hitl_decision_node in pipeline mocked tests

## Decisions Made

- `hitl_decision_node` checks `retrain_done` flag: after retrain completes and graph loops back through `drift -> report -> hitl`, the node returns `no_drift` to route to `alert_worker` (prevents infinite interrupt loop)
- `route_after_drift` simplified to always return `report_worker` — the HITL node handles the retrain-or-not decision after the human sees the report. This is a topology change from Phase 2.
- `test_agent_graph.py` `test_routing_with_drift` updated: `route_after_drift(drift=True)` now correctly returns `report_worker` (not `retrain_worker`)
- `test_full_pipeline_with_drift` updated to mock `hitl_decision_node` returning `approve` (then `no_drift` after retrain)
- `alert_worker` uses lazy imports inside the function body to support test mocking of `slack_sdk.WebClient`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_agent_graph.py routing tests for new topology**
- **Found during:** Task 1 (after changing route_after_drift)
- **Issue:** `test_routing_with_drift` asserted `route_after_drift(drift=True)` returns `retrain_worker`, but the new topology routes to `report_worker` always. `test_full_pipeline_with_drift` would fail because the graph no longer goes directly to retrain.
- **Fix:** Updated `test_routing_with_drift` docstring + assertion to expect `report_worker`. Updated `test_full_pipeline_with_drift` to mock `hitl_decision_node` returning `approve` first call and `no_drift` after retrain. Updated `test_graph_compiles` to include `hitl_decision_node` in expected nodes.
- **Files modified:** tests/test_agent_graph.py
- **Committed in:** 85e618d (Task 1 commit)

**2. [Rule 3 - Blocking] Checked out Plan 02 implementations from parallel worktree**
- **Found during:** Pre-execution setup
- **Issue:** This worktree branch was at `bb98e1a` which had Plan 02 _stubs_ (not implementations). The implementations (blocks.py, bot.py) were committed in the `worktree-agent-a8598b53` branch. The test branch's `bb98e1a` commit has Plan 01 stubs for these files.
- **Fix:** Used `git checkout <commit> -- <files>` to check out implemented versions from the parallel worktree commits (b629fcd, f0c77e6). Committed as prerequisite before Task 1.
- **Files modified:** src/ml605_slack/blocks.py, bot.py, __init__.py, __main__.py, tests/test_slack_blocks.py, test_slack_commands.py, test_agent_state.py
- **Committed in:** a82bc82 (prerequisite commit)

---

**Total deviations:** 2 auto-fixed (1 Rule 1 - Bug, 1 Rule 3 - Blocking)
**Impact on plan:** Both necessary. Rule 1 corrects outdated routing test assertions. Rule 3 unblocks execution by restoring Plan 02 implementations needed for Plan 03 tests.

## Issues Encountered

- Worktree branch `worktree-agent-aa7be7c4` was initialized from `bf27189 Initial commit` (LICENSE + README only). Required `git reset --hard test` then cherry-picking implementation files from parallel worktree commits.
- The `test` branch has Plan 02 stubs (from `e06a06d feat(04-01): scaffold ml605_slack package`). The implementations (b629fcd, f0c77e6) are in a different branch. This is a parallel execution coordination issue.

## Known Stubs

None — all specified functionality is fully implemented with real logic.

## Next Phase Readiness

- `hitl_decision_node` is wired and tested — Plan 04-04 can integrate the full bot flow
- `alert_worker` posts real Slack messages — Plan 04-04 can test end-to-end
- `_get_or_create_graph()` in bot.py provides the shared graph instance for the background pipeline runner
- Outstanding: `_run_pipeline_background` in bot.py still uses `build_graph()` without MemorySaver — Plan 04-04 should update it to use `_get_or_create_graph()` for HITL support

## Self-Check: PASSED

- FOUND: src/ml605_agent/graph.py (hitl_decision_node at line 377, route_after_hitl at line 422)
- FOUND: src/ml605_slack/bot.py (_get_or_create_graph, start_hitl_timeout, approve_retrain, reject_retrain handlers)
- FOUND: tests/test_slack_hitl.py (7 tests, all GREEN)
- FOUND: tests/test_alert_worker.py (6 tests, all GREEN)
- FOUND: commit 85e618d (Task 1 - hitl_decision_node + topology)
- FOUND: commit 13b73f6 (Task 2 - alert_worker + button handlers)

---
*Phase: 04-slack-integration*
*Completed: 2026-04-07*
