---
phase: 04-slack-integration
plan: 02
subsystem: slack
tags: [slack-bolt, slack-sdk, block-kit, slash-commands, hitl, socket-mode, threading]

# Dependency graph
requires:
  - phase: 04-slack-integration/04-01
    provides: ml605_slack package scaffold, slack-bolt dependency, PipelineState fields
  - phase: 03-analysis-explainability
    provides: report_worker SHAP output in PipelineState, registry.py for model promotion
provides:
  - Block Kit message builders (build_drift_alert_blocks, build_no_drift_blocks)
  - Slash command dispatcher (handle_ml605_command) with all 6 subcommands
  - SocketModeHandler entry point (__main__.py)
affects: [04-03, 04-04]

# Tech tracking
tech-stack:
  added: [slack-bolt>=1.27.0, slack-sdk (transitive)]
  patterns:
    - ack()-first pattern in every slash command handler (Slack 3s timeout compliance)
    - background daemon threading for long-running pipeline operations
    - module-level command handler for direct testability without app.command decorator
    - pure Block Kit builder functions (no side effects, fully unit testable)

key-files:
  created:
    - src/ml605_slack/__init__.py
    - src/ml605_slack/blocks.py
    - src/ml605_slack/bot.py
    - src/ml605_slack/__main__.py
    - tests/test_slack_blocks.py
    - tests/test_slack_commands.py
  modified:
    - src/ml605_agent/state.py
    - tests/test_agent_state.py
    - pyproject.toml
    - uv.lock

key-decisions:
  - "handle_ml605_command is a module-level function (not nested in create_app) to enable direct import and testing without mocking the full Bolt App lifecycle"
  - "PipelineState extended with shap_top_features and hitl_decision fields to support Phase 4 HITL flow and alert_worker SHAP access"
  - "slack-bolt 1.28.0 installed (>=1.27.0 satisfied) — no pin needed, tested with 1.28.0"
  - "daemon=True on background threads ensures process exits cleanly if bot shuts down"

patterns-established:
  - "Block Kit builder pattern: pure functions returning list[dict], no imports of Slack client"
  - "Slash command dispatching: single /ml605 handler, dispatch on text.split()[0]"
  - "Background pipeline: ack immediately, run in daemon thread, post results to channel"

requirements-completed: [SLACK-01, SLACK-02, SLACK-03, SLACK-04, SLACK-05, HITL-01]

# Metrics
duration: 12min
completed: 2026-04-07
---

# Phase 4 Plan 2: Slack Block Kit Builders and Slash Commands Summary

**Block Kit builders for drift alert (with HITL approve/reject buttons) and no-drift summary, plus a 6-subcommand /ml605 slash command dispatcher via SocketMode — 16 tests GREEN**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-07T19:13:53Z
- **Completed:** 2026-04-07T19:25:59Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Implemented `build_drift_alert_blocks` — returns 7 Block Kit blocks: header, divider, drifted features section, SHAP section, metrics section, divider, and HITL actions block with approve/reject buttons (action_id "approve_retrain" / "reject_retrain", both carrying thread_id value)
- Implemented `build_no_drift_blocks` — returns 4 Block Kit blocks: header, metrics (RMSE+MAE), threshold confirmation, production version
- Implemented `handle_ml605_command` as a module-level function dispatching all 6 subcommands (run, status, promote, retrain, report, history) with ack-first pattern
- /ml605 run and /ml605 retrain spawn daemon background threads; /ml605 promote calls transition_model_stage; /ml605 report calls files_upload_v2
- Added `shap_top_features` and `hitl_decision` fields to PipelineState TypedDict

## Task Commits

1. **Task 1: Block Kit builders + test_slack_blocks GREEN** - `b629fcd` (feat)
2. **Task 2: Slash command handlers + test_slack_commands GREEN** - `f0c77e6` (feat)

**Plan metadata:** (docs commit — see below)

## Files Created/Modified

- `src/ml605_slack/__init__.py` — Package marker
- `src/ml605_slack/blocks.py` — `build_drift_alert_blocks`, `build_no_drift_blocks` (pure Block Kit functions)
- `src/ml605_slack/bot.py` — `handle_ml605_command` dispatcher + `create_app()` + helper functions
- `src/ml605_slack/__main__.py` — `SocketModeHandler` entry point
- `tests/test_slack_blocks.py` — 8 tests for Block Kit structure assertions
- `tests/test_slack_commands.py` — 8 tests for slash command handler behavior
- `src/ml605_agent/state.py` — Added `shap_top_features: Optional[list[str]]` and `hitl_decision: Optional[str]`
- `tests/test_agent_state.py` — Updated expected field count from 19 to 21 (includes new fields)
- `pyproject.toml` — Added slack-bolt dependency, added src/ml605_slack to hatch wheel packages

## Decisions Made

- Made `handle_ml605_command` module-level (not nested inside `create_app`) so tests can import and call it directly with mock args — avoids need to extract from app._listeners
- Used `daemon=True` on background threads so the bot process exits cleanly on shutdown
- `_get_latest_report_path()` uses `pathlib.Path("reports").glob("*.html")` sorted by mtime — simple and robust
- `_build_status_blocks()` and `_build_history_blocks()` wrapped in try/except to prevent command failures crashing the bot

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_agent_state.py schema assertion**
- **Found during:** Task 2 (after adding shap_top_features + hitl_decision to PipelineState)
- **Issue:** test_state_schema checked for exactly 19 fields by name; adding 2 new fields caused assertion failure ("Extra: shap_top_features, hitl_decision")
- **Fix:** Updated expected_fields set to include shap_top_features and hitl_decision; updated docstring from "19 required fields" to "21 required fields"
- **Files modified:** tests/test_agent_state.py
- **Verification:** uv run pytest tests/test_agent_state.py — 2 passed
- **Committed in:** f0c77e6 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Required fix — test was enforcing an outdated field count. The new fields are core to Phase 4 HITL functionality.

## Issues Encountered

- Worktree branch (`worktree-agent-a533e4d8`) was branched from initial commit (LICENSE + README only). Required `git reset --hard origin/test` to obtain current project code before execution.
- The `ml605_slack` package didn't exist yet (Plan 04-01 not executed). Plan 04-02 absorbed the scaffolding work from Plan 04-01 (pyproject.toml update, __init__.py, state.py fix).

## Known Stubs

None — all builder functions fully implemented with real Block Kit JSON. Command handlers use real MLflow/Slack mocks in tests and real implementations in production code.

## Next Phase Readiness

- `blocks.py` provides the two Block Kit builder functions needed by alert_worker (Plan 04-03)
- `bot.py` `create_app()` is the entry point for the full Slack bot (Plan 04-04 integration wiring)
- `PipelineState.shap_top_features` field is now declared — alert_worker can safely read it
- `PipelineState.hitl_decision` field is declared for HITL node output (Plan 04-03)
- Outstanding: alert_worker implementation (currently stub in workers.py) — Plan 04-03

---
*Phase: 04-slack-integration*
*Completed: 2026-04-07*
