---
phase: 04-slack-integration
plan: 01
subsystem: slack-package-scaffold
tags: [slack-bolt, ml605_slack, PipelineState, TDD-RED, HITL]
dependency_graph:
  requires: []
  provides: [ml605_slack-package, slack-bolt-dependency, shap_top_features-field, hitl_decision-field, RED-test-stubs]
  affects: [src/ml605_agent/state.py, pyproject.toml, tests/]
tech_stack:
  added: [slack-bolt==1.28.0, slack-sdk==3.41.0]
  patterns: [TDD-RED-stubs, TypedDict-extension]
key_files:
  created:
    - src/ml605_slack/__init__.py
    - src/ml605_slack/blocks.py
    - src/ml605_slack/bot.py
    - src/ml605_slack/__main__.py
    - tests/test_slack_blocks.py
    - tests/test_slack_commands.py
    - tests/test_slack_hitl.py
    - tests/test_alert_worker.py
  modified:
    - pyproject.toml
    - src/ml605_agent/state.py
    - tests/test_agent_state.py
decisions:
  - "[04-slack-integration]: slack-bolt 1.28.0 installed (satisfies >=1.27.0); slack-sdk 3.41.0 as transitive dependency"
  - "[04-slack-integration]: PipelineState shap_top_features and hitl_decision fields added early (Pitfall 3 from RESEARCH.md)"
  - "[04-slack-integration]: TDD RED stubs use pytest.fail() (not xfail) so failure is visible in test output"
  - "[04-slack-integration]: test_agent_state.py updated to expect 21 fields (was 19) after adding shap_top_features and hitl_decision"
metrics:
  duration: 9 min
  completed: 2026-04-07
  tasks_completed: 2
  files_created: 8
  files_modified: 3
---

# Phase 04 Plan 01: Scaffold ml605_slack Package and RED Test Stubs Summary

Scaffolded the ml605_slack package with slack-bolt>=1.27.0 installed, fixed the silent PipelineState bug (missing shap_top_features), and created 29 RED test stubs across 4 test files covering all 8 requirements.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Install slack-bolt, scaffold ml605_slack, fix PipelineState | f881e0c | pyproject.toml, src/ml605_slack/{__init__,blocks,bot,__main__}.py, src/ml605_agent/state.py |
| 2 | Create RED test stubs for all 8 requirements | 35785d2 | tests/test_slack_{blocks,commands,hitl}.py, tests/test_alert_worker.py, tests/test_agent_state.py |

## Acceptance Criteria Verification

- pyproject.toml contains `"slack-bolt>=1.27.0"` in dependencies: PASS (slack-bolt==1.28.0 installed)
- pyproject.toml hatch wheel packages contains `"src/ml605_slack"`: PASS
- `src/ml605_slack/__init__.py` exists: PASS
- `src/ml605_slack/blocks.py` contains `def build_drift_alert_blocks(` and `def build_no_drift_blocks(`: PASS
- `src/ml605_slack/bot.py` contains `def create_app(`: PASS
- `src/ml605_slack/__main__.py` contains `def main(`: PASS
- `src/ml605_agent/state.py` contains `shap_top_features: Optional[list[str]]`: PASS
- `src/ml605_agent/state.py` contains `hitl_decision: Optional[str]`: PASS
- `uv run python -c "from slack_bolt import App"` exits 0: PASS
- All 29 new tests FAIL with "RED: not implemented" (not import errors): PASS
- `tests/test_slack_blocks.py`: TestDriftAlertBlocks (5 tests) + TestNoDriftBlocks (3 tests): PASS
- `tests/test_slack_commands.py`: 8 test classes with 8 tests total: PASS
- `tests/test_slack_hitl.py`: TestHITLInterrupt (4) + TestHITLTimeout (1) + TestHITLLogging (2): PASS
- `tests/test_alert_worker.py`: TestAlertWorkerDrift (2) + TestAlertWorkerNoDrift (2) + TestAlertWorkerError (2): PASS

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated test_agent_state.py to include new PipelineState fields**
- **Found during:** Task 2 verification
- **Issue:** `test_state_schema` had a hardcoded set of 19 expected fields; adding `shap_top_features` and `hitl_decision` to PipelineState caused this test to fail with "Extra: shap_top_features, hitl_decision"
- **Fix:** Updated expected_fields set in `test_state_schema` to include both new fields (19 -> 21) and updated docstring count
- **Files modified:** tests/test_agent_state.py
- **Commit:** 35785d2 (included in Task 2 commit)

### Pre-existing Failures (out of scope, not regressions from this plan)

Two pre-existing test failures were identified during verification. Both were confirmed to exist before any plan changes:

1. `tests/test_agent_workers.py::test_test_worker` — `actual_intensity` column missing from test fixture
2. `tests/test_report_worker.py::test_forecast_chart_embedded` — forecast chart assertion failure

These are documented in `.planning/phases/04-slack-integration/deferred-items.md`.

## Known Stubs

All stubs in this plan are intentional TDD RED stubs. Each function raises `NotImplementedError("Plan 04-02 implements this")`. These are tracked for Plan 04-02 and 04-03 to implement:

- `src/ml605_slack/blocks.py`: `build_drift_alert_blocks`, `build_no_drift_blocks` — to be implemented in Plan 04-02
- `src/ml605_slack/bot.py`: `create_app` — to be implemented in Plan 04-02
- `src/ml605_slack/__main__.py`: `main` — to be implemented in Plan 04-02
- All 29 test stubs — to become GREEN in Plans 04-02 and 04-03

## Self-Check: PASSED

All created files exist and commits are valid.
