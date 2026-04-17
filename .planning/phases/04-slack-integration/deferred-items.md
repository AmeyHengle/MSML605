# Deferred Items — Phase 04 Slack Integration

## Pre-existing Test Failures (out of scope for 04-01)

These failures existed before Plan 04-01 execution and are NOT caused by changes in this plan.

### 1. test_agent_workers.py::test_test_worker
- **Status:** Pre-existing failure (confirmed by testing without state.py changes)
- **Error:** `AssertionError: Expected 'eval_result', got: ['status', 'error']` — `actual_intensity` column missing from test DataFrame fixture
- **Cause:** Test fixture `_featured_df` does not include `actual_intensity` column that `test_worker` expects
- **Impact:** Does not affect the current plan's deliverables
- **Suggested fix:** Add `actual_intensity` column to the `_featured_df` fixture in `tests/test_agent_workers.py`

### 2. test_report_worker.py::test_forecast_chart_embedded
- **Status:** Pre-existing failure (confirmed by testing without any changes)
- **Error:** AssertionError in forecast chart embedding
- **Impact:** Does not affect the current plan's deliverables
- **Suggested fix:** To be investigated in a future quick task or Phase 04 bug fix

### 3. test_agent_workers.py::test_retrain_worker (discovered during 04-04)
- **Status:** Pre-existing failure (confirmed present before 04-04 changes)
- **Error:** `['actual_intensity'] not in DataFrame` — workers.py TARGET_COL="actual_intensity" but test fixture uses "intensity.actual"  
- **Cause:** workers.py was changed to use "actual_intensity" in fix(03-uat) but test_agent_workers.py was not updated to match
- **Impact:** Does not affect Slack/HITL functionality or Phase 4 deliverables
- **Suggested fix:** Update test fixture data["intensity.actual"] to data["actual_intensity"] in tests/test_agent_workers.py line 207

### 4. test_mcp.py failures (4 tests, discovered during 04-04)
- **Status:** Pre-existing failures (MCP server connectivity tests)
- **Error:** Async test framework issues + server not running
- **Impact:** Does not affect Phase 4 deliverables
- **Suggested fix:** Investigate asyncio_mode config or test_mcp.py fixture issues in Phase 5

### 5. test_report_helpers.py::test_forecast_chart_returns_nonempty_string (04-04 scan)
- **Status:** Pre-existing failure
- **Error:** Empty string returned from forecast_chart function
- **Impact:** Does not affect Phase 4 deliverables

## Items Deferred to Future Plans

- Block Kit implementation (blocks.py stubs → Plan 04-02)
- Slack Bolt App implementation (bot.py skeleton → Plan 04-02)
- HITL interrupt/resume node (graph.py → Plan 04-03)
- alert_worker Slack integration (→ Plan 04-03)
