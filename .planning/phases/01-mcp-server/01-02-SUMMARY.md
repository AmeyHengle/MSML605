---
phase: 01-mcp-server
plan: 02
subsystem: mcp-server
tags: [fastmcp, mcp-tools, streamable-http, carbon-intensity, langchain-mcp-adapters]

# Dependency graph
requires:
  - ml605_mcp package directory (from plan 01-01)
  - tests/test_mcp.py stubs in RED state (from plan 01-01)
  - fastmcp>=3.1.1, langchain-mcp-adapters>=0.2.2 (from plan 01-01)
provides:
  - src/ml605_mcp/server.py with FastMCP server exposing fetch_intensity and fetch_generation_mix
  - Working MCP endpoint at /mcp over Streamable HTTP transport
  - /health liveness endpoint
  - Clean run_pipeline.py (no stray breakpoint)
affects:
  - Phase 2 agents (can now discover and call carbon intensity tools via MCP)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "FastMCP sync tools in thread pool — def not async def for requests-based data.py"
    - "_df_to_records: dt.strftime for Timestamps, isnan check for NaN -> None"
    - "_resolve_window: supports both hours_back rolling window and explicit ISO8601 range"
    - "MultiServerMCPClient direct API: instantiate + get_tools() without async context manager (langchain-mcp-adapters >= 0.1.0)"

key-files:
  created:
    - src/ml605_mcp/server.py
  modified:
    - tests/test_mcp.py

key-decisions:
  - "Used sync def for MCP tools (not async def) — FastMCP runs sync tools in a thread pool; data.py uses requests which is blocking"
  - "langchain-mcp-adapters 0.2.2 requires direct API for MultiServerMCPClient (no async context manager support)"

# Metrics
duration: 4min
completed: 2026-03-27
---

# Phase 1 Plan 02: MCP Server Implementation Summary

**FastMCP server with fetch_intensity and fetch_generation_mix tools over Streamable HTTP on port 8000 — all 35 tests passing (29 pre-existing + 6 MCP unit/integration tests)**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-27T19:02:27Z
- **Completed:** 2026-03-27T19:06:15Z
- **Tasks:** 2
- **Files modified:** 2 (src/ml605_mcp/server.py created, tests/test_mcp.py updated)

## Accomplishments

- Implemented `src/ml605_mcp/server.py` with `FastMCP("carbon-intensity")` instance
- `fetch_intensity` tool: accepts `hours_back` (default 12) or `start_dt`/`end_dt`; returns `{readings, start, end, count, factors}`
- `fetch_generation_mix` tool: same signature; returns `{readings, start, end, count}` — no `factors` key; filters to generation mix columns only
- `_resolve_window` helper: parses ISO8601 strings (handles `Z` suffix) or computes rolling window ending at UTC now
- `_df_to_records` helper: converts pandas Timestamp columns via `dt.strftime` (no raw Timestamp serialization), replaces `float('nan')` with `None`
- `/health` custom route returns `PlainTextResponse("OK")` — liveness probe for integration test subprocess fixture
- Entry point: `mcp.run(transport="http", host="0.0.0.0", port=8000)` for standalone execution
- Verified run_pipeline.py has no stray `breakpoint()` — already clean from prior work
- Fixed `test_client_tool_discovery` to use langchain-mcp-adapters 0.2.2 direct API

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement src/ml605_mcp/server.py** - `df34e34` (feat)
2. **Task 2: Fix run_pipeline.py breakpoint and run full test suite** - `9824a18` (fix)

**Plan metadata:** (committed with final docs commit)

## Files Created/Modified

- `src/ml605_mcp/server.py` - Full FastMCP server: mcp instance, _resolve_window, _df_to_records, fetch_intensity tool, fetch_generation_mix tool, /health route, __main__ entry point
- `tests/test_mcp.py` - Updated test_client_tool_discovery to use direct MultiServerMCPClient API (no async context manager)

## Decisions Made

- Used `def` (synchronous) for MCP tool functions, not `async def`. FastMCP runs sync tools in a thread pool. Using async def would require `asyncio.to_thread()` wrapping around the requests-based `data.py` calls.
- `langchain-mcp-adapters 0.2.2` removed the `async with MultiServerMCPClient(...)` context manager API. Updated test to use `client = MultiServerMCPClient(...); tools = await client.get_tools()` directly.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed MultiServerMCPClient async context manager API removed in langchain-mcp-adapters >= 0.1.0**
- **Found during:** Task 2 (integration test run)
- **Issue:** `test_client_tool_discovery` used `async with MultiServerMCPClient(...)` which raises `NotImplementedError` in langchain-mcp-adapters 0.2.2. The plan's test stubs were written against an older API.
- **Fix:** Changed to direct instantiation + `await client.get_tools()` without context manager — this is the documented migration path in the library's error message.
- **Files modified:** tests/test_mcp.py
- **Commit:** 9824a18

**2. run_pipeline.py already clean (no deviation needed)**
- **Found during:** Task 2 (Step 1: search for breakpoint)
- **Issue:** Plan documented removing a stray `breakpoint()` from line 10. `grep -n "breakpoint()"` returned no output — the file was already clean.
- **Action:** No change needed; Task 2 still committed as it includes the integration test fix.

---

**Total deviations:** 1 auto-fixed (Rule 1 - broken API usage due to library version)
**Impact on plan:** Minor. The fix is a 3-line change in the test file. Functionality is identical — MultiServerMCPClient still discovers tools correctly, just with a different call pattern.

## Issues Encountered

- langchain-mcp-adapters 0.2.2 is a breaking change from the API assumed in plan 01-01 test stubs. Resolved by Rule 1 auto-fix.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Phase 2 agents can begin immediately: use `MultiServerMCPClient` with `transport="http"`, `url="http://localhost:8000/mcp"` to discover and call `fetch_intensity` and `fetch_generation_mix`
- Start the MCP server with: `uv run python src/ml605_mcp/server.py`
- Health check: `curl http://localhost:8000/health` returns `OK`
- All 35 tests green — no regression risk

---
*Phase: 01-mcp-server*
*Completed: 2026-03-27*

## Self-Check: PASSED

- src/ml605_mcp/server.py: FOUND
- tests/test_mcp.py: FOUND
- 01-02-SUMMARY.md: FOUND
- Commit df34e34 (feat - server.py): FOUND
- Commit 9824a18 (fix - integration test): FOUND
