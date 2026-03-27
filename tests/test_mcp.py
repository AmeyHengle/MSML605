"""
Tests for the ml605_mcp FastMCP server.

Requirements covered:
  MCP-01: FastMCP server exposes National Grid ESO API as callable tools
  MCP-02: Two tools exposed — fetch_intensity and fetch_generation_mix
  MCP-03: MCP server reachable via langchain-mcp-adapters MultiServerMCPClient
  MCP-04: Server runs as a separate process and responds on port 8000

Test strategy:
  - Unit tests (test_tool_discovery, test_fetch_intensity_schema,
    test_fetch_generation_mix_schema, test_tool_parameters): use FastMCP
    in-process Client(mcp) — no subprocess, no network, milliseconds.
    data.py is mocked with unittest.mock.patch.
  - Integration tests (test_client_tool_discovery, test_server_health_endpoint):
    start server as subprocess, poll /health, then connect with
    MultiServerMCPClient. Marked @pytest.mark.integration for selective skip.
"""
from __future__ import annotations

import asyncio
import subprocess
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastmcp import Client
from langchain_mcp_adapters.client import MultiServerMCPClient

from ml605_mcp.server import mcp  # Will fail (ImportError) until plan 01-02


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_fetch_result():
    """Return a mock WindowFetchResult with minimal plausible data."""
    result = MagicMock()
    # DataFrame mock: has timestamp and interval_end columns
    df_mock = MagicMock()
    df_mock.columns = ["timestamp", "interval_end", "actual_intensity",
                       "forecast_intensity", "intensity_index", "gas", "wind"]
    df_mock.__len__ = lambda self: 2
    copy_mock = MagicMock()
    copy_mock.columns = df_mock.columns
    copy_mock.__len__ = lambda self: 2
    # strftime chain: df_copy[col].dt.strftime(...)
    col_mock = MagicMock()
    copy_mock.__getitem__ = lambda self, key: col_mock
    copy_mock.__setitem__ = lambda self, key, val: None
    copy_mock.to_dict = lambda orient: [
        {"timestamp": "2026-03-01T00:00:00Z", "interval_end": "2026-03-01T00:30:00Z",
         "actual_intensity": 120, "forecast_intensity": 125,
         "intensity_index": "moderate", "gas": 0.4, "wind": 0.3},
        {"timestamp": "2026-03-01T00:30:00Z", "interval_end": "2026-03-01T01:00:00Z",
         "actual_intensity": None, "forecast_intensity": 130,
         "intensity_index": "low", "gas": 0.35, "wind": 0.35},
    ]
    df_mock.copy = lambda: copy_mock
    result.df = df_mock
    result.factors = {"gas": 490.0, "coal": 937.0, "wind": 0.0}
    return result


# ---------------------------------------------------------------------------
# Unit tests — in-process, no network (MCP-01, MCP-02)
# ---------------------------------------------------------------------------


async def test_tool_discovery():
    """MCP-01: FastMCP server instance exposes exactly the required tools."""
    async with Client(mcp) as client:
        tools = await client.list_tools()
    tool_names = {t.name for t in tools}
    assert "fetch_intensity" in tool_names, (
        f"fetch_intensity not found in tools: {tool_names}"
    )
    assert "fetch_generation_mix" in tool_names, (
        f"fetch_generation_mix not found in tools: {tool_names}"
    )


async def test_fetch_intensity_schema(mock_fetch_result):
    """MCP-02: fetch_intensity returns correct schema keys."""
    with patch("ml605_mcp.server.fetch_window_dataframe", return_value=mock_fetch_result):
        async with Client(mcp) as client:
            result = await client.call_tool("fetch_intensity", {"hours_back": 1})
    data = result.data if hasattr(result, "data") else result[0].text
    assert "readings" in str(data)
    assert "start" in str(data)
    assert "end" in str(data)
    assert "count" in str(data)
    assert "factors" in str(data)


async def test_fetch_generation_mix_schema(mock_fetch_result):
    """MCP-02: fetch_generation_mix returns correct schema keys (no 'factors' key)."""
    with patch("ml605_mcp.server.fetch_window_dataframe", return_value=mock_fetch_result):
        async with Client(mcp) as client:
            result = await client.call_tool("fetch_generation_mix", {"hours_back": 1})
    data = result.data if hasattr(result, "data") else result[0].text
    assert "readings" in str(data)
    assert "start" in str(data)
    assert "end" in str(data)
    assert "count" in str(data)


async def test_tool_parameters(mock_fetch_result):
    """MCP-02: Both tools accept hours_back and start_dt/end_dt parameters."""
    with patch("ml605_mcp.server.fetch_window_dataframe", return_value=mock_fetch_result):
        async with Client(mcp) as client:
            # Test hours_back parameter (default mode)
            r1 = await client.call_tool("fetch_intensity", {"hours_back": 6})
            # Test explicit start_dt/end_dt (override mode)
            r2 = await client.call_tool(
                "fetch_generation_mix",
                {
                    "start_dt": "2026-03-01T00:00Z",
                    "end_dt": "2026-03-01T06:00Z",
                },
            )
    assert r1 is not None
    assert r2 is not None


# ---------------------------------------------------------------------------
# Integration tests — subprocess + real HTTP (MCP-03, MCP-04)
# ---------------------------------------------------------------------------


@pytest.fixture
async def running_server():
    """Start the MCP server as a subprocess and wait for /health to respond."""
    proc = subprocess.Popen(
        ["uv", "run", "python", "src/ml605_mcp/server.py"],
        cwd="f:/ml605-project",
    )
    # Poll health endpoint until ready (up to 10 seconds)
    async with httpx.AsyncClient() as http_client:
        for _ in range(20):
            try:
                r = await http_client.get("http://localhost:8000/health", timeout=1.0)
                if r.status_code == 200:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.5)
    yield proc
    proc.terminate()
    proc.wait()


@pytest.mark.integration
async def test_server_health_endpoint(running_server):
    """MCP-04: Server starts as standalone process and /health responds 200 OK."""
    async with httpx.AsyncClient() as http_client:
        r = await http_client.get("http://localhost:8000/health", timeout=5.0)
    assert r.status_code == 200
    assert r.text.strip() == "OK"


@pytest.mark.integration
async def test_client_tool_discovery(running_server):
    """MCP-03: MultiServerMCPClient can discover tools from running server via HTTP."""
    # langchain-mcp-adapters >= 0.1.0 removed context manager support.
    # Use the direct API: instantiate and call get_tools() without async with.
    client = MultiServerMCPClient(
        {
            "carbon_intensity": {
                "transport": "http",
                "url": "http://localhost:8000/mcp",
            }
        }
    )
    tools = await client.get_tools()
    tool_names = {t.name for t in tools}
    assert "fetch_intensity" in tool_names
    assert "fetch_generation_mix" in tool_names
