"""FastMCP server exposing National Grid ESO carbon intensity API as MCP tools.

Exposes two tools over Streamable HTTP transport on port 8000:
  - fetch_intensity: returns actual/forecast intensity + emission factors
  - fetch_generation_mix: returns generation mix (fuel-type percentages only)

Run standalone:
    uv run python src/ml605_mcp/server.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import PlainTextResponse

# Ensure the src/ directory is on sys.path so ml605_pipeline can be imported
# when this file is run as __main__ (e.g. `uv run python src/ml605_mcp/server.py`)
_SRC = Path(__file__).resolve().parent.parent  # src/ml605_mcp/../../ -> project root? No.
# Path(__file__) = .../src/ml605_mcp/server.py
# .parent         = .../src/ml605_mcp/
# .parent.parent  = .../src/          <- this is the src dir we want on sys.path
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ml605_pipeline.data import fetch_window_dataframe  # noqa: E402

# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

mcp = FastMCP("carbon-intensity")


# ---------------------------------------------------------------------------
# Private helpers (not MCP tools)
# ---------------------------------------------------------------------------


def _resolve_window(
    hours_back: int,
    start_dt: str | None,
    end_dt: str | None,
) -> tuple[datetime, datetime]:
    """Return (start, end) UTC datetimes for the requested window.

    If both start_dt and end_dt are provided, parse them as ISO8601.
    Otherwise, compute a rolling window ending at now.
    """
    if start_dt is not None and end_dt is not None:
        start = datetime.fromisoformat(start_dt.replace("Z", "+00:00"))
        end = datetime.fromisoformat(end_dt.replace("Z", "+00:00"))
    else:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours_back)
    return start, end


def _df_to_records(df) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe dicts.

    Handles two pitfalls:
    1. pandas Timestamp columns — converts to ISO strings via dt.strftime
       to avoid serialisation failures.
    2. float NaN values — replaces with None so JSON output is valid.
    """
    df_copy = df.copy()
    for col in ["timestamp", "interval_end"]:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = df_copy.to_dict("records")
    return [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        for row in rows
    ]


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool
def fetch_intensity(
    hours_back: int = 12,
    start_dt: str | None = None,
    end_dt: str | None = None,
) -> dict:
    """Fetch carbon intensity readings with emission factors for a time window.

    Includes actual/forecast intensity, intensity index, and CO2 emission
    factors per fuel type baked into the response.

    Args:
        hours_back: Hours before now to fetch (default 12). Ignored when
                    start_dt and end_dt are both provided.
        start_dt: ISO8601 start datetime (e.g. '2026-03-01T00:00Z').
        end_dt:   ISO8601 end datetime (e.g. '2026-03-01T12:00Z').
    """
    start, end = _resolve_window(hours_back, start_dt, end_dt)
    result = fetch_window_dataframe(start, end)
    return {
        "readings": _df_to_records(result.df),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "count": len(result.df),
        "factors": result.factors,
    }


@mcp.tool
def fetch_generation_mix(
    hours_back: int = 12,
    start_dt: str | None = None,
    end_dt: str | None = None,
) -> dict:
    """Fetch generation mix (fuel-type percentages) for a time window.

    Returns only generation mix columns — no intensity or emission factor data.

    Args:
        hours_back: Hours before now to fetch (default 12). Ignored when
                    start_dt and end_dt are both provided.
        start_dt: ISO8601 start datetime (e.g. '2026-03-01T00:00Z').
        end_dt:   ISO8601 end datetime (e.g. '2026-03-01T12:00Z').
    """
    start, end = _resolve_window(hours_back, start_dt, end_dt)
    result = fetch_window_dataframe(start, end)

    intensity_cols = {
        "timestamp",
        "interval_end",
        "actual_intensity",
        "forecast_intensity",
        "intensity_index",
    }
    mix_cols = [c for c in result.df.columns if c not in intensity_cols]
    mix_df = result.df[["timestamp", "interval_end"] + mix_cols]

    return {
        "readings": _df_to_records(mix_df),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "count": len(mix_df),
    }


# ---------------------------------------------------------------------------
# Custom routes
# ---------------------------------------------------------------------------


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> PlainTextResponse:
    """Simple liveness probe for the MCP server."""
    return PlainTextResponse("OK")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8000)
