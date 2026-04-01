from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests


INTENSITY_API_BASE = "https://api.carbonintensity.org.uk/intensity"
GENERATION_API_BASE = "https://api.carbonintensity.org.uk/generation"
FACTORS_API_URL = "https://api.carbonintensity.org.uk/intensity/factors"


def to_api_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%MZ")


def fetch_intensity_factors(session: requests.Session | None = None) -> dict[str, float]:
    s = session or requests.Session()
    resp = s.get(FACTORS_API_URL, headers={"Accept": "application/json"}, timeout=30)
    resp.raise_for_status()
    payload = resp.json().get("data", [])
    return payload[0] if payload else {}


def _get_json(session: requests.Session, url: str, retries: int = 3) -> Any:
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - intentionally broad for retries
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed GET after {retries} retries: {url}") from last_exc


@dataclass(frozen=True)
class WindowFetchResult:
    df: pd.DataFrame
    factors: dict[str, float]
    raw_factors_json: str


def fetch_window_dataframe(
    start_dt: datetime,
    end_dt: datetime,
    *,
    session: requests.Session | None = None,
) -> WindowFetchResult:
    """
    Fetch intensity + generationmix for [start_dt, end_dt] and merge on interval start time ("from").

    Note: Carbon Intensity API resolution is not 30s. Your pipeline interval_seconds is an execution cadence,
    not necessarily the API sampling period.
    """
    s = session or requests.Session()

    intensity_url = f"{INTENSITY_API_BASE}/{to_api_datetime(start_dt)}/{to_api_datetime(end_dt)}"
    generation_url = f"{GENERATION_API_BASE}/{to_api_datetime(start_dt)}/{to_api_datetime(end_dt)}"

    intensity_payload = _get_json(s, intensity_url).get("data", [])
    generation_payload = _get_json(s, generation_url).get("data", [])

    generation_by_from = {item.get("from"): item for item in generation_payload}

    rows: list[dict[str, Any]] = []
    for item in intensity_payload:
        intensity = item.get("intensity", {}) or {}
        generation_item = generation_by_from.get(item.get("from"), {})
        mix = generation_item.get("generationmix", []) or []

        row: dict[str, Any] = {
            "timestamp": item.get("from"),
            "interval_end": item.get("to"),
            "actual_intensity": intensity.get("actual"),
            "forecast_intensity": intensity.get("forecast"),
            "intensity_index": intensity.get("index"),
        }
        for fuel_item in mix:
            fuel = fuel_item.get("fuel")
            if fuel:
                row[fuel] = fuel_item.get("perc")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    factors = fetch_intensity_factors(s)
    raw_factors_json = json.dumps(factors, indent=2)

    return WindowFetchResult(df=df, factors=factors, raw_factors_json=raw_factors_json)