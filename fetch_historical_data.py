from __future__ import annotations

import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import mlflow
import pandas as pd
import requests

from log_tracking import setup_run_logging


INTENSITY_API_BASE = "https://api.carbonintensity.org.uk/intensity"
GENERATION_API_BASE = "https://api.carbonintensity.org.uk/generation"
FACTORS_API_URL = "https://api.carbonintensity.org.uk/intensity/factors"

START_DATE = datetime(2020, 3, 24, 0, 0, 0, tzinfo=UTC)
END_DATE = datetime(2026, 3, 24, 23, 59, 0, tzinfo=UTC)
OUTPUT_CSV = Path("historical_data.csv")
FACTORS_JSON = Path("intensity_factors.json")


def to_api_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%MZ")


def get_month_end(value: datetime) -> datetime:
    next_month = value.replace(day=28) + timedelta(days=4)
    month_end = next_month - timedelta(days=next_month.day)
    return month_end.replace(hour=23, minute=59, second=59, microsecond=0)


def count_month_chunks(start_dt: datetime, end_dt: datetime) -> int:
    if start_dt > end_dt:
        return 0
    count = 0
    cursor = start_dt
    while cursor <= end_dt:
        count += 1
        cursor = (get_month_end(cursor) + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    return count


def fetch_intensity_factors() -> dict[str, float]:
    response = requests.get(
        FACTORS_API_URL,
        headers={"Accept": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json().get("data", [])
    if not payload:
        return {}
    return payload[0]


def normalize_factor_name(name: str) -> str:
    # Convert API keys like "Gas (Combined Cycle)" -> "factor_gas_combined_cycle".
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"factor_{normalized}"


def fetch_data_for_range(start_dt: datetime, end_dt: datetime) -> list[dict]:
    intensity_url = (
        f"{INTENSITY_API_BASE}/{to_api_datetime(start_dt)}/{to_api_datetime(end_dt)}"
    )
    generation_url = (
        f"{GENERATION_API_BASE}/{to_api_datetime(start_dt)}/{to_api_datetime(end_dt)}"
    )

    intensity_response = requests.get(intensity_url, timeout=30)
    intensity_response.raise_for_status()
    intensity_payload = intensity_response.json().get("data", [])

    generation_response = requests.get(generation_url, timeout=30)
    generation_response.raise_for_status()
    generation_payload = generation_response.json().get("data", [])

    generation_by_from = {item.get("from"): item for item in generation_payload}

    records: list[dict] = []
    for item in intensity_payload:
        intensity = item.get("intensity", {}) or {}
        generation_item = generation_by_from.get(item.get("from"), {})
        mix = generation_item.get("generationmix", []) or []

        record = {
            "timestamp": item.get("from"),
            "interval_end": item.get("to"),
            "actual_intensity": intensity.get("actual"),
            "forecast_intensity": intensity.get("forecast"),
            "intensity_index": intensity.get("index"),
        }
        for fuel_item in mix:
            fuel = fuel_item.get("fuel")
            if fuel:
                record[fuel] = fuel_item.get("perc")
        records.append(record)
    return records


def main() -> None:
    logger = setup_run_logging("fetch_historical_data")

    mlflow.set_experiment("historical-data-pipeline")

    all_historical_data: list[dict] = []
    current_chunk_start_date = START_DATE
    chunk_count = 0
    total_chunks = count_month_chunks(START_DATE, END_DATE)

    logger.info(
        "Starting historical data retrieval from %s to %s",
        START_DATE.strftime("%Y-%m-%d %H:%M"),
        END_DATE.strftime("%Y-%m-%d %H:%M"),
    )
    logger.info("Total chunks to fetch: %s", total_chunks)

    with mlflow.start_run(run_name="monthly_historical_data_pull"):
        mlflow.log_param("start_date", START_DATE.isoformat())
        mlflow.log_param("end_date", END_DATE.isoformat())
        mlflow.log_param("chunk_granularity", "monthly")

        factors = fetch_intensity_factors()
        FACTORS_JSON.write_text(json.dumps(factors, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(FACTORS_JSON))
        mlflow.log_metric("factor_count", len(factors))
        factor_columns = {
            normalize_factor_name(key): value for key, value in factors.items()
        }

        while current_chunk_start_date <= END_DATE:
            current_month_end = get_month_end(current_chunk_start_date)
            current_chunk_end_date = min(current_month_end, END_DATE)

            if current_chunk_start_date > current_chunk_end_date:
                break

            chunk_data = fetch_data_for_range(
                current_chunk_start_date, current_chunk_end_date
            )
            if chunk_data:
                all_historical_data.extend(chunk_data)

            chunk_count += 1
            progress_pct = (chunk_count / total_chunks) * 100 if total_chunks else 100.0
            logger.info(
                "[%s/%s] %6.2f%% | %s -> %s | chunk_rows=%s total_rows=%s",
                chunk_count,
                total_chunks,
                progress_pct,
                current_chunk_start_date.strftime("%Y-%m-%d"),
                current_chunk_end_date.strftime("%Y-%m-%d"),
                len(chunk_data),
                len(all_historical_data),
            )
            current_chunk_start_date = (current_month_end + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )

        df_historical = pd.DataFrame(all_historical_data)
        if df_historical.empty:
            mlflow.log_metric("rows_fetched", 0)
            mlflow.log_metric("chunks_processed", chunk_count)
            logger.warning("No historical data fetched.")
            return

        df_historical["timestamp"] = pd.to_datetime(
            df_historical["timestamp"], utc=True, errors="coerce"
        )
        df_historical["interval_end"] = pd.to_datetime(
            df_historical["interval_end"], utc=True, errors="coerce"
        )
        df_historical = df_historical.dropna(subset=["timestamp"])
        df_historical = df_historical.sort_values(by="timestamp").reset_index(drop=True)

        core_cols = {
            "timestamp",
            "interval_end",
            "actual_intensity",
            "forecast_intensity",
            "intensity_index",
        }
        fuel_cols = sorted(
            {key for row in all_historical_data for key in row.keys() if key not in core_cols}
        )

        for fuel in fuel_cols:
            if fuel not in df_historical.columns:
                df_historical[fuel] = 0.0
        if fuel_cols:
            df_historical[fuel_cols] = df_historical[fuel_cols].fillna(0.0)

        # Add /intensity/factors fields as constant feature columns for each row.
        for factor_col, factor_value in factor_columns.items():
            df_historical[factor_col] = factor_value

        df_historical.to_csv(OUTPUT_CSV, index=False)

        mlflow.log_metric("rows_fetched", int(len(df_historical)))
        mlflow.log_metric("chunks_processed", chunk_count)
        mlflow.log_metric("num_fuel_types", len(fuel_cols))
        mlflow.log_metric("num_factor_features_added", len(factor_columns))
        mlflow.log_artifact(str(OUTPUT_CSV))

        logger.info("Successfully fetched %s historical records.", len(df_historical))
        logger.info("Head:\n%s", df_historical.head().to_string())


if __name__ == "__main__":
    main()
