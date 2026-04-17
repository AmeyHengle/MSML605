from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.ml605_pipeline import data as pipeline_data


@dataclass(frozen=True)
class BenchmarkResult:
    label: str
    runs: int
    times_s: list[float]
    rows: int

    @property
    def avg_s(self) -> float:
        return statistics.mean(self.times_s)

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) >= 2 else 0.0


def _sequential_fetch_window_dataframe(
    start_dt: datetime,
    end_dt: datetime,
) -> pipeline_data.WindowFetchResult:
    """
    A single-threaded version of fetch_window_dataframe().

    It deliberately performs the same three network calls (intensity, generation, factors)
    but sequentially, to quantify the impact of ThreadPoolExecutor.
    """
    intensity_url = (
        f"{pipeline_data.INTENSITY_API_BASE}/"
        f"{pipeline_data.to_api_datetime(start_dt)}/"
        f"{pipeline_data.to_api_datetime(end_dt)}"
    )
    generation_url = (
        f"{pipeline_data.GENERATION_API_BASE}/"
        f"{pipeline_data.to_api_datetime(start_dt)}/"
        f"{pipeline_data.to_api_datetime(end_dt)}"
    )

    intensity_payload = pipeline_data._get_json_requests(intensity_url).get("data", [])  # type: ignore[attr-defined]
    generation_payload = pipeline_data._get_json_requests(generation_url).get("data", [])  # type: ignore[attr-defined]
    factors = pipeline_data.fetch_intensity_factors(None)

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

    import pandas as pd

    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    raw_factors_json = json.dumps(factors, indent=2)
    return pipeline_data.WindowFetchResult(df=df, factors=factors, raw_factors_json=raw_factors_json)


def _run_benchmark(
    label: str,
    func,
    start_dt: datetime,
    end_dt: datetime,
    runs: int,
) -> BenchmarkResult:
    times: list[float] = []
    rows = 0
    for _ in range(runs):
        t0 = time.perf_counter()
        result = func(start_dt, end_dt)
        dt = time.perf_counter() - t0
        rows = len(result.df)
        times.append(dt)
    return BenchmarkResult(label=label, runs=runs, times_s=times, rows=rows)


def _save_plots(
    out_dir: Path,
    threaded: BenchmarkResult,
    sequential: BenchmarkResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = [threaded.label, sequential.label]
    avgs = [threaded.avg_s, sequential.avg_s]

    # 1) Bar chart (average)
    plt.figure(figsize=(7, 4))
    plt.bar(labels, avgs)
    plt.ylabel("Average time (s)")
    plt.title("Fetch window timing (avg over runs)")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_avg_bar.png", dpi=200)
    plt.close()

    # 2) Box plot (distribution)
    plt.figure(figsize=(7, 4))
    plt.boxplot([threaded.times_s, sequential.times_s], tick_labels=labels, showmeans=True)
    plt.ylabel("Time (s)")
    plt.title("Fetch window timing distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_times_boxplot.png", dpi=200)
    plt.close()

    # 3) Per-run line plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, threaded.runs + 1), threaded.times_s, marker="o", label=threaded.label)
    plt.plot(range(1, sequential.runs + 1), sequential.times_s, marker="o", label=sequential.label)
    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.title("Fetch window timing per run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_times_per_run.png", dpi=200)
    plt.close()


def _print_metrics(threaded: BenchmarkResult, sequential: BenchmarkResult) -> dict[str, Any]:
    delta_s = sequential.avg_s - threaded.avg_s
    pct_faster = (delta_s / sequential.avg_s * 100.0) if sequential.avg_s else float("nan")
    speedup = (sequential.avg_s / threaded.avg_s) if threaded.avg_s else float("inf")

    metrics = {
        "threaded": {
            "runs": threaded.runs,
            "rows": threaded.rows,
            "avg_s": threaded.avg_s,
            "median_s": threaded.median_s,
            "stdev_s": threaded.stdev_s,
            "times_s": threaded.times_s,
        },
        "sequential": {
            "runs": sequential.runs,
            "rows": sequential.rows,
            "avg_s": sequential.avg_s,
            "median_s": sequential.median_s,
            "stdev_s": sequential.stdev_s,
            "times_s": sequential.times_s,
        },
        "improvement": {
            "avg_delta_s": delta_s,
            "avg_percent_faster": pct_faster,
            "avg_speedup_x": speedup,
        },
    }

    print("=== Benchmark summary ===")
    print(f"Threaded   avg={threaded.avg_s:.6f}s  median={threaded.median_s:.6f}s  stdev={threaded.stdev_s:.6f}s")
    print(f"Sequential avg={sequential.avg_s:.6f}s  median={sequential.median_s:.6f}s  stdev={sequential.stdev_s:.6f}s")
    print("--- Improvement (avg) ---")
    print(f"Delta (sequential - threaded): {delta_s:.6f}s")
    print(f"Percent faster: {pct_faster:.3f}%")
    print(f"Speedup: {speedup:.3f}x")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark threaded vs sequential window fetching.")
    parser.add_argument("--runs", type=int, default=5, help="Number of repetitions per mode.")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=1.0,
        help="Window size (hours) for the API fetch.",
    )
    parser.add_argument(
        "--start-utc",
        type=str,
        default="2026-04-03T12:00:00Z",
        help="Start time in UTC, e.g. 2026-04-03T12:00:00Z",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="performance_evaluation",
        help="Directory to write plots/metrics into.",
    )
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
    end = start + timedelta(hours=float(args.window_hours))

    threaded = _run_benchmark(
        label="Threaded",
        func=lambda s, e: pipeline_data.fetch_window_dataframe(s, e),
        start_dt=start,
        end_dt=end,
        runs=int(args.runs),
    )
    sequential = _run_benchmark(
        label="Sequential",
        func=_sequential_fetch_window_dataframe,
        start_dt=start,
        end_dt=end,
        runs=int(args.runs),
    )

    out_dir = Path(args.out_dir)
    _save_plots(out_dir, threaded=threaded, sequential=sequential)

    metrics = _print_metrics(threaded=threaded, sequential=sequential)
    (out_dir / "benchmark_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

