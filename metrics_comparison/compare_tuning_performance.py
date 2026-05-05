from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


DEFAULT_BASELINE_EXPERIMENT = "intensity-model-training"
DEFAULT_TUNED_EXPERIMENT = "intensity-model-automl"
DEFAULT_METRIC = "rmse"
DEFAULT_TARGET = 40.0


def metric_hits_target(value: float, threshold: float, higher_is_better: bool) -> bool:
    return value >= threshold if higher_is_better else value <= threshold


def load_runs(experiment_name: str, metric_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )

    metric_col = f"metrics.{metric_name}"
    required = [metric_col, "start_time", "end_time", "run_id", "tags.mlflow.runName"]
    available = [c for c in required if c in runs.columns]
    df = runs[available].copy()

    if metric_col not in df.columns:
        raise ValueError(
            f"Metric '{metric_name}' was not found in experiment '{experiment_name}'."
        )

    df = df.dropna(subset=[metric_col]).sort_values("start_time").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No runs with metric '{metric_name}' found in '{experiment_name}'.")
    return df


def compute_time_to_target(
    experiment_name: str,
    metric_name: str,
    threshold: float,
    higher_is_better: bool,
    label: str,
) -> dict[str, Any]:
    df = load_runs(experiment_name, metric_name)
    metric_col = f"metrics.{metric_name}"

    first_start = df.iloc[0]["start_time"]
    reached_idx = None
    for idx, row in df.iterrows():
        if metric_hits_target(float(row[metric_col]), threshold, higher_is_better):
            reached_idx = idx
            break

    best_metric = float(df[metric_col].max() if higher_is_better else df[metric_col].min())

    if reached_idx is None:
        return {
            "Workflow": label,
            "Experiment": experiment_name,
            "Metric": metric_name,
            "Target": threshold,
            "Reached Target": False,
            "Runs Needed": None,
            "Best Metric": best_metric,
            "Time To Target Seconds": None,
            "Reached Run Name": None,
        }

    reached = df.iloc[reached_idx]
    reached_end = reached.get("end_time", reached["start_time"])
    time_to_target_seconds = float((reached_end - first_start).total_seconds())

    return {
        "Workflow": label,
        "Experiment": experiment_name,
        "Metric": metric_name,
        "Target": threshold,
        "Reached Target": True,
        "Runs Needed": int(reached_idx + 1),
        "Best Metric": best_metric,
        "Time To Target Seconds": time_to_target_seconds,
        "Reached Run Name": reached.get("tags.mlflow.runName"),
    }


def compare_time_to_target(
    baseline_experiment: str,
    tuned_experiment: str,
    metric_name: str,
    threshold: float,
    higher_is_better: bool,
) -> pd.DataFrame:
    baseline = compute_time_to_target(
        baseline_experiment,
        metric_name,
        threshold,
        higher_is_better,
        "Before Tuning (Baseline)",
    )
    tuned = compute_time_to_target(
        tuned_experiment,
        metric_name,
        threshold,
        higher_is_better,
        "After Tuning (MLflow AutoML)",
    )

    df = pd.DataFrame([baseline, tuned])
    if df["Reached Target"].all():
        before_s = float(df.iloc[0]["Time To Target Seconds"])
        after_s = float(df.iloc[1]["Time To Target Seconds"])
        saved = before_s - after_s
        pct_saved = (saved / before_s * 100.0) if before_s > 0 else 0.0
        df["Absolute Time Saved Seconds"] = [saved, saved]
        df["Percent Time Saved"] = [pct_saved, pct_saved]
    else:
        df["Absolute Time Saved Seconds"] = None
        df["Percent Time Saved"] = None
    return df


def run_command(command: list[str], cwd: Path) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), check=True)
    return time.perf_counter() - started


def compare_single_run_runtime(project_root: Path) -> pd.DataFrame:
    before_seconds = run_command(["uv", "run", "python", "train_with_mlflow.py"], project_root)
    after_seconds = run_command(["uv", "run", "python", "train_automl.py"], project_root)

    saved = before_seconds - after_seconds
    pct_saved = (saved / before_seconds * 100.0) if before_seconds > 0 else 0.0

    return pd.DataFrame(
        [
            {"Workflow": "Before Tuning (Baseline)", "Runtime Seconds": before_seconds},
            {"Workflow": "After Tuning (MLflow AutoML)", "Runtime Seconds": after_seconds},
            {"Workflow": "Difference (Before - After)", "Runtime Seconds": saved, "Percent Saved": pct_saved},
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare performance before/after hyperparameter tuning with MLflow."
    )
    parser.add_argument("--baseline-experiment", default=DEFAULT_BASELINE_EXPERIMENT)
    parser.add_argument("--tuned-experiment", default=DEFAULT_TUNED_EXPERIMENT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Set this for metrics like r2 where larger is better.",
    )
    parser.add_argument(
        "--output-csv",
        default="metrics_improvements/performance_before_after.csv",
        help="Output CSV for MLflow history time-to-target comparison.",
    )
    parser.add_argument(
        "--benchmark-runtime",
        action="store_true",
        help="Also execute both scripts once and compare wall-clock runtime.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root used when --benchmark-runtime is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    comparison_df = compare_time_to_target(
        baseline_experiment=args.baseline_experiment,
        tuned_experiment=args.tuned_experiment,
        metric_name=args.metric,
        threshold=args.target,
        higher_is_better=args.higher_is_better,
    )

    print("\n=== Time-to-Target Comparison (MLflow History) ===")
    print(comparison_df.to_string(index=False))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    if args.benchmark_runtime:
        runtime_df = compare_single_run_runtime(Path(args.project_root).resolve())
        runtime_csv = output_csv.with_name("single_run_runtime_before_after.csv")
        runtime_df.to_csv(runtime_csv, index=False)
        print("\n=== Single-Run Runtime Benchmark ===")
        print(runtime_df.to_string(index=False))
        print(f"\nSaved: {runtime_csv}")


if __name__ == "__main__":
    main()

