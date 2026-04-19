from __future__ import annotations

import argparse
import logging
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
BASELINE_LABEL = "Without MLflow Tuning (Baseline Workflow)"
TUNED_LABEL = "With MLflow Tuning (AutoML Workflow)"

logger = logging.getLogger(__name__)


def _run_command(command: list[str], cwd: Path, timeout: float | None = None) -> float:
    started = time.perf_counter()
    subprocess.run(command, cwd=str(cwd), check=True, timeout=timeout)
    return time.perf_counter() - started


def _metric_meets_target(value: float, threshold: float, higher_is_better: bool) -> bool:
    return value >= threshold if higher_is_better else value <= threshold


def _get_experiment_runs(experiment_name: str, metric_name: str) -> pd.DataFrame:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        available = [exp.name for exp in mlflow.search_experiments()]
        raise ValueError(
            f"Experiment '{experiment_name}' not found in MLflow. Available: {available}"
        )

    df_runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        output_format="pandas",
    )
    metric_col = f"metrics.{metric_name}"
    if metric_col not in df_runs.columns:
        raise ValueError(
            f"Metric '{metric_name}' was not found in experiment '{experiment_name}'."
        )

    required_cols = [metric_col, "start_time", "end_time", "run_id", "tags.mlflow.runName"]
    available_cols = [c for c in required_cols if c in df_runs.columns]
    df = df_runs[available_cols].copy()
    df = df.dropna(subset=[metric_col]).sort_values("start_time").reset_index(drop=True)
    return df


def _run_duration_seconds(row: pd.Series) -> float:
    """Compute wall-clock duration for a single run.

    Falls back to 0 when end_time is missing or precedes start_time
    (e.g., interrupted runs), so failed runs don't skew the total.
    """
    start = row.get("start_time")
    end = row.get("end_time")
    if pd.isna(start) or pd.isna(end):
        return 0.0
    delta = (end - start).total_seconds()
    return max(0.0, float(delta))


def compute_time_to_target_from_history(
    experiment_name: str,
    metric_name: str,
    threshold: float,
    higher_is_better: bool,
    workflow_label: str,
) -> dict[str, Any]:
    """Compute time-to-target from MLflow run history in one experiment.

    Time-to-target is the sum of per-run durations from the first run up to and
    including the first run that meets the target. This measures compute effort
    rather than wall-clock idle time between runs.
    """
    df = _get_experiment_runs(experiment_name, metric_name)
    metric_col = f"metrics.{metric_name}"

    base_result: dict[str, Any] = {
        "Workflow": workflow_label,
        "Experiment": experiment_name,
        "Metric Name": metric_name,
        "Target Threshold": threshold,
        "Higher Is Better": higher_is_better,
        "Best Metric": None,
        "Reached Target": False,
        "Runs Needed": None,
        "Time To Target Seconds": None,
        "Time To Target Minutes": None,
        "Reached Run Name": None,
    }

    if df.empty:
        logger.warning("No runs with metric '%s' in experiment '%s'", metric_name, experiment_name)
        return base_result

    best_metric = float(df[metric_col].max() if higher_is_better else df[metric_col].min())
    base_result["Best Metric"] = best_metric

    reached_idx = None
    for idx, row in df.iterrows():
        if _metric_meets_target(float(row[metric_col]), threshold, higher_is_better):
            reached_idx = idx
            break

    if reached_idx is None:
        return base_result

    cumulative_seconds = sum(
        _run_duration_seconds(df.iloc[i]) for i in range(reached_idx + 1)
    )
    reached_row = df.iloc[reached_idx]

    return {
        **base_result,
        "Reached Target": True,
        "Runs Needed": int(reached_idx + 1),
        "Time To Target Seconds": cumulative_seconds,
        "Time To Target Minutes": cumulative_seconds / 60.0,
        "Reached Run Name": reached_row.get("tags.mlflow.runName"),
    }


def compare_workflows(
    baseline_experiment: str,
    tuned_experiment: str,
    metric_name: str,
    threshold: float,
    higher_is_better: bool,
) -> pd.DataFrame:
    baseline = compute_time_to_target_from_history(
        experiment_name=baseline_experiment,
        metric_name=metric_name,
        threshold=threshold,
        higher_is_better=higher_is_better,
        workflow_label=BASELINE_LABEL,
    )
    tuned = compute_time_to_target_from_history(
        experiment_name=tuned_experiment,
        metric_name=metric_name,
        threshold=threshold,
        higher_is_better=higher_is_better,
        workflow_label=TUNED_LABEL,
    )

    df = pd.DataFrame([baseline, tuned])
    if df["Reached Target"].all():
        baseline_t = float(df.iloc[0]["Time To Target Seconds"])
        tuned_t = float(df.iloc[1]["Time To Target Seconds"])
        saved = baseline_t - tuned_t
        pct = (saved / baseline_t * 100.0) if baseline_t > 0 else 0.0
        df["Absolute Time Saved Seconds"] = [saved, saved]
        df["Percent Time Saved"] = [pct, pct]
    else:
        df["Absolute Time Saved Seconds"] = None
        df["Percent Time Saved"] = None
    return df


def trigger_original_pipelines(
    project_root: Path,
    baseline_command: list[str],
    tuned_command: list[str],
    timeout: float | None = None,
) -> pd.DataFrame:
    baseline_s = _run_command(baseline_command, cwd=project_root, timeout=timeout)
    tuned_s = _run_command(tuned_command, cwd=project_root, timeout=timeout)
    saved = baseline_s - tuned_s
    pct = (saved / baseline_s * 100.0) if baseline_s > 0 else 0.0

    return pd.DataFrame(
        [
            {"Workflow": BASELINE_LABEL, "Runtime Seconds": baseline_s, "Percent Time Saved": None},
            {"Workflow": TUNED_LABEL, "Runtime Seconds": tuned_s, "Percent Time Saved": None},
            {"Workflow": "Time Savings", "Runtime Seconds": saved, "Percent Time Saved": pct},
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare time-to-target performance with vs without MLflow tuning."
    )
    parser.add_argument("--baseline-experiment", default=DEFAULT_BASELINE_EXPERIMENT)
    parser.add_argument("--tuned-experiment", default=DEFAULT_TUNED_EXPERIMENT)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--target",
        type=float,
        nargs="?",
        const=DEFAULT_TARGET,
        default=DEFAULT_TARGET,
        help="Target threshold for the chosen metric. Defaults to %(default)s.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Use when larger metric is better (e.g., r2).",
    )
    parser.add_argument(
        "--output-csv",
        default="metrics_improvements/time_to_target_comparison.csv",
    )
    parser.add_argument(
        "--trigger-original-pipelines",
        action="store_true",
        help=(
            "Also execute original scripts and log one-shot runtime. "
            "Baseline: uv run python train_with_mlflow.py, "
            "Tuned: uv run python train_automl.py"
        ),
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repo root used when executing baseline/tuned scripts.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    project_root = Path(args.project_root).resolve()

    logger.info(
        "Analyzing time-to-target metric=%s target=%s higher_is_better=%s",
        args.metric, args.target, args.higher_is_better,
    )
    comparison_df = compare_workflows(
        baseline_experiment=args.baseline_experiment,
        tuned_experiment=args.tuned_experiment,
        metric_name=args.metric,
        threshold=args.target,
        higher_is_better=args.higher_is_better,
    )
    logger.info("MLflow history time-to-target comparison:\n%s", comparison_df.to_string(index=False))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    logger.info("Saved comparison CSV: %s", output_csv)

    if args.trigger_original_pipelines:
        logger.info("Triggering original pipeline scripts for one-shot runtime benchmark...")
        runtime_df = trigger_original_pipelines(
            project_root=project_root,
            baseline_command=["uv", "run", "python", "train_with_mlflow.py"],
            tuned_command=["uv", "run", "python", "train_automl.py"],
        )
        runtime_csv = output_csv.with_name("pipeline_runtime_comparison.csv")
        runtime_df.to_csv(runtime_csv, index=False)
        logger.info("Runtime comparison:\n%s", runtime_df.to_string(index=False))
        logger.info("Saved runtime CSV: %s", runtime_csv)


if __name__ == "__main__":
    main()
