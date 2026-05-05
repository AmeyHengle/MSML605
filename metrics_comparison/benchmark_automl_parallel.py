"""
Benchmark: parallel vs sequential AutoML candidate training.

Measures the wall-clock difference between:
  - run_automl(..., parallel=True)   -> candidates trained in a ThreadPoolExecutor
  - run_automl(..., parallel=False)  -> candidates trained one after another

The benchmark uses a synthetic regression dataset (sklearn.datasets.make_regression)
so it is fully self-contained and reproducible. MLflow runs are written to a
throwaway sqlite file inside the output directory so the main experiment store
(mlflow.db) is not polluted.

Outputs (written under --out-dir, default: performance_evaluation/automl/):
  - benchmark_automl_avg_bar.png        bar chart of average time per mode
  - benchmark_automl_boxplot.png        distribution per mode
  - benchmark_automl_per_run.png        per-run line chart
  - benchmark_automl_metrics.json       full metrics (times, speedup, rmse per mode)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml605_pipeline.automl import run_automl, CANDIDATE_MODELS  # noqa: E402


@dataclass(frozen=True)
class AutoMLBenchmarkResult:
    label: str
    runs: int
    times_s: list[float]
    best_rmse: float
    best_name: str

    @property
    def avg_s(self) -> float:
        return statistics.mean(self.times_s)

    @property
    def median_s(self) -> float:
        return statistics.median(self.times_s)

    @property
    def stdev_s(self) -> float:
        return statistics.stdev(self.times_s) if len(self.times_s) >= 2 else 0.0


def _build_dataset(
    n_samples: int,
    n_features: int,
    noise: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_arr, y_arr = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )
    feature_names = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    y = pd.Series(y_arr, name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def _run_one(
    label: str,
    *,
    parallel: bool,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    runs: int,
    experiment_name: str,
) -> AutoMLBenchmarkResult:
    times: list[float] = []
    best_rmse = float("inf")
    best_name = ""
    for i in range(runs):
        with mlflow.start_run(
            run_name=f"{label}_run{i + 1}",
            experiment_id=mlflow.set_experiment(experiment_name).experiment_id,
        ):
            t0 = time.perf_counter()
            res = run_automl(X_train, y_train, X_test, y_test, parallel=parallel)
            dt = time.perf_counter() - t0
        times.append(dt)
        if res.best.rmse < best_rmse:
            best_rmse = res.best.rmse
            best_name = res.best.name
        print(
            f"  [{label}] run {i + 1}/{runs}: {dt:.3f}s "
            f"(best={res.best.name}, rmse={res.best.rmse:.4f})"
        )
    return AutoMLBenchmarkResult(
        label=label, runs=runs, times_s=times, best_rmse=best_rmse, best_name=best_name
    )


def _save_plots(
    out_dir: Path,
    parallel_res: AutoMLBenchmarkResult,
    sequential_res: AutoMLBenchmarkResult,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = [parallel_res.label, sequential_res.label]
    avgs = [parallel_res.avg_s, sequential_res.avg_s]

    plt.figure(figsize=(7, 4))
    bars = plt.bar(labels, avgs, color=["#2E86AB", "#E63946"])
    plt.ylabel("Average time (s)")
    plt.title(
        f"AutoML: parallel vs sequential (avg over {parallel_res.runs} runs, "
        f"{len(CANDIDATE_MODELS)} candidates)"
    )
    for bar, value in zip(bars, avgs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}s",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_automl_avg_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.boxplot(
        [parallel_res.times_s, sequential_res.times_s],
        tick_labels=labels,
        showmeans=True,
    )
    plt.ylabel("Time (s)")
    plt.title("AutoML timing distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_automl_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(
        range(1, parallel_res.runs + 1),
        parallel_res.times_s,
        marker="o",
        label=parallel_res.label,
        color="#2E86AB",
    )
    plt.plot(
        range(1, sequential_res.runs + 1),
        sequential_res.times_s,
        marker="s",
        label=sequential_res.label,
        color="#E63946",
    )
    plt.xlabel("Run #")
    plt.ylabel("Time (s)")
    plt.title("AutoML timing per run")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "benchmark_automl_per_run.png", dpi=200)
    plt.close()


def _print_summary(
    parallel_res: AutoMLBenchmarkResult,
    sequential_res: AutoMLBenchmarkResult,
) -> dict[str, Any]:
    delta_s = sequential_res.avg_s - parallel_res.avg_s
    pct_faster = (delta_s / sequential_res.avg_s * 100.0) if sequential_res.avg_s else float("nan")
    speedup = (sequential_res.avg_s / parallel_res.avg_s) if parallel_res.avg_s else float("inf")

    metrics = {
        "candidate_count": len(CANDIDATE_MODELS),
        "candidate_names": list(CANDIDATE_MODELS.keys()),
        "parallel": {
            "runs": parallel_res.runs,
            "avg_s": parallel_res.avg_s,
            "median_s": parallel_res.median_s,
            "stdev_s": parallel_res.stdev_s,
            "times_s": parallel_res.times_s,
            "best_model": parallel_res.best_name,
            "best_rmse": parallel_res.best_rmse,
        },
        "sequential": {
            "runs": sequential_res.runs,
            "avg_s": sequential_res.avg_s,
            "median_s": sequential_res.median_s,
            "stdev_s": sequential_res.stdev_s,
            "times_s": sequential_res.times_s,
            "best_model": sequential_res.best_name,
            "best_rmse": sequential_res.best_rmse,
        },
        "improvement": {
            "avg_delta_s": delta_s,
            "avg_percent_faster": pct_faster,
            "avg_speedup_x": speedup,
        },
    }

    print()
    print("=" * 60)
    print("AutoML parallel-vs-sequential benchmark summary")
    print("=" * 60)
    print(
        f"Candidates trained per run : {len(CANDIDATE_MODELS)} "
        f"({', '.join(CANDIDATE_MODELS.keys())})"
    )
    print(
        f"Parallel   avg={parallel_res.avg_s:.3f}s  "
        f"median={parallel_res.median_s:.3f}s  stdev={parallel_res.stdev_s:.3f}s"
    )
    print(
        f"Sequential avg={sequential_res.avg_s:.3f}s  "
        f"median={sequential_res.median_s:.3f}s  stdev={sequential_res.stdev_s:.3f}s"
    )
    print("-" * 60)
    print(f"Delta (sequential - parallel) : {delta_s:.3f}s")
    print(f"Percent faster                : {pct_faster:.2f}%")
    print(f"Speedup                       : {speedup:.2f}x")
    print(
        f"Best model (parallel)   : {parallel_res.best_name} "
        f"(rmse={parallel_res.best_rmse:.4f})"
    )
    print(
        f"Best model (sequential) : {sequential_res.best_name} "
        f"(rmse={sequential_res.best_rmse:.4f})"
    )
    print("=" * 60)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark parallel vs sequential AutoML candidate training."
    )
    parser.add_argument("--runs", type=int, default=3, help="Repetitions per mode.")
    parser.add_argument(
        "--n-samples", type=int, default=2000, help="Synthetic rows for make_regression."
    )
    parser.add_argument(
        "--n-features", type=int, default=30, help="Synthetic feature count."
    )
    parser.add_argument("--noise", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="performance_evaluation/automl",
        help="Directory to write plots/metrics into.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help=(
            "Run one untimed warmup of each mode before measuring. Recommended when "
            "imports, JIT paths, and BLAS workers would otherwise pollute the first run."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Isolate MLflow writes from the main experiment store.
    tracking_db = (out_dir / "benchmark_mlflow.db").resolve()
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tracking_db.as_posix()}"
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment_name = "benchmark-automl-parallel"
    print(f"MLflow tracking URI : {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment   : {experiment_name}")

    print(
        f"Building synthetic dataset: n_samples={args.n_samples}, "
        f"n_features={args.n_features}, noise={args.noise}"
    )
    X_train, X_test, y_train, y_test = _build_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        noise=args.noise,
        random_state=args.random_state,
    )
    print(f"Train shape={X_train.shape}, Test shape={X_test.shape}")

    if args.warmup:
        print("Warmup run (parallel)...")
        _run_one(
            "warmup_parallel",
            parallel=True,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            runs=1,
            experiment_name=experiment_name,
        )
        print("Warmup run (sequential)...")
        _run_one(
            "warmup_sequential",
            parallel=False,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            runs=1,
            experiment_name=experiment_name,
        )

    print(f"\nRunning parallel mode ({args.runs} runs)...")
    parallel_res = _run_one(
        "parallel",
        parallel=True,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        runs=int(args.runs),
        experiment_name=experiment_name,
    )

    print(f"\nRunning sequential mode ({args.runs} runs)...")
    sequential_res = _run_one(
        "sequential",
        parallel=False,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        runs=int(args.runs),
        experiment_name=experiment_name,
    )

    _save_plots(out_dir, parallel_res=parallel_res, sequential_res=sequential_res)
    metrics = _print_summary(parallel_res=parallel_res, sequential_res=sequential_res)
    (out_dir / "benchmark_automl_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(f"\nArtifacts written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
