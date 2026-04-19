"""Manual hyperparameter grid search vs AutoML - wall-clock comparison.

This script is a *standalone* benchmark. It is NOT part of the main training
pipeline and it does not log to MLflow or the Model Registry.

It answers the question:

    "If I had to tune hyperparameters the old way (plain grid search, one
     model family at a time), how long would it take me to reach the same
     quality that our AutoML workflow produces in a single pass?"

How it works
------------
1. Load `historical_data.csv` and build the same train/test split used by
   the production training scripts (`ml605_pipeline.features` +
   `ml605_pipeline.modeling.time_split`).
2. Run a manual grid search over RandomForest / GradientBoosting / Ridge /
   ElasticNet. Every combination is fit once on the training split and
   scored on the test split. Fit time and RMSE are recorded per combo.
3. Run the AutoML candidate set from `ml605_pipeline.automl.CANDIDATE_MODELS`
   the same way (no MLflow side effects). AutoML's best RMSE becomes the
   "target" the manual search has to match.
4. Emit a summary CSV, a per-combo detailed CSV, and a markdown report
   showing wall-clock time, best RMSE, and time-to-target for both sides.

Typical usage:

    uv run python metrics_improvements/manual_vs_automl_comparison.py \\
        --preset large
"""

from __future__ import annotations

import argparse
import itertools
import logging
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error

from ml605_pipeline.features import (
    add_time_features,
    ensure_feature_columns,
    load_feature_list,
    one_hot_intensity_index,
)
from ml605_pipeline.modeling import time_split


logger = logging.getLogger(__name__)

MANUAL_LABEL = "Manual Grid Search (no MLflow / no AutoML)"
AUTOML_LABEL = "AutoML (ml605_pipeline.automl)"


# ---------------------------------------------------------------------------
# Hyperparameter grid presets.
# Numbers were picked so LARGE lands in the 300+ combo range.
# ---------------------------------------------------------------------------

SMALL_GRID: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 300],
        "max_depth": [10, 14, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
}  # 2*3*3 + 4 = 22

MEDIUM_GRID: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [6, 10, 14, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },
}  # 4*4*3 + 3*3*3 + 6 = 48 + 27 + 6 = 81

LARGE_GRID: dict[str, dict[str, list[Any]]] = {
    "random_forest": {
        "n_estimators": [100, 200, 300, 500, 800],
        "max_depth": [6, 10, 14, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2, 4, 8],
    },  # 5 * 5 * 2 * 4 = 200
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.8, 1.0],
    },  # 3 * 3 * 4 * 2 = 72
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    },  # 6
    "elastic_net": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    },  # 25
}  # total = 303

PRESETS: dict[str, dict[str, dict[str, list[Any]]]] = {
    "small": SMALL_GRID,
    "medium": MEDIUM_GRID,
    "large": LARGE_GRID,
}


def _rf(params: dict[str, Any]) -> RandomForestRegressor:
    return RandomForestRegressor(random_state=42, n_jobs=-1, **params)


def _gbr(params: dict[str, Any]) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(random_state=42, **params)


def _ridge(params: dict[str, Any]) -> Ridge:
    return Ridge(**params)


def _elastic_net(params: dict[str, Any]) -> ElasticNet:
    return ElasticNet(random_state=42, max_iter=10_000, **params)


ESTIMATOR_FACTORIES: dict[str, Callable[[dict[str, Any]], Any]] = {
    "random_forest": _rf,
    "gradient_boosting": _gbr,
    "ridge": _ridge,
    "elastic_net": _elastic_net,
}


@dataclass
class ComboResult:
    combo_index: int
    model_family: str
    params: dict[str, Any]
    rmse: float
    fit_seconds: float
    cumulative_seconds: float


@dataclass
class ApproachSummary:
    approach: str
    combos_tried: int
    wall_clock_seconds: float
    best_rmse: float
    target_rmse: float
    reached_target: bool
    time_to_target_seconds: float | None
    combos_to_target: int | None
    winning_family: str
    winning_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Data loading - mirrors what train_automl.py does, without MLflow.
# ---------------------------------------------------------------------------


def load_data_split(
    data_path: Path, features_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Feature list file not found: {features_path}")

    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError(f"{data_path} is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = (
        df.dropna(subset=["timestamp", "actual_intensity"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    df = add_time_features(df)
    df = one_hot_intensity_index(df)

    feature_cols = load_feature_list(features_path)
    df = ensure_feature_columns(df, feature_cols)

    X_train, X_test, y_train, y_test = time_split(df, feature_cols)
    if len(X_test) == 0:
        raise ValueError("Not enough rows to create a test split.")
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Manual grid search.
# ---------------------------------------------------------------------------


def _iter_grid(
    grid: dict[str, dict[str, list[Any]]]
) -> list[tuple[str, dict[str, Any]]]:
    combos: list[tuple[str, dict[str, Any]]] = []
    for family, param_grid in grid.items():
        keys = list(param_grid.keys())
        for values in itertools.product(*[param_grid[k] for k in keys]):
            combos.append((family, dict(zip(keys, values))))
    return combos


def run_manual_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    grid: dict[str, dict[str, list[Any]]],
    log_every: int = 10,
) -> list[ComboResult]:
    combos = _iter_grid(grid)
    logger.info("Manual grid search: %d combinations", len(combos))

    results: list[ComboResult] = []
    cumulative = 0.0
    for idx, (family, params) in enumerate(combos, start=1):
        factory = ESTIMATOR_FACTORIES[family]
        model = factory(params)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elapsed = time.perf_counter() - started
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)
        cumulative += elapsed

        results.append(
            ComboResult(
                combo_index=idx,
                model_family=family,
                params=params,
                rmse=rmse,
                fit_seconds=elapsed,
                cumulative_seconds=cumulative,
            )
        )
        if idx == 1 or idx % log_every == 0 or idx == len(combos):
            best_so_far = min(r.rmse for r in results)
            logger.info(
                "  [%4d/%d] %-18s rmse=%.4f fit=%.2fs cum=%.1fs best=%.4f",
                idx, len(combos), family, rmse, elapsed, cumulative, best_so_far,
            )
    return results


# ---------------------------------------------------------------------------
# AutoML reference - uses the same candidate set as ml605_pipeline.automl,
# but evaluates without any MLflow activity.
# ---------------------------------------------------------------------------


def run_automl_candidates(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[ComboResult]:
    from ml605_pipeline.automl import CANDIDATE_MODELS

    results: list[ComboResult] = []
    cumulative = 0.0
    for idx, (name, template) in enumerate(CANDIDATE_MODELS.items(), start=1):
        model = clone(template)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        elapsed = time.perf_counter() - started
        rmse = float(mean_squared_error(y_test, preds) ** 0.5)
        cumulative += elapsed

        results.append(
            ComboResult(
                combo_index=idx,
                model_family=name,
                params=template.get_params(),  # type: ignore[union-attr]
                rmse=rmse,
                fit_seconds=elapsed,
                cumulative_seconds=cumulative,
            )
        )
        logger.info(
            "  [automl %d/%d] %-24s rmse=%.4f fit=%.2fs",
            idx, len(CANDIDATE_MODELS), name, rmse, elapsed,
        )
    return results


# ---------------------------------------------------------------------------
# Summarisation.
# ---------------------------------------------------------------------------


def summarise(
    label: str, results: list[ComboResult], target_rmse: float, wall_clock_seconds: float
) -> ApproachSummary:
    best = min(results, key=lambda r: r.rmse)
    reached = next((r for r in results if r.rmse <= target_rmse), None)
    return ApproachSummary(
        approach=label,
        combos_tried=len(results),
        wall_clock_seconds=wall_clock_seconds,
        best_rmse=best.rmse,
        target_rmse=target_rmse,
        reached_target=reached is not None,
        time_to_target_seconds=(reached.cumulative_seconds if reached else None),
        combos_to_target=(reached.combo_index if reached else None),
        winning_family=best.model_family,
        winning_params=best.params,
    )


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 60:
        return f"{value:.1f}s"
    return f"{value / 60.0:.2f}m"


def write_markdown_report(
    manual: ApproachSummary,
    automl: ApproachSummary,
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Manual Grid Search vs AutoML - Time-to-Target Report\n")
    lines.append(
        "This report compares a **manual hyperparameter grid search** (the old "
        "way: no MLflow, no AutoML, just nested loops) against the project's "
        "**AutoML** candidate set on the same train/test split.\n"
    )
    lines.append("## Summary\n")
    lines.append("| Approach | Combos Tried | Wall Clock | Best RMSE | Reached Target | Time To Target | Combos To Target |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in (manual, automl):
        lines.append(
            f"| {s.approach} | {s.combos_tried} | {_fmt_seconds(s.wall_clock_seconds)} | "
            f"{s.best_rmse:.4f} | {s.reached_target} | "
            f"{_fmt_seconds(s.time_to_target_seconds)} | "
            f"{s.combos_to_target if s.combos_to_target is not None else 'N/A'} |"
        )
    lines.append("")

    if manual.reached_target and automl.reached_target:
        saved = float(manual.time_to_target_seconds) - float(automl.time_to_target_seconds)  # type: ignore[arg-type]
        denom = float(manual.time_to_target_seconds)  # type: ignore[arg-type]
        pct = (saved / denom * 100.0) if denom > 0 else 0.0
        if saved >= 0:
            lines.append(
                f"**Conclusion:** AutoML reached RMSE <= {manual.target_rmse:.4f} "
                f"in `{_fmt_seconds(automl.time_to_target_seconds)}` vs "
                f"`{_fmt_seconds(manual.time_to_target_seconds)}` for the manual "
                f"grid search - a time saving of `{saved:.1f}s` ({pct:.2f}%).\n"
            )
        else:
            lines.append(
                f"**Conclusion:** On this run the manual grid happened to reach "
                f"RMSE <= {manual.target_rmse:.4f} faster than AutoML "
                f"({_fmt_seconds(manual.time_to_target_seconds)} vs "
                f"{_fmt_seconds(automl.time_to_target_seconds)}). Try a larger "
                f"grid or a tighter target.\n"
            )
    else:
        lines.append(
            "**Conclusion:** At least one approach did not reach the target RMSE. "
            "Loosen `--target-rmse` or widen the grid to get a clean comparison.\n"
        )

    lines.append("## Winning configuration per approach\n")
    for s in (manual, automl):
        lines.append(f"- **{s.approach}** - family=`{s.winning_family}`, params=`{s.winning_params}`")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown report: %s", output_path)


def write_csv_outputs(
    manual: ApproachSummary,
    automl: ApproachSummary,
    manual_results: list[ComboResult],
    automl_results: list[ComboResult],
    summary_csv: Path,
    detailed_csv: Path,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([asdict(manual), asdict(automl)]).to_csv(summary_csv, index=False)
    logger.info("Wrote summary CSV: %s", summary_csv)

    detailed_rows: list[dict[str, Any]] = []
    for approach, results in ((MANUAL_LABEL, manual_results), (AUTOML_LABEL, automl_results)):
        for r in results:
            row = asdict(r)
            row["approach"] = approach
            detailed_rows.append(row)
    pd.DataFrame(detailed_rows).to_csv(detailed_csv, index=False)
    logger.info("Wrote detailed combo CSV: %s", detailed_csv)


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a manual hyperparameter grid search against the project's "
            "AutoML pipeline. Standalone - does not touch MLflow."
        )
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="large",
        help="Grid size preset. 'large' is ~300 combos (recommended for the real story).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("historical_data.csv"),
    )
    parser.add_argument(
        "--features-path",
        type=Path,
        default=Path("features_used.txt"),
    )
    parser.add_argument(
        "--target-rmse",
        type=float,
        default=None,
        help=(
            "RMSE threshold used to measure time-to-target. "
            "If omitted, defaults to AutoML's best RMSE (so the manual grid "
            "has to *match AutoML's quality*)."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("metrics_improvements/manual_vs_automl_summary.csv"),
    )
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=Path("metrics_improvements/manual_vs_automl_detailed.csv"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("MANUAL_VS_AUTOML_REPORT.md"),
    )
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Skip the AutoML side (useful for iterating on the grid).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()

    logger.info("Loading data split from %s", args.data_path)
    X_train, y_train, X_test, y_test = load_data_split(args.data_path, args.features_path)
    logger.info("Train rows=%d  Test rows=%d  Features=%d", len(X_train), len(X_test), X_train.shape[1])

    grid = PRESETS[args.preset]
    total_combos = 0
    for pg in grid.values():
        family_total = 1
        for values in pg.values():
            family_total *= len(values)
        total_combos += family_total
    logger.info("Grid preset: %s (%d combos)", args.preset, total_combos)

    # ----- AutoML first (cheap, and we may use its best RMSE as the target) -----
    automl_summary: ApproachSummary | None = None
    automl_results: list[ComboResult] = []
    if not args.manual_only:
        logger.info("=== AutoML candidate sweep ===")
        automl_started = time.perf_counter()
        automl_results = run_automl_candidates(X_train, y_train, X_test, y_test)
        automl_wall = time.perf_counter() - automl_started
        # Placeholder target used just so summarise() can compute reached=True for
        # the best AutoML run. We overwrite this below after we know the real target.
        automl_best = min(r.rmse for r in automl_results)
        automl_summary = summarise(AUTOML_LABEL, automl_results, automl_best, automl_wall)

    target_rmse = args.target_rmse
    if target_rmse is None:
        if automl_summary is None:
            raise SystemExit(
                "Cannot default --target-rmse because --manual-only was set. "
                "Pass --target-rmse explicitly."
            )
        target_rmse = automl_summary.best_rmse
        logger.info("Target RMSE auto-set to AutoML best = %.4f", target_rmse)

    # ----- Manual grid search -----
    logger.info("=== Manual grid search (no MLflow, no AutoML) ===")
    manual_started = time.perf_counter()
    manual_results = run_manual_grid_search(X_train, y_train, X_test, y_test, grid)
    manual_wall = time.perf_counter() - manual_started

    manual_summary = summarise(MANUAL_LABEL, manual_results, target_rmse, manual_wall)
    if automl_summary is not None:
        # Re-summarise AutoML against the *final* target so reporting is consistent.
        automl_summary = summarise(AUTOML_LABEL, automl_results, target_rmse, automl_summary.wall_clock_seconds)

    # ----- Emit results -----
    logger.info("Manual: %s", manual_summary)
    if automl_summary is not None:
        logger.info("AutoML: %s", automl_summary)

        write_csv_outputs(
            manual=manual_summary,
            automl=automl_summary,
            manual_results=manual_results,
            automl_results=automl_results,
            summary_csv=args.output_csv,
            detailed_csv=args.detailed_csv,
        )
        write_markdown_report(
            manual=manual_summary, automl=automl_summary, output_path=args.report_md
        )
    else:
        # manual-only mode: still dump what we have.
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([asdict(manual_summary)]).to_csv(args.output_csv, index=False)
        pd.DataFrame([asdict(r) | {"approach": MANUAL_LABEL} for r in manual_results]).to_csv(
            args.detailed_csv, index=False
        )


if __name__ == "__main__":
    main()
