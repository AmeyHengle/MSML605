"""Manual grid search vs AutoML - fixed RMSE <= 10.0 time-to-target benchmark.

This is a variant of ``manual_vs_automl_comparison.py`` that pins the RMSE
target to a fixed value (default 10.0) so that *both* the manual grid search
and the AutoML sweep actually reach the threshold and we get a clean
"time-to-target" comparison.

Why this exists
---------------
The default script uses AutoML's best RMSE (e.g. 8.39 from Ridge) as the
target. In practice the manual tree-based grid never gets below ~10.3 on this
dataset, so time-to-target is always N/A for the manual side. That makes the
comparison meaningless.

Setting target RMSE to 10.0 gives us a threshold both approaches can clear,
so we can actually measure:
  * how many combos each approach needed to reach RMSE <= 10.0
  * how many wall-clock seconds each approach spent before hitting it
  * absolute and percentage time saved by AutoML

Typical usage:

    uv run python metrics_improvements/manual_vs_automl_target10.py
    uv run python metrics_improvements/manual_vs_automl_target10.py --preset medium
    uv run python metrics_improvements/manual_vs_automl_target10.py --target-rmse 9.5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

# Allow running as a plain script (python metrics_improvements/manual_vs_automl_target10.py)
# as well as a module (python -m metrics_improvements.manual_vs_automl_target10).
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from metrics_improvements.manual_vs_automl_comparison import (  # type: ignore
        AUTOML_LABEL,
        MANUAL_LABEL,
        PRESETS,
        ApproachSummary,
        ComboResult,
        load_data_split,
        run_automl_candidates,
        run_manual_grid_search,
        summarise,
        write_csv_outputs,
    )
except ModuleNotFoundError:
    from manual_vs_automl_comparison import (  # type: ignore
        AUTOML_LABEL,
        MANUAL_LABEL,
        PRESETS,
        ApproachSummary,
        ComboResult,
        load_data_split,
        run_automl_candidates,
        run_manual_grid_search,
        summarise,
        write_csv_outputs,
    )


logger = logging.getLogger(__name__)


DEFAULT_TARGET_RMSE = 10.0


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value < 60:
        return f"{value:.1f}s"
    return f"{value / 60.0:.2f}m"


def write_markdown_report(
    manual: ApproachSummary,
    automl: ApproachSummary,
    target_rmse: float,
    output_path: Path,
) -> None:
    """Markdown report focused on time-to-target at a fixed RMSE threshold."""
    lines: list[str] = []
    lines.append(
        f"# Manual Grid Search vs AutoML - Time to RMSE <= {target_rmse:.2f}\n"
    )
    lines.append(
        "Both approaches were run on the same train/test split. The target "
        f"RMSE was **fixed at {target_rmse:.2f}** so that the manual grid "
        "search can actually reach it and we get a clean wall-clock "
        "comparison, instead of an N/A.\n"
    )

    lines.append("## Summary\n")
    lines.append(
        "| Approach | Combos Tried | Wall Clock | Best RMSE | Reached Target | "
        "Time To Target | Combos To Target |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for s in (manual, automl):
        lines.append(
            f"| {s.approach} | {s.combos_tried} | "
            f"{_fmt_seconds(s.wall_clock_seconds)} | "
            f"{s.best_rmse:.4f} | {s.reached_target} | "
            f"{_fmt_seconds(s.time_to_target_seconds)} | "
            f"{s.combos_to_target if s.combos_to_target is not None else 'N/A'} |"
        )
    lines.append("")

    if manual.reached_target and automl.reached_target:
        manual_t = float(manual.time_to_target_seconds)  # type: ignore[arg-type]
        automl_t = float(automl.time_to_target_seconds)  # type: ignore[arg-type]
        saved = manual_t - automl_t
        pct = (saved / manual_t * 100.0) if manual_t > 0 else 0.0
        if saved >= 0:
            lines.append(
                f"**Conclusion:** AutoML reached RMSE <= {target_rmse:.2f} in "
                f"`{_fmt_seconds(automl_t)}` (`{automl.combos_to_target}` "
                f"models) vs `{_fmt_seconds(manual_t)}` "
                f"(`{manual.combos_to_target}` models) for the manual grid "
                f"search - a time saving of `{saved:.1f}s` (`{pct:.2f}%`).\n"
            )
        else:
            lines.append(
                f"**Conclusion:** On this run the manual grid happened to "
                f"reach RMSE <= {target_rmse:.2f} faster than AutoML "
                f"({_fmt_seconds(manual_t)} vs {_fmt_seconds(automl_t)}).\n"
            )
    else:
        missed = [s.approach for s in (manual, automl) if not s.reached_target]
        lines.append(
            f"**Conclusion:** Target RMSE {target_rmse:.2f} was not reached "
            f"by: {', '.join(missed)}. Raise `--target-rmse` or widen the "
            "grid for a clean comparison.\n"
        )

    lines.append("## Winning configuration per approach\n")
    for s in (manual, automl):
        lines.append(
            f"- **{s.approach}** - family=`{s.winning_family}`, "
            f"params=`{s.winning_params}`"
        )
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote markdown report: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual grid search vs AutoML with a fixed RMSE target (default "
            f"{DEFAULT_TARGET_RMSE}). Produces a time-to-target comparison "
            "that both approaches can actually reach."
        )
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="large",
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
        default=DEFAULT_TARGET_RMSE,
        help=(
            f"Fixed RMSE threshold for the time-to-target metric "
            f"(default {DEFAULT_TARGET_RMSE})."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("metrics_improvements/manual_vs_automl_target10_summary.csv"),
    )
    parser.add_argument(
        "--detailed-csv",
        type=Path,
        default=Path("metrics_improvements/manual_vs_automl_target10_detailed.csv"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("MANUAL_VS_AUTOML_TARGET10_REPORT.md"),
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help=(
            "Stop the manual grid as soon as it hits the target. Useful when "
            "you only care about 'how fast did manual catch up', not the full "
            "303-combo sweep."
        ),
    )
    return parser.parse_args()


def _run_manual_with_optional_early_stop(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    grid,
    target_rmse: float,
    early_stop: bool,
) -> list[ComboResult]:
    """Run the manual grid, optionally stopping the moment we reach the target.

    When ``early_stop`` is False we just reuse the standard sweep. When True
    we re-implement the loop so we can break out early.
    """
    if not early_stop:
        return run_manual_grid_search(X_train, y_train, X_test, y_test, grid)

    import itertools
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.metrics import mean_squared_error

    try:
        from metrics_improvements.manual_vs_automl_comparison import (  # type: ignore
            ESTIMATOR_FACTORIES,
        )
    except ModuleNotFoundError:
        from manual_vs_automl_comparison import ESTIMATOR_FACTORIES  # type: ignore

    combos: list[tuple[str, dict]] = []
    for family, param_grid in grid.items():
        keys = list(param_grid.keys())
        for values in itertools.product(*[param_grid[k] for k in keys]):
            combos.append((family, dict(zip(keys, values))))
    logger.info(
        "Manual grid search (early-stop at RMSE <= %.4f): %d combinations max",
        target_rmse,
        len(combos),
    )

    results: list[ComboResult] = []
    cumulative = 0.0
    for idx, (family, params) in enumerate(combos, start=1):
        model = ESTIMATOR_FACTORIES[family](params)
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
        if idx == 1 or idx % 10 == 0:
            best = min(r.rmse for r in results)
            logger.info(
                "  [%4d/%d] %-18s rmse=%.4f fit=%.2fs cum=%.1fs best=%.4f",
                idx, len(combos), family, rmse, elapsed, cumulative, best,
            )
        if rmse <= target_rmse:
            logger.info(
                "Early stop: combo %d (%s) hit RMSE %.4f <= target %.4f after %.1fs",
                idx, family, rmse, target_rmse, cumulative,
            )
            break
    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()

    logger.info("Loading data split from %s", args.data_path)
    X_train, y_train, X_test, y_test = load_data_split(
        args.data_path, args.features_path
    )
    logger.info(
        "Train rows=%d  Test rows=%d  Features=%d",
        len(X_train), len(X_test), X_train.shape[1],
    )
    logger.info("Fixed target RMSE: %.4f", args.target_rmse)

    grid = PRESETS[args.preset]
    total_combos = 0
    for pg in grid.values():
        family_total = 1
        for values in pg.values():
            family_total *= len(values)
        total_combos += family_total
    logger.info("Grid preset: %s (%d combos)", args.preset, total_combos)

    logger.info("=== AutoML candidate sweep ===")
    automl_started = time.perf_counter()
    automl_results = run_automl_candidates(X_train, y_train, X_test, y_test)
    automl_wall = time.perf_counter() - automl_started

    logger.info("=== Manual grid search ===")
    manual_started = time.perf_counter()
    manual_results = _run_manual_with_optional_early_stop(
        X_train, y_train, X_test, y_test,
        grid=grid,
        target_rmse=args.target_rmse,
        early_stop=args.early_stop,
    )
    manual_wall = time.perf_counter() - manual_started

    manual_summary = summarise(MANUAL_LABEL, manual_results, args.target_rmse, manual_wall)
    automl_summary = summarise(AUTOML_LABEL, automl_results, args.target_rmse, automl_wall)

    logger.info("Manual: %s", manual_summary)
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
        manual=manual_summary,
        automl=automl_summary,
        target_rmse=args.target_rmse,
        output_path=args.report_md,
    )


if __name__ == "__main__":
    main()
