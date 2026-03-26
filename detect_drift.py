# detect_drift.py
"""
Drift detection script.

Compares the current live data window against historical_data.csv (reference).
Logs per-feature PSI, KS statistics, and overall drift flag to MLflow.

Exit codes:
  0 — no significant drift detected
  1 — drift detected (retrain recommended)
  2 — reference data or features_used.txt not found
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

# Make src/ importable when running as a script (common local layout).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mlflow
import pandas as pd

from log_tracking import setup_run_logging
from ml605_pipeline.config import load_config_from_env
from ml605_pipeline.data import fetch_window_dataframe
from ml605_pipeline.drift import detect_drift
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    load_feature_list,
    one_hot_intensity_index,
)


DATA_PATH = Path("historical_data.csv")
FEATURES_PATH = Path("features_used.txt")
DRIFT_EXPERIMENT = "carbon-intensity-drift"


def main() -> int:
    logger = setup_run_logging("detect_drift")
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mlflow(\.*)?")

    if not DATA_PATH.exists():
        logger.error("Reference data not found: %s", DATA_PATH)
        return 2
    if not FEATURES_PATH.exists():
        logger.error("Features list not found: %s", FEATURES_PATH)
        return 2

    cfg = load_config_from_env()
    feature_cols = load_feature_list(FEATURES_PATH)

    # Load reference (training) distribution
    ref_df = pd.read_csv(DATA_PATH)
    ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], utc=True, errors="coerce")
    ref_df = ref_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ref_df = add_time_features(ref_df)
    ref_df = one_hot_intensity_index(ref_df)

    # Fetch current live window
    result = fetch_window_dataframe(cfg.window_start_utc, cfg.window_end_utc)
    live_df = result.df
    if live_df.empty:
        logger.warning("No live data fetched for drift check; skipping.")
        return 0

    live_df = add_time_features(live_df)
    live_df = one_hot_intensity_index(live_df)
    live_df = apply_factor_columns(live_df, result.factors)
    ref_df = apply_factor_columns(ref_df, result.factors)

    # Only test numeric features present in both DataFrames
    numeric_cols = [
        c for c in feature_cols
        if c in ref_df.select_dtypes("number").columns
        and c in live_df.select_dtypes("number").columns
    ]

    mlflow.set_experiment(DRIFT_EXPERIMENT)
    with mlflow.start_run(run_name=f"drift_check_{cfg.window_label}"):
        mlflow.set_tag("window_start", cfg.window_start_utc.isoformat())
        mlflow.set_tag("window_end", cfg.window_end_utc.isoformat())
        mlflow.log_param("reference_rows", len(ref_df))
        mlflow.log_param("live_rows", len(live_df))
        mlflow.log_param("features_tested", len(numeric_cols))

        report = detect_drift(ref_df, live_df, feature_cols=numeric_cols)

        # Log per-feature metrics
        for fr in report.feature_results:
            mlflow.log_metric(f"psi_{fr.feature}", fr.psi)
            mlflow.log_metric(f"ks_stat_{fr.feature}", fr.ks_statistic)
            mlflow.log_metric(f"ks_pval_{fr.feature}", fr.ks_p_value)

        mlflow.log_metric("drift_score", report.drift_score)
        mlflow.log_metric("drifted_feature_count", len(report.drifted_features))
        mlflow.log_param("overall_drift", str(report.overall_drift))

        if report.overall_drift:
            logger.warning(
                "DRIFT DETECTED — %d/%d features drifted. Score=%.4f. Drifted: %s",
                len(report.drifted_features),
                len(numeric_cols),
                report.drift_score,
                ", ".join(report.drifted_features),
            )
            mlflow.set_tag("alert", "drift_detected")
        else:
            logger.info(
                "No drift detected. Max PSI=%.4f across %d features.",
                report.drift_score,
                len(numeric_cols),
            )

    return 1 if report.overall_drift else 0


if __name__ == "__main__":
    sys.exit(main())
