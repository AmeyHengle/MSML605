# run_pipeline.py
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import logging
import warnings
import mlflow
import mlflow.sklearn

from log_tracking import setup_run_logging
from ml605_pipeline.automl import run_automl
from ml605_pipeline.config import load_config_from_env
from ml605_pipeline.data import fetch_window_dataframe
from ml605_pipeline.drift import detect_drift
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
    one_hot_intensity_index,
)
from ml605_pipeline.modeling import time_split
from ml605_pipeline.registry import register_model, transition_model_stage

import pandas as pd


def main() -> None:
    logger = setup_run_logging("run_pipeline")
    cfg = load_config_from_env()

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mlflow(\.*)?")

    features_path = Path(cfg.features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            "features_used.txt not found. Generate it once from your dataset, then rerun."
        )

    mlflow.set_experiment(cfg.mlflow_experiment)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)

    start_dt = cfg.window_start_utc
    end_dt = cfg.window_end_utc

    logger.info(
        "Pipeline window: %s -> %s (window_hours=%s)",
        start_dt.isoformat(), end_dt.isoformat(), cfg.window_hours,
    )

    with mlflow.start_run(run_name=f"{cfg.run_name_prefix}_{cfg.window_label}"):
        mlflow.set_tag("pipeline", "daily")
        mlflow.log_param("window_hours", cfg.window_hours)
        mlflow.log_param("data_resolution_minutes", cfg.data_resolution_minutes)
        mlflow.log_param("window_start_utc", start_dt.isoformat())
        mlflow.log_param("window_end_utc", end_dt.isoformat())

        result = fetch_window_dataframe(
            start_dt,
            end_dt,
            resolution_minutes=cfg.data_resolution_minutes,
        )
        df = result.df

        if df.empty:
            logger.warning("No rows fetched for window; exiting.")
            mlflow.log_metric("rows_fetched", 0)
            mlflow.set_tag("pipeline_outcome", "no_data_fetched")
            return

        # Feature engineering
        df = add_time_features(df)
        df = apply_factor_columns(df, result.factors)
        df = one_hot_intensity_index(df)

        out_csv = cfg.output_csv
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))
        mlflow.log_metric("rows_fetched", int(len(df)))

        feature_cols = load_feature_list(features_path)
        df = ensure_feature_columns(df, feature_cols)

        # --- Drift detection ---
        should_retrain = True  # Default: always retrain if no reference data
        data_path = _ROOT / "historical_data.csv"
        if data_path.exists():
            ref_df = pd.read_csv(data_path)
            ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], utc=True, errors="coerce")
            ref_df = ref_df.dropna(subset=["timestamp"]).reset_index(drop=True)
            ref_df = add_time_features(ref_df)
            ref_df = apply_factor_columns(ref_df, result.factors)
            ref_df = one_hot_intensity_index(ref_df)

            numeric_feature_cols = [
                c for c in feature_cols
                if c in ref_df.select_dtypes("number").columns
                and c in df.select_dtypes("number").columns
            ]
            drift_report = detect_drift(ref_df, df, feature_cols=numeric_feature_cols)

            mlflow.log_metric("drift_score", drift_report.drift_score)
            mlflow.log_metric("drifted_feature_count", len(drift_report.drifted_features))
            mlflow.log_param("drift_detected", str(drift_report.overall_drift))

            if drift_report.overall_drift:
                logger.warning(
                    "Drift detected (%d features). Retraining triggered.",
                    len(drift_report.drifted_features),
                )
                mlflow.set_tag("retrain_reason", "drift_detected")
            else:
                logger.info(
                    "No drift detected (max PSI=%.4f). Skipping retraining.",
                    drift_report.drift_score,
                )
                should_retrain = False
        else:
            logger.info("No reference data found; retraining unconditionally.")

        if not should_retrain:
            logger.info("Pipeline complete — no retraining needed.")
            mlflow.set_tag("pipeline_outcome", "skipped_no_drift")
            return

        # --- AutoML retraining ---
        X_train, X_test, y_train, y_test = time_split(df, feature_cols)
        if len(X_test) == 0:
            raise ValueError("Not enough rows to create a test split (need > 1 row).")

        automl_result = run_automl(X_train, y_train, X_test, y_test)

        mlflow.log_param("best_model", automl_result.best.name)
        for k, v in automl_result.best.metrics.items():
            mlflow.log_metric(k, v)
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_artifact(str(features_path))

        logger.info("Train/test rows: %s/%s", len(X_train), len(X_test))
        logger.info("Best model: %s  RMSE=%.4f", automl_result.best.name, automl_result.best.rmse)

        # Register the winner — use the child run's run_id where the model artifact lives
        version = register_model(run_id=automl_result.best.run_id)
        transition_model_stage(version=version, stage="Production")
        mlflow.set_tag("pipeline_outcome", "retrained")
        logger.info("Model v%s promoted to Production in MLflow Registry.", version)


if __name__ == "__main__":
    main()
