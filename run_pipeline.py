from __future__ import annotations

from pathlib import Path
import sys

# Make src/ importable when running as a script (common local layout).
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import logging
import warnings
import mlflow
import mlflow.sklearn
from log_tracking import setup_run_logging
from ml605_pipeline.config import load_config_from_env
from ml605_pipeline.data import fetch_window_dataframe
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
    one_hot_intensity_index,
)
from ml605_pipeline.modeling import time_split, train_random_forest


def main() -> None:
    logger = setup_run_logging("run_pipeline")
    cfg = load_config_from_env()

    # Reduce noise from MLflow stderr logging (PowerShell may surface it as an error).
    logging.getLogger("mlflow").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking").setLevel(logging.WARNING)
    logging.getLogger("mlflow.tracking.fluent").setLevel(logging.WARNING)
    logging.getLogger("mlflow.utils.autologging_utils").setLevel(logging.ERROR)
    logging.getLogger("mlflow.sklearn").setLevel(logging.ERROR)
    logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)

    # Suppress MLflow's noisy warnings that can be surfaced by uv/PowerShell as errors.
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mlflow(\..*)?")

    features_path = Path(cfg.features_path)
    if not features_path.exists():
        raise FileNotFoundError(
            "features_used.txt not found. Generate it once from your dataset, then rerun."
        )

    mlflow.set_experiment(cfg.mlflow_experiment)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    start_dt = cfg.window_start_utc
    end_dt = cfg.window_end_utc

    logger.info(
        "Pipeline window: %s -> %s (window_hours=%s interval_seconds=%s)",
        start_dt.isoformat(),
        end_dt.isoformat(),
        cfg.window_hours,
        cfg.interval_seconds,
    )

    with mlflow.start_run(run_name=f"{cfg.run_name_prefix}_{cfg.window_label}"):
        mlflow.set_tag("pipeline", "daily")
        mlflow.log_param("window_hours", cfg.window_hours)
        mlflow.log_param("interval_seconds", cfg.interval_seconds)
        mlflow.log_param("window_start_utc", start_dt.isoformat())
        mlflow.log_param("window_end_utc", end_dt.isoformat())

        result = fetch_window_dataframe(start_dt, end_dt)
        df = result.df

        if df.empty:
            logger.warning("No rows fetched for window; exiting.")
            mlflow.log_metric("rows_fetched", 0)
            return

        # Feature engineering
        df = add_time_features(df)
        df = apply_factor_columns(df, result.factors)
        df = one_hot_intensity_index(df)

        # Persist and log fetched dataset + factors
        out_csv = cfg.output_csv
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        mlflow.log_artifact(str(out_csv))
        mlflow.log_text(result.raw_factors_json, "intensity_factors.json")
        mlflow.log_metric("rows_fetched", int(len(df)))

        feature_cols = load_feature_list(features_path)
        df = ensure_feature_columns(df, feature_cols)

        X_train, X_test, y_train, y_test = time_split(df, feature_cols)
        if len(X_test) == 0:
            raise ValueError("Not enough rows to create a test split (need > 1 row).")

        train_result = train_random_forest(X_train, y_train, X_test, y_test)

        for k, v in train_result.metrics.items():
            mlflow.log_metric(k, v)

        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_artifact(str(features_path))

        logger.info("Train/test rows: %s/%s", len(X_train), len(X_test))
        logger.info("Metrics: %s", train_result.metrics)


if __name__ == "__main__":
    main()

