# train_automl.py
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn

from log_tracking import setup_run_logging
from ml605_pipeline.automl import run_automl
from ml605_pipeline.features import (
    add_time_features,
    ensure_feature_columns,
    load_feature_list,
    one_hot_intensity_index,
)
from ml605_pipeline.modeling import time_split
from ml605_pipeline.registry import register_model, transition_model_stage

import pandas as pd


DATA_PATH = Path("historical_data.csv")
FEATURES_PATH = Path("features_used.txt")
EXPERIMENT = "intensity-model-automl"


def main() -> None:
    logger = setup_run_logging("train_automl")

    logging.getLogger("mlflow").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore", category=UserWarning, module=r"mlflow(\.*)?")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "historical_data.csv not found. Run fetch_historical_data.py first."
        )
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "features_used.txt not found. Run fetch_historical_data.py first."
        )

    mlflow.set_experiment(EXPERIMENT)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("historical_data.csv is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "actual_intensity"]).sort_values("timestamp").reset_index(drop=True)

    df = add_time_features(df)
    df = one_hot_intensity_index(df)

    feature_cols = load_feature_list(FEATURES_PATH)
    df = ensure_feature_columns(df, feature_cols)

    X_train, X_test, y_train, y_test = time_split(df, feature_cols)
    if len(X_test) == 0:
        raise ValueError("Not enough rows to create a test split.")

    with mlflow.start_run(run_name="automl_model_selection") as run:
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # Forecast baseline (naive comparison point)
        if "forecast_intensity" in df.columns:
            from sklearn.metrics import mean_squared_error
            fc_test = df["forecast_intensity"].iloc[int(len(df) * 0.8):].fillna(df["forecast_intensity"].mean())
            baseline_rmse = float(mean_squared_error(y_test, fc_test) ** 0.5)
            mlflow.log_metric("baseline_rmse_forecast", baseline_rmse)

        result = run_automl(X_train, y_train, X_test, y_test)

        # Log best model summary to parent run
        mlflow.log_param("best_model", result.best.name)
        for k, v in result.best.metrics.items():
            mlflow.log_metric(f"best_{k}", v)

        # Rank summary
        ranking = sorted(result.all_candidates, key=lambda c: c.rmse)
        logger.info("AutoML results (ranked by test RMSE):")
        for i, c in enumerate(ranking, 1):
            logger.info("  %d. %-30s RMSE=%.4f  R2=%.4f", i, c.name, c.rmse, c.metrics.get("r2", float("nan")))
        logger.info("Winner: %s (RMSE=%.4f)", result.best.name, result.best.rmse)

        run_id = run.info.run_id

    # Register the best model in MLflow Model Registry
    version = register_model(run_id=result.best.run_id)
    transition_model_stage(version=version, stage="Staging")
    logger.info("Model v%s registered as Staging in MLflow Model Registry.", version)
    logger.info("To promote to Production: transition_model_stage('%s', 'Production')", version)


if __name__ == "__main__":
    main()
