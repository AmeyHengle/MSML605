from __future__ import annotations

from pathlib import Path

import mlflow

from log_tracking import setup_run_logging
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = Path("historical_data.csv")
FEATURES_PATH = Path("features_used.txt")


def main() -> None:
    logger = setup_run_logging("train_with_mlflow")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "historical_data.csv not found. Run fetch_historical_data.py first."
        )
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(
            "features_used.txt not found. Please generate or provide the feature list."
        )

    mlflow.set_experiment("intensity-model-training")
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    df = pd.read_csv(DATA_PATH)
    if df.empty:
        raise ValueError("historical_data.csv is empty.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "interval_end" in df.columns:
        df["interval_end"] = pd.to_datetime(df["interval_end"], utc=True, errors="coerce")

    df = df.dropna(subset=["timestamp", "actual_intensity"]).reset_index(drop=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Time features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Categorical feature from API (e.g. low, moderate, high, very high)
    if "intensity_index" in df.columns:
        encoded = pd.get_dummies(df["intensity_index"], prefix="intensity_index")
        df = pd.concat([df.drop(columns=["intensity_index"]), encoded], axis=1)

    selected_features = [
        line.strip()
        for line in FEATURES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0.0

    feature_cols = selected_features
    X = df[feature_cols].fillna(0.0)
    y = df["actual_intensity"]

    # Time-aware split to avoid leakage from future into past.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    factor_cols = [c for c in feature_cols if c.startswith("factor_")]
    fuel_mix_cols = [
        c
        for c in feature_cols
        if c
        not in {
            "forecast_intensity",
            "hour",
            "day_of_week",
            "month",
            "day_of_year",
            "is_weekend",
        }
        and not c.startswith("intensity_index_")
        and not c.startswith("factor_")
    ]

    with mlflow.start_run(run_name="random_forest_baseline"):
        # Log baseline RMSE using forecast_intensity if available
        if "forecast_intensity" in df.columns:
            fc_test = df["forecast_intensity"].iloc[split_idx:].fillna(df["forecast_intensity"].mean())
            baseline_rmse = float(mean_squared_error(y_test, fc_test) ** 0.5)
            mlflow.log_metric("baseline_rmse_forecast", baseline_rmse)

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=14,
            random_state=42,
            oob_score=True,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100

        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("mape", float(mape))

        # Log training metrics
        train_preds = model.predict(X_train)
        rmse_train = mean_squared_error(y_train, train_preds) ** 0.5
        r2_train = r2_score(y_train, train_preds)
        mlflow.log_metric("rmse_train", float(rmse_train))
        mlflow.log_metric("r2_train", float(r2_train))

        # Log OOB score
        mlflow.log_metric("oob_score", float(model.oob_score_))
        mlflow.log_param("feature_count", len(feature_cols))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("num_factor_features", len(factor_cols))
        mlflow.log_param("num_fuel_mix_features", len(fuel_mix_cols))
        mlflow.log_param("feature_list_path", str(FEATURES_PATH))
        mlflow.log_artifact(str(FEATURES_PATH))

        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(
            ascending=False
        )
        top_features = importances.head(10).to_dict()
        mlflow.log_dict(top_features, "top_10_feature_importance.json")

        logger.info("Rows: train=%s test=%s", len(X_train), len(X_test))
        logger.info(
            "Features: total=%s fuel_mix=%s factors=%s",
            len(feature_cols),
            len(fuel_mix_cols),
            len(factor_cols),
        )
        logger.info("RMSE: %.4f", rmse)
        logger.info("MAE : %.4f", mae)
        logger.info("R2  : %.4f", r2)
        logger.info("MAPE: %.2f%%", mape)
        if "forecast_intensity" in df.columns:
            logger.info("Baseline RMSE (forecast_intensity): %.4f", baseline_rmse)
        logger.info("Train RMSE: %.4f", rmse_train)
        logger.info("Train R2: %.4f", r2_train)
        logger.info("OOB Score: %.4f", model.oob_score_)


if __name__ == "__main__":
    main()
