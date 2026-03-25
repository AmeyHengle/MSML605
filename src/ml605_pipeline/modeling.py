from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class TrainResult:
    model: RandomForestRegressor
    metrics: dict[str, float]


def time_split(df: pd.DataFrame, feature_cols: list[str], target_col: str = "actual_intensity"):
    split_idx = int(len(df) * 0.8)
    X = df[feature_cols].fillna(0.0)
    y = df[target_col]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    return X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = 42,
) -> TrainResult:
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=14,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = float(np.mean(np.abs((y_test - preds) / y_test.replace(0, np.nan))) * 100)

    return TrainResult(
        model=model,
        metrics={
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "mape": float(mape),
        },
    )

