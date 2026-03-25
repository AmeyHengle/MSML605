from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


_INTENSITY_CATEGORIES = ["very low", "low", "moderate", "high", "very high"]


def normalize_factor_name(name: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"factor_{normalized}"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def one_hot_intensity_index(df: pd.DataFrame) -> pd.DataFrame:
    if "intensity_index" not in df.columns:
        return df
    df = df.copy()
    cat = pd.CategoricalDtype(categories=_INTENSITY_CATEGORIES, ordered=False)
    encoded = pd.get_dummies(df["intensity_index"].astype(cat), prefix="intensity_index")
    df = pd.concat([df.drop(columns=["intensity_index"]), encoded], axis=1)
    return df


def apply_factor_columns(df: pd.DataFrame, factors: dict[str, float]) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    cols = {normalize_factor_name(k): v for k, v in factors.items()}
    for c, v in cols.items():
        df[c] = v
    return df


def load_feature_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_feature_columns(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
    return df

