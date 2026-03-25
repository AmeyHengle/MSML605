from __future__ import annotations

import numpy as np
import pandas as pd

from ml605_pipeline.modeling import time_split, train_random_forest


def test_time_split_sizes() -> None:
    """Test that time_split returns correct train/test sizes (80/20 split)."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "feat_a": np.random.rand(10),
            "actual_intensity": np.random.rand(10),
        }
    )
    X_train, X_test, y_train, y_test = time_split(df, ["feat_a"])

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2


def test_time_split_no_overlap() -> None:
    """Test that time_split maintains chronological order (train before test)."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=10, freq="h"),
            "feat_a": np.random.rand(10),
            "actual_intensity": np.random.rand(10),
        }
    )
    X_train, X_test, y_train, y_test = time_split(df, ["feat_a"])

    # Train indices should all be before test indices
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.max() < y_test.index.min()


def test_train_random_forest_returns_finite_metrics() -> None:
    """Test that train_random_forest returns a TrainResult with finite metrics and expected keys."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="h"),
            "f1": np.arange(20, dtype=float),
            "actual_intensity": np.arange(20, dtype=float) * 2 + 1,
        }
    )

    X_train, X_test, y_train, y_test = time_split(df, ["f1"])
    result = train_random_forest(X_train, y_train, X_test, y_test)

    # Assert all metrics are finite (not NaN, not inf)
    assert all(np.isfinite(v) for v in result.metrics.values()), \
        f"Some metrics are not finite: {result.metrics}"

    # Assert expected metric keys are present
    expected_keys = {"rmse", "mae", "r2", "mape", "rmse_train", "r2_train", "oob_score"}
    assert set(result.metrics.keys()) == expected_keys, \
        f"Metric keys mismatch. Expected {expected_keys}, got {set(result.metrics.keys())}"

    # Assert model is not None
    assert result.model is not None
