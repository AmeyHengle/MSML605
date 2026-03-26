from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml605_pipeline.evaluate import compute_metrics, EvalResult


def test_compute_metrics_perfect_prediction() -> None:
    y = pd.Series([100.0, 200.0, 150.0, 180.0])
    preds = np.array([100.0, 200.0, 150.0, 180.0])
    result = compute_metrics(y, preds)
    assert isinstance(result, EvalResult)
    assert result.rmse == pytest.approx(0.0, abs=1e-9)
    assert result.r2 == pytest.approx(1.0, abs=1e-9)
    assert result.mae == pytest.approx(0.0, abs=1e-9)


def test_compute_metrics_known_error() -> None:
    y = pd.Series([100.0, 200.0])
    preds = np.array([110.0, 190.0])
    result = compute_metrics(y, preds)
    assert result.rmse == pytest.approx(10.0, abs=1e-6)
    assert result.mae == pytest.approx(10.0, abs=1e-6)


def test_compute_metrics_excludes_zero_targets_from_mape() -> None:
    y = pd.Series([0.0, 100.0])
    preds = np.array([0.0, 110.0])
    result = compute_metrics(y, preds)
    # MAPE should be finite (zero target excluded)
    assert np.isfinite(result.mape)


def test_eval_result_to_dict_has_required_keys() -> None:
    y = pd.Series([100.0, 200.0])
    preds = np.array([105.0, 195.0])
    result = compute_metrics(y, preds)
    d = result.to_dict()
    assert {"rmse", "mae", "r2", "mape"}.issubset(d.keys())
    assert all(np.isfinite(v) for v in d.values())


def test_compute_metrics_all_zero_targets_mape_is_nan() -> None:
    """When all targets are zero, MAPE is undefined (NaN). Document this as expected behavior."""
    y = pd.Series([0.0, 0.0, 0.0])
    preds = np.array([1.0, 2.0, 3.0])
    result = compute_metrics(y, preds)
    assert np.isnan(result.mape)
