from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml605_pipeline.automl import (
    AutoMLResult,
    CANDIDATE_MODELS,
    ModelCandidate,
    _evaluate_candidate,
)


def _make_regression_data(n: int = 60):
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    y = pd.Series(100.0 + 3.0 * X["f1"] - 2.0 * X["f2"] + rng.normal(0, 5, n))
    split = int(n * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def test_candidate_models_has_at_least_three_entries() -> None:
    assert len(CANDIDATE_MODELS) >= 3


def test_evaluate_candidate_returns_model_candidate() -> None:
    X_train, X_test, y_train, y_test = _make_regression_data()
    name, model = next(iter(CANDIDATE_MODELS.items()))
    result = _evaluate_candidate(name, model, X_train, y_train, X_test, y_test)
    assert isinstance(result, ModelCandidate)
    assert result.name == name
    assert result.rmse > 0
    assert np.isfinite(result.rmse)
    assert "rmse" in result.metrics
    assert "r2" in result.metrics


def test_evaluate_candidate_metrics_are_finite() -> None:
    X_train, X_test, y_train, y_test = _make_regression_data()
    for name, model in CANDIDATE_MODELS.items():
        result = _evaluate_candidate(name, model, X_train, y_train, X_test, y_test)
        assert all(np.isfinite(v) for v in result.metrics.values()), (
            f"Non-finite metric in {name}: {result.metrics}"
        )
