"""Tests for report_worker helper functions in graph.py (ANALYSIS-01).

TDD RED phase for Plan 03-03 Task 1:
- _compute_shap
- _fig_to_base64
- _shap_bar_to_base64
- _forecast_chart_to_base64
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def test_compute_shap_rf_returns_values_and_features() -> None:
    """_compute_shap with RF returns (shap_values, top_features) where len(top_features) == 10."""
    from ml605_agent.graph import _compute_shap

    feature_cols = [f"f{i}" for i in range(12)]
    n = 30
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)

    shap_values, top_features = _compute_shap(model, X, max_display=10)

    assert shap_values is not None
    assert len(top_features) == 10


def test_compute_shap_ridge_fallback_no_raise() -> None:
    """_compute_shap with Ridge (shap.Explainer raises) returns (None, []) without raising."""
    from ml605_agent.graph import _compute_shap

    feature_cols = ["hour", "day_of_week"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = Ridge()
    model.fit(X, y)

    shap_values, top_features = _compute_shap(model, X)

    assert shap_values is None
    assert top_features == []


def test_fig_to_base64_returns_nonempty_string() -> None:
    """_fig_to_base64 returns a non-empty ASCII string for any matplotlib Figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml605_agent.graph import _fig_to_base64

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])

    result = _fig_to_base64(fig)

    assert isinstance(result, str)
    assert len(result) > 0
    # Should be valid base64 (ASCII only)
    import base64
    decoded = base64.b64decode(result)
    assert len(decoded) > 0


def test_forecast_chart_returns_nonempty_string() -> None:
    """_forecast_chart_to_base64 returns a non-empty base64 string for valid input."""
    from ml605_agent.graph import _forecast_chart_to_base64

    feature_cols = ["hour", "day_of_week", "month"]
    n = 60
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    result = _forecast_chart_to_base64(df, feature_cols, model)

    assert isinstance(result, str)
    assert len(result) > 0


def test_forecast_chart_does_not_call_plt_show(monkeypatch) -> None:
    """_forecast_chart_to_base64 does not call plt.show() — runs in headless mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ml605_agent.graph import _forecast_chart_to_base64

    show_called = []
    original_show = plt.show
    monkeypatch.setattr(plt, "show", lambda *a, **kw: show_called.append(True))

    feature_cols = ["hour", "day_of_week"]
    n = 20
    X = pd.DataFrame({col: np.random.rand(n) for col in feature_cols})
    y = np.random.rand(n)
    model = RandomForestRegressor(n_estimators=5, random_state=0)
    model.fit(X, y)
    df = X.copy()
    df["intensity.actual"] = y

    _forecast_chart_to_base64(df, feature_cols, model)

    assert show_called == [], "plt.show() must not be called in headless mode"
