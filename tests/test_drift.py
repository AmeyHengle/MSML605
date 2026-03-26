# tests/test_drift.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml605_pipeline.drift import (
    DriftReport,
    FeatureDriftResult,
    PSI_HIGH,
    PSI_LOW,
    _compute_psi,
    detect_drift,
)


def test_psi_identical_distributions_is_near_zero() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(100, 15, 1000)
    psi = _compute_psi(data, data.copy())
    assert psi < PSI_LOW


def test_psi_very_different_distributions_is_high() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(100, 15, 1000)
    cur = rng.normal(200, 15, 1000)  # completely different mean
    psi = _compute_psi(ref, cur)
    assert psi >= PSI_HIGH


def test_detect_drift_no_drift_on_same_data() -> None:
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "hour": rng.integers(0, 24, n).astype(float),
        "actual_intensity": rng.normal(200, 30, n),
    })
    report = detect_drift(df, df.copy(), feature_cols=["hour", "actual_intensity"])
    assert isinstance(report, DriftReport)
    assert not report.overall_drift
    assert report.drift_score < PSI_HIGH


def test_detect_drift_detects_large_shift() -> None:
    rng = np.random.default_rng(42)
    n = 500
    ref = pd.DataFrame({"actual_intensity": rng.normal(200, 20, n)})
    cur = pd.DataFrame({"actual_intensity": rng.normal(350, 20, n)})  # big shift
    report = detect_drift(ref, cur, feature_cols=["actual_intensity"])
    assert report.overall_drift
    assert "actual_intensity" in report.drifted_features


def test_detect_drift_skips_missing_columns() -> None:
    ref = pd.DataFrame({"col_a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    cur = pd.DataFrame({"col_b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    # col_a not in cur, col_b not in ref — should not crash
    report = detect_drift(ref, cur, feature_cols=["col_a", "col_b"])
    assert isinstance(report, DriftReport)
    assert len(report.feature_results) == 0


def test_feature_drift_result_fields() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(100, 10, 200)})
    cur = pd.DataFrame({"x": rng.normal(150, 10, 200)})
    report = detect_drift(ref, cur, feature_cols=["x"])
    assert len(report.feature_results) == 1
    r = report.feature_results[0]
    assert isinstance(r, FeatureDriftResult)
    assert r.feature == "x"
    assert 0.0 <= r.ks_statistic <= 1.0
    assert 0.0 <= r.ks_p_value <= 1.0
    assert r.psi >= 0.0
