from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


PSI_LOW = 0.1
PSI_HIGH = 0.25


@dataclass(frozen=True)
class FeatureDriftResult:
    feature: str
    ks_statistic: float
    ks_p_value: float
    psi: float
    drift_detected: bool


@dataclass(frozen=True)
class DriftReport:
    feature_results: list[FeatureDriftResult]
    overall_drift: bool
    drifted_features: list[str]

    @property
    def drift_score(self) -> float:
        if not self.feature_results:
            return 0.0
        return max(r.psi for r in self.feature_results)


def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Public PSI helper shared across batch/API flows.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    eps = 1e-8
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(current) + eps * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def ks_test(reference: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> tuple[float, float, bool]:
    """
    Public KS helper for consistent thresholding.
    Returns (ks_stat, p_value, drift_flag).
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(reference) < 5 or len(current) < 5:
        return 0.0, 1.0, False

    ks_stat, ks_p = stats.ks_2samp(reference, current)
    return float(ks_stat), float(ks_p), bool(ks_p < alpha)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = PSI_HIGH,
    ks_alpha: float = 0.05,
) -> DriftReport:
    results: list[FeatureDriftResult] = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        psi = compute_psi(ref_vals, cur_vals, bins=10)
        drift = psi >= psi_threshold or ks_p < ks_alpha

        results.append(
            FeatureDriftResult(
                feature=col,
                ks_statistic=float(ks_stat),
                ks_p_value=float(ks_p),
                psi=float(psi),
                drift_detected=drift,
            )
        )

    drifted = [r.feature for r in results if r.drift_detected]
    return DriftReport(
        feature_results=results,
        overall_drift=len(drifted) > 0,
        drifted_features=drifted,
    )   