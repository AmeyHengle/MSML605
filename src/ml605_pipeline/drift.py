# src/ml605_pipeline/drift.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


PSI_LOW = 0.1   # Below: no significant change
PSI_HIGH = 0.25  # At or above: significant change — consider retraining


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
        """Maximum PSI across all features. 0.0 if no features were tested."""
        if not self.feature_results:
            return 0.0
        return max(r.psi for r in self.feature_results)


def _compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no change. 0.1-0.25: moderate. >= 0.25: significant drift.
    """
    breakpoints = np.percentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Epsilon prevents log(0); effectively regularises empty bins
    eps = 1e-8
    ref_pct = (ref_counts + eps) / (len(reference) + eps * len(ref_counts))
    cur_pct = (cur_counts + eps) / (len(current) + eps * len(cur_counts))

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = PSI_HIGH,
    ks_alpha: float = 0.05,
) -> DriftReport:
    """
    Compare current data distribution against reference (training) distribution.

    A feature is flagged as drifted if PSI >= psi_threshold OR KS p-value < ks_alpha.
    Overall drift is True when at least one feature drifts.

    Args:
        reference_df: Reference distribution (historical training data).
        current_df: Incoming live data window to compare.
        feature_cols: Numeric columns to test. Non-numeric and absent columns are skipped.
        psi_threshold: PSI at or above which drift is flagged (default 0.25).
        ks_alpha: KS test significance level (default 0.05).

    Returns:
        DriftReport with per-feature results and aggregated drift flag.
    """
    results: list[FeatureDriftResult] = []

    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref_vals = reference_df[col].dropna().to_numpy(dtype=float)
        cur_vals = current_df[col].dropna().to_numpy(dtype=float)

        # Need at least 5 points per group for meaningful statistical tests
        if len(ref_vals) < 5 or len(cur_vals) < 5:
            continue

        ks_stat, ks_p = stats.ks_2samp(ref_vals, cur_vals)
        psi = _compute_psi(ref_vals, cur_vals)
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
