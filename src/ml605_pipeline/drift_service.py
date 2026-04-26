from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ml605_pipeline.drift import compute_psi


@dataclass(frozen=True)
class DriftDecision:
    ks_stat: float
    ks_p_value: float
    psi: float
    drift_detected: bool


def decide_drift(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    ks_stat_threshold: float = 0.10,
    ks_alpha: float = 0.05,
    psi_threshold: float = 0.25,
    psi_bins: int = 10,
) -> DriftDecision:
    """
    Unified PSI/KS-only drift decision.

    Drift fires if:
      - KS statistic >= ks_stat_threshold
      OR
      - KS p-value < ks_alpha
      OR
      - PSI >= psi_threshold
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)

    if len(ref) < 5 or len(cur) < 5:
        return DriftDecision(ks_stat=0.0, ks_p_value=1.0, psi=0.0, drift_detected=False)

    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    psi = compute_psi(ref, cur, bins=psi_bins)

    drift = bool(
        (ks_stat >= ks_stat_threshold)
        or (ks_p < ks_alpha)
        or (psi >= psi_threshold)
    )

    return DriftDecision(
        ks_stat=float(ks_stat),
        ks_p_value=float(ks_p),
        psi=float(psi),
        drift_detected=drift,
    )
