"""
ml_monitor/config.py
--------------------
All configuration dataclasses.
Centralised here so every other module imports from one place.
Swap values via environment variables or a config file in later iterations.
"""

from dataclasses import dataclass


@dataclass
class ThresholdConfig:
    """
    Configurable breach thresholds.

    Severity upgrade rules:
      A metric that crosses the *critical_* threshold will escalate
      the whole alert to CRITICAL regardless of other metrics.
    """
    # Warning thresholds
    min_accuracy: float    = 0.80
    max_drift_score: float = 0.30
    max_latency_ms: float  = 300.0

    # Critical thresholds (subset of warning)
    critical_accuracy: float   = 0.70
    critical_drift: float      = 0.50
    critical_latency_ms: float = 600.0
