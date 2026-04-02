"""
ml_monitor/severity.py
----------------------
Severity enum, Alert dataclass, and SeverityClassifier.

Kept separate from metrics.py to avoid circular imports
(metrics.py imports from here, not the other way around).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Optional

from .config import ThresholdConfig

if TYPE_CHECKING:
    from .metrics import ModelMetrics


class Severity(Enum):
    """
    Image 2 — Severity routing legend:
      LOW      → Slack only
      MEDIUM   → Slack + email
      CRITICAL → All channels + auto-rollback
    """
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """Fired when one or more thresholds are breached."""
    severity: Severity
    metrics: "ModelMetrics"
    reasons: list[str]
    fired_at: datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        return f"[{self.severity.value}] {', '.join(self.reasons)} | {self.metrics}"


class SeverityClassifier:
    """
    Maps metric violations → Severity level.

    Rules
    -----
    1. If ANY metric crosses its *critical_* threshold  → CRITICAL
    2. Else if 2+ metrics breach warning thresholds     → MEDIUM
    3. Else if exactly 1 metric breaches warning        → LOW
    4. No breach                                        → (None, [])
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config

    def classify(
        self, metrics: "ModelMetrics"
    ) -> tuple[Optional[Severity], list[str]]:
        reasons: list[str] = []
        is_critical = False

        if metrics.accuracy < self.config.min_accuracy:
            reasons.append(
                f"accuracy={metrics.accuracy:.3f} < {self.config.min_accuracy}"
            )
            if metrics.accuracy < self.config.critical_accuracy:
                is_critical = True

        if metrics.drift_score > self.config.max_drift_score:
            reasons.append(
                f"drift={metrics.drift_score:.3f} > {self.config.max_drift_score}"
            )
            if metrics.drift_score > self.config.critical_drift:
                is_critical = True

        if metrics.latency_ms > self.config.max_latency_ms:
            reasons.append(
                f"latency={metrics.latency_ms:.1f}ms > {self.config.max_latency_ms}ms"
            )
            if metrics.latency_ms > self.config.critical_latency_ms:
                is_critical = True

        if not reasons:
            return None, []

        if is_critical:
            return Severity.CRITICAL, reasons
        if len(reasons) >= 2:
            return Severity.MEDIUM, reasons
        return Severity.LOW, reasons
