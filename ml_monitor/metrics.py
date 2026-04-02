"""
ml_monitor/metrics.py
---------------------
ModelMetrics dataclass + ModelMonitor.

ModelMonitor is the ONLY class that talks to the live model.
Replace _collect_metrics() with real instrumentation:
  - Prometheus / Grafana pull
  - Evidently AI report
  - Alibi Detect drift detector
  - Your own evaluation harness
"""

from __future__ import annotations

import random
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .config import ThresholdConfig
from .severity import Alert, SeverityClassifier

log = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """A single snapshot of live model health."""
    accuracy: float     # 0.0 – 1.0
    drift_score: float  # 0.0 – 1.0  (higher = more drift)
    latency_ms: float   # milliseconds
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self) -> str:
        return (
            f"accuracy={self.accuracy:.3f}  "
            f"drift={self.drift_score:.3f}  "
            f"latency={self.latency_ms:.1f}ms"
        )


class ModelMonitor:
    """
    Diagram: Continuous monitoring — accuracy, drift, latency.

    Responsibilities:
      1. Collect metrics from the live model  (_collect_metrics)
      2. Check collected metrics against thresholds (check_thresholds)

    Keeps NO state about alerts or notifications — that belongs to the pipeline.
    """

    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config
        self._classifier = SeverityClassifier(config)

    def collect(self) -> ModelMetrics:
        """Public entry point. Override or monkey-patch in tests."""
        metrics = self._collect_metrics()
        log.info("Metrics  →  %s", metrics)
        return metrics

    def check_thresholds(self, metrics: ModelMetrics) -> Optional[Alert]:
        """Return an Alert if any threshold is breached, else None."""
        severity, reasons = self._classifier.classify(metrics)
        if severity is None:
            return None
        return Alert(severity=severity, metrics=metrics, reasons=reasons)

    # ── stub: replace in production ──────────────────────────────────────
    def _collect_metrics(self) -> ModelMetrics:
        """
        STUB — simulates fetching metrics from a live model endpoint.

        Production replacement examples
        --------------------------------
        # Prometheus:
        acc  = prometheus_client.Gauge('model_accuracy', ...).get()
        drift = evidently_report.get_metric(DataDriftMetric())
        lat  = requests.get(f"{endpoint}/metrics").json()["p99_latency_ms"]
        return ModelMetrics(accuracy=acc, drift_score=drift, latency_ms=lat)
        """
        return ModelMetrics(
            accuracy=random.uniform(0.65, 0.99),
            drift_score=random.uniform(0.00, 0.60),
            latency_ms=random.uniform(80, 700),
        )
