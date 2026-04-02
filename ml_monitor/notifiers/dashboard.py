"""
ml_monitor/notifiers/dashboard.py
-----------------------------------
Dashboard notification channel (Grafana / MLflow).

Diagram: Dashboard — Grafana / MLflow

Production wiring (Grafana Annotations API)
--------------------------------------------
import requests

class DashboardNotifier(NotificationChannel):
    def __init__(self, grafana_url: str, api_key: str):
        self.grafana_url = grafana_url
        self.headers = {"Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"}

    def send(self, alert: Alert) -> None:
        annotation = {
            "text":      alert.summary(),
            "tags":      ["ml-monitor", alert.severity.value.lower()],
            "isRegion":  False,
            "time":      int(alert.fired_at.timestamp() * 1000),
        }
        resp = requests.post(
            f"{self.grafana_url}/api/annotations",
            json=annotation,
            headers=self.headers,
            timeout=5,
        )
        resp.raise_for_status()

MLflow alternative
------------------
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
with mlflow.start_run(run_name="monitor-alert"):
    mlflow.log_param("severity", alert.severity.value)
    mlflow.log_metrics({
        "accuracy":   alert.metrics.accuracy,
        "drift":      alert.metrics.drift_score,
        "latency_ms": alert.metrics.latency_ms,
    })
"""

import logging
from .base import NotificationChannel
from ..severity import Alert

log = logging.getLogger(__name__)


class DashboardNotifier(NotificationChannel):
    """Stub implementation — replace body of send() with Grafana / MLflow call."""

    def __init__(
        self,
        grafana_url: str = "http://localhost:3000",
        api_key: str = "MOCK_GRAFANA_KEY",
    ) -> None:
        self.grafana_url = grafana_url
        self.api_key = api_key

    def send(self, alert: Alert) -> None:
        # TODO: swap with Grafana Annotations API or mlflow.log_metrics call
        log.info("  📊 Dashboard → logged to Grafana | %s", alert.summary())
