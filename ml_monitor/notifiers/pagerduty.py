"""
ml_monitor/notifiers/pagerduty.py
----------------------------------
PagerDuty notification channel.

Diagram: PagerDuty — Escalate if severe

Production wiring (Events API v2)
-----------------------------------
import requests, json

EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"

class PagerDutyNotifier(NotificationChannel):
    def __init__(self, routing_key: str):
        self.routing_key = routing_key

    def send(self, alert: Alert) -> None:
        payload = {
            "routing_key":  self.routing_key,
            "event_action": "trigger",
            "payload": {
                "summary":   alert.summary(),
                "severity":  alert.severity.value.lower(),
                "source":    "ml-monitor",
                "timestamp": alert.fired_at.isoformat(),
                "custom_details": {"reasons": alert.reasons},
            },
        }
        resp = requests.post(EVENTS_URL, json=payload, timeout=5)
        resp.raise_for_status()
"""

import logging
from .base import NotificationChannel
from ..severity import Alert

log = logging.getLogger(__name__)


class PagerDutyNotifier(NotificationChannel):
    """Stub implementation — replace body of send() with real PD Events API call."""

    def __init__(self, routing_key: str = "MOCK_PD_ROUTING_KEY") -> None:
        self.routing_key = routing_key

    def send(self, alert: Alert) -> None:
        # TODO: swap with PagerDuty Events API v2 call
        log.warning("  🚨 PagerDuty → ESCALATING | %s", alert.summary())
