"""
ml_monitor/router.py
---------------------
AlertRouter — severity-based notification dispatch.

Diagram (Image 2):
  Monitoring system → Alert router → Slack / Email / Dashboard
  Severity routing:
    LOW      → Slack only
    MEDIUM   → Slack + email
    CRITICAL → all channels + auto-rollback (rollback handled by pipeline)
"""

import logging
from .severity import Alert, Severity
from .notifiers import (
    NotificationChannel,
    SlackNotifier,
    EmailNotifier,
    PagerDutyNotifier,
    DashboardNotifier,
)

log = logging.getLogger(__name__)


class AlertRouter:
    """
    Routes an Alert to the correct set of channels based on severity.

    The routing table is built in __init__ and can be overridden by
    passing custom channel instances — useful for testing or extending
    with new channels (e.g. OpsGenie, Teams webhook).
    """

    def __init__(
        self,
        slack: SlackNotifier,
        email: EmailNotifier,
        pagerduty: PagerDutyNotifier,
        dashboard: DashboardNotifier,
    ) -> None:
        self._slack     = slack
        self._email     = email
        self._pagerduty = pagerduty
        self._dashboard = dashboard

        # Image 2 — severity routing table
        self._routing: dict[Severity, list[NotificationChannel]] = {
            Severity.LOW:      [self._slack],
            Severity.MEDIUM:   [self._slack, self._email],
            Severity.CRITICAL: [self._slack, self._email,
                                self._pagerduty, self._dashboard],
        }

    def route(self, alert: Alert) -> None:
        """Fire the alert to every channel mapped to its severity level."""
        log.warning("AlertRouter  → firing %s", alert.summary())
        for channel in self._routing[alert.severity]:
            channel.send(alert)
