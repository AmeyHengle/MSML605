"""
ml_monitor/notifiers/slack.py
------------------------------
Slack notification channel.

Diagram: Slack / email — Notify on-call team (#ml-alerts)

Production wiring
-----------------
pip install slack-sdk

from slack_sdk.webhook import WebhookClient

class SlackNotifier(NotificationChannel):
    def __init__(self, webhook_url: str, channel: str = "#ml-alerts"):
        self.client  = WebhookClient(webhook_url)
        self.channel = channel

    def send(self, alert: Alert) -> None:
        self.client.send(
            text=f":rotating_light: *{alert.severity.value}* alert\n{alert.summary()}"
        )
"""

import logging
from .base import NotificationChannel
from ..severity import Alert

log = logging.getLogger(__name__)


class SlackNotifier(NotificationChannel):
    """Stub implementation — replace body of send() with real Slack SDK call."""

    def __init__(
        self,
        channel: str = "#ml-alerts",
        webhook_url: str = "https://hooks.slack.com/services/MOCK",
    ) -> None:
        self.channel = channel
        self.webhook_url = webhook_url

    def send(self, alert: Alert) -> None:
        # TODO: swap with slack_sdk.webhook.WebhookClient call
        log.info("  📣 Slack  → %s | %s", self.channel, alert.summary())
