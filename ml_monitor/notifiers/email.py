"""
ml_monitor/notifiers/email.py
------------------------------
Email notification channel.

Diagram: ml-team@company.com

Production wiring (smtplib)
----------------------------
import smtplib
from email.mime.text import MIMEText

class EmailNotifier(NotificationChannel):
    def __init__(self, to: str, smtp_host: str, smtp_port: int = 587,
                 username: str = "", password: str = ""):
        self.to        = to
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username  = username
        self.password  = password

    def send(self, alert: Alert) -> None:
        msg = MIMEText(alert.summary())
        msg["Subject"] = f"[ML Monitor] {alert.severity.value} Alert"
        msg["From"]    = self.username
        msg["To"]      = self.to
        with smtplib.SMTP(self.smtp_host, self.smtp_port) as s:
            s.starttls()
            s.login(self.username, self.password)
            s.send_message(msg)
"""

import logging
from .base import NotificationChannel
from ..severity import Alert

log = logging.getLogger(__name__)


class EmailNotifier(NotificationChannel):
    """Stub implementation — replace body of send() with real smtplib/SES call."""

    def __init__(
        self,
        to: str = "ml-team@company.com",
        smtp_host: str = "localhost",
    ) -> None:
        self.to = to
        self.smtp_host = smtp_host

    def send(self, alert: Alert) -> None:
        # TODO: swap with smtplib or boto3 SES call
        log.info("  📧 Email  → %s | %s", self.to, alert.summary())
