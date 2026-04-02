"""
ml_monitor/notifiers/base.py
-----------------------------
Abstract base class for all notification channels.
New channels (OpsGenie, Teams, webhooks …) only need to implement send().
"""

from abc import ABC, abstractmethod
from ..severity import Alert


class NotificationChannel(ABC):
    """All notifiers must implement this interface."""

    @abstractmethod
    def send(self, alert: Alert) -> None:
        """Dispatch the alert through this channel."""
        ...
