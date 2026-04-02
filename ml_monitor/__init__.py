"""
ml_monitor
==========
ML monitoring system — public API surface.

Quick start
-----------
from ml_monitor import build_pipeline

pipeline = build_pipeline(poll_interval_s=10.0)
pipeline.run()
"""

from .config   import ThresholdConfig
from .metrics  import ModelMetrics, ModelMonitor
from .severity import Alert, Severity, SeverityClassifier
from .registry import ModelVersion, ModelRegistry
from .router   import AlertRouter
from .pipeline import MonitoringPipeline, build_pipeline
from .notifiers import (
    NotificationChannel,
    SlackNotifier,
    EmailNotifier,
    PagerDutyNotifier,
    DashboardNotifier,
)

__all__ = [
    # config
    "ThresholdConfig",
    # metrics
    "ModelMetrics", "ModelMonitor",
    # severity
    "Alert", "Severity", "SeverityClassifier",
    # registry
    "ModelVersion", "ModelRegistry",
    # routing
    "AlertRouter",
    # pipeline
    "MonitoringPipeline", "build_pipeline",
    # notifiers
    "NotificationChannel",
    "SlackNotifier", "EmailNotifier",
    "PagerDutyNotifier", "DashboardNotifier",
]
