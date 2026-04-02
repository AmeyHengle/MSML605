"""
tests/test_pipeline.py
-----------------------
Integration tests for MonitoringPipeline.
Uses monkeypatching to inject deterministic metrics instead of random ones.
"""

import pytest
from datetime import datetime

from ml_monitor import (
    ThresholdConfig, ModelMetrics, ModelMonitor,
    ModelRegistry, AlertRouter, MonitoringPipeline,
    SlackNotifier, EmailNotifier, PagerDutyNotifier, DashboardNotifier,
    Severity,
)


# ── helpers ───────────────────────────────────────────────────────────────

def make_pipeline(metrics: ModelMetrics) -> tuple[MonitoringPipeline, list]:
    """
    Build a pipeline where _collect_metrics() always returns `metrics`.
    Returns (pipeline, sent_alerts) where sent_alerts is populated on dispatch.
    """
    sent: list[dict] = []

    class CapturingSlack(SlackNotifier):
        def send(self, alert):
            sent.append({"channel": "slack", "severity": alert.severity})

    class CapturingEmail(EmailNotifier):
        def send(self, alert):
            sent.append({"channel": "email", "severity": alert.severity})

    class CapturingPD(PagerDutyNotifier):
        def send(self, alert):
            sent.append({"channel": "pagerduty", "severity": alert.severity})

    class CapturingDash(DashboardNotifier):
        def send(self, alert):
            sent.append({"channel": "dashboard", "severity": alert.severity})

    cfg      = ThresholdConfig()
    monitor  = ModelMonitor(cfg)
    monitor._collect_metrics = lambda: metrics   # inject fixed metrics

    router = AlertRouter(
        slack=CapturingSlack(),
        email=CapturingEmail(),
        pagerduty=CapturingPD(),
        dashboard=CapturingDash(),
    )

    registry = ModelRegistry()
    for v in ["v0", "v1", "v2", "v3"]:
        registry.register(v, f"s3://models/{v}/model.pkl")
    registry.promote_to_live("v3")

    pipeline = MonitoringPipeline(monitor=monitor, router=router,
                                  registry=registry, poll_interval_s=0)
    return pipeline, sent


def m(accuracy=0.95, drift=0.10, latency=100.0):
    return ModelMetrics(accuracy=accuracy, drift_score=drift,
                        latency_ms=latency, timestamp=datetime.now())


# ── tests ─────────────────────────────────────────────────────────────────

class TestOkCycle:
    def test_no_alerts_when_all_ok(self):
        pipeline, sent = make_pipeline(m())
        result = pipeline.run_once()
        assert result["status"] == "ok"
        assert sent == []


class TestLowSeverity:
    def test_only_slack_notified(self):
        pipeline, sent = make_pipeline(m(drift=0.40))  # LOW
        pipeline.run_once()
        channels = [s["channel"] for s in sent]
        assert "slack" in channels
        assert "email" not in channels
        assert "pagerduty" not in channels


class TestMediumSeverity:
    def test_slack_and_email_notified(self):
        pipeline, sent = make_pipeline(m(accuracy=0.75, drift=0.40))  # MEDIUM
        pipeline.run_once()
        channels = [s["channel"] for s in sent]
        assert "slack" in channels
        assert "email" in channels
        assert "pagerduty" not in channels


class TestCriticalSeverity:
    def test_all_channels_notified(self):
        pipeline, sent = make_pipeline(m(accuracy=0.65))  # CRITICAL
        pipeline.run_once()
        channels = {s["channel"] for s in sent}
        assert channels == {"slack", "email", "pagerduty", "dashboard"}

    def test_auto_rollback_triggered(self):
        pipeline, _ = make_pipeline(m(accuracy=0.65))
        result = pipeline.run_once()
        assert result["rolled_back_to"] == "v2"
        assert pipeline.registry.get_live().version == "v2"

    def test_no_rollback_on_medium(self):
        pipeline, _ = make_pipeline(m(accuracy=0.75, drift=0.40))
        result = pipeline.run_once()
        assert result["rolled_back_to"] is None
        assert pipeline.registry.get_live().version == "v3"
