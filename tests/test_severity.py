"""
tests/test_severity.py
-----------------------
Unit tests for SeverityClassifier.
"""

import pytest
from ml_monitor import ThresholdConfig, SeverityClassifier, Severity, ModelMetrics
from datetime import datetime


@pytest.fixture
def cfg():
    return ThresholdConfig()


@pytest.fixture
def classifier(cfg):
    return SeverityClassifier(cfg)


def make_metrics(accuracy=0.95, drift=0.10, latency=100.0):
    return ModelMetrics(accuracy=accuracy, drift_score=drift,
                        latency_ms=latency, timestamp=datetime.now())


class TestNoBreaches:
    def test_all_ok_returns_none(self, classifier):
        m = make_metrics(accuracy=0.95, drift=0.10, latency=100.0)
        severity, reasons = classifier.classify(m)
        assert severity is None
        assert reasons == []


class TestLow:
    def test_single_accuracy_breach_is_low(self, classifier):
        m = make_metrics(accuracy=0.75)   # below 0.80 but above 0.70
        severity, reasons = classifier.classify(m)
        assert severity == Severity.LOW
        assert len(reasons) == 1

    def test_single_drift_breach_is_low(self, classifier):
        m = make_metrics(drift=0.40)      # above 0.30 but below 0.50
        severity, reasons = classifier.classify(m)
        assert severity == Severity.LOW

    def test_single_latency_breach_is_low(self, classifier):
        m = make_metrics(latency=400.0)   # above 300 but below 600
        severity, reasons = classifier.classify(m)
        assert severity == Severity.LOW


class TestMedium:
    def test_two_warning_breaches_is_medium(self, classifier):
        # accuracy + drift both breached (non-critical values)
        m = make_metrics(accuracy=0.75, drift=0.40)
        severity, reasons = classifier.classify(m)
        assert severity == Severity.MEDIUM
        assert len(reasons) == 2


class TestCritical:
    def test_critical_accuracy_escalates(self, classifier):
        m = make_metrics(accuracy=0.65)   # below critical threshold 0.70
        severity, _ = classifier.classify(m)
        assert severity == Severity.CRITICAL

    def test_critical_drift_escalates(self, classifier):
        m = make_metrics(drift=0.55)      # above critical threshold 0.50
        severity, _ = classifier.classify(m)
        assert severity == Severity.CRITICAL

    def test_critical_latency_escalates(self, classifier):
        m = make_metrics(latency=650.0)   # above critical threshold 600
        severity, _ = classifier.classify(m)
        assert severity == Severity.CRITICAL
