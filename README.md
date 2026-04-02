# ml-monitor

A modular ML monitoring system for production models.  
Tracks accuracy, drift, and latency — fires severity-routed alerts and auto-rollbacks on critical failures.

## Project structure

```
ml-monitor/
├── main.py                        # Entrypoint
├── requirements.txt
├── ml_monitor/
│   ├── __init__.py                # Public API surface
│   ├── config.py                  # ThresholdConfig dataclass
│   ├── metrics.py                 # ModelMetrics + ModelMonitor
│   ├── severity.py                # Severity enum, Alert, SeverityClassifier
│   ├── registry.py                # ModelVersion + ModelRegistry
│   ├── router.py                  # AlertRouter (severity → channels)
│   ├── pipeline.py                # MonitoringPipeline + build_pipeline()
│   └── notifiers/
│       ├── __init__.py
│       ├── base.py                # NotificationChannel ABC
│       ├── slack.py               # SlackNotifier
│       ├── email.py               # EmailNotifier
│       ├── pagerduty.py           # PagerDutyNotifier
│       └── dashboard.py           # DashboardNotifier (Grafana / MLflow)
└── tests/
    ├── test_severity.py           # SeverityClassifier unit tests
    ├── test_registry.py           # ModelRegistry unit tests
    └── test_pipeline.py           # MonitoringPipeline integration tests
```

## Monitoring flow

```
Live model in production
        │
        ▼
Continuous monitoring (accuracy, drift, latency)
        │
  Threshold breached?
   ├─ No  ──► Log OK, keep serving
   └─ Yes ──► Fire alert
               ├─ LOW      → Slack only
               ├─ MEDIUM   → Slack + Email
               └─ CRITICAL → Slack + Email + PagerDuty + Dashboard
                              + Auto-rollback from model registry
```

## Severity rules

| Condition                          | Severity |
|------------------------------------|----------|
| 1 metric breaches warning threshold | LOW     |
| 2+ metrics breach warning threshold | MEDIUM  |
| Any metric breaches critical threshold | CRITICAL |

Default thresholds (`ThresholdConfig`):

| Metric    | Warning | Critical |
|-----------|---------|----------|
| accuracy  | < 0.80  | < 0.70   |
| drift     | > 0.30  | > 0.50   |
| latency   | > 300ms | > 600ms  |

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Running tests

```bash
pytest tests/ -v
```

## Wiring in real integrations

Every stub has a drop-in comment at the top of its file. The key ones:

| File | Replace | With |
|------|---------|------|
| `metrics.py` | `_collect_metrics()` | Prometheus / Evidently / your eval harness |
| `notifiers/slack.py` | `send()` body | `slack_sdk.webhook.WebhookClient` |
| `notifiers/email.py` | `send()` body | `smtplib` or `boto3` SES |
| `notifiers/pagerduty.py` | `send()` body | PagerDuty Events API v2 |
| `notifiers/dashboard.py` | `send()` body | Grafana Annotations API or `mlflow` |
| `pipeline.py` `build_pipeline()` | registry seed loop | MLflow Model Registry loader |

## Extending

**Add a new notification channel**
1. Create `ml_monitor/notifiers/teams.py` subclassing `NotificationChannel`
2. Add it to `notifiers/__init__.py`
3. Wire it into `AlertRouter._routing` in `router.py`

**Change thresholds at runtime**
```python
from ml_monitor import build_pipeline, ThresholdConfig

pipeline = build_pipeline(
    thresholds=ThresholdConfig(min_accuracy=0.85, max_drift_score=0.20)
)
pipeline.run()
```
