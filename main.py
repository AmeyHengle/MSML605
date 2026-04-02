"""
main.py
-------
Entrypoint. Run with:
  python main.py
"""

import logging
from ml_monitor import build_pipeline, ThresholdConfig

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)

if __name__ == "__main__":
    thresholds = ThresholdConfig(
        min_accuracy=0.80,
        max_drift_score=0.30,
        max_latency_ms=300.0,
        critical_accuracy=0.70,
        critical_drift=0.50,
        critical_latency_ms=600.0,
    )

    pipeline = build_pipeline(
        poll_interval_s=2.0,
        thresholds=thresholds,
        slack_channel="#ml-alerts",
        alert_email="ml-team@company.com",
    )

    print("Registry at startup:", pipeline.registry)
    print()
    pipeline.run(max_cycles=6)
    print("\nRegistry at end:    ", pipeline.registry)
