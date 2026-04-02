"""
ml_monitor/pipeline.py
-----------------------
MonitoringPipeline — the main polling loop that wires everything together.
build_pipeline()   — convenience factory for quick startup.

Diagram (Image 1) flow per cycle:
  collect metrics
    ↓
  threshold breached?
    → No  : log OK, continue serving
    → Yes : AlertRouter.route(alert)
              if CRITICAL → ModelRegistry.rollback()
"""

from __future__ import annotations

import time
import logging
from typing import Optional

from .config import ThresholdConfig
from .metrics import ModelMonitor
from .registry import ModelRegistry
from .router import AlertRouter
from .severity import Severity
from .notifiers import (
    SlackNotifier,
    EmailNotifier,
    PagerDutyNotifier,
    DashboardNotifier,
)

log = logging.getLogger(__name__)


class MonitoringPipeline:
    """
    Orchestrates one monitoring cycle:
      1. Collect metrics  (ModelMonitor)
      2. Check thresholds (ModelMonitor)
      3. Route alert      (AlertRouter)       — if breached
      4. Auto-rollback    (ModelRegistry)     — if CRITICAL

    run_once() executes a single cycle and returns a status dict.
    run()      loops indefinitely (or for max_cycles) with poll_interval_s sleep.
    """

    def __init__(
        self,
        monitor: ModelMonitor,
        router: AlertRouter,
        registry: ModelRegistry,
        poll_interval_s: float = 5.0,
    ) -> None:
        self.monitor          = monitor
        self.router           = router
        self.registry         = registry
        self.poll_interval_s  = poll_interval_s
        self._running         = False

    # ── public API ───────────────────────────────────────────────────────

    def run_once(self) -> dict:
        """
        Execute one monitoring cycle.
        Returns a status dict — useful for testing and dashboards.
        """
        metrics = self.monitor.collect()
        alert   = self.monitor.check_thresholds(metrics)

        if alert is None:
            log.info("Status   →  ✅ All thresholds OK — keeping current model\n")
            return {"status": "ok", "metrics": metrics}

        self.router.route(alert)

        rolled_back_to: Optional[str] = None
        if alert.severity == Severity.CRITICAL:
            log.warning("Pipeline →  🔴 CRITICAL — triggering auto-rollback")
            prev = self.registry.rollback()
            if prev:
                rolled_back_to = prev.version
                log.warning("Pipeline →  Restored to %s\n", prev.version)
            else:
                log.error("Pipeline →  Rollback failed — no previous version!\n")
        else:
            log.info("")   # blank line between cycles

        return {
            "status":         "alert",
            "severity":       alert.severity.value,
            "reasons":        alert.reasons,
            "metrics":        metrics,
            "rolled_back_to": rolled_back_to,
        }

    def run(self, max_cycles: Optional[int] = None) -> None:
        """
        Blocking polling loop.
        Pass max_cycles=N for a finite run (handy in CI / smoke tests).
        """
        self._running = True
        cycle = 0
        log.info(
            "Pipeline →  🟢 Monitoring started  (interval=%ss)", self.poll_interval_s
        )
        try:
            while self._running:
                cycle += 1
                log.info("── Cycle %d ──────────────────────────────────", cycle)
                self.run_once()

                if max_cycles and cycle >= max_cycles:
                    break

                time.sleep(self.poll_interval_s)
        except KeyboardInterrupt:
            log.info("Pipeline →  stopped by user")
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the run() loop to exit after the current cycle."""
        self._running = False


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def build_pipeline(
    poll_interval_s: float = 5.0,
    thresholds: Optional[ThresholdConfig] = None,
    slack_channel: str = "#ml-alerts",
    alert_email: str = "ml-team@company.com",
) -> MonitoringPipeline:
    """
    Wire all components and return a ready-to-run MonitoringPipeline.

    Override any constructor argument to swap in real integrations:
      pipeline = build_pipeline()
      pipeline.monitor._collect_metrics = my_real_collector
      pipeline.run()
    """
    thresholds = thresholds or ThresholdConfig()

    monitor  = ModelMonitor(config=thresholds)
    router   = AlertRouter(
        slack     = SlackNotifier(channel=slack_channel),
        email     = EmailNotifier(to=alert_email),
        pagerduty = PagerDutyNotifier(),
        dashboard = DashboardNotifier(),
    )
    registry = ModelRegistry()

    # Seed registry — replace with real loader (MLflow, S3 manifest …)
    for ver in ["v0", "v1", "v2", "v3"]:
        registry.register(ver, artifact_path=f"s3://models/{ver}/model.pkl")
    registry.promote_to_live("v3")

    return MonitoringPipeline(
        monitor=monitor,
        router=router,
        registry=registry,
        poll_interval_s=poll_interval_s,
    )
