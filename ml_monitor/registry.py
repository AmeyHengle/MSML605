"""
ml_monitor/registry.py
----------------------
ModelVersion dataclass + ModelRegistry.

Diagram: Model registry (v0, v1, v2, v3-live)

In production, back this with MLflow Model Registry, SageMaker Model Registry,
or a simple S3 manifest file. The interface stays the same.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """A single versioned model artifact."""
    version: str          # e.g. "v3"
    artifact_path: str    # S3 URI, local path, MLflow run ID …
    registered_at: datetime = field(default_factory=datetime.now)
    is_live: bool = False

    def __repr__(self) -> str:
        tag = " ← LIVE" if self.is_live else ""
        return f"ModelVersion({self.version}{tag})"


class ModelRegistry:
    """
    In-memory versioned model store.

    Public interface
    ----------------
    register(version, artifact_path)  → ModelVersion
    promote_to_live(version)          → ModelVersion
    get_live()                        → ModelVersion | None
    get_previous()                    → ModelVersion | None
    rollback()                        → ModelVersion | None   ← used by pipeline
    list_versions()                   → list[ModelVersion]
    """

    def __init__(self) -> None:
        self._versions: list[ModelVersion] = []

    # ── write operations ─────────────────────────────────────────────────

    def register(self, version: str, artifact_path: str) -> ModelVersion:
        mv = ModelVersion(version=version, artifact_path=artifact_path)
        self._versions.append(mv)
        log.info("Registry  ← registered %s at '%s'", version, artifact_path)
        return mv

    def promote_to_live(self, version: str) -> ModelVersion:
        for mv in self._versions:
            mv.is_live = False
        target = self._get(version)
        target.is_live = True
        log.info("Registry  ← promoted %s to LIVE", version)
        return target

    def rollback(self) -> Optional[ModelVersion]:
        """
        Demote current live → promote the version registered just before it.
        Diagram: 'Rollback to prev. version — From model registry'
        """
        prev = self.get_previous()
        if prev is None:
            log.warning("Registry  ← no previous version to roll back to!")
            return None

        current = self.get_live()
        if current:
            current.is_live = False
            log.warning("Registry  ← demoted %s", current.version)

        prev.is_live = True
        log.warning("Registry  ← ROLLBACK → restored %s", prev.version)
        return prev

    # ── read operations ──────────────────────────────────────────────────

    def get_live(self) -> Optional[ModelVersion]:
        for mv in reversed(self._versions):
            if mv.is_live:
                return mv
        return None

    def get_previous(self) -> Optional[ModelVersion]:
        """Return the version registered immediately before the current live one."""
        live_idx: Optional[int] = None
        for i, mv in enumerate(self._versions):
            if mv.is_live:
                live_idx = i
        if live_idx is not None and live_idx > 0:
            return self._versions[live_idx - 1]
        return None

    def list_versions(self) -> list[ModelVersion]:
        return list(self._versions)

    # ── helpers ──────────────────────────────────────────────────────────

    def _get(self, version: str) -> ModelVersion:
        for mv in self._versions:
            if mv.version == version:
                return mv
        raise KeyError(f"Version '{version}' not in registry")

    def __repr__(self) -> str:
        return (
            "ModelRegistry("
            + ", ".join(repr(v) for v in self._versions)
            + ")"
        )
