from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    # Runtime
    window_hours: int = 12
    interval_seconds: int = 30

    # Drift thresholds (PSI/KS only)
    ks_threshold: float = 0.10
    psi_threshold: float = 0.25

    # Project root + canonical paths
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    windows_dir: Path = Path("data/windows")
    reference_data_path: Path = Path("data/historical_data.csv")
    features_path: Path = Path("features_used.txt")

    # MCP
    mcp_base_url: str = "http://localhost:8001"

    @property
    def window_end_utc(self) -> datetime:
        return datetime.now(tz=UTC)

    @property
    def window_start_utc(self) -> datetime:
        return self.window_end_utc - timedelta(hours=self.window_hours)

    @property
    def window_label(self) -> str:
        end = self.window_end_utc.strftime("%Y%m%d_%H%M%S")
        return f"last_{self.window_hours}h_until_{end}Z"

    @property
    def output_csv(self) -> Path:
        self.windows_dir.mkdir(parents=True, exist_ok=True)
        return self.windows_dir / f"window_{self.window_label}.csv"


def _resolve_path(root: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (root / p)


def load_config_from_env() -> PipelineConfig:
    root = Path(getenv("PROJECT_ROOT", ".")).resolve()

    data_dir = _resolve_path(root, getenv("DATA_DIR", "data"))
    windows_dir = _resolve_path(root, getenv("WINDOWS_DIR", "data/windows"))
    reference_data_path = _resolve_path(root, getenv("REFERENCE_DATA_PATH", "data/historical_data.csv"))
    features_path = _resolve_path(root, getenv("FEATURES_PATH", "features_used.txt"))

    return PipelineConfig(
        window_hours=int(getenv("PIPELINE_WINDOW_HOURS", "12")),
        interval_seconds=int(getenv("PIPELINE_INTERVAL_SECONDS", "30")),
        ks_threshold=float(getenv("PIPELINE_KS_THRESHOLD", "0.10")),
        psi_threshold=float(getenv("PIPELINE_PSI_THRESHOLD", "0.25")),
        project_root=root,
        data_dir=data_dir,
        windows_dir=windows_dir,
        reference_data_path=reference_data_path,
        features_path=features_path,
        mcp_base_url=getenv("MCP_BASE_URL", "http://localhost:8001"),
    )
