from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    # Runtime window
    window_hours: int = 12
    interval_seconds: int = 30
    data_resolution_minutes: int = 5

    # IO
    data_dir: Path = Path("data")
    features_path: Path = Path("features_used.txt")

    # MLflow
    mlflow_experiment: str = "daily-intensity-pipeline"
    run_name_prefix: str = "daily_run"

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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir / f"historical_{self.window_label}.csv"


def load_config_from_env() -> PipelineConfig:
    # Keep it simple: only a few env overrides.
    window_hours = int(getenv("PIPELINE_WINDOW_HOURS", "12"))
    interval_seconds = int(getenv("PIPELINE_INTERVAL_SECONDS", "30"))
    data_resolution_minutes = int(getenv("PIPELINE_DATA_RESOLUTION_MINUTES", "5"))
    mlflow_experiment = getenv("MLFLOW_EXPERIMENT", "daily-intensity-pipeline")
    return PipelineConfig(
        window_hours=window_hours,
        interval_seconds=interval_seconds,
        data_resolution_minutes=data_resolution_minutes,
        mlflow_experiment=mlflow_experiment,
    )