from __future__ import annotations

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

MODEL_NAME = "carbon-intensity-model"


def register_model(run_id: str, model_artifact_path: str = "model") -> str:
    """Register a trained model in MLflow Model Registry. Returns version string."""
    model_uri = f"runs:/{run_id}/{model_artifact_path}"
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    return result.version


def transition_model_stage(version: str, stage: str) -> None:
    """Transition model version to 'Staging' or 'Production'."""
    client = MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME, version=version, stage=stage, archive_existing_versions=True
    )


def get_production_model_uri() -> str | None:
    """Return URI of the current Production model, or None if none registered."""
    client = MlflowClient()
    try:
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        if not versions:
            return None
        return f"models:/{MODEL_NAME}/Production"
    except Exception:
        return None


def load_production_model():
    """Load and return the current Production model. Returns None if none registered."""
    uri = get_production_model_uri()
    if uri is None:
        return None
    return mlflow.sklearn.load_model(uri)
