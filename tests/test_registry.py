from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml605_pipeline.registry import (
    MODEL_NAME,
    get_production_model_uri,
    load_production_model,
    register_model,
    transition_model_stage,
)


def test_model_name_constant_is_set() -> None:
    assert isinstance(MODEL_NAME, str)
    assert len(MODEL_NAME) > 0


def test_get_production_model_uri_returns_none_when_no_versions() -> None:
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result is None


def test_get_production_model_uri_returns_uri_when_version_exists() -> None:
    mock_version = MagicMock()
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result == f"models:/{MODEL_NAME}/Production"


def test_get_production_model_uri_returns_none_on_exception() -> None:
    mock_client = MagicMock()
    mock_client.get_latest_versions.side_effect = Exception("registry unavailable")
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result is None


def test_register_model_returns_version_string() -> None:
    mock_result = MagicMock()
    mock_result.version = "3"
    with patch("ml605_pipeline.registry.mlflow.register_model", return_value=mock_result):
        version = register_model(run_id="abc123")
    assert version == "3"


def test_transition_model_stage_calls_client_correctly() -> None:
    mock_client = MagicMock()
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        transition_model_stage(version="1", stage="Production")
    mock_client.transition_model_version_stage.assert_called_once_with(
        name=MODEL_NAME, version="1", stage="Production", archive_existing_versions=True
    )


def test_load_production_model_returns_none_when_no_production_model() -> None:
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = load_production_model()
    assert result is None
