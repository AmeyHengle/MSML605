from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ml605_pipeline.registry import (
    MODEL_NAME,
    get_production_model_uri,
    register_model,
)


def test_model_name_constant_is_set() -> None:
    assert isinstance(MODEL_NAME, str)
    assert len(MODEL_NAME) > 0


def test_get_production_model_uri_returns_none_when_no_versions(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = []
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result is None


def test_get_production_model_uri_returns_uri_when_version_exists(monkeypatch) -> None:
    mock_version = MagicMock()
    mock_client = MagicMock()
    mock_client.get_latest_versions.return_value = [mock_version]
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result == f"models:/{MODEL_NAME}/Production"


def test_get_production_model_uri_returns_none_on_exception(monkeypatch) -> None:
    mock_client = MagicMock()
    mock_client.get_latest_versions.side_effect = Exception("registry unavailable")
    with patch("ml605_pipeline.registry.MlflowClient", return_value=mock_client):
        result = get_production_model_uri()
    assert result is None
