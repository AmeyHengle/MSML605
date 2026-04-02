"""
tests/test_registry.py
-----------------------
Unit tests for ModelRegistry.
"""

import pytest
from ml_monitor import ModelRegistry


@pytest.fixture
def registry():
    r = ModelRegistry()
    for v in ["v0", "v1", "v2", "v3"]:
        r.register(v, f"s3://models/{v}/model.pkl")
    r.promote_to_live("v3")
    return r


def test_live_version_is_v3(registry):
    assert registry.get_live().version == "v3"


def test_previous_version_is_v2(registry):
    assert registry.get_previous().version == "v2"


def test_rollback_makes_v2_live(registry):
    prev = registry.rollback()
    assert prev.version == "v2"
    assert registry.get_live().version == "v2"


def test_rollback_demotes_v3(registry):
    registry.rollback()
    v3 = next(v for v in registry.list_versions() if v.version == "v3")
    assert not v3.is_live


def test_double_rollback(registry):
    registry.rollback()   # v3 → v2
    registry.rollback()   # v2 → v1
    assert registry.get_live().version == "v1"


def test_rollback_at_oldest_version_returns_none():
    r = ModelRegistry()
    r.register("v0", "s3://models/v0/model.pkl")
    r.promote_to_live("v0")
    result = r.rollback()
    assert result is None
