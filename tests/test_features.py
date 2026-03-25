from __future__ import annotations

import pandas as pd

from ml605_pipeline.features import (
    add_time_features,
    normalize_factor_name,
    one_hot_intensity_index,
)


def test_normalize_factor_name() -> None:
    assert normalize_factor_name("Gas (Combined Cycle)") == "factor_gas_combined_cycle"
    assert normalize_factor_name("Dutch Imports") == "factor_dutch_imports"


def test_add_time_features() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-03-01T00:00:00Z", "2026-03-01T12:34:00Z"], utc=True
            )
        }
    )
    out = add_time_features(df)
    assert set(["hour", "day_of_week", "month", "day_of_year", "is_weekend"]).issubset(
        out.columns
    )


def test_one_hot_intensity_index() -> None:
    df = pd.DataFrame({"intensity_index": ["low", "high", "low"]})
    out = one_hot_intensity_index(df)
    assert "intensity_index" not in out.columns
    assert "intensity_index_low" in out.columns
    assert "intensity_index_high" in out.columns
