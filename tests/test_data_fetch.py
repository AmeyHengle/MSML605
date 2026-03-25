from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from ml605_pipeline.data import fetch_window_dataframe


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, url_to_payload: dict[str, dict]):
        self._map = url_to_payload

    def get(self, url: str, *_, **__):
        # Prefer the most specific (longest) prefix match.
        for key in sorted(self._map.keys(), key=len, reverse=True):
            if url.startswith(key):
                return _FakeResponse(self._map[key])
        raise KeyError(f"Unexpected URL: {url}")


def test_fetch_window_dataframe_merges_intensity_and_generation() -> None:
    start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    end = datetime(2026, 3, 1, 1, 0, tzinfo=UTC)

    intensity_payload = {
        "data": [
            {
                "from": "2026-03-01T00:00Z",
                "to": "2026-03-01T00:30Z",
                "intensity": {"forecast": 100, "actual": 110, "index": "low"},
            }
        ]
    }
    generation_payload = {
        "data": [
            {
                "from": "2026-03-01T00:00Z",
                "to": "2026-03-01T00:30Z",
                "generationmix": [{"fuel": "wind", "perc": 12.3}],
            }
        ]
    }
    factors_payload = {"data": [{"Coal": 937, "Wind": 0}]}

    session = _FakeSession(
        {
            "https://api.carbonintensity.org.uk/intensity/": intensity_payload,
            "https://api.carbonintensity.org.uk/generation/": generation_payload,
            "https://api.carbonintensity.org.uk/intensity/factors": factors_payload,
        }
    )

    result = fetch_window_dataframe(start, end, session=session)  # type: ignore[arg-type]
    df = result.df
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df.loc[0, "actual_intensity"] == 110
    assert df.loc[0, "forecast_intensity"] == 100
    assert df.loc[0, "intensity_index"] == "low"
    assert df.loc[0, "wind"] == 12.3
    assert result.factors["Coal"] == 937
