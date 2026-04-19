from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics_improvements.generate_report import (  # noqa: E402
    _build_conclusion,
    _fmt_metric,
    _fmt_percent,
    _fmt_seconds,
    generate_tuning_report,
)
from metrics_improvements.mlflow_metrics_comparison import (  # noqa: E402
    BASELINE_LABEL,
    TUNED_LABEL,
    _metric_meets_target,
    _run_duration_seconds,
    compute_time_to_target_from_history,
)


def _ts(seconds_from_epoch: float) -> pd.Timestamp:
    return pd.Timestamp(seconds_from_epoch, unit="s")


def _fake_runs_df(metrics: list[float], durations_s: list[float]) -> pd.DataFrame:
    assert len(metrics) == len(durations_s)
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    cursor = 1_700_000_000.0
    for d in durations_s:
        starts.append(_ts(cursor))
        ends.append(_ts(cursor + d))
        cursor += d + 3600.0
    return pd.DataFrame(
        {
            "metrics.rmse": metrics,
            "start_time": starts,
            "end_time": ends,
            "run_id": [f"r{i}" for i in range(len(metrics))],
            "tags.mlflow.runName": [f"run-{i}" for i in range(len(metrics))],
        }
    )


def test_metric_meets_target_lower_is_better() -> None:
    assert _metric_meets_target(39.0, 40.0, higher_is_better=False)
    assert _metric_meets_target(40.0, 40.0, higher_is_better=False)
    assert not _metric_meets_target(40.1, 40.0, higher_is_better=False)


def test_metric_meets_target_higher_is_better() -> None:
    assert _metric_meets_target(0.91, 0.90, higher_is_better=True)
    assert not _metric_meets_target(0.89, 0.90, higher_is_better=True)


def test_run_duration_seconds_handles_missing_end() -> None:
    row = pd.Series({"start_time": _ts(100.0), "end_time": pd.NaT})
    assert _run_duration_seconds(row) == 0.0


def test_run_duration_seconds_handles_negative_delta() -> None:
    row = pd.Series({"start_time": _ts(200.0), "end_time": _ts(100.0)})
    assert _run_duration_seconds(row) == 0.0


def test_run_duration_seconds_positive_case() -> None:
    row = pd.Series({"start_time": _ts(100.0), "end_time": _ts(130.0)})
    assert _run_duration_seconds(row) == pytest.approx(30.0)


def test_compute_time_to_target_sums_compute_time_not_wall_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _fake_runs_df(metrics=[50.0, 45.0, 39.0, 35.0], durations_s=[60.0, 90.0, 120.0, 150.0])

    def fake_get_runs(_name: str, _metric: str) -> pd.DataFrame:
        return df

    monkeypatch.setattr(
        "metrics_improvements.mlflow_metrics_comparison._get_experiment_runs",
        fake_get_runs,
    )

    result = compute_time_to_target_from_history(
        experiment_name="fake",
        metric_name="rmse",
        threshold=40.0,
        higher_is_better=False,
        workflow_label=BASELINE_LABEL,
    )

    assert result["Reached Target"] is True
    assert result["Runs Needed"] == 3
    assert result["Time To Target Seconds"] == pytest.approx(60.0 + 90.0 + 120.0)
    assert result["Best Metric"] == pytest.approx(35.0)


def test_compute_time_to_target_not_reached(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _fake_runs_df(metrics=[50.0, 45.0], durations_s=[60.0, 90.0])
    monkeypatch.setattr(
        "metrics_improvements.mlflow_metrics_comparison._get_experiment_runs",
        lambda _n, _m: df,
    )

    result = compute_time_to_target_from_history(
        experiment_name="fake",
        metric_name="rmse",
        threshold=40.0,
        higher_is_better=False,
        workflow_label=BASELINE_LABEL,
    )

    assert result["Reached Target"] is False
    assert result["Runs Needed"] is None
    assert result["Time To Target Seconds"] is None
    assert result["Best Metric"] == pytest.approx(45.0)


def test_compute_time_to_target_empty_df(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = _fake_runs_df(metrics=[], durations_s=[])
    monkeypatch.setattr(
        "metrics_improvements.mlflow_metrics_comparison._get_experiment_runs",
        lambda _n, _m: empty,
    )

    result = compute_time_to_target_from_history(
        experiment_name="fake",
        metric_name="rmse",
        threshold=40.0,
        higher_is_better=False,
        workflow_label=BASELINE_LABEL,
    )

    assert result["Reached Target"] is False
    assert result["Best Metric"] is None


def test_fmt_helpers() -> None:
    assert _fmt_seconds(None) == "N/A"
    assert _fmt_seconds(30.0) == "30.0s"
    assert _fmt_seconds(120.0) == "2.00m"
    assert _fmt_metric(1.2345) == "1.2345"
    assert _fmt_metric(None) == "N/A"
    assert _fmt_percent(12.5) == "12.50%"
    assert _fmt_percent(None) == "N/A"


def test_build_conclusion_exact_prefix_match() -> None:
    """Regression guard: str.contains('With') previously matched 'Without' too."""
    df = pd.DataFrame(
        [
            {"Workflow": BASELINE_LABEL, "Time To Target Seconds": 300.0},
            {"Workflow": TUNED_LABEL, "Time To Target Seconds": 120.0},
        ]
    )
    out = _build_conclusion(df)
    assert "saved `180.0` seconds" in out
    assert "60.00%" in out


def test_build_conclusion_missing_workflow() -> None:
    df = pd.DataFrame([{"Workflow": TUNED_LABEL, "Time To Target Seconds": 120.0}])
    out = _build_conclusion(df)
    assert "missing" in out.lower()


def test_build_conclusion_target_not_reached() -> None:
    df = pd.DataFrame(
        [
            {"Workflow": BASELINE_LABEL, "Time To Target Seconds": None},
            {"Workflow": TUNED_LABEL, "Time To Target Seconds": 120.0},
        ]
    )
    out = _build_conclusion(df)
    assert "not yet reached" in out.lower()


def test_generate_tuning_report_writes_markdown(tmp_path: Path) -> None:
    df_comp = pd.DataFrame(
        [
            {
                "Workflow": BASELINE_LABEL,
                "Experiment": "baseline",
                "Metric Name": "rmse",
                "Target Threshold": 40.0,
                "Reached Target": True,
                "Runs Needed": 3,
                "Best Metric": 35.0,
                "Time To Target Seconds": 300.0,
                "Absolute Time Saved Seconds": 180.0,
                "Percent Time Saved": 60.0,
            },
            {
                "Workflow": TUNED_LABEL,
                "Experiment": "tuned",
                "Metric Name": "rmse",
                "Target Threshold": 40.0,
                "Reached Target": True,
                "Runs Needed": 2,
                "Best Metric": 34.0,
                "Time To Target Seconds": 120.0,
                "Absolute Time Saved Seconds": 180.0,
                "Percent Time Saved": 60.0,
            },
        ]
    )
    target = tmp_path / "report.md"
    generate_tuning_report(df_comp=df_comp, target_file=str(target))
    text = target.read_text(encoding="utf-8")
    assert "Time-to-Target Report" in text
    assert "saved `180.0` seconds" in text
    assert "60.00%" in text
