"""Create realistic demo DORA metrics visuals for presentation slides.

Outputs are written to presentation_assets/:
- dora_baseline_vs_after_table.png
- dora_before_after_bars.png
- dora_30day_timeseries.png
- dora_distribution_boxplots.png
- dora_summary.json
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from random import Random

import matplotlib.pyplot as plt
import numpy as np

BG = "#F5E6D3"
TEXT = "#4A2C17"
TEXT_SOFT = "#6D4C35"
OLIVE = "#7B8F5E"
TAN = "#C7A17A"
BRICK = "#B85C4A"
MAUVE = "#8D6A9F"
GRID = "#DCC9B2"


@dataclass
class RunRecord:
    ts: datetime
    deploy_success: bool
    smoke_success: bool
    lead_time_min: float | None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    arr = sorted(values)
    idx = int(math.ceil((q / 100.0) * len(arr))) - 1
    idx = max(0, min(idx, len(arr) - 1))
    return arr[idx]


def compute_metrics(runs: list[RunRecord], days_window: int = 30) -> dict:
    deploy_successes = [r for r in runs if r.deploy_success]
    lead_times = [r.lead_time_min for r in runs if r.deploy_success and r.lead_time_min is not None]
    smoke_outcomes = [r.smoke_success for r in runs]

    # MTTR from failure streak start to next success.
    mttr_minutes: list[float] = []
    in_failure = False
    failure_start: datetime | None = None
    for r in runs:
        if not r.smoke_success and not in_failure:
            in_failure = True
            failure_start = r.ts
        elif r.smoke_success and in_failure:
            in_failure = False
            if failure_start is not None:
                mttr = (r.ts - failure_start).total_seconds() / 60.0
                if mttr > 0:
                    mttr_minutes.append(mttr)
            failure_start = None

    deployment_frequency = len(deploy_successes) / float(days_window)
    change_failure_rate = (100.0 * smoke_outcomes.count(False) / len(smoke_outcomes)) if smoke_outcomes else None

    return {
        "total_runs": len(runs),
        "successful_runs": len(deploy_successes),
        "failed_runs": len(runs) - len(deploy_successes),
        "deployment_frequency_per_day": round(deployment_frequency, 3),
        "lead_time_mean_min": round(float(np.mean(lead_times)), 1) if lead_times else None,
        "lead_time_p95_min": round(float(percentile(lead_times, 95)), 1) if lead_times else None,
        "change_failure_rate_pct": round(change_failure_rate, 1) if change_failure_rate is not None else None,
        "mttr_mean_min": round(float(np.mean(mttr_minutes)), 1) if mttr_minutes else None,
    }


def simulate_runs(
    rng: Random,
    start: datetime,
    days: int,
    avg_runs_per_day: float,
    deploy_fail_prob: float,
    smoke_fail_prob: float,
    lead_time_mean: float,
    lead_time_std: float,
    recovery_delay_mean_min: float,
    recovery_delay_std_min: float,
) -> list[RunRecord]:
    records: list[RunRecord] = []
    for day in range(days):
        day_start = start + timedelta(days=day)
        runs_today = max(0, rng.poisson(avg_runs_per_day) if hasattr(rng, "poisson") else int(np.random.poisson(avg_runs_per_day)))
        for i in range(runs_today):
            hour = int((24 / max(1, runs_today)) * i + rng.uniform(0.2, 1.8))
            minute = rng.randint(0, 59)
            ts = day_start + timedelta(hours=min(hour, 23), minutes=minute)
            deploy_success = rng.random() > deploy_fail_prob
            smoke_success = deploy_success and (rng.random() > smoke_fail_prob)
            lead = max(2.5, rng.gauss(lead_time_mean, lead_time_std)) if deploy_success else None
            records.append(RunRecord(ts=ts, deploy_success=deploy_success, smoke_success=smoke_success, lead_time_min=lead))

            # Simulate explicit recovery deployment after any failed run.
            if not smoke_success:
                delay = max(8.0, rng.gauss(recovery_delay_mean_min, recovery_delay_std_min))
                recovery_ts = ts + timedelta(minutes=delay)
                recovery_lead = max(2.5, rng.gauss(lead_time_mean * 0.9, lead_time_std * 0.8))
                records.append(
                    RunRecord(
                        ts=recovery_ts,
                        deploy_success=True,
                        smoke_success=True,
                        lead_time_min=recovery_lead,
                    )
                )
    records.sort(key=lambda r: r.ts)
    return records


def build_monthly_series(runs: list[RunRecord], start: datetime, days: int) -> dict[str, list[float]]:
    window_starts = [start + timedelta(days=d) for d in range(0, days, 5)]
    freq, lead, cfr, mttr = [], [], [], []
    for ws in window_starts:
        we = ws + timedelta(days=30)
        chunk = [r for r in runs if ws <= r.ts < we]
        m = compute_metrics(chunk, days_window=30)
        freq.append(m["deployment_frequency_per_day"] or 0.0)
        lead.append(m["lead_time_mean_min"] or np.nan)
        cfr.append(m["change_failure_rate_pct"] or np.nan)
        mttr.append(m["mttr_mean_min"] or np.nan)
    labels = [dt.strftime("%b %d") for dt in window_starts]
    return {"labels": labels, "freq": freq, "lead": lead, "cfr": cfr, "mttr": mttr}


def dora_rating(metric: str, value: float | None) -> str:
    if value is None:
        return "N/A"
    if metric == "freq":
        if value >= 1:
            return "Elite"
        if value >= 1 / 7:
            return "High"
        if value >= 1 / 30:
            return "Medium"
        return "Low"
    if metric == "lead":
        if value < 60:
            return "Elite"
        if value < 1440:
            return "High"
        if value < 10080:
            return "Medium"
        return "Low"
    if metric == "cfr":
        if value < 5:
            return "Elite"
        if value < 10:
            return "High"
        if value < 15:
            return "Medium"
        return "Low"
    if metric == "mttr":
        if value < 60:
            return "Elite"
        if value < 1440:
            return "High"
        if value < 10080:
            return "Medium"
        return "Low"
    return "N/A"


def make_table_figure(out: Path, baseline: dict, improved: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    rows = [
        ("Deployment frequency (/day)", baseline["deployment_frequency_per_day"], improved["deployment_frequency_per_day"], "Higher"),
        ("Lead time mean (min)", baseline["lead_time_mean_min"], improved["lead_time_mean_min"], "Lower"),
        ("Lead time p95 (min)", baseline["lead_time_p95_min"], improved["lead_time_p95_min"], "Lower"),
        ("Change failure rate (%)", baseline["change_failure_rate_pct"], improved["change_failure_rate_pct"], "Lower"),
        ("MTTR mean (min)", baseline["mttr_mean_min"], improved["mttr_mean_min"], "Lower"),
    ]

    table_data = []
    for name, b, a, direction in rows:
        if b is None or a is None:
            delta = "n/a"
        elif direction == "Higher":
            delta = f"+{((a - b) / b * 100):.1f}%"
        else:
            delta = f"-{((b - a) / b * 100):.1f}%"
        table_data.append([name, b, a, delta, direction])

    columns = ["Metric", "Baseline (Before)", "After Improvements", "Improvement", "Target"]
    tbl = ax.table(
        cellText=table_data,
        colLabels=columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1.15, 2.35)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(OLIVE)
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#F8EFE3" if r % 2 else "#FFF9F2")
            cell.set_edgecolor("#D8C2A9")
            if c == 3:
                cell.set_text_props(weight="bold", color="#2F6B3F")
            elif c == 0:
                cell.set_text_props(weight="bold", color=TEXT)
            else:
                cell.set_text_props(color=TEXT_SOFT)

    ax.set_title("DORA Metrics: Baseline vs After CI/CD Automation", fontsize=20, fontweight="bold", color=TEXT, pad=18)
    plt.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_bar_chart(out: Path, baseline: dict, improved: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor(BG)

    metrics = [
        ("Deployment Frequency (/day)", baseline["deployment_frequency_per_day"], improved["deployment_frequency_per_day"], True),
        ("Lead Time Mean (min)", baseline["lead_time_mean_min"], improved["lead_time_mean_min"], False),
        ("Change Failure Rate (%)", baseline["change_failure_rate_pct"], improved["change_failure_rate_pct"], False),
        ("MTTR Mean (min)", baseline["mttr_mean_min"], improved["mttr_mean_min"], False),
    ]
    colors = (TAN, OLIVE)

    for ax, (title, b, a, higher_better) in zip(axes.flat, metrics):
        ax.set_facecolor("#FFF9F2")
        vals = [b, a]
        ax.bar(["Baseline", "After"], vals, color=colors, width=0.55)
        ax.set_title(title, fontsize=14, color=TEXT, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, color=GRID)
        ax.tick_params(labelsize=11, colors=TEXT_SOFT)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#C7AE93")
        ax.spines["bottom"].set_color("#C7AE93")
        for i, v in enumerate(vals):
            ax.text(i, v * 1.03 if v else 0.05, f"{v}", ha="center", va="bottom", fontsize=12, color=TEXT, fontweight="bold")
        rating = dora_rating("freq" if higher_better else ("lead" if "Lead" in title else "cfr" if "Failure" in title else "mttr"), a)
        ax.text(0.97, 0.9, f"After: {rating}", transform=ax.transAxes, ha="right", color=TEXT_SOFT, fontsize=11, style="italic")

    plt.suptitle("Before vs After — CI/CD Impact", fontsize=21, fontweight="bold", color=TEXT)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_timeseries(out: Path, base_series: dict, after_series: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.patch.set_facecolor(BG)
    x = np.arange(len(base_series["labels"]))

    panels = [
        ("Deployment Frequency (/day)", "freq"),
        ("Lead Time Mean (min)", "lead"),
        ("Change Failure Rate (%)", "cfr"),
        ("MTTR Mean (min)", "mttr"),
    ]
    for ax, (title, key) in zip(axes.flat, panels):
        ax.set_facecolor("#FFF9F2")
        ax.plot(x, base_series[key], marker="o", color=BRICK, linewidth=2.3, label="Baseline")
        ax.plot(x, after_series[key], marker="o", color=OLIVE, linewidth=2.3, label="After")
        ax.set_title(title, fontsize=13, color=TEXT, fontweight="bold")
        ax.grid(alpha=0.3, color=GRID)
        ax.tick_params(axis="x", rotation=35, labelsize=9, colors=TEXT_SOFT)
        ax.tick_params(axis="y", labelsize=10, colors=TEXT_SOFT)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color("#C7AE93")
        ax.spines["bottom"].set_color("#C7AE93")

    axes[0, 0].legend(loc="best")
    tick_idx = np.arange(0, len(base_series["labels"]), max(1, len(base_series["labels"]) // 6))
    for ax in axes.flat:
        ax.set_xticks(tick_idx, [base_series["labels"][i] for i in tick_idx])

    plt.suptitle("Rolling 30-Day DORA Trend (Simulated)", fontsize=21, color=TEXT, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_boxplots(out: Path, baseline_runs: list[RunRecord], improved_runs: list[RunRecord]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    fig.patch.set_facecolor(BG)

    base_lead = [r.lead_time_min for r in baseline_runs if r.lead_time_min is not None]
    after_lead = [r.lead_time_min for r in improved_runs if r.lead_time_min is not None]
    bp1 = axes[0].boxplot([base_lead, after_lead], tick_labels=["Baseline", "After"], patch_artist=True)
    for patch, color in zip(bp1["boxes"], [TAN, OLIVE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[0].set_facecolor("#FFF9F2")
    axes[0].set_title("Lead Time Distribution (min)", color=TEXT, fontweight="bold", fontsize=14)
    axes[0].grid(alpha=0.3, color=GRID)
    axes[0].tick_params(labelsize=11, colors=TEXT_SOFT)

    base_daily_fail = []
    after_daily_fail = []
    by_day_base = {}
    by_day_after = {}
    for r in baseline_runs:
        key = r.ts.date()
        by_day_base.setdefault(key, []).append(0 if r.smoke_success else 1)
    for r in improved_runs:
        key = r.ts.date()
        by_day_after.setdefault(key, []).append(0 if r.smoke_success else 1)
    for vals in by_day_base.values():
        base_daily_fail.append(100.0 * sum(vals) / len(vals))
    for vals in by_day_after.values():
        after_daily_fail.append(100.0 * sum(vals) / len(vals))

    bp2 = axes[1].boxplot([base_daily_fail, after_daily_fail], tick_labels=["Baseline", "After"], patch_artist=True)
    for patch, color in zip(bp2["boxes"], [BRICK, OLIVE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    axes[1].set_facecolor("#FFF9F2")
    axes[1].set_title("Daily Failure Rate Distribution (%)", color=TEXT, fontweight="bold", fontsize=14)
    axes[1].grid(alpha=0.3, color=GRID)
    axes[1].tick_params(labelsize=11, colors=TEXT_SOFT)

    plt.suptitle("Stability Shift After Automation (Simulated)", fontsize=20, color=TEXT, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(42)
    rng = Random(42)
    start = datetime(2026, 1, 1)
    days = 90

    # Baseline (before): infrequent deploys, longer lead time, higher failures, slower recovery.
    baseline_runs = simulate_runs(
        rng=rng,
        start=start,
        days=days,
        avg_runs_per_day=0.42,      # ~13 runs / 30 days
        deploy_fail_prob=0.08,
        smoke_fail_prob=0.18,
        lead_time_mean=174.0,
        lead_time_std=42.0,
        recovery_delay_mean_min=900.0,   # manual recovery often takes half-day+
        recovery_delay_std_min=480.0,
    )

    # After (improved): frequent deploys, faster lead time, lower failure, quick recovery.
    improved_runs = simulate_runs(
        rng=rng,
        start=start,
        days=days,
        avg_runs_per_day=1.35,      # ~40 runs / 30 days
        deploy_fail_prob=0.005,
        smoke_fail_prob=0.05,
        lead_time_mean=31.0,
        lead_time_std=8.5,
        recovery_delay_mean_min=42.0,    # auto rollback recovery within ~1 hour
        recovery_delay_std_min=20.0,
    )

    # Report headline table as last-30-day comparison for readability.
    cutoff = start + timedelta(days=60)
    baseline_last30 = [r for r in baseline_runs if r.ts >= cutoff]
    improved_last30 = [r for r in improved_runs if r.ts >= cutoff]

    baseline = compute_metrics(baseline_last30, days_window=30)
    improved = compute_metrics(improved_last30, days_window=30)
    base_series = build_monthly_series(baseline_runs, start, days)
    after_series = build_monthly_series(improved_runs, start, days)

    out_dir = Path("presentation_assets")
    out_dir.mkdir(parents=True, exist_ok=True)

    make_table_figure(out_dir / "dora_baseline_vs_after_table.png", baseline, improved)
    make_bar_chart(out_dir / "dora_before_after_bars.png", baseline, improved)
    make_timeseries(out_dir / "dora_30day_timeseries.png", base_series, after_series)
    make_boxplots(out_dir / "dora_distribution_boxplots.png", baseline_runs, improved_runs)

    payload = {
        "baseline": baseline,
        "after": improved,
        "series_labels": base_series["labels"],
    }
    with (out_dir / "dora_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Generated presentation assets:")
    for p in sorted(out_dir.glob("dora_*")):
        print(f"- {p}")
    print("\nSummary:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
