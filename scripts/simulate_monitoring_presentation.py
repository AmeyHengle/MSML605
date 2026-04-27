"""Generate monitoring/load-testing presentation assets (simulated).

Creates themed Matplotlib visuals that show:
- auto-scaling helps absorb load until saturation
- latency/error behavior across load phases
- KPI comparison: fixed capacity vs auto-scaling
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BG = "#F5E6D3"
TEXT = "#4A2C17"
TEXT_SOFT = "#6D4C35"
OLIVE = "#7B8F5E"
TAN = "#C7A17A"
BRICK = "#B85C4A"
AMBER = "#C4884A"
GRID = "#DCC9B2"


@dataclass
class SimPoint:
    t: int
    phase: str
    users: float
    target_rps: float
    capacity_rps: float
    instances: int
    utilization: float
    p50_ms: float
    p95_ms: float
    error_rate_pct: float
    success_rps: float


def phase_at(t: int) -> str:
    if t < 300:
        return "Light"
    if t < 600:
        return "Medium"
    if t < 900:
        return "Heavy"
    return "Stress"


def users_target(phase: str, t: int) -> float:
    # Smooth ramps inside each phase.
    if phase == "Light":
        return 8 + 6 * (t / 300)
    if phase == "Medium":
        return 18 + 16 * ((t - 300) / 300)
    if phase == "Heavy":
        return 34 + 18 * ((t - 600) / 300)
    return 52 + 30 * ((t - 900) / 300)


def simulate(auto_scaling: bool) -> list[SimPoint]:
    rng = np.random.default_rng(7 if auto_scaling else 11)
    points: list[SimPoint] = []

    capacity_per_instance = 55.0
    instances = 1
    pending_scale_until: int | None = None
    over80_counter = 0
    max_instances = 2
    provision_delay_sec = 90 if auto_scaling else 0

    for t in range(0, 1200, 5):  # 20 min, every 5 seconds
        phase = phase_at(t)
        users = users_target(phase, t)
        target_rps = max(1.0, users * 2.4 + rng.normal(0, 1.2))

        capacity_rps = instances * capacity_per_instance
        utilization = target_rps / capacity_rps

        # Auto-scaling trigger when sustained >80% utilization.
        if auto_scaling and instances < max_instances:
            if utilization > 0.8:
                over80_counter += 1
            else:
                over80_counter = max(0, over80_counter - 1)
            if over80_counter >= 8 and pending_scale_until is None:  # ~40 sec sustained
                pending_scale_until = t + provision_delay_sec

        if auto_scaling and pending_scale_until is not None and t >= pending_scale_until:
            instances += 1
            pending_scale_until = None
            over80_counter = 0
            capacity_rps = instances * capacity_per_instance
            utilization = target_rps / capacity_rps

        # Queueing-inspired latency behavior.
        if utilization <= 0.75:
            queue_mult = 1.0
        elif utilization <= 1.0:
            queue_mult = 1.0 + (utilization - 0.75) * 4.2
        else:
            queue_mult = 2.05 + (utilization - 1.0) * 7.5

        base_p50 = 120 + rng.normal(0, 6)
        p50 = max(55, base_p50 * queue_mult)
        p95 = p50 * (1.55 + max(0.0, utilization - 0.9) * 1.2)

        # Error profile grows quickly after saturation.
        if utilization < 0.85:
            err = max(0.0, rng.normal(0.35, 0.18))
        elif utilization < 1.0:
            err = 0.8 + (utilization - 0.85) * 9 + abs(rng.normal(0, 0.2))
        else:
            err = 2.3 + (utilization - 1.0) * 15 + abs(rng.normal(0, 0.5))
        err = min(35.0, err)

        success_rps = target_rps * (1 - err / 100.0)

        points.append(
            SimPoint(
                t=t,
                phase=phase,
                users=users,
                target_rps=target_rps,
                capacity_rps=capacity_rps,
                instances=instances,
                utilization=utilization * 100,
                p50_ms=p50,
                p95_ms=p95,
                error_rate_pct=err,
                success_rps=success_rps,
            )
        )

    return points


def summarize(points: list[SimPoint]) -> dict:
    arr_p95 = np.array([p.p95_ms for p in points])
    arr_err = np.array([p.error_rate_pct for p in points])
    arr_util = np.array([p.utilization for p in points])
    arr_success = np.array([p.success_rps for p in points])
    arr_target = np.array([p.target_rps for p in points])
    slo_ok = np.logical_and(arr_p95 < 600, arr_err < 2.0)
    return {
        "avg_p95_ms": round(float(arr_p95.mean()), 1),
        "peak_p95_ms": round(float(arr_p95.max()), 1),
        "avg_error_pct": round(float(arr_err.mean()), 2),
        "peak_error_pct": round(float(arr_err.max()), 2),
        "avg_util_pct": round(float(arr_util.mean()), 1),
        "peak_util_pct": round(float(arr_util.max()), 1),
        "throughput_efficiency_pct": round(float((arr_success.sum() / arr_target.sum()) * 100), 1),
        "slo_compliance_pct": round(float(slo_ok.mean() * 100), 1),
        "max_instances": int(max(p.instances for p in points)),
    }


def plot_timeline(out: Path, fixed: list[SimPoint], scaled: list[SimPoint]) -> None:
    t = np.array([p.t / 60 for p in fixed])  # minutes
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.patch.set_facecolor(BG)

    fixed_rps = np.array([p.target_rps for p in fixed])
    scaled_rps = np.array([p.target_rps for p in scaled])
    fixed_cap = np.array([p.capacity_rps for p in fixed])
    scaled_cap = np.array([p.capacity_rps for p in scaled])
    fixed_p95 = np.array([p.p95_ms for p in fixed])
    scaled_p95 = np.array([p.p95_ms for p in scaled])
    fixed_err = np.array([p.error_rate_pct for p in fixed])
    scaled_err = np.array([p.error_rate_pct for p in scaled])

    for ax in axes:
        ax.set_facecolor("#FFF9F2")
        ax.grid(alpha=0.3, color=GRID)
        ax.tick_params(colors=TEXT_SOFT, labelsize=10)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#C7AE93")
        ax.spines["bottom"].set_color("#C7AE93")

    axes[0].plot(t, scaled_rps, color=TAN, linewidth=2.2, label="Incoming RPS")
    axes[0].plot(t, fixed_cap, color=BRICK, linestyle="--", linewidth=2, label="Capacity (No autoscale)")
    axes[0].plot(t, scaled_cap, color=OLIVE, linestyle="--", linewidth=2, label="Capacity (Autoscale)")
    axes[0].set_ylabel("RPS", color=TEXT)
    axes[0].set_title("Load vs Capacity Timeline", fontsize=14, fontweight="bold", color=TEXT)
    axes[0].legend(loc="upper left", fontsize=9)

    axes[1].plot(t, fixed_p95, color=BRICK, linewidth=2.1, label="No autoscale")
    axes[1].plot(t, scaled_p95, color=OLIVE, linewidth=2.1, label="Autoscale")
    axes[1].axhline(600, color=AMBER, linestyle=":", linewidth=1.8, label="p95 SLO (600ms)")
    axes[1].set_ylabel("p95 Latency (ms)", color=TEXT)
    axes[1].set_title("Latency Response", fontsize=14, fontweight="bold", color=TEXT)
    axes[1].legend(loc="upper left", fontsize=9)

    axes[2].plot(t, fixed_err, color=BRICK, linewidth=2.1, label="No autoscale")
    axes[2].plot(t, scaled_err, color=OLIVE, linewidth=2.1, label="Autoscale")
    axes[2].axhline(2.0, color=AMBER, linestyle=":", linewidth=1.8, label="Error SLO (2%)")
    axes[2].set_ylabel("Error Rate (%)", color=TEXT)
    axes[2].set_xlabel("Time (minutes)", color=TEXT)
    axes[2].set_title("Reliability Under Load", fontsize=14, fontweight="bold", color=TEXT)
    axes[2].legend(loc="upper left", fontsize=9)

    for xline, label in [(5, "Light"), (10, "Medium"), (15, "Heavy"), (20, "Stress")]:
        for ax in axes:
            ax.axvline(xline, color="#D8C5AE", linestyle=":", linewidth=1)
        axes[0].text(xline - 0.8, axes[0].get_ylim()[1] * 0.92, label, color=TEXT_SOFT, fontsize=9)

    plt.suptitle("Auto-Scaling Demonstration: Holds Performance Until Saturation", fontsize=20, fontweight="bold", color=TEXT)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_kpi_bars(out: Path, fixed_metrics: dict, scaled_metrics: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.patch.set_facecolor(BG)
    items = [
        ("SLO Compliance (%)", fixed_metrics["slo_compliance_pct"], scaled_metrics["slo_compliance_pct"], True),
        ("Throughput Efficiency (%)", fixed_metrics["throughput_efficiency_pct"], scaled_metrics["throughput_efficiency_pct"], True),
        ("Avg p95 Latency (ms)", fixed_metrics["avg_p95_ms"], scaled_metrics["avg_p95_ms"], False),
        ("Avg Error Rate (%)", fixed_metrics["avg_error_pct"], scaled_metrics["avg_error_pct"], False),
    ]

    for ax, (title, a, b, higher_better) in zip(axes.flat, items):
        ax.set_facecolor("#FFF9F2")
        ax.grid(axis="y", alpha=0.3, color=GRID)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color("#C7AE93")
        ax.spines["bottom"].set_color("#C7AE93")
        vals = [a, b]
        ax.bar(["No autoscale", "Autoscale"], vals, color=[TAN, OLIVE], width=0.58)
        ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT)
        ax.tick_params(labelsize=10, colors=TEXT_SOFT)
        for i, v in enumerate(vals):
            ax.text(i, v * 1.03 if v > 0 else 0.05, f"{v}", ha="center", va="bottom", fontsize=11, color=TEXT, fontweight="bold")
        delta = ((b - a) / a * 100) if a != 0 else 0.0
        sign = "+" if delta >= 0 else ""
        good = (delta > 0 and higher_better) or (delta < 0 and not higher_better)
        ax.text(
            0.98,
            0.9,
            f"{sign}{delta:.1f}%",
            transform=ax.transAxes,
            ha="right",
            color=OLIVE if good else BRICK,
            fontsize=11,
            fontweight="bold",
        )

    plt.suptitle("Monitoring KPI Comparison: Fixed Capacity vs Auto-Scaling", fontsize=20, fontweight="bold", color=TEXT)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_utilization_curve(out: Path, fixed: list[SimPoint], scaled: list[SimPoint]) -> None:
    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#FFF9F2")
    ax.grid(alpha=0.3, color=GRID)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#C7AE93")
    ax.spines["bottom"].set_color("#C7AE93")

    fixed_util = np.array([p.utilization for p in fixed])
    scaled_util = np.array([p.utilization for p in scaled])
    fixed_p95 = np.array([p.p95_ms for p in fixed])
    scaled_p95 = np.array([p.p95_ms for p in scaled])

    ax.scatter(fixed_util, fixed_p95, s=26, alpha=0.45, color=BRICK, label="No autoscale")
    ax.scatter(scaled_util, scaled_p95, s=26, alpha=0.45, color=OLIVE, label="Autoscale")
    ax.axvline(80, color=AMBER, linestyle="--", linewidth=2, label="Scale trigger (~80% util)")
    ax.axvline(100, color=BRICK, linestyle=":", linewidth=2, label="Saturation (~100% util)")
    ax.axhline(600, color="#8E7A61", linestyle=":", linewidth=1.8, label="p95 SLO")

    ax.set_xlabel("Utilization (%)", fontsize=12, color=TEXT)
    ax.set_ylabel("p95 Latency (ms)", fontsize=12, color=TEXT)
    ax.tick_params(colors=TEXT_SOFT, labelsize=10)
    ax.set_title("Latency vs Utilization: Where Auto-Scaling Helps", fontsize=18, fontweight="bold", color=TEXT)
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_table(out: Path, fixed_metrics: dict, scaled_metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    rows = [
        ("Average p95 latency (ms)", fixed_metrics["avg_p95_ms"], scaled_metrics["avg_p95_ms"], "Lower"),
        ("Peak p95 latency (ms)", fixed_metrics["peak_p95_ms"], scaled_metrics["peak_p95_ms"], "Lower"),
        ("Average error rate (%)", fixed_metrics["avg_error_pct"], scaled_metrics["avg_error_pct"], "Lower"),
        ("SLO compliance (%)", fixed_metrics["slo_compliance_pct"], scaled_metrics["slo_compliance_pct"], "Higher"),
        ("Throughput efficiency (%)", fixed_metrics["throughput_efficiency_pct"], scaled_metrics["throughput_efficiency_pct"], "Higher"),
        ("Peak utilization (%)", fixed_metrics["peak_util_pct"], scaled_metrics["peak_util_pct"], "Lower"),
        ("Max instances used", fixed_metrics["max_instances"], scaled_metrics["max_instances"], "Adaptive"),
    ]

    table_data = []
    for metric, base, after, target in rows:
        if target == "Higher" and base:
            imp = f"+{((after - base) / base) * 100:.1f}%"
        elif target == "Lower" and base:
            imp = f"-{((base - after) / base) * 100:.1f}%"
        else:
            imp = "n/a"
        table_data.append([metric, base, after, imp, target])

    tbl = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Fixed Capacity", "Auto-Scaling", "Change", "Target"],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.5)
    tbl.scale(1.12, 2.25)

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

    ax.set_title("Monitoring Metrics: Fixed Capacity vs Auto-Scaling", fontsize=20, color=TEXT, fontweight="bold", pad=18)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fixed = simulate(auto_scaling=False)
    scaled = simulate(auto_scaling=True)

    fixed_metrics = summarize(fixed)
    scaled_metrics = summarize(scaled)

    out = Path("presentation_assets")
    out.mkdir(parents=True, exist_ok=True)

    make_table(out / "monitoring_baseline_vs_autoscale_table.png", fixed_metrics, scaled_metrics)
    plot_timeline(out / "monitoring_autoscale_timeline.png", fixed, scaled)
    plot_kpi_bars(out / "monitoring_kpi_comparison_bars.png", fixed_metrics, scaled_metrics)
    plot_utilization_curve(out / "monitoring_latency_vs_utilization.png", fixed, scaled)

    payload = {
        "fixed_capacity": fixed_metrics,
        "auto_scaling": scaled_metrics,
    }
    with (out / "monitoring_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Generated monitoring presentation assets:")
    for p in sorted(out.glob("monitoring_*")):
        print(f"- {p}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

