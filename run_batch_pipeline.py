from __future__ import annotations

import html
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ml605_pipeline.config import load_config_from_env
from ml605_pipeline.data import fetch_window_dataframe
from ml605_pipeline.drift import detect_drift
from ml605_pipeline.drift_service import decide_drift
from ml605_pipeline.evaluate import compute_metrics
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
)


def _fallback_narrative(
    *,
    overall_drift: bool,
    drift_score: float,
    drifted_features: list[str],
    rows_fetched: int,
    retrain_performed: bool,
) -> str:
    drift_text = "detected" if overall_drift else "not detected"
    features_text = ", ".join(drifted_features[:5]) if drifted_features else "none"
    retrain_text = "triggered" if retrain_performed else "not triggered"
    return (
        f"Window processed with {rows_fetched} rows. Drift was {drift_text} "
        f"(max PSI={drift_score:.4f}); top drifted features: {features_text}. "
        f"Retrain was {retrain_text}."
    )


def _generate_groq_narrative(
    *,
    overall_drift: bool,
    drift_score: float,
    drifted_features: list[str],
    rows_fetched: int,
    retrain_performed: bool,
    ks_threshold: float,
    psi_threshold: float,
) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return _fallback_narrative(
            overall_drift=overall_drift,
            drift_score=drift_score,
            drifted_features=drifted_features,
            rows_fetched=rows_fetched,
            retrain_performed=retrain_performed,
        )

    prompt = {
        "overall_drift": overall_drift,
        "drift_score_max_psi": round(drift_score, 6),
        "drifted_features_top10": drifted_features[:10],
        "rows_fetched": rows_fetched,
        "retrain_performed": retrain_performed,
        "thresholds": {
            "ks_threshold": ks_threshold,
            "psi_threshold": psi_threshold,
        },
    }

    body = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 280,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an MLOps analyst. Write a concise but thorough 4-6 sentence summary "
                    "covering drift diagnosis, likely bottlenecks/root causes, risk impact, "
                    "and retrain recommendation."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt),
            },
        ],
    }

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from Groq")
        msg = (choices[0].get("message") or {}).get("content", "").strip()
        if not msg:
            raise RuntimeError("Empty narrative returned from Groq")
        return msg
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
        return _fallback_narrative(
            overall_drift=overall_drift,
            drift_score=drift_score,
            drifted_features=drifted_features,
            rows_fetched=rows_fetched,
            retrain_performed=retrain_performed,
        )


def _histogram_overlay(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = 20,
) -> dict[str, list[float]]:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return {"x": [], "reference": [], "current": []}

    lo = float(min(ref.min(), cur.min()))
    hi = float(max(ref.max(), cur.max()))
    if np.isclose(lo, hi):
        lo -= 1.0
        hi += 1.0

    edges = np.linspace(lo, hi, bins + 1)
    ref_hist, _ = np.histogram(ref, bins=edges, density=True)
    cur_hist, _ = np.histogram(cur, bins=edges, density=True)
    mids = ((edges[:-1] + edges[1:]) / 2.0).tolist()
    return {
        "x": [float(v) for v in mids],
        "reference": [float(v) for v in ref_hist.tolist()],
        "current": [float(v) for v in cur_hist.tolist()],
    }


def _compute_model_comparison(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict | None:
    target_col = "actual_intensity"
    ref = reference_df.dropna(subset=[target_col]).copy()
    cur = current_df.dropna(subset=[target_col]).copy()
    if len(ref) < 30 or len(cur) < 20:
        return None

    eval_size = max(10, int(len(cur) * 0.2))
    if eval_size >= len(cur):
        return None

    cur_train = cur.iloc[:-eval_size].copy()
    eval_df = cur.iloc[-eval_size:].copy()
    if len(cur_train) < 10:
        return None

    X_ref = ref[feature_cols].fillna(0.0)
    y_ref = ref[target_col].astype(float)
    X_new = pd.concat([ref[feature_cols], cur_train[feature_cols]], axis=0).fillna(0.0)
    y_new = pd.concat([ref[target_col], cur_train[target_col]], axis=0).astype(float)
    X_eval = eval_df[feature_cols].fillna(0.0)
    y_eval = eval_df[target_col].astype(float)

    old_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        random_state=42,
        n_jobs=-1,
    )
    new_model = RandomForestRegressor(
        n_estimators=250,
        max_depth=14,
        random_state=43,
        n_jobs=-1,
    )
    old_model.fit(X_ref, y_ref)
    new_model.fit(X_new, y_new)

    old_pred = old_model.predict(X_eval)
    new_pred = new_model.predict(X_eval)
    old_metrics = compute_metrics(y_eval, old_pred).to_dict()
    new_metrics = compute_metrics(y_eval, new_pred).to_dict()

    idx = eval_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S").tolist() if "timestamp" in eval_df.columns else []
    return {
        "eval_points": len(eval_df),
        "old_metrics": old_metrics,
        "new_metrics": new_metrics,
        "delta": {
            "rmse": float(new_metrics["rmse"] - old_metrics["rmse"]),
            "mae": float(new_metrics["mae"] - old_metrics["mae"]),
            "r2": float(new_metrics["r2"] - old_metrics["r2"]),
            "mape": float(new_metrics["mape"] - old_metrics["mape"]),
        },
        "times": idx,
        "y_true": [float(v) for v in y_eval.tolist()],
        "old_pred": [float(v) for v in old_pred.tolist()],
        "new_pred": [float(v) for v in new_pred.tolist()],
    }


def _build_observations(
    *,
    report,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    model_comparison: dict | None,
) -> list[str]:
    observations: list[str] = []
    observations.append(
        f"Drift score (max PSI) is {report.drift_score:.4f}; "
        f"{'drift detected' if report.overall_drift else 'no feature crossed configured threshold'}."
    )
    if report.drifted_features:
        observations.append("Top drifted features: " + ", ".join(report.drifted_features[:5]) + ".")

    missing = []
    for col in feature_cols:
        if col in current_df.columns:
            miss_rate = float(current_df[col].isna().mean())
            missing.append((col, miss_rate))
    missing = sorted(missing, key=lambda t: t[1], reverse=True)[:5]
    high_missing = [f"{c} ({m*100:.1f}%)" for c, m in missing if m > 0.05]
    if high_missing:
        observations.append("Potential data bottlenecks (missingness >5%): " + ", ".join(high_missing) + ".")

    if model_comparison is None:
        observations.append(
            "Model comparison is unavailable due to limited rows; collect a larger window "
            "for robust old-vs-new evaluation."
        )
    else:
        d = model_comparison["delta"]
        rmse_dir = "improved" if d["rmse"] < 0 else "degraded"
        observations.append(
            f"Candidate retrain vs old model on holdout: RMSE {rmse_dir} by {abs(d['rmse']):.4f}, "
            f"R2 delta {d['r2']:+.4f}."
        )
    return observations


def _build_retrain_decision(
    *,
    report,
    model_comparison: dict | None,
    rows_fetched: int,
) -> dict[str, object]:
    drift_strength = min(report.drift_score / 0.3, 2.0) if report.drift_score > 0 else 0.0
    affected_feature_score = min(len(report.drifted_features) / 4.0, 1.5)
    sample_score = min(rows_fetched / 120.0, 1.0)

    perf_score = 0.0
    expected_impact = "insufficient evidence"
    if model_comparison is not None:
        rmse_delta = float(model_comparison["delta"]["rmse"])
        r2_delta = float(model_comparison["delta"]["r2"])
        if rmse_delta < -0.02 or r2_delta > 0.02:
            perf_score = 1.0
            expected_impact = "high improvement likely"
        elif rmse_delta < 0:
            perf_score = 0.6
            expected_impact = "moderate improvement likely"
        elif rmse_delta <= 0.02:
            perf_score = 0.2
            expected_impact = "neutral to mild improvement"
        else:
            perf_score = -0.6
            expected_impact = "regression risk"

    confidence_raw = (0.45 * drift_strength) + (0.20 * affected_feature_score) + (0.20 * sample_score) + (0.15 * perf_score)
    confidence_score = max(0.0, min(confidence_raw, 1.0))
    confidence_pct = int(round(confidence_score * 100))

    should_retrain = bool(report.overall_drift)
    action = "investigate_before_retrain"
    go_no_go = "NO-GO"
    if should_retrain:
        if model_comparison is None:
            action = "retrain_recommended_with_guardrails"
            go_no_go = "GO"
        else:
            rmse_delta = float(model_comparison["delta"]["rmse"])
            if rmse_delta <= 0.02:
                action = "retrain_recommended"
                go_no_go = "GO"
            else:
                action = "investigate_before_retrain"
                go_no_go = "NO-GO"

    rationale = [
        f"Drift score is {report.drift_score:.4f} with {len(report.drifted_features)} drifted feature(s).",
        f"Window rows fetched: {rows_fetched}.",
    ]
    if model_comparison is not None:
        rationale.append(
            "Holdout deltas (new-old): "
            f"RMSE {model_comparison['delta']['rmse']:+.4f}, "
            f"R2 {model_comparison['delta']['r2']:+.4f}, "
            f"MAPE {model_comparison['delta']['mape']:+.4f}."
        )
    else:
        rationale.append("Model comparison unavailable; recommendation uses drift and data quality signals only.")

    risk_if_deferred = (
        "Elevated drift may degrade forecast reliability and downstream planning decisions."
        if report.overall_drift
        else "Low immediate risk; continue monitoring for trend escalation."
    )

    return {
        "action": action,
        "go_no_go": go_no_go,
        "confidence_pct": confidence_pct,
        "expected_impact": expected_impact,
        "risk_if_deferred": risk_if_deferred,
        "rationale": rationale,
    }


def _build_html_report(
    *,
    window_csv: Path,
    report,
    psi_threshold: float,
    ks_threshold: float,
    gas_decision,
    narrative: str,
    model_comparison: dict | None,
    observations: list[str],
    decision: dict[str, object],
    histogram_payload: dict[str, dict[str, list[float]]],
) -> str:
    rows = []
    sorted_results = sorted(report.feature_results, key=lambda r: r.psi, reverse=True)
    for r in sorted_results:
        rows.append(
            f"""
            <tr>
              <td>{html.escape(r.feature)}</td>
              <td>{r.ks_statistic:.4f}</td>
              <td>{r.ks_p_value:.6f}</td>
              <td>{r.psi:.4f}</td>
              <td>{'YES' if r.drift_detected else 'NO'}</td>
            </tr>
            """
        )
    rows_html = "\n".join(rows) if rows else "<tr><td colspan='5'>No eligible numeric features</td></tr>"
    top_results = sorted_results[:10]
    drift_plot_labels = [r.feature for r in top_results]
    drift_plot_psi = [round(float(r.psi), 6) for r in top_results]
    drift_plot_ks = [round(float(r.ks_statistic), 6) for r in top_results]

    gas_html = ""
    if gas_decision is not None:
        gas_html = f"""
        <h3>Single-feature gas check</h3>
        <ul>
          <li>KS stat: <strong>{gas_decision.ks_stat:.4f}</strong></li>
          <li>KS p-value: <strong>{gas_decision.ks_p_value:.6f}</strong></li>
          <li>PSI: <strong>{gas_decision.psi:.4f}</strong></li>
          <li>Drift detected: <strong>{'YES' if gas_decision.drift_detected else 'NO'}</strong></li>
        </ul>
        """

    observations_html = "".join(f"<li>{html.escape(item)}</li>" for item in observations)
    decision_rationale_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in decision["rationale"])

    model_html = "<p><em>Model comparison unavailable for this run.</em></p>"
    model_chart_payload = {"times": [], "y_true": [], "old_pred": [], "new_pred": []}
    if model_comparison is not None:
        om = model_comparison["old_metrics"]
        nm = model_comparison["new_metrics"]
        dd = model_comparison["delta"]
        model_html = f"""
        <table>
          <thead>
            <tr><th>Metric</th><th>Old model</th><th>Candidate retrained model</th><th>Delta (new-old)</th></tr>
          </thead>
          <tbody>
            <tr><td>RMSE</td><td>{om['rmse']:.4f}</td><td>{nm['rmse']:.4f}</td><td>{dd['rmse']:+.4f}</td></tr>
            <tr><td>MAE</td><td>{om['mae']:.4f}</td><td>{nm['mae']:.4f}</td><td>{dd['mae']:+.4f}</td></tr>
            <tr><td>R2</td><td>{om['r2']:.4f}</td><td>{nm['r2']:.4f}</td><td>{dd['r2']:+.4f}</td></tr>
            <tr><td>MAPE</td><td>{om['mape']:.4f}</td><td>{nm['mape']:.4f}</td><td>{dd['mape']:+.4f}</td></tr>
          </tbody>
        </table>
        <p class="meta">Evaluation points: {int(model_comparison['eval_points'])}</p>
        """
        model_chart_payload = {
            "times": model_comparison["times"],
            "y_true": model_comparison["y_true"],
            "old_pred": model_comparison["old_pred"],
            "new_pred": model_comparison["new_pred"],
        }

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Drift Report</title>
  <style>
    body {{ font-family: Inter, Arial, sans-serif; margin: 24px; color: #1f2937; }}
    h1 {{ margin-bottom: 6px; }}
    .meta {{ color: #4b5563; margin-bottom: 16px; }}
    .badge {{ display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; }}
    .yes {{ background: #fee2e2; color: #991b1b; }}
    .no {{ background: #dcfce7; color: #166534; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px; text-align: left; font-size: 14px; }}
    th {{ background: #f9fafb; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .panel {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
    .chart {{ width: 100%; height: 280px; }}
    .decision-go {{ background: #dcfce7; color: #166534; }}
    .decision-no-go {{ background: #fee2e2; color: #991b1b; }}
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
  <h1>Data Drift Report</h1>
  <div class="meta">Window CSV: {html.escape(str(window_csv))}</div>
  <div class="meta">KS threshold: {ks_threshold:.4f} | PSI threshold: {psi_threshold:.4f}</div>
  <div>
    Overall drift:
    <span class="badge {'yes' if report.overall_drift else 'no'}">
      {'YES' if report.overall_drift else 'NO'}
    </span>
  </div>
  <div class="meta">Drift score (max PSI): <strong>{report.drift_score:.4f}</strong></div>
  <div class="meta">Drifted features: <strong>{', '.join(report.drifted_features) if report.drifted_features else 'None'}</strong></div>
  <h3>Retrain Decision Card</h3>
  <div class="panel">
    <div>
      Decision:
      <span class="badge {'decision-go' if decision['go_no_go'] == 'GO' else 'decision-no-go'}">
        {html.escape(str(decision['go_no_go']))}
      </span>
      <span class="meta"><strong>{html.escape(str(decision['action']))}</strong></span>
    </div>
    <div class="meta">Confidence score: <strong>{int(decision['confidence_pct'])}%</strong></div>
    <div class="meta">Expected impact: <strong>{html.escape(str(decision['expected_impact']))}</strong></div>
    <div class="meta">Risk if deferred: {html.escape(str(decision['risk_if_deferred']))}</div>
    <ul>{decision_rationale_html}</ul>
  </div>
  <h3>Narrative Summary</h3>
  <p>{html.escape(narrative)}</p>
  <h3>Key Observations</h3>
  <ul>{observations_html}</ul>
  {gas_html}
  <div class="grid">
    <div class="panel">
      <h3>Per-feature diagnostics</h3>
      <table>
        <thead>
          <tr>
            <th>Feature</th>
            <th>KS Statistic</th>
            <th>KS p-value</th>
            <th>PSI</th>
            <th>Drift?</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    <div class="panel">
      <h3>Top drifted features: PSI and KS</h3>
      <canvas id="chartDrift" class="chart"></canvas>
    </div>
    <div class="panel">
      <h3>Model performance comparison (old vs candidate retrain)</h3>
      {model_html}
      <canvas id="chartModel" class="chart"></canvas>
    </div>
    <div class="panel">
      <h3>Distribution overlays (top drifted features)</h3>
      <div id="distribution-plots"></div>
    </div>
  </div>
  <script>
    const driftLabels = {json.dumps(drift_plot_labels)};
    const driftPsi = {json.dumps(drift_plot_psi)};
    const driftKs = {json.dumps(drift_plot_ks)};
    const modelChart = {json.dumps(model_chart_payload)};
    const distributionCharts = {json.dumps(histogram_payload)};

    const driftCtx = document.getElementById('chartDrift');
    if (driftCtx && driftLabels.length > 0) {{
      new Chart(driftCtx, {{
        type: 'bar',
        data: {{
          labels: driftLabels,
          datasets: [
            {{ label: 'PSI', data: driftPsi, backgroundColor: 'rgba(245,166,35,0.6)' }},
            {{ label: 'KS statistic', data: driftKs, backgroundColor: 'rgba(91,143,255,0.6)' }}
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false }}
      }});
    }}

    const modelCtx = document.getElementById('chartModel');
    if (modelCtx && modelChart.times.length > 0) {{
      new Chart(modelCtx, {{
        type: 'line',
        data: {{
          labels: modelChart.times,
          datasets: [
            {{ label: 'Actual', data: modelChart.y_true, borderColor: '#111827', fill: false }},
            {{ label: 'Old model', data: modelChart.old_pred, borderColor: '#e05252', fill: false }},
            {{ label: 'Candidate retrained', data: modelChart.new_pred, borderColor: '#2eb67d', fill: false }}
          ]
        }},
        options: {{ responsive: true, maintainAspectRatio: false }}
      }});
    }}

    const distRoot = document.getElementById('distribution-plots');
    if (distRoot) {{
      Object.entries(distributionCharts).forEach(([feat, payload], idx) => {{
        const wrap = document.createElement('div');
        wrap.style.marginBottom = '16px';
        const title = document.createElement('div');
        title.textContent = feat;
        title.style.margin = '6px 0';
        title.style.fontWeight = '600';
        const canvas = document.createElement('canvas');
        canvas.id = `dist_${{idx}}`;
        canvas.className = 'chart';
        wrap.appendChild(title);
        wrap.appendChild(canvas);
        distRoot.appendChild(wrap);

        new Chart(canvas, {{
          type: 'line',
          data: {{
            labels: payload.x,
            datasets: [
              {{ label: 'Reference', data: payload.reference, borderColor: '#5b8fff', fill: false }},
              {{ label: 'Current', data: payload.current, borderColor: '#e05252', fill: false }}
            ]
          }},
          options: {{ responsive: true, maintainAspectRatio: false }}
        }});
      }});
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    cfg = load_config_from_env()

    print(f"[batch] window_hours={cfg.window_hours}")
    print(f"[batch] fetching {cfg.window_start_utc.isoformat()} -> {cfg.window_end_utc.isoformat()}")

    result = fetch_window_dataframe(cfg.window_start_utc, cfg.window_end_utc)
    df = result.df

    if df.empty:
        print("[batch] no data fetched")
        return

    df = add_time_features(df)
    df = apply_factor_columns(df, result.factors)

    out_csv = cfg.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[batch] window_csv={out_csv}")

    if not cfg.features_path.exists():
        print(f"[batch] features file not found: {cfg.features_path}")
        print("[batch] cannot compute drift without features list")
        return

    feature_cols = load_feature_list(cfg.features_path)
    df = ensure_feature_columns(df, feature_cols)

    if not cfg.reference_data_path.exists():
        print(f"[batch] reference file not found: {cfg.reference_data_path}")
        print("[batch] cannot perform drift comparison yet")
        return

    ref_df = pd.read_csv(cfg.reference_data_path)
    if "timestamp" in ref_df.columns:
        ref_df["timestamp"] = pd.to_datetime(ref_df["timestamp"], utc=True, errors="coerce")
    ref_df = add_time_features(ref_df)
    ref_df = apply_factor_columns(ref_df, result.factors)
    ref_df = ensure_feature_columns(ref_df, feature_cols)

    numeric_cols = [
        c for c in feature_cols
        if c in ref_df.select_dtypes("number").columns
        and c in df.select_dtypes("number").columns
    ]

    report = detect_drift(
        reference_df=ref_df,
        current_df=df,
        feature_cols=numeric_cols,
        psi_threshold=cfg.psi_threshold,
    )

    gas_decision = None
    # Also compute unified single-feature decision on gas when available
    if "gas" in ref_df.columns and "gas" in df.columns:
        gas_decision = decide_drift(
            ref_df["gas"].dropna().to_numpy(),
            df["gas"].dropna().to_numpy(),
            ks_stat_threshold=cfg.ks_threshold,
            ks_alpha=0.05,
            psi_threshold=cfg.psi_threshold,
            psi_bins=10,
        )
        print(
            "[batch] gas_ks_stat="
            f"{gas_decision.ks_stat:.4f} gas_ks_p={gas_decision.ks_p_value:.6f} gas_psi={gas_decision.psi:.4f}"
        )

    print(f"[batch] rows_fetched={len(df)}")
    print(f"[batch] overall_drift={report.overall_drift}")
    print(f"[batch] drift_score(max_psi)={report.drift_score:.4f}")
    print(f"[batch] drifted_features={report.drifted_features[:10]}")
    # Current batch flow does not train a new model artifact; treat drift detection
    # as the retrain trigger signal used by CI notifications.
    retrain_performed = bool(report.overall_drift)
    print(f"[batch] retrain_performed={str(retrain_performed).lower()}")
    narrative = _generate_groq_narrative(
        overall_drift=report.overall_drift,
        drift_score=report.drift_score,
        drifted_features=report.drifted_features,
        rows_fetched=len(df),
        retrain_performed=retrain_performed,
        ks_threshold=cfg.ks_threshold,
        psi_threshold=cfg.psi_threshold,
    )
    print(f"[batch] narrative_json={json.dumps(narrative)}")
    model_comparison = _compute_model_comparison(ref_df, df, numeric_cols)
    decision = _build_retrain_decision(
        report=report,
        model_comparison=model_comparison,
        rows_fetched=len(df),
    )
    observations = _build_observations(
        report=report,
        current_df=df,
        feature_cols=numeric_cols,
        model_comparison=model_comparison,
    )
    if model_comparison is not None:
        old_rmse = float(model_comparison["old_metrics"]["rmse"])
        new_rmse = float(model_comparison["new_metrics"]["rmse"])
        rmse_delta = float(model_comparison["delta"]["rmse"])
        print(f"[batch] proxy_old_rmse={old_rmse:.4f}")
        print(f"[batch] proxy_new_rmse={new_rmse:.4f}")
        print(f"[batch] proxy_rmse_delta={rmse_delta:+.4f}")
    recommendation_action = str(decision["action"])
    print(f"[batch] recommendation_action={recommendation_action}")
    print(f"[batch] decision_go_no_go={decision['go_no_go']}")
    print(f"[batch] decision_confidence_pct={decision['confidence_pct']}")
    print(f"[batch] decision_expected_impact={decision['expected_impact']}")
    print(f"[batch] bottlenecks_json={json.dumps(observations[:5])}")

    histogram_payload = {}
    for feat in report.drifted_features[:3]:
        if feat in ref_df.columns and feat in df.columns:
            histogram_payload[feat] = _histogram_overlay(
                ref_df[feat].dropna().to_numpy(dtype=float),
                df[feat].dropna().to_numpy(dtype=float),
            )

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"drift_report_{cfg.window_label}.html"
    report_html = _build_html_report(
        window_csv=out_csv,
        report=report,
        psi_threshold=cfg.psi_threshold,
        ks_threshold=cfg.ks_threshold,
        gas_decision=gas_decision,
        narrative=narrative,
        model_comparison=model_comparison,
        observations=observations,
        decision=decision,
        histogram_payload=histogram_payload,
    )
    report_path.write_text(report_html, encoding="utf-8")
    print(f"[batch] report_html={report_path.resolve()}")


if __name__ == "__main__":
    main()
