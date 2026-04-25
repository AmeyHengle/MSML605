from __future__ import annotations

import html
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from ml605_pipeline.config import load_config_from_env
from ml605_pipeline.data import fetch_window_dataframe
from ml605_pipeline.drift import detect_drift
from ml605_pipeline.drift_service import decide_drift
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
)


def _build_html_report(
    *,
    window_csv: Path,
    report,
    psi_threshold: float,
    ks_threshold: float,
    gas_decision,
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
  </style>
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
  {gas_html}
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

    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = cfg.reports_dir / f"drift_report_{cfg.window_label}.html"
    report_html = _build_html_report(
        window_csv=out_csv,
        report=report,
        psi_threshold=cfg.psi_threshold,
        ks_threshold=cfg.ks_threshold,
        gas_decision=gas_decision,
    )
    report_path.write_text(report_html, encoding="utf-8")
    print(f"[batch] report_html={report_path.resolve()}")


if __name__ == "__main__":
    main()
