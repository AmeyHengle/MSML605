from __future__ import annotations

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
from ml605_pipeline.features import (
    add_time_features,
    apply_factor_columns,
    ensure_feature_columns,
    load_feature_list,
)


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

    print(f"[batch] rows_fetched={len(df)}")
    print(f"[batch] overall_drift={report.overall_drift}")
    print(f"[batch] drift_score(max_psi)={report.drift_score:.4f}")
    print(f"[batch] drifted_features={report.drifted_features[:10]}")


if __name__ == "__main__":
    main()
