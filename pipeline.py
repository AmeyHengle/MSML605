# pipeline.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Optional

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from ml605_pipeline.drift import compute_psi as shared_compute_psi
from ml605_pipeline.features import add_time_features, ensure_feature_columns

import pickle, os

ENERGY_FEATURES = [
    'biomass', 'coal', 'imports', 'gas',
    'nuclear', 'other', 'hydro', 'solar', 'wind'
]

# ── Severity helpers ──────────────────────────────────────────────────────────
def ks_severity(ks: float) -> str:
    if   ks < 0.05: return 'none'
    elif ks < 0.10: return 'low'
    elif ks < 0.15: return 'high'
    else:           return 'critical'

def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    # Delegate PSI computation to shared drift kernel for consistency across
    # interactive API simulation and batch pipeline execution.
    return shared_compute_psi(ref, cur, bins=n_bins)

def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def kde_curve(samples: np.ndarray, n_points: int = 200):
    samples = np.asarray(samples, dtype=float)
    # Daily slices can be tiny (or degenerate). gaussian_kde requires >=2 values
    # with non-zero variance, so return a stable placeholder curve when needed.
    if samples.size == 0:
        x = np.linspace(0.0, 1.0, n_points)
        y = np.zeros(n_points)
        return x.tolist(), y.tolist()
    if samples.size < 2 or np.allclose(samples, samples[0]):
        center = float(samples[0])
        lo = max(0.0, center - 1.0)
        hi = center + 1.0
        x = np.linspace(lo, hi, n_points)
        y = np.zeros(n_points)
        y[n_points // 2] = 1.0
        return x.tolist(), y.tolist()

    try:
        kde = stats.gaussian_kde(samples, bw_method='silverman')
        lo  = max(0.0, float(samples.min()) - 1.0)
        hi  = float(samples.max()) + 1.0
        x   = np.linspace(lo, hi, n_points)
        return x.tolist(), kde(x).tolist()
    except Exception:  # noqa: BLE001
        center = float(np.mean(samples))
        lo = max(0.0, center - 1.0)
        hi = center + 1.0
        x = np.linspace(lo, hi, n_points)
        y = np.zeros(n_points)
        y[n_points // 2] = 1.0
        return x.tolist(), y.tolist()

def quantile_sample_idx(x: np.ndarray, n: int) -> np.ndarray:
    n  = min(n, len(x))
    bp = np.quantile(x, np.linspace(0, 1, n + 1))
    idxs = []
    for j in range(n):
        lo, hi = bp[j], bp[j + 1]
        mask   = (x >= lo) & (x <= hi) if j == n - 1 else (x >= lo) & (x < hi)
        idx_m  = np.where(mask)[0]
        if len(idx_m) == 0:
            idx_m = np.array([np.argmin(np.abs(x - (lo + hi) / 2))])
        med = np.median(x[idx_m])
        idxs.append(idx_m[np.argmin(np.abs(x[idx_m] - med))])
    return np.array(idxs)

def train_lr(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    m = LinearRegression()
    m.fit(X, y)
    return m

def save_model(model: LinearRegression, version: int,
               period_str: str, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'model_v{version:02d}_{period_str}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

def drift_pills(exp_map: dict, cur_map: dict, ks_threshold: float) -> dict:
    pills = {}
    for feat in ENERGY_FEATURES:
        ref_s = np.array(exp_map.get(feat, []))
        cur_s = np.array(cur_map.get(feat, []))
        if len(ref_s) < 5 or len(cur_s) < 5:
            pills[feat] = 'none'
        else:
            ks, _ = stats.ks_2samp(ref_s, cur_s)
            pills[feat] = ks_severity(ks)
    return pills

def pca_line_data(pca: PCA, scaler: StandardScaler,
                  model: LinearRegression,
                  pc1_range: list, n_points: int = 120):
    pc1_vals   = np.linspace(pc1_range[0], pc1_range[1], n_points)
    pca_coords = np.zeros((n_points, pca.n_components_))
    pca_coords[:, 0] = pc1_vals
    X_recon = scaler.inverse_transform(pca.inverse_transform(pca_coords))
    y_line  = model.predict(X_recon)
    return pc1_vals.tolist(), y_line.tolist()


class PipelineState:
    def __init__(self, config: dict):
        self.feature_x    = config.get('feature_x',    'gas')
        self.feature_y    = config.get('feature_y',    'forecast_intensity')
        self.ks_threshold = config.get('ks_threshold',  0.10)
        self.n_init       = config.get('n_init',        50)
        self.n_monthly    = config.get('n_monthly',     5)
        self.speed        = config.get('speed',         1.0)
        self.models_dir   = config.get('models_dir',   'models')
        self.data_path    = config.get('data_path',    'data/historical_data.csv')
        self.baseline_years = int(config.get('baseline_years', 4))
        self.retrain_cooldown_days = int(config.get('retrain_cooldown_days', 14))
        # Keep simulations bounded for UI responsiveness and legacy test runtime.
        self.max_sim_days = int(config.get('max_sim_days', 180))

        df = pd.read_csv(self.data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.dropna(subset=['timestamp']).reset_index(drop=True)
        df = add_time_features(df)
        df = ensure_feature_columns(df, ENERGY_FEATURES + [self.feature_y])
        df = df.drop(columns=[c for c in df.columns if c.startswith('factor_')], errors='ignore')
        df = df.sort_values('timestamp').reset_index(drop=True)
        self.df = df

        # Build a long baseline window (first N years) to initialize a stable model.
        start_ts = df['timestamp'].min()
        baseline_cutoff = start_ts + pd.DateOffset(years=self.baseline_years)
        baseline_df = df[df['timestamp'] < baseline_cutoff].copy()
        sim_df = df[df['timestamp'] >= baseline_cutoff].copy()

        # Fallback for short datasets: first 70% as baseline, remaining 30% for simulation.
        if len(baseline_df) < 10 or len(sim_df) < 2:
            split_idx = max(1, int(len(df) * 0.7))
            baseline_df = df.iloc[:split_idx].copy()
            sim_df = df.iloc[split_idx:].copy()

        # Final fallback: if still no simulation slice, reserve last row for simulation.
        if len(sim_df) == 0 and len(df) > 1:
            baseline_df = df.iloc[:-1].copy()
            sim_df = df.iloc[-1:].copy()

        baseline_df['day'] = baseline_df['timestamp'].dt.floor('D')
        sim_df['day'] = sim_df['timestamp'].dt.floor('D')
        self.baseline_df = baseline_df
        self.sim_df = sim_df
        all_days = sorted(sim_df['day'].unique())
        self.days = all_days[:self.max_sim_days]
        if len(self.days) > 0:
            self.sim_df = self.sim_df[self.sim_df['day'].isin(self.days)].copy()

        # Fit scaler + PCA once on entire dataset — never refit during simulation
        full = df[ENERGY_FEATURES + [self.feature_y]].dropna()
        self.scaler  = StandardScaler()
        X_all_sc     = self.scaler.fit_transform(full[ENERGY_FEATURES].values)
        self.pca     = PCA(n_components=len(ENERGY_FEATURES))
        self.pca.fit(X_all_sc)

        # Global fixed axis ranges — computed once, sent to frontend at init
        pc1_all = self.pca.transform(X_all_sc)[:, 0]
        y_all   = full[self.feature_y].values
        p_pc1   = (pc1_all.max() - pc1_all.min()) * 0.05
        p_y     = (y_all.max()   - y_all.min())   * 0.05
        self.pc1_range       = [
            float(pc1_all.min() - p_pc1),
            float(pc1_all.max() + p_pc1)
        ]
        self.intensity_range = [
            float(y_all.min() - p_y),
            float(y_all.max() + p_y)
        ]

        # Simulation accumulators
        self.exp_X   = []
        self.exp_y   = []
        self.exp_gas = []
        self.exp_map = {f: [] for f in ENERGY_FEATURES}

        self.model         = None
        self.model_version = 0
        self.model_log     = []
        self.current_day = 0
        self.last_retrain_day = None

    # Compatibility shim: legacy tests/UI still reference current_month.
    @property
    def current_month(self) -> int:
        # Keep old semantics where this points to "next tick index" + 1 after init.
        return self.current_day + 1

    @staticmethod
    def _fmt_day(day_val) -> str:
        return pd.Timestamp(day_val).strftime('%Y-%m-%d')

    def _day_clean(self, idx: int):
        day_val = self.days[idx]
        rows = self.sim_df[self.sim_df['day'] == day_val]
        clean  = rows[ENERGY_FEATURES + [self.feature_y]].dropna()
        return clean[ENERGY_FEATURES].values, clean[self.feature_y].values, rows

    def _baseline_clean(self):
        rows = self.baseline_df
        clean = rows[ENERGY_FEATURES + [self.feature_y]].dropna()
        return clean[ENERGY_FEATURES].values, clean[self.feature_y].values, rows

    def _project(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X))[:, 0]

    def _eval(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.model.predict(X)
        r2 = float(r2_score(y, y_pred))
        rmse = float(compute_rmse(y, y_pred))
        # Guard API payloads from NaN/Inf values on degenerate slices.
        if not np.isfinite(r2):
            r2 = 0.0
        if not np.isfinite(rmse):
            rmse = 0.0
        return r2, rmse, y_pred

    def _plot_sample(self, X: np.ndarray, y: np.ndarray,
                     y_pred: np.ndarray, n: int) -> dict:
        pc1 = self._project(X)
        idx = quantile_sample_idx(pc1, n)
        return {
            'pca_x':    [float(v) for v in pc1[idx]],
            'pca_y':    [float(v) for v in y[idx]],
            'pred_x':   [float(v) for v in y_pred[idx]],
            'actual_y': [float(v) for v in y[idx]],
        }

    def initialize(self) -> dict:
        X_m, y_m, rows = self._baseline_clean()

        self.model         = train_lr(X_m, y_m)
        self.model_version = 1
        baseline_end = rows['timestamp'].max() if len(rows) else self.df['timestamp'].max()
        baseline_end_s = pd.Timestamp(baseline_end).strftime('%Y-%m-%d')
        path = save_model(self.model, self.model_version,
                          baseline_end_s, self.models_dir)

        r2, rmse, y_pred = self._eval(X_m, y_m)
        line_pc1, line_y = pca_line_data(self.pca, self.scaler,
                                         self.model, self.pc1_range)
        pts = self._plot_sample(X_m, y_m, y_pred, self.n_init)

        gas_vals     = rows[self.feature_x].dropna().values
        kde_x, kde_y = kde_curve(gas_vals)

        self.exp_X.extend(X_m.tolist())
        self.exp_y.extend(y_m.tolist())
        self.exp_gas.extend(gas_vals.tolist())
        for feat in ENERGY_FEATURES:
            self.exp_map[feat].extend(rows[feat].dropna().values.tolist())

        self.model_log.append({'version': 1, 'period': baseline_end_s,
                               'r2': r2, 'rmse': rmse})
        self.current_day = 0

        first_day = self._fmt_day(self.days[0]) if self.days else baseline_end_s
        days_list = [self._fmt_day(d) for d in self.days]

        return {
            # Keep legacy key names for frontend compatibility; values are daily.
            'month':           first_day,
            'total_months':    len(self.days),
            'model_version':   int(self.model_version),
            'r2':              round(r2, 4),
            'rmse':            round(rmse, 2),
            'ks_stat':         None,
            'psi':             None,
            'checkpoint':      path,
            'feature_x':       self.feature_x,
            'feature_y':       self.feature_y,
            'pca_x':           pts['pca_x'],
            'pca_y':           pts['pca_y'],
            'pred_x':          pts['pred_x'],
            'actual_y':        pts['actual_y'],
            'line_pc1':        line_pc1,
            'line_y':          line_y,
            'kde_x':           kde_x,
            'kde_ref_y':       kde_y,
            'pc1_range':       self.pc1_range,
            'intensity_range': self.intensity_range,
            'months_list':     days_list,
        }

    def tick(self) -> Optional[dict]:
        i = self.current_day
        if i >= len(self.days):
            return None

        day_val = self.days[i]
        period = self._fmt_day(day_val)
        X_m, y_m, rows = self._day_clean(i)

        cur_gas = rows[self.feature_x].dropna().values
        if len(self.exp_gas) >= 5 and len(cur_gas) >= 5:
            ks_stat, _ = stats.ks_2samp(self.exp_gas, cur_gas)
            psi_val = compute_psi(np.array(self.exp_gas), cur_gas)
            drifted = bool(ks_stat >= self.ks_threshold)
        else:
            ks_stat = 0.0
            psi_val = 0.0
            drifted = False

        cur_map = {f: rows[f].dropna().values for f in ENERGY_FEATURES}
        pills   = drift_pills(self.exp_map, cur_map, self.ks_threshold)

        kde_rx, kde_ry = kde_curve(np.array(self.exp_gas))
        kde_cx, kde_cy = kde_curve(cur_gas)

        # Accumulate before retrain so new model trains on current month too
        self.exp_X.extend(X_m.tolist())
        self.exp_y.extend(y_m.tolist())
        self.exp_gas.extend(cur_gas.tolist())
        for feat in ENERGY_FEATURES:
            self.exp_map[feat].extend(cur_map[feat].tolist())

        retrained  = False
        checkpoint = None
        new_line   = None

        if drifted:
            can_retrain = True
            if self.last_retrain_day is not None:
                delta_days = (pd.Timestamp(day_val) - pd.Timestamp(self.last_retrain_day)).days
                can_retrain = delta_days >= self.retrain_cooldown_days

            if can_retrain:
                self.model = train_lr(np.array(self.exp_X), np.array(self.exp_y))
                self.model_version += 1
                checkpoint = save_model(self.model, self.model_version,
                                        str(period), self.models_dir)
                lx, ly = pca_line_data(self.pca, self.scaler,
                                       self.model, self.pc1_range)
                new_line = {'line_pc1': lx, 'line_y': ly}
                retrained = True
                self.last_retrain_day = day_val

        r2, rmse, y_pred = self._eval(X_m, y_m)
        pts = self._plot_sample(X_m, y_m, y_pred, self.n_monthly)

        if retrained:
            log = (f"⚠ Drift  KS={ks_stat:.4f} ≥ {self.ks_threshold}"
                   f"  →  Retrained v{self.model_version}"
                   f"  |  R²={r2:.4f}  RMSE={rmse:.2f}")
        elif drifted:
            log = (f"⚠ Drift  KS={ks_stat:.4f} ≥ {self.ks_threshold}"
                   f"  |  Retrain cooldown active ({self.retrain_cooldown_days}d)"
                   f"  |  R²={r2:.4f}  RMSE={rmse:.2f}")
        else:
            log = (f"✓  No drift  KS={ks_stat:.4f}"
                   f"  |  R²={r2:.4f}  RMSE={rmse:.2f}")

        self.current_day += 1

        return {
            'month':          str(period),
            'month_idx':      int(i),
            'total_months':   int(len(self.days)),
            'ks_stat':        round(float(ks_stat), 4),
            'psi':            round(float(psi_val), 4),
            'drift_detected': bool(drifted),
            'retrained':      bool(retrained),
            'model_version':  int(self.model_version),
            'r2':             round(float(r2), 4),
            'rmse':           round(float(rmse), 2),
            'pca_x':          pts['pca_x'],
            'pca_y':          pts['pca_y'],
            'pred_x':         pts['pred_x'],
            'actual_y':       pts['actual_y'],
            'new_line':       new_line,
            'kde_ref_x':      kde_rx,
            'kde_ref_y':      kde_ry,
            'kde_cur_x':      kde_cx,
            'kde_cur_y':      kde_cy,
            'drift_pills':    pills,
            'checkpoint':     checkpoint,
            'log':            log,
            'done':           bool(self.current_day >= len(self.days)),
        }