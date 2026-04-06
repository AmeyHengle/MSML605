# pipeline.py
# Core ML pipeline logic — data loading, drift detection, training, sampling.
# All computation lives here; FastAPI routes call into this module.

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle, os
from typing import Optional


ENERGY_FEATURES = [
    'biomass', 'coal', 'imports', 'gas',
    'nuclear', 'other', 'hydro', 'solar', 'wind'
]

# ── KS severity thresholds ────────────────────────────────────────────────────
def ks_severity(ks: float) -> str:
    if   ks < 0.05: return 'none'
    elif ks < 0.10: return 'low'
    elif ks < 0.15: return 'high'
    else:           return 'critical'

# ── PSI ───────────────────────────────────────────────────────────────────────
def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    bins = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(bins) < 2:
        return 0.0
    ref_c, _ = np.histogram(ref, bins=bins)
    cur_c, _ = np.histogram(cur, bins=bins)
    ref_p = np.where(ref_c == 0, 1e-6, ref_c / len(ref))
    cur_p = np.where(cur_c == 0, 1e-6, cur_c / len(cur))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

# ── KDE for a 1-D array ───────────────────────────────────────────────────────
def kde_curve(samples: np.ndarray, n_points: int = 200):
    """Return (x, y) arrays suitable for a smooth KDE curve."""
    kde  = stats.gaussian_kde(samples, bw_method='silverman')
    lo   = max(0.0, float(samples.min()) - 1.0)
    hi   = float(samples.max()) + 1.0
    x    = np.linspace(lo, hi, n_points)
    return x.tolist(), kde(x).tolist()

# ── Quantile-stratified sample ────────────────────────────────────────────────
def quantile_sample(x: np.ndarray, y: np.ndarray, n: int):
    """
    Pick n points that cover the full spread of x.
    Divides x into n equal-quantile buckets and picks the point closest
    to each bucket's median — avoids clustering in dense regions.
    """
    n  = min(n, len(x))
    bp = np.quantile(x, np.linspace(0, 1, n + 1))
    sx, sy = [], []
    for j in range(n):
        lo, hi = bp[j], bp[j + 1]
        mask   = (x >= lo) & (x <= hi) if j == n - 1 else (x >= lo) & (x < hi)
        bx, by = x[mask], y[mask]
        if len(bx) == 0:
            idx = np.argmin(np.abs(x - (lo + hi) / 2))
            sx.append(float(x[idx])); sy.append(float(y[idx]))
        else:
            idx = np.argmin(np.abs(bx - np.median(bx)))
            sx.append(float(bx[idx])); sy.append(float(by[idx]))
    return sx, sy

# ── Linear regression helpers ─────────────────────────────────────────────────
def train_lr(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    m = LinearRegression()
    m.fit(x.reshape(-1, 1), y)
    return m

def get_r2(model: LinearRegression, x: np.ndarray, y: np.ndarray) -> float:
    return float(r2_score(y, model.predict(x.reshape(-1, 1))))

def save_model(model: LinearRegression, version: int,
               period_str: str, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'model_v{version:02d}_{period_str}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path

# ── Drift pills (per-feature severity for all 9 features) ────────────────────
def drift_pills(exp_x_map: dict, cur_map: dict,
                ks_threshold: float) -> dict:
    """
    For each energy feature compute KS against the expanding reference.
    Returns {feature: severity_string} for the UI pill indicators.
    """
    pills = {}
    for feat in ENERGY_FEATURES:
        ref_s = np.array(exp_x_map.get(feat, []))
        cur_s = np.array(cur_map.get(feat, []))
        if len(ref_s) < 5 or len(cur_s) < 5:
            pills[feat] = 'none'
        else:
            ks, _ = stats.ks_2samp(ref_s, cur_s)
            pills[feat] = ks_severity(ks)
    return pills

# ── Main pipeline state class ─────────────────────────────────────────────────
class PipelineState:
    """
    Holds all mutable state for a single simulation run.
    Instantiated fresh on every /api/initialize call.
    """

    def __init__(self, config: dict):
        self.feature_x    = config.get('feature_x',    'gas')
        self.feature_y    = config.get('feature_y',    'forecast_intensity')
        self.ks_threshold = config.get('ks_threshold',  0.10)
        self.n_init       = config.get('n_init',        50)
        self.n_monthly    = config.get('n_monthly',     5)
        self.speed        = config.get('speed',         1.0)   # seconds between events
        self.models_dir   = config.get('models_dir',   'models')
        self.data_path    = config.get('data_path',    'data/historical_data.csv')

        # Load & prep data
        df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        drop = [c for c in df.columns if c.startswith('factor_')]
        df   = df.drop(columns=drop)
        df['period']  = df['timestamp'].dt.to_period('M')
        self.df       = df
        self.months   = sorted(df['period'].unique())

        # Simulation accumulators
        self.exp_x     = []          # all gas values seen so far
        self.exp_y     = []          # all forecast_intensity values seen so far
        self.exp_map   = {f: [] for f in ENERGY_FEATURES}  # all values per feature

        self.rep_x     = []          # representative scatter pts
        self.rep_y     = []

        self.model          = None
        self.model_version  = 0
        self.model_log      = []     # list of version dicts
        self.current_month  = 0      # index into self.months

    def month_data(self, idx: int):
        """Return full and sampled data for a given month index."""
        period = self.months[idx]
        valid  = self.df[self.df['period'] == period]

        # Drop NaNs jointly so x and y are guaranteed same length
        clean  = valid[[self.feature_x, self.feature_y]].dropna()
        x_full = clean[self.feature_x].values
        y_full = clean[self.feature_y].values
        return x_full, y_full, valid

    def initialize(self) -> dict:
        """
        Train the baseline model on month 0 (Jan 2020).
        Returns the initial state payload sent to the frontend.
        """
        x_m, y_m, valid = self.month_data(0)

        # Representative sample (n_init points)
        sx, sy = quantile_sample(x_m, y_m, self.n_init)
        self.rep_x.extend(sx); self.rep_y.extend(sy)

        # Train baseline model
        self.model = train_lr(x_m, y_m)
        self.model_version = 1
        path = save_model(self.model, self.model_version,
                          str(self.months[0]), self.models_dir)
        r2   = get_r2(self.model, x_m, y_m)

        # Accumulators
        self.exp_x.extend(x_m.tolist())
        self.exp_y.extend(y_m.tolist())
        for feat in ENERGY_FEATURES:
            vals = valid[feat].dropna().values
            self.exp_map[feat].extend(vals.tolist())

        # KDE for primary feature
        kde_x, kde_y = kde_curve(x_m)

        # Regression line endpoints
        x_lo = float(min(x_m)) - 1
        x_hi = float(max(x_m)) + 1
        slope     = float(self.model.coef_[0])
        intercept = float(self.model.intercept_)

        version_entry = {
            'version':   self.model_version,
            'period':    str(self.months[0]),
            'slope':     slope,
            'intercept': intercept,
            'r2':        r2,
            'n_train':   len(x_m),
        }
        self.model_log.append(version_entry)
        self.current_month = 1   # next tick starts at month index 1

        return {
            'month':         str(self.months[0]),
            'total_months':  len(self.months),
            'model_version': self.model_version,
            'r2':            round(r2, 4),
            'slope':         round(slope, 4),
            'intercept':     round(intercept, 4),
            'n_train':       len(x_m),
            'scatter_x':     sx,
            'scatter_y':     sy,
            'kde_x':         kde_x,
            'kde_ref_y':     kde_y,
            'line_x':        [x_lo, x_hi],
            'line_y':        [slope * x_lo + intercept,
                              slope * x_hi + intercept],
            'checkpoint':    path,
            'feature_x':     self.feature_x,
            'feature_y':     self.feature_y,
            'months_list':   [str(m) for m in self.months],
        }

    def tick(self) -> Optional[dict]:
        """
        Advance one month. Returns the SSE event payload or None if done.
        """
        i = self.current_month
        if i >= len(self.months):
            return None

        period  = self.months[i]
        x_m, y_m, valid = self.month_data(i)

        # KS drift on primary feature (expanding reference)
        ks_stat, _ = stats.ks_2samp(self.exp_x, x_m)
        drifted    = ks_stat >= self.ks_threshold

        # Per-feature pills
        cur_map = {f: valid[f].dropna().values for f in ENERGY_FEATURES}
        pills   = drift_pills(self.exp_map, cur_map, self.ks_threshold)

        # KDE curves (reference vs current)
        ref_arr = np.array(self.exp_x)
        kde_rx, kde_ry = kde_curve(ref_arr)
        kde_cx, kde_cy = kde_curve(x_m)

        # Representative sample — always added every month
        sx, sy = quantile_sample(x_m, y_m, self.n_monthly)
        self.rep_x.extend(sx); self.rep_y.extend(sy)

        # Accumulate into expanding history
        self.exp_x.extend(x_m.tolist())
        self.exp_y.extend(y_m.tolist())
        for feat in ENERGY_FEATURES:
            self.exp_map[feat].extend(cur_map[feat].tolist())

        retrained  = False
        checkpoint = None
        new_line   = None
        r2         = get_r2(self.model, x_m, y_m)

        if drifted:
            # Retrain on all accumulated data
            exp_x_arr = np.array(self.exp_x)
            exp_y_arr = np.array(self.exp_y)
            self.model         = train_lr(exp_x_arr, exp_y_arr)
            self.model_version += 1
            checkpoint = save_model(self.model, self.model_version,
                                    str(period), self.models_dir)
            r2    = get_r2(self.model, x_m, y_m)
            slope = float(self.model.coef_[0])
            icept = float(self.model.intercept_)
            x_lo  = float(exp_x_arr.min()) - 1
            x_hi  = float(exp_x_arr.max()) + 1
            new_line = {
                'slope':     round(slope, 4),
                'intercept': round(icept, 4),
                'line_x':    [x_lo, x_hi],
                'line_y':    [slope * x_lo + icept, slope * x_hi + icept],
            }
            self.model_log.append({
                'version':   self.model_version,
                'period':    str(period),
                'slope':     slope,
                'intercept': icept,
                'r2':        r2,
                'n_train':   len(self.exp_x),
            })
            retrained = True

        # Log message
        if retrained:
            log = (f"⚠ Drift detected  KS={ks_stat:.4f} ≥ {self.ks_threshold}"
                   f"  →  Retrained  |  Model v{self.model_version}"
                   f"  |  R²={r2:.4f}")
        else:
            log = f"✓  No drift  |  KS={ks_stat:.4f}  |  R²={r2:.4f}"

        self.current_month += 1

        return {
            'month':            str(period),
            'month_idx':        i,
            'total_months':     len(self.months),
            'ks_stat':          round(ks_stat, 4),
            'drift_detected':   drifted,
            'retrained':        retrained,
            'model_version':    self.model_version,
            'r2':               round(r2, 4),
            'new_scatter_x':    sx,
            'new_scatter_y':    sy,
            'kde_ref_x':        kde_rx,
            'kde_ref_y':        kde_ry,
            'kde_cur_x':        kde_cx,
            'kde_cur_y':        kde_cy,
            'new_line':         new_line,
            'drift_pills':      pills,
            'checkpoint':       checkpoint,
            'log':              log,
            'done':             self.current_month >= len(self.months),
        }
