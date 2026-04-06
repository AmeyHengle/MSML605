# pipeline.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle, os
from typing import Optional

ENERGY_FEATURES = [
    'biomass', 'coal', 'imports', 'gas',
    'nuclear', 'other', 'hydro', 'solar', 'wind'
]

def ks_severity(ks: float) -> str:
    if   ks < 0.05: return 'none'
    elif ks < 0.10: return 'low'
    elif ks < 0.15: return 'high'
    else:           return 'critical'

def compute_psi(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    bins = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)))
    if len(bins) < 2:
        return 0.0
    ref_c, _ = np.histogram(ref, bins=bins)
    cur_c, _ = np.histogram(cur, bins=bins)
    ref_p = np.where(ref_c == 0, 1e-6, ref_c / len(ref))
    cur_p = np.where(cur_c == 0, 1e-6, cur_c / len(cur))
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

def kde_curve(samples: np.ndarray, n_points: int = 200):
    kde = stats.gaussian_kde(samples, bw_method='silverman')
    lo  = max(0.0, float(samples.min()) - 1.0)
    hi  = float(samples.max()) + 1.0
    x   = np.linspace(lo, hi, n_points)
    return x.tolist(), kde(x).tolist()

def quantile_sample(x: np.ndarray, y: np.ndarray, n: int):
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

def train_lr(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    m = LinearRegression()
    m.fit(X, y)
    return m

def get_metrics(model: LinearRegression, X: np.ndarray, y: np.ndarray):
    y_pred = model.predict(X)
    r2   = float(r2_score(y, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    return r2, rmse

def save_model(model, version: int, period_str: str, models_dir: str) -> str:
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

        df = pd.read_csv(self.data_path, parse_dates=['timestamp'])
        drop = [c for c in df.columns if c.startswith('factor_')]
        df   = df.drop(columns=drop)
        df['period'] = df['timestamp'].dt.to_period('M')
        self.df      = df
        self.months  = sorted(df['period'].unique())

        # ── PCA fitted ONCE on entire dataset ─────────────────────────────────
        # A fixed projection ensures the PC1 axis has consistent meaning
        # across all months — the x-axis never shifts under the user's feet.
        full_clean  = df[ENERGY_FEATURES + [self.feature_y]].dropna()
        full_feats  = full_clean[ENERGY_FEATURES].values
        self.scaler = StandardScaler()
        full_scaled = self.scaler.fit_transform(full_feats)
        self.pca    = PCA(n_components=1)
        self.pca.fit(full_scaled)

        # ── Global axis ranges (fixed for entire simulation) ──────────────────
        pc1_all = self.pca.transform(full_scaled).ravel()
        y_all   = full_clean[self.feature_y].values
        pad_pc1 = (pc1_all.max() - pc1_all.min()) * 0.05
        pad_y   = (y_all.max()   - y_all.min())   * 0.05
        self.axis_ranges = {
            'pc1_min': float(pc1_all.min()) - pad_pc1,
            'pc1_max': float(pc1_all.max()) + pad_pc1,
            'y_min':   float(y_all.min())   - pad_y,
            'y_max':   float(y_all.max())   + pad_y,
        }

        # Accumulators
        self.exp_feats = []      # full 9-feature rows
        self.exp_y     = []      # forecast_intensity
        self.exp_pc1   = []      # projected PC1 (for scatter)
        self.exp_map   = {f: [] for f in ENERGY_FEATURES}

        self.model         = None
        self.model_version = 0
        self.model_log     = []
        self.current_month = 0

    def _project(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(self.scaler.transform(X)).ravel()

    def _regression_line(self):
        """Sweep PC1 grid → inverse_transform → model.predict."""
        pc1_grid  = np.linspace(
            self.axis_ranges['pc1_min'],
            self.axis_ranges['pc1_max'], 300
        ).reshape(-1, 1)
        feat_grid = self.scaler.inverse_transform(
            self.pca.inverse_transform(pc1_grid)
        )
        y_line = self.model.predict(feat_grid)
        return pc1_grid.ravel().tolist(), y_line.tolist()

    def _month_arrays(self, idx: int):
        period = self.months[idx]
        valid  = self.df[self.df['period'] == period]
        clean  = valid[ENERGY_FEATURES + [self.feature_y]].dropna()
        X      = clean[ENERGY_FEATURES].values
        y      = clean[self.feature_y].values
        return X, y, valid

    def initialize(self) -> dict:
        X_m, y_m, valid = self._month_arrays(0)
        pc1_m = self._project(X_m)

        sx, sy = quantile_sample(pc1_m, y_m, self.n_init)

        self.model         = train_lr(X_m, y_m)
        self.model_version = 1
        path = save_model(self.model, self.model_version,
                          str(self.months[0]), self.models_dir)
        r2, rmse = get_metrics(self.model, X_m, y_m)

        line_pc1, line_y = self._regression_line()

        # Predicted vs Actual — subsample for payload size
        n_s    = min(300, len(X_m))
        idx_s  = np.random.choice(len(X_m), n_s, replace=False)
        y_pred = self.model.predict(X_m[idx_s]).tolist()
        y_act  = y_m[idx_s].tolist()

        kde_x, kde_y = kde_curve(valid[self.feature_x].dropna().values)

        self.exp_feats.extend(X_m.tolist())
        self.exp_y.extend(y_m.tolist())
        self.exp_pc1.extend(pc1_m.tolist())
        for feat in ENERGY_FEATURES:
            self.exp_map[feat].extend(valid[feat].dropna().values.tolist())

        self.model_log.append({
            'version': self.model_version,
            'period':  str(self.months[0]),
            'r2': r2, 'rmse': rmse,
        })
        self.current_month = 1

        return {
            'month':          str(self.months[0]),
            'total_months':   len(self.months),
            'model_version':  self.model_version,
            'r2':             round(r2,   4),
            'rmse':           round(rmse, 4),
            'ks_stat':        0.0,
            'psi':            0.0,
            'scatter_pc1':    sx,
            'scatter_y':      sy,
            'line_pc1':       line_pc1,
            'line_y':         line_y,
            'pred_y':         y_pred,
            'actual_y':       y_act,
            'kde_x':          kde_x,
            'kde_ref_y':      kde_y,
            'axis_ranges':    self.axis_ranges,
            'feature_x':      self.feature_x,
            'feature_y':      self.feature_y,
            'months_list':    [str(m) for m in self.months],
        }

    def tick(self) -> Optional[dict]:
        i = self.current_month
        if i >= len(self.months):
            return None

        period         = self.months[i]
        X_m, y_m, valid = self._month_arrays(i)
        pc1_m          = self._project(X_m)

        cur_gas = valid[self.feature_x].dropna().values
        exp_gas = np.array(self.exp_map[self.feature_x])
        ks_stat, _ = stats.ks_2samp(exp_gas, cur_gas)
        drifted    = bool(ks_stat >= self.ks_threshold)
        psi_val    = compute_psi(exp_gas, cur_gas) if len(exp_gas) >= 10 else 0.0

        cur_map = {f: valid[f].dropna().values for f in ENERGY_FEATURES}
        pills   = drift_pills(self.exp_map, cur_map, self.ks_threshold)

        kde_rx, kde_ry = kde_curve(exp_gas)
        kde_cx, kde_cy = kde_curve(cur_gas)

        sx, sy = quantile_sample(pc1_m, y_m, self.n_monthly)

        # Accumulate before possible retrain
        self.exp_feats.extend(X_m.tolist())
        self.exp_y.extend(y_m.tolist())
        self.exp_pc1.extend(pc1_m.tolist())
        for feat in ENERGY_FEATURES:
            self.exp_map[feat].extend(cur_map[feat].tolist())

        retrained  = False
        checkpoint = None
        new_line   = None

        if drifted:
            exp_X = np.array(self.exp_feats)
            exp_y = np.array(self.exp_y)
            self.model         = train_lr(exp_X, exp_y)
            self.model_version += 1
            checkpoint = save_model(self.model, self.model_version,
                                    str(period), self.models_dir)
            lx, ly   = self._regression_line()
            new_line  = {'line_pc1': lx, 'line_y': ly}
            retrained = True
            self.model_log.append({
                'version': self.model_version, 'period': str(period)
            })

        r2, rmse = get_metrics(self.model, X_m, y_m)

        n_s    = min(300, len(X_m))
        idx_s  = np.random.choice(len(X_m), n_s, replace=False)
        y_pred = self.model.predict(X_m[idx_s]).tolist()
        y_act  = y_m[idx_s].tolist()

        if retrained:
            log = (f"⚠ Drift  KS={ks_stat:.4f} ≥ {self.ks_threshold}"
                   f"  →  Retrained v{self.model_version}"
                   f"  |  R²={r2:.3f}  RMSE={rmse:.2f}")
        else:
            log = (f"✓ No drift  KS={ks_stat:.4f}"
                   f"  |  R²={r2:.3f}  RMSE={rmse:.2f}")

        self.current_month += 1

        return {
            'month':           str(period),
            'month_idx':       i,
            'total_months':    len(self.months),
            'ks_stat':         round(ks_stat, 4),
            'psi':             round(psi_val, 4),
            'drift_detected':  drifted,
            'retrained':       retrained,
            'model_version':   self.model_version,
            'r2':              round(r2,   4),
            'rmse':            round(rmse, 4),
            'new_scatter_pc1': sx,
            'new_scatter_y':   sy,
            'new_line':        new_line,
            'pred_y':          y_pred,
            'actual_y':        y_act,
            'kde_ref_x':       kde_rx,
            'kde_ref_y':       kde_ry,
            'kde_cur_x':       kde_cx,
            'kde_cur_y':       kde_cy,
            'drift_pills':     pills,
            'checkpoint':      checkpoint,
            'log':             log,
            'done':            self.current_month >= len(self.months),
        }