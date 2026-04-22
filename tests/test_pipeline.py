# tests/test_pipeline.py
# Unit tests for pipeline.py — no server needed, pure Python logic.
# Run locally: pytest tests/test_pipeline.py -v

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from pipeline import (
    compute_psi,
    ks_severity,
    kde_curve,
    quantile_sample_idx,
    pca_line_data,
    drift_pills,
    PipelineState,
    ENERGY_FEATURES,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'historical_data.csv')

@pytest.fixture(scope='module')
def state():
    """Initialised PipelineState — expensive to create, reused across tests."""
    cfg = {
        'feature_x':    'gas',
        'feature_y':    'forecast_intensity',
        'ks_threshold':  0.10,
        'n_init':        20,
        'n_monthly':     5,
        'speed':         0.0,
        'models_dir':    '/tmp/test_models',
        'data_path':     DATA_PATH,
    }
    s = PipelineState(cfg)
    s.initialize()
    return s


# ── PSI tests ─────────────────────────────────────────────────────────────────
class TestComputePSI:
    def test_identical_distributions_near_zero(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(5, 1, 1000)
        assert compute_psi(ref, ref) < 0.01

    def test_very_different_distributions_large(self):
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        cur = rng.normal(10, 1, 1000)
        assert compute_psi(ref, cur) > 0.5

    def test_returns_float(self):
        rng = np.random.default_rng(0)
        ref = rng.normal(0, 1, 200)
        cur = rng.normal(0.5, 1, 200)
        result = compute_psi(ref, cur)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_non_negative(self):
        rng = np.random.default_rng(7)
        ref = rng.uniform(0, 20, 500)
        cur = rng.uniform(2, 18, 500)
        assert compute_psi(ref, cur) >= 0.0

    def test_single_bin_returns_zero(self):
        # All values identical → bins collapse → should not raise
        ref = np.ones(100)
        cur = np.ones(100)
        result = compute_psi(ref, cur)
        assert result == 0.0


# ── KS severity tests ─────────────────────────────────────────────────────────
class TestKsSeverity:
    @pytest.mark.parametrize('ks,expected', [
        (0.00, 'none'),
        (0.04, 'none'),
        (0.049, 'none'),
        (0.05, 'low'),
        (0.08, 'low'),
        (0.099, 'low'),
        (0.10, 'high'),
        (0.14, 'high'),
        (0.149, 'high'),
        (0.15, 'critical'),
        (0.99, 'critical'),
    ])
    def test_thresholds(self, ks, expected):
        assert ks_severity(ks) == expected

    def test_returns_string(self):
        assert isinstance(ks_severity(0.07), str)


# ── KDE curve tests ───────────────────────────────────────────────────────────
class TestKdeCurve:
    def test_returns_two_equal_length_lists(self):
        rng = np.random.default_rng(1)
        samples = rng.normal(5, 2, 300)
        x, y = kde_curve(samples)
        assert len(x) == len(y)

    def test_default_200_points(self):
        rng = np.random.default_rng(2)
        samples = rng.normal(5, 2, 300)
        x, y = kde_curve(samples)
        assert len(x) == 200

    def test_y_values_non_negative(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(5, 2, 300)
        _, y = kde_curve(samples)
        assert all(v >= 0 for v in y)

    def test_x_range_covers_data(self):
        samples = np.array([2.0, 5.0, 8.0, 10.0])
        x, _ = kde_curve(samples)
        assert min(x) < samples.min()
        assert max(x) > samples.max()


# ── Quantile sample index tests ───────────────────────────────────────────────
class TestQuantileSampleIdx:
    def test_returns_exactly_n_indices(self):
        rng = np.random.default_rng(10)
        x = rng.uniform(0, 20, 500)
        for n in [5, 10, 20, 50]:
            idx = quantile_sample_idx(x, n)
            assert len(idx) == n, f"Expected {n} indices, got {len(idx)}"

    def test_indices_within_bounds(self):
        rng = np.random.default_rng(11)
        x = rng.uniform(0, 20, 300)
        idx = quantile_sample_idx(x, 30)
        assert all(0 <= i < len(x) for i in idx)

    def test_covers_spread(self):
        x = np.linspace(0, 100, 1000)
        idx = quantile_sample_idx(x, 50)
        sampled = x[idx]
        assert sampled.min() < 10,  "Sample misses low end of distribution"
        assert sampled.max() > 90,  "Sample misses high end of distribution"

    def test_n_larger_than_array_clips_to_len(self):
        x = np.array([1.0, 2.0, 3.0])
        idx = quantile_sample_idx(x, 100)
        assert len(idx) == len(x)

    def test_no_duplicate_indices_for_spread_data(self):
        x = np.linspace(0, 100, 1000)
        idx = quantile_sample_idx(x, 50)
        assert len(set(idx)) == len(idx), "Duplicate indices returned"


# ── PCA line data tests ───────────────────────────────────────────────────────
class TestPcaLineData:
    def test_returns_equal_length_lists(self, state):
        lx, ly = pca_line_data(state.pca, state.scaler,
                               state.model, state.pc1_range)
        assert len(lx) == len(ly)

    def test_default_120_points(self, state):
        lx, ly = pca_line_data(state.pca, state.scaler,
                               state.model, state.pc1_range)
        assert len(lx) == 120

    def test_x_spans_pc1_range(self, state):
        lx, _ = pca_line_data(state.pca, state.scaler,
                               state.model, state.pc1_range)
        assert abs(min(lx) - state.pc1_range[0]) < 0.01
        assert abs(max(lx) - state.pc1_range[1]) < 0.01

    def test_y_values_finite(self, state):
        _, ly = pca_line_data(state.pca, state.scaler,
                               state.model, state.pc1_range)
        assert all(np.isfinite(v) for v in ly)

    def test_y_values_plausible_range(self, state):
        """Predictions should stay within rough UK grid intensity bounds."""
        _, ly = pca_line_data(state.pca, state.scaler,
                               state.model, state.pc1_range)
        assert min(ly) > -200,  "Prediction unrealistically low"
        assert max(ly) < 800,   "Prediction unrealistically high"


# ── Drift pills tests ─────────────────────────────────────────────────────────
class TestDriftPills:
    def test_returns_all_nine_features(self):
        rng = np.random.default_rng(20)
        exp_map = {f: rng.normal(5, 1, 500).tolist() for f in ENERGY_FEATURES}
        cur_map = {f: rng.normal(5, 1, 100) for f in ENERGY_FEATURES}
        pills = drift_pills(exp_map, cur_map, 0.10)
        assert set(pills.keys()) == set(ENERGY_FEATURES)

    def test_severity_values_are_valid(self):
        rng = np.random.default_rng(21)
        exp_map = {f: rng.normal(5, 1, 500).tolist() for f in ENERGY_FEATURES}
        cur_map = {f: rng.normal(5, 1, 100) for f in ENERGY_FEATURES}
        pills = drift_pills(exp_map, cur_map, 0.10)
        valid = {'none', 'low', 'high', 'critical'}
        for feat, sev in pills.items():
            assert sev in valid, f"{feat} severity '{sev}' not in {valid}"

    def test_sparse_data_returns_none(self):
        exp_map = {f: [1.0, 2.0] for f in ENERGY_FEATURES}  # < 5 points
        cur_map = {f: np.array([1.0, 2.0]) for f in ENERGY_FEATURES}
        pills = drift_pills(exp_map, cur_map, 0.10)
        for feat in ENERGY_FEATURES:
            assert pills[feat] == 'none', \
                f"Expected 'none' for sparse data, got '{pills[feat]}'"

    def test_highly_different_distributions_flagged(self):
        rng = np.random.default_rng(22)
        exp_map = {f: rng.normal(0, 1, 1000).tolist() for f in ENERGY_FEATURES}
        cur_map = {f: rng.normal(10, 1, 200) for f in ENERGY_FEATURES}
        pills = drift_pills(exp_map, cur_map, 0.10)
        for feat in ENERGY_FEATURES:
            assert pills[feat] in {'high', 'critical'}, \
                f"{feat} should be flagged but got '{pills[feat]}'"


# ── PipelineState tests ───────────────────────────────────────────────────────
class TestPipelineState:
    def test_initialize_returns_required_keys(self, state):
        """
        This test would have caught the pc1_range undefined bug.
        Verifies every key the frontend depends on is present.
        """
        cfg = {
            'feature_x': 'gas', 'feature_y': 'forecast_intensity',
            'ks_threshold': 0.10, 'n_init': 10, 'n_monthly': 5,
            'speed': 0.0, 'models_dir': '/tmp/test_models2',
            'data_path': DATA_PATH,
        }
        fresh = PipelineState(cfg)
        result = fresh.initialize()
        required = [
            'month', 'total_months', 'model_version',
            'r2', 'rmse', 'checkpoint',
            'pca_x', 'pca_y', 'pred_x', 'actual_y',
            'line_pc1', 'line_y',
            'kde_x', 'kde_ref_y',
            'pc1_range', 'intensity_range',   # the keys that caused the bug
            'months_list',
        ]
        for key in required:
            assert key in result, f"Missing key in initialize() response: '{key}'"

    def test_pc1_range_is_valid(self, state):
        assert state.pc1_range[0] < state.pc1_range[1]
        assert all(np.isfinite(v) for v in state.pc1_range)

    def test_intensity_range_is_valid(self, state):
        assert state.intensity_range[0] < state.intensity_range[1]
        assert all(np.isfinite(v) for v in state.intensity_range)

    def test_model_version_starts_at_one(self, state):
        assert state.model_version == 1

    def test_current_month_starts_at_one_after_init(self, state):
        assert state.current_month == 1

    def test_tick_advances_month(self, state):
        before = state.current_month
        state.tick()
        assert state.current_month == before + 1

    def test_tick_returns_required_keys(self, state):
        required = [
            'month', 'month_idx', 'total_months',
            'ks_stat', 'psi', 'drift_detected', 'retrained',
            'model_version', 'r2', 'rmse',
            'pca_x', 'pca_y', 'pred_x', 'actual_y',
            'kde_ref_x', 'kde_ref_y', 'kde_cur_x', 'kde_cur_y',
            'drift_pills', 'log', 'done',
        ]
        result = state.tick()
        assert result is not None
        for key in required:
            assert key in result, f"Missing key in tick() response: '{key}'"

    def test_tick_ks_stat_in_valid_range(self, state):
        result = state.tick()
        assert result is not None
        assert 0.0 <= result['ks_stat'] <= 1.0

    def test_tick_r2_is_finite(self, state):
        result = state.tick()
        assert result is not None
        assert np.isfinite(result['r2'])

    def test_tick_returns_none_when_exhausted(self):
        cfg = {
            'feature_x': 'gas', 'feature_y': 'forecast_intensity',
            'ks_threshold': 0.10, 'n_init': 5, 'n_monthly': 3,
            'speed': 0.0, 'models_dir': '/tmp/test_models3',
            'data_path': DATA_PATH,
        }
        s = PipelineState(cfg)
        s.initialize()
        result = None
        for _ in range(200):   # more than enough months
            result = s.tick()
            if result is None:
                break
        assert result is None, "tick() should return None when all months exhausted"

    def test_model_version_increments_only_on_drift(self):
        cfg = {
            'feature_x': 'gas', 'feature_y': 'forecast_intensity',
            'ks_threshold': 0.001,   # very low — forces retrain on first tick
            'n_init': 5, 'n_monthly': 3,
            'speed': 0.0, 'models_dir': '/tmp/test_models4',
            'data_path': DATA_PATH,
        }
        s = PipelineState(cfg)
        s.initialize()
        assert s.model_version == 1
        result = s.tick()
        assert result is not None
        if result['retrained']:
            assert s.model_version == 2
        else:
            assert s.model_version == 1

    def test_model_version_never_decrements(self):
        cfg = {
            'feature_x': 'gas', 'feature_y': 'forecast_intensity',
            'ks_threshold': 0.001,   # forces frequent retrains
            'n_init': 5, 'n_monthly': 3,
            'speed': 0.0, 'models_dir': '/tmp/test_models5',
            'data_path': DATA_PATH,
        }
        s = PipelineState(cfg)
        s.initialize()
        prev_version = s.model_version
        for _ in range(10):
            result = s.tick()
            if result is None:
                break
            assert s.model_version >= prev_version, \
                "Model version decremented — this should never happen"
            prev_version = s.model_version