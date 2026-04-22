# tests/test_api.py
# Smoke tests for the FastAPI endpoints.
# Run against a live server — set BASE_URL to your Render or local URL.
#
# Local:   BASE_URL=http://localhost:8000 pytest tests/test_api.py -v
# Render:  BASE_URL=https://msml605.onrender.com pytest tests/test_api.py -v
#
# These run as a post-deploy gate in CI — if they fail, the deploy is marked
# as a failure and contributes to change failure rate tracking.

import os
import json
import pytest
import requests

BASE_URL = os.getenv('BASE_URL', 'http://localhost:8000')

VALID_CONFIG = {
    'feature_x':    'gas',
    'feature_y':    'forecast_intensity',
    'ks_threshold':  0.10,
    'n_init':        10,
    'n_monthly':     3,
    'speed':         0.0,
    'models_dir':    'models',
    'data_path':     'data/historical_data.csv',
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def initialize(config=None):
    return requests.post(
        f'{BASE_URL}/api/initialize',
        json=config or VALID_CONFIG,
        timeout=60       # initialization can take 10–30s on cold start
    )

def reset():
    return requests.post(f'{BASE_URL}/api/reset', timeout=10)

def status():
    return requests.get(f'{BASE_URL}/api/status', timeout=10)


# ── Fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def clean_state():
    """Reset server state before each test so tests are independent."""
    reset()
    yield
    reset()


# ── /api/status tests ─────────────────────────────────────────────────────────
class TestStatus:
    def test_returns_200(self):
        r = status()
        assert r.status_code == 200

    def test_initialized_false_before_init(self):
        r = status().json()
        assert r['initialized'] is False

    def test_running_false_before_init(self):
        r = status().json()
        assert r['running'] is False

    def test_month_idx_none_before_init(self):
        r = status().json()
        assert r['month_idx'] is None

    def test_initialized_true_after_init(self):
        initialize()
        r = status().json()
        assert r['initialized'] is True

    def test_total_months_positive_after_init(self):
        initialize()
        r = status().json()
        assert isinstance(r['total'], int)
        assert r['total'] > 0


# ── /api/initialize tests ─────────────────────────────────────────────────────
class TestInitialize:
    def test_returns_200(self):
        r = initialize()
        assert r.status_code == 200

    def test_status_field_is_ok(self):
        r = initialize().json()
        assert r['status'] == 'ok'

    def test_data_field_present(self):
        r = initialize().json()
        assert 'data' in r

    # Critical field checks — these would catch the pc1_range bug
    @pytest.mark.parametrize('key', [
        'month', 'total_months', 'model_version',
        'r2', 'rmse',
        'pca_x', 'pca_y', 'pred_x', 'actual_y',
        'line_pc1', 'line_y',
        'kde_x', 'kde_ref_y',
        'pc1_range', 'intensity_range',
        'months_list',
    ])
    def test_required_key_present(self, key):
        d = initialize().json()['data']
        assert key in d, f"Missing required key: '{key}'"

    def test_pc1_range_is_ordered(self):
        d = initialize().json()['data']
        assert d['pc1_range'][0] < d['pc1_range'][1]

    def test_intensity_range_is_ordered(self):
        d = initialize().json()['data']
        assert d['intensity_range'][0] < d['intensity_range'][1]

    def test_model_version_is_one(self):
        d = initialize().json()['data']
        assert d['model_version'] == 1

    def test_r2_is_finite_float(self):
        d = initialize().json()['data']
        assert isinstance(d['r2'], (int, float))
        assert -10 < d['r2'] <= 1.0   # finite and plausible

    def test_rmse_is_positive(self):
        d = initialize().json()['data']
        assert d['rmse'] > 0

    def test_pca_x_and_pca_y_same_length(self):
        d = initialize().json()['data']
        assert len(d['pca_x']) == len(d['pca_y'])

    def test_line_pc1_and_line_y_same_length(self):
        d = initialize().json()['data']
        assert len(d['line_pc1']) == len(d['line_y'])

    def test_total_months_is_integer(self):
        d = initialize().json()['data']
        assert isinstance(d['total_months'], int)
        assert d['total_months'] > 0

    def test_months_list_length_matches_total(self):
        d = initialize().json()['data']
        assert len(d['months_list']) == d['total_months']

    def test_double_initialize_does_not_crash(self):
        r1 = initialize()
        r2 = initialize()
        assert r1.status_code == 200
        assert r2.status_code == 200

    def test_invalid_feature_x_returns_error(self):
        bad_config = {**VALID_CONFIG, 'feature_x': 'not_a_real_feature'}
        r = initialize(bad_config)
        # Should return 4xx or 5xx, not silently succeed
        assert r.status_code >= 400 or 'error' in r.json()


# ── /api/reset tests ──────────────────────────────────────────────────────────
class TestReset:
    def test_returns_200(self):
        r = reset()
        assert r.status_code == 200

    def test_status_field_is_reset(self):
        r = reset().json()
        assert r['status'] == 'reset'

    def test_clears_initialized_state(self):
        initialize()
        assert status().json()['initialized'] is True
        reset()
        assert status().json()['initialized'] is False

    def test_reset_without_init_does_not_crash(self):
        reset()
        r = reset()   # double reset
        assert r.status_code == 200


# ── /api/pause and /api/resume tests ──────────────────────────────────────────
class TestPauseResume:
    def test_pause_returns_200(self):
        initialize()
        r = requests.post(f'{BASE_URL}/api/pause', timeout=10)
        assert r.status_code == 200

    def test_pause_status_field(self):
        initialize()
        r = requests.post(f'{BASE_URL}/api/pause', timeout=10).json()
        assert r['status'] == 'paused'

    def test_resume_returns_200(self):
        initialize()
        requests.post(f'{BASE_URL}/api/pause', timeout=10)
        r = requests.post(f'{BASE_URL}/api/resume', timeout=10)
        assert r.status_code == 200

    def test_resume_status_field(self):
        initialize()
        requests.post(f'{BASE_URL}/api/pause', timeout=10)
        r = requests.post(f'{BASE_URL}/api/resume', timeout=10).json()
        assert r['status'] == 'resumed'


# ── /api/simulate SSE tests ───────────────────────────────────────────────────
class TestSimulate:
    def _get_first_n_events(self, n=3):
        """Open SSE stream and collect first n events, then close."""
        initialize()
        events = []
        with requests.get(
            f'{BASE_URL}/api/simulate',
            stream=True, timeout=120
        ) as r:
            assert r.status_code == 200
            for line in r.iter_lines():
                if line and line.startswith(b'data:'):
                    payload = json.loads(line[5:].strip())
                    if not payload.get('paused'):
                        events.append(payload)
                    if len(events) >= n:
                        break
        return events

    def test_first_event_has_required_keys(self):
        events = self._get_first_n_events(1)
        required = [
            'month', 'month_idx', 'ks_stat', 'psi',
            'drift_detected', 'retrained', 'model_version',
            'r2', 'rmse', 'pca_x', 'pca_y',
            'pred_x', 'actual_y', 'drift_pills', 'log', 'done',
        ]
        for key in required:
            assert key in events[0], f"SSE event missing key: '{key}'"

    def test_month_advances_sequentially(self):
        events = self._get_first_n_events(3)
        months = [e['month_idx'] for e in events]
        assert months == sorted(months), "Month indices not advancing sequentially"
        assert months[0] < months[-1], "Month index not advancing at all"

    def test_ks_stat_in_valid_range(self):
        events = self._get_first_n_events(3)
        for e in events:
            assert 0.0 <= e['ks_stat'] <= 1.0, \
                f"KS stat {e['ks_stat']} out of [0, 1]"

    def test_psi_non_negative(self):
        events = self._get_first_n_events(3)
        for e in events:
            assert e['psi'] >= 0.0, f"PSI {e['psi']} is negative"

    def test_r2_is_finite(self):
        events = self._get_first_n_events(3)
        for e in events:
            import math
            assert math.isfinite(e['r2']), f"R² {e['r2']} is not finite"

    def test_model_version_at_least_one(self):
        events = self._get_first_n_events(3)
        for e in events:
            assert e['model_version'] >= 1

    def test_drift_pills_has_all_features(self):
        from pipeline import ENERGY_FEATURES
        events = self._get_first_n_events(1)
        pills = events[0]['drift_pills']
        assert set(pills.keys()) == set(ENERGY_FEATURES)

    def test_drift_pills_values_are_valid_severities(self):
        events = self._get_first_n_events(1)
        valid = {'none', 'low', 'high', 'critical'}
        for feat, sev in events[0]['drift_pills'].items():
            assert sev in valid, f"'{feat}' has invalid severity '{sev}'"

    def test_simulate_without_init_returns_error(self):
        # No initialize() call here — state was reset by autouse fixture
        with requests.get(
            f'{BASE_URL}/api/simulate',
            stream=True, timeout=30
        ) as r:
            for line in r.iter_lines():
                if line and line.startswith(b'data:'):
                    payload = json.loads(line[5:].strip())
                    assert 'error' in payload, \
                        "Expected error payload when simulating without init"
                    break