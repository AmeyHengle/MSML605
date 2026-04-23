# locust/locustfile.py
# Load test for the /api/predict endpoint on App Runner.
#
# Install:   pip install locust
# Run ramp:  locust -f locust/locustfile.py \
#              --host https://YOUR-APP-RUNNER-URL \
#              --headless -u 300 -r 10 --run-time 10m
#
# Run spike: locust -f locust/locustfile.py \
#              --host https://YOUR-APP-RUNNER-URL \
#              --headless -u 500 -r 100 --run-time 5m
#
# Run with web UI (recommended for demo):
#            locust -f locust/locustfile.py \
#              --host https://YOUR-APP-RUNNER-URL
#            → open http://localhost:8089

import random
from locust import HttpUser, task, between, events

# Realistic energy mix samples from the UK grid dataset
SAMPLE_FEATURES = [
    [9.1,  0.0, 11.0, 44.0, 18.2, 0.0, 3.1, 0.0,  14.6],
    [10.8, 0.0, 12.1, 29.8, 17.9, 0.0, 1.6, 20.5,  7.2],
    [9.5,  0.0, 10.6, 22.8, 18.0, 0.0, 1.0, 0.0,  38.1],
    [4.3,  3.7, 16.8, 37.4, 12.0, 0.0, 0.4, 0.0,  25.4],
    [7.2,  0.4, 18.4, 51.2, 18.9, 0.0, 0.4, 0.0,   3.5],
    [8.7,  0.0,  6.5, 45.5, 22.7, 0.0, 1.6, 0.0,  14.9],
    [7.4,  0.0, 13.2, 29.8, 13.3, 0.0, 0.6, 21.8, 14.0],
    [6.3,  5.0,  5.6, 55.1, 16.0, 0.0, 7.9, 0.2,   3.9],
    [6.8,  0.0,  3.4, 19.9, 26.2, 0.0, 1.7, 0.0,  42.1],
    [4.4,  0.0,  3.2,  9.8, 14.5, 0.0, 0.1, 18.9, 49.0],
]


class PredictUser(HttpUser):
    """
    Simulates a user hitting the prediction endpoint.
    wait_time=between(0.05, 0.2) → each user sends 5-20 req/s.
    """
    wait_time = between(0.05, 0.2)

    def on_start(self):
        """Initialize the model before sending predictions."""
        with self.client.post(
            '/api/initialize',
            json={
                'feature_x':    'gas',
                'feature_y':    'forecast_intensity',
                'ks_threshold':  0.10,
                'n_init':        10,
                'n_monthly':     3,
                'speed':         0.0,
            },
            catch_response=True,
            timeout=60,
        ) as r:
            if r.status_code != 200:
                r.failure(f'Initialize failed: {r.status_code}')

    @task(10)
    def predict(self):
        """Main task — hit /api/predict with a random sample."""
        features = random.choice(SAMPLE_FEATURES)
        with self.client.post(
            '/api/predict',
            json={'features': features},
            catch_response=True,
            timeout=10,
            name='/api/predict',
        ) as r:
            if r.status_code == 200:
                data = r.json()
                if 'forecast_intensity' not in data:
                    r.failure('Missing forecast_intensity in response')
            else:
                r.failure(f'Status {r.status_code}')

    @task(1)
    def check_status(self):
        """Occasional status check — background health polling."""
        with self.client.get(
            '/api/status',
            catch_response=True,
            timeout=5,
            name='/api/status',
        ) as r:
            if r.status_code != 200:
                r.failure(f'Status check failed: {r.status_code}')


@events.quitting.add_listener
def on_quit(environment, **kwargs):
    """Print summary when load test ends."""
    stats = environment.runner.stats.total
    print('\n── Load test summary ─────────────────────────────────')
    print(f'  Total requests    : {stats.num_requests:,}')
    print(f'  Failures          : {stats.num_failures:,}')
    print(f'  Failure rate      : {stats.fail_ratio*100:.1f}%')
    print(f'  Median latency    : {stats.median_response_time:.0f}ms')
    print(f'  p95 latency       : {stats.get_response_time_percentile(0.95):.0f}ms')
    print(f'  p99 latency       : {stats.get_response_time_percentile(0.99):.0f}ms')
    print(f'  Peak RPS          : {stats.max_rps:.1f}')
    print('──────────────────────────────────────────────────────')