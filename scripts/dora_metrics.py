# scripts/dora_metrics.py
# Computes DORA metrics from GitHub Actions workflow run history.
# Called automatically by the Deploy workflow after every push.
# Can also be run locally:
#   GITHUB_TOKEN=... GITHUB_REPO=AmeyHengle/MSML605 python scripts/dora_metrics.py

import os
import json
import requests
from datetime import datetime, timezone, timedelta
from dateutil import parser as dateparser

GITHUB_TOKEN  = os.environ['GITHUB_TOKEN']
GITHUB_REPO   = os.getenv('GITHUB_REPO',   'AmeyHengle/MSML605')
WORKFLOW_NAME = os.getenv('WORKFLOW_NAME', 'Deploy')
DEPLOY_JOB    = os.getenv('DEPLOY_JOB',   'deploy-backend')
SMOKE_JOB     = os.getenv('SMOKE_JOB',    'smoke-tests')
DAYS_WINDOW   = int(os.getenv('DAYS_WINDOW', '30'))

GH_HEADERS = {
    'Authorization': f'Bearer {GITHUB_TOKEN}',
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
}
BASE = f'https://api.github.com/repos/{GITHUB_REPO}'


def gh_get(url, params=None):
    r = requests.get(url, headers=GH_HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def get_workflow_runs(since: datetime) -> list:
    """Fetch all workflow runs for WORKFLOW_NAME within the time window."""
    runs = []
    page = 1
    while True:
        data = gh_get(f'{BASE}/actions/runs', params={
            'per_page': 100,
            'page': page,
        })
        for run in data.get('workflow_runs', []):
            if run['name'] != WORKFLOW_NAME:
                continue
            created = dateparser.parse(run['created_at'])
            if created < since:
                return runs   # runs are ordered newest-first, stop here
            runs.append(run)
        if len(data.get('workflow_runs', [])) < 100:
            break
        page += 1
    return runs


def get_job_status(run_id: int, job_name: str) -> dict:
    """Return the job record for a named job within a workflow run."""
    data = gh_get(f'{BASE}/actions/runs/{run_id}/jobs')
    for job in data.get('jobs', []):
        if job['name'] == job_name or job_name in job['name']:
            return job
    return {}


def compute_dora(runs: list) -> dict:
    """
    Compute all four DORA metrics from workflow run history.

    Definitions used:
      Deployment frequency  = successful deploy-backend jobs / window days
      Lead time             = time from push (run created_at) to
                              deploy-backend job completed_at
      Change failure rate   = runs where smoke-tests failed / total runs
      MTTR                  = mean time between first failure in a streak
                              and the next successful run that follows it
    """
    if not runs:
        return {
            'window_days':          DAYS_WINDOW,
            'total_runs':           0,
            'deployment_frequency': 0.0,
            'lead_time_mean_min':   None,
            'lead_time_p95_min':    None,
            'change_failure_rate':  None,
            'mttr_mean_min':        None,
            'failed_runs':          0,
            'successful_runs':      0,
        }

    lead_times   = []
    outcomes     = []   # True = success (smoke passed), False = failure
    failure_timestamps = []
    recovery_timestamps = []
    in_failure_streak = False
    failure_start = None

    for run in reversed(runs):   # chronological order
        run_id  = run['id']
        push_ts = dateparser.parse(run['created_at'])

        # --- deploy-backend job outcome ---
        deploy_job = get_job_status(run_id, DEPLOY_JOB)
        deploy_ok  = deploy_job.get('conclusion') == 'success'

        # --- smoke-test job outcome (defines "change failure") ---
        smoke_job = get_job_status(run_id, SMOKE_JOB)
        smoke_ok  = smoke_job.get('conclusion') == 'success'
        smoke_ran = smoke_job.get('conclusion') is not None

        # Lead time: push → deploy job completed
        if deploy_ok and deploy_job.get('completed_at'):
            deploy_ts = dateparser.parse(deploy_job['completed_at'])
            lead_sec  = (deploy_ts - push_ts).total_seconds()
            if lead_sec > 0:
                lead_times.append(lead_sec / 60)

        # Change failure / MTTR tracking
        if smoke_ran:
            success = smoke_ok
            outcomes.append(success)

            if not success and not in_failure_streak:
                in_failure_streak = True
                failure_start     = push_ts
            elif success and in_failure_streak:
                in_failure_streak = False
                recovery_ts       = dateparser.parse(
                    smoke_job.get('completed_at') or run['updated_at']
                )
                mttr_min = (recovery_ts - failure_start).total_seconds() / 60
                if mttr_min > 0:
                    recovery_timestamps.append(mttr_min)
                failure_start = None

    # --- Deployment frequency ---
    successful_deploys = sum(
        1 for r in runs
        if get_job_status(r['id'], DEPLOY_JOB).get('conclusion') == 'success'
    )
    freq = successful_deploys / DAYS_WINDOW

    # --- Lead time ---
    lt_mean = round(sum(lead_times) / len(lead_times), 1) if lead_times else None
    lt_p95  = None
    if lead_times:
        sorted_lt = sorted(lead_times)
        p95_idx   = int(len(sorted_lt) * 0.95)
        lt_p95    = round(sorted_lt[min(p95_idx, len(sorted_lt) - 1)], 1)

    # --- Change failure rate ---
    cfr = None
    if outcomes:
        cfr = round(outcomes.count(False) / len(outcomes) * 100, 1)

    # --- MTTR ---
    mttr_mean = None
    if recovery_timestamps:
        mttr_mean = round(sum(recovery_timestamps) / len(recovery_timestamps), 1)

    return {
        'window_days':          DAYS_WINDOW,
        'total_runs':           len(runs),
        'successful_runs':      successful_deploys,
        'failed_runs':          len(runs) - successful_deploys,
        'deployment_frequency': round(freq, 3),
        'lead_time_mean_min':   lt_mean,
        'lead_time_p95_min':    lt_p95,
        'change_failure_rate':  cfr,
        'mttr_mean_min':        mttr_mean,
    }


def dora_rating(metrics: dict) -> dict:
    """
    Map DORA metrics to Elite / High / Medium / Low ratings
    using the standard DORA benchmark thresholds.
    """
    ratings = {}

    freq = metrics['deployment_frequency']
    if   freq >= 1:    ratings['deployment_frequency'] = 'Elite'
    elif freq >= 1/7:  ratings['deployment_frequency'] = 'High'
    elif freq >= 1/30: ratings['deployment_frequency'] = 'Medium'
    else:              ratings['deployment_frequency'] = 'Low'

    lt = metrics['lead_time_mean_min']
    if lt is None:     ratings['lead_time'] = 'N/A'
    elif lt < 60:      ratings['lead_time'] = 'Elite'
    elif lt < 1440:    ratings['lead_time'] = 'High'
    elif lt < 10080:   ratings['lead_time'] = 'Medium'
    else:              ratings['lead_time'] = 'Low'

    cfr = metrics['change_failure_rate']
    if cfr is None:    ratings['change_failure_rate'] = 'N/A'
    elif cfr < 5:      ratings['change_failure_rate'] = 'Elite'
    elif cfr < 10:     ratings['change_failure_rate'] = 'High'
    elif cfr < 15:     ratings['change_failure_rate'] = 'Medium'
    else:              ratings['change_failure_rate'] = 'Low'

    mttr = metrics['mttr_mean_min']
    if mttr is None:   ratings['mttr'] = 'N/A'
    elif mttr < 60:    ratings['mttr'] = 'Elite'
    elif mttr < 1440:  ratings['mttr'] = 'High'
    elif mttr < 10080: ratings['mttr'] = 'Medium'
    else:              ratings['mttr'] = 'Low'

    return ratings


if __name__ == '__main__':
    since = datetime.now(timezone.utc) - timedelta(days=DAYS_WINDOW)
    print(f'Fetching workflow runs for the last {DAYS_WINDOW} days...')

    runs    = get_workflow_runs(since)
    metrics = compute_dora(runs)
    ratings = dora_rating(metrics)

    print('\n── DORA Metrics ─────────────────────────────────────────')
    print(f"  Window              : last {DAYS_WINDOW} days")
    print(f"  Total runs          : {metrics['total_runs']}")
    print(f"  Successful deploys  : {metrics['successful_runs']}")
    print(f"  Failed runs         : {metrics['failed_runs']}")
    print()
    print(f"  Deployment freq     : {metrics['deployment_frequency']:.3f} /day"
          f"  [{ratings['deployment_frequency']}]")
    print(f"  Lead time (mean)    : {metrics['lead_time_mean_min']} min"
          f"  [{ratings['lead_time']}]")
    print(f"  Lead time (p95)     : {metrics['lead_time_p95_min']} min")
    print(f"  Change failure rate : {metrics['change_failure_rate']}%"
          f"  [{ratings['change_failure_rate']}]")
    print(f"  MTTR (mean)         : {metrics['mttr_mean_min']} min"
          f"  [{ratings['mttr']}]")
    print('─────────────────────────────────────────────────────────')

    # Write JSON summary for downstream use (e.g. /api/dora endpoint)
    out_path = 'dora_metrics.json'
    with open(out_path, 'w') as f:
        json.dump({'metrics': metrics, 'ratings': ratings}, f, indent=2)
    print(f'\nSaved → {out_path}')