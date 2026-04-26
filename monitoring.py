# monitoring.py
# Polls AWS CloudWatch for App Runner metrics.
# Imported by main.py — adds /api/cloudwatch/metrics and /api/cloudwatch/stream.

import os
import json
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Optional

import boto3
from fastapi import Request
from fastapi.responses import StreamingResponse

AWS_REGION      = os.getenv('AWS_REGION',          'us-east-1')
APP_RUNNER_ARN  = os.getenv('APP_RUNNER_SERVICE_ARN', '')

# Metric namespace and dimension App Runner uses in CloudWatch
CW_NAMESPACE = 'AWS/AppRunner'

cw_client = boto3.client('cloudwatch', region_name=AWS_REGION)
ar_client = boto3.client('apprunner',  region_name=AWS_REGION)


def _service_name_from_arn(arn: str) -> str:
    """Extract service name from App Runner ARN."""
    return arn.split('/')[-2] if arn else 'unknown'


def _get_metric(metric_name: str, stat: str,
                minutes: int = 5, period: int = 30) -> list:
    """
    Fetch a single CloudWatch metric for the App Runner service.
    Returns list of {timestamp, value} dicts sorted oldest-first.
    """
    if not APP_RUNNER_ARN:
        return []

    service_name = _service_name_from_arn(APP_RUNNER_ARN)
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)

    resp = cw_client.get_metric_statistics(
        Namespace=CW_NAMESPACE,
        MetricName=metric_name,
        Dimensions=[{'Name': 'ServiceName', 'Value': service_name}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=[stat],
    )

    points = sorted(resp['Datapoints'], key=lambda p: p['Timestamp'])
    return [
        {
            'timestamp': p['Timestamp'].isoformat(),
            'value':     round(p[stat], 4),
        }
        for p in points
    ]


def _get_instance_count() -> int:
    """
    App Runner doesn't expose instance count directly in CloudWatch.
    Estimate from RequestLatency sample count vs max concurrency.
    Falls back to 1 if no data.
    """
    points = _get_metric('Requests', 'Sum', minutes=2, period=60)
    if not points:
        return 1
    req_per_min = points[-1]['value'] if points else 0
    # Each instance handles ~100 req concurrently; rough estimate
    return max(1, round(req_per_min / 6000))


def get_current_metrics() -> dict:
    """
    Snapshot of the last 5 minutes of App Runner metrics.
    Returns a dict ready to JSON-serialize and send to the frontend.
    """
    req_series    = _get_metric('Requests',           'Sum',     minutes=10, period=30)
    lat_p50       = _get_metric('RequestLatency',     'Average', minutes=10, period=30)
    lat_p99       = _get_metric('RequestLatency',     'p99',     minutes=10, period=30)
    err_4xx       = _get_metric('Http4xxRequests',    'Sum',     minutes=10, period=30)
    err_5xx       = _get_metric('Http5xxRequests',    'Sum',     minutes=10, period=30)
    cpu_series    = _get_metric('CPUUtilization',     'Average', minutes=10, period=30)
    memory_series = _get_metric('MemoryUtilization',  'Average', minutes=10, period=30)

    # Latest snapshot values
    def latest(series, default=0.0):
        return series[-1]['value'] if series else default

    total_req  = sum(p['value'] for p in req_series) if req_series else 0
    total_4xx  = sum(p['value'] for p in err_4xx)    if err_4xx   else 0
    total_5xx  = sum(p['value'] for p in err_5xx)    if err_5xx   else 0
    error_rate = round((total_4xx + total_5xx) / max(total_req, 1) * 100, 2)

    # Requests per second (average over last period)
    rps = round(latest(req_series) / 30, 2) if req_series else 0.0

    return {
        'timestamp':      datetime.now(timezone.utc).isoformat(),
        'rps':            rps,
        'lat_p50_ms':     round(latest(lat_p50),    2),
        'lat_p99_ms':     round(latest(lat_p99),    2),
        'error_rate_pct': error_rate,
        'cpu_pct':        round(latest(cpu_series), 2),
        'memory_pct':     round(latest(memory_series), 2),
        'instance_count': _get_instance_count(),
        'series': {
            'requests':    req_series,
            'lat_p50':     lat_p50,
            'lat_p99':     lat_p99,
            'cpu':         cpu_series,
            'memory':      memory_series,
        }
    }


async def cloudwatch_stream(request: Request):
    """
    SSE stream that polls CloudWatch every 10 seconds.
    Attach to GET /api/cloudwatch/stream in main.py.
    """
    async def event_gen():
        while True:
            if await request.is_disconnected():
                break
            try:
                payload = get_current_metrics()
                yield f'data: {json.dumps(payload)}\n\n'
            except Exception as e:
                yield f'data: {{"error": "{str(e)}"}}\n\n'
            await asyncio.sleep(10)

    return StreamingResponse(
        event_gen(),
        media_type='text/event-stream',
        headers={
            'Cache-Control':               'no-cache',
            'X-Accel-Buffering':           'no',
            'Access-Control-Allow-Origin': '*',
        }
    )