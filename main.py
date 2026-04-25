# main.py
import asyncio
import json
import numpy as np
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from pipeline import PipelineState, ENERGY_FEATURES

try:
    from monitoring import get_current_metrics, cloudwatch_stream
    MONITORING_ENABLED = True
except Exception:
    MONITORING_ENABLED = False

app = FastAPI(title='CarbonWatch MLOps')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Shared state ──────────────────────────────────────────────────────────────
_state: Optional[PipelineState] = None
_simulation_running: bool = False
_agent_lock = threading.Lock()
_agent_job = {
    'available': (Path(__file__).resolve().parent / 'run_batch_pipeline.py').exists(),
    'running': False,
    'job_id': None,
    'status': 'idle',
    'started_at': None,
    'ended_at': None,
    'exit_code': None,
    'error': None,
    'report_path': None,
    'logs': deque(maxlen=200),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_status_payload() -> dict:
    report_path = _agent_job.get('report_path')
    report_available = bool(report_path and Path(report_path).exists())
    return {
        'available': _agent_job['available'],
        'running': _agent_job['running'],
        'job_id': _agent_job['job_id'],
        'status': _agent_job['status'],
        'started_at': _agent_job['started_at'],
        'ended_at': _agent_job['ended_at'],
        'exit_code': _agent_job['exit_code'],
        'error': _agent_job['error'],
        'report_path': report_path,
        'report_available': report_available,
        'report_url': '/api/agent/report/latest' if report_available else None,
        'log_lines': len(_agent_job['logs']),
    }


def _run_agent_job(job_id: str) -> None:
    script_path = Path(__file__).resolve().parent / 'run_batch_pipeline.py'
    cmd = [sys.executable, '-u', str(script_path)]
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(script_path.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout is not None:
            for line in proc.stdout:
                with _agent_lock:
                    log_line = line.rstrip()
                    _agent_job['logs'].append(log_line)
                    marker = '[batch] report_html='
                    if log_line.startswith(marker):
                        _agent_job['report_path'] = log_line.split(marker, 1)[1].strip()
        exit_code = proc.wait()
        with _agent_lock:
            _agent_job['running'] = False
            _agent_job['status'] = 'succeeded' if exit_code == 0 else 'failed'
            _agent_job['exit_code'] = exit_code
            _agent_job['ended_at'] = _utc_now_iso()
            if exit_code != 0 and not _agent_job['error']:
                _agent_job['error'] = f'Agent job exited with code {exit_code}'
    except Exception as exc:  # noqa: BLE001
        with _agent_lock:
            _agent_job['running'] = False
            _agent_job['status'] = 'failed'
            _agent_job['ended_at'] = _utc_now_iso()
            _agent_job['error'] = str(exc)
    finally:
        if proc is not None and proc.stdout is not None:
            proc.stdout.close()


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)


class InitConfig(BaseModel):
    feature_x:    str   = 'gas'
    feature_y:    str   = 'forecast_intensity'
    ks_threshold: float = 0.10
    n_init:       int   = 50
    n_monthly:    int   = 5
    speed:        float = 1.0
    models_dir:   str   = 'models'
    data_path:    str   = 'data/historical_data.csv'


class PredictRequest(BaseModel):
    features: List[float]


# ── Pipeline routes ───────────────────────────────────────────────────────────
@app.post('/api/initialize')
async def initialize(config: InitConfig):
    global _state, _simulation_running

    # Validate early so bad input returns 4xx instead of internal KeyError.
    if config.feature_x not in ENERGY_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature_x '{config.feature_x}'. Must be one of: {', '.join(ENERGY_FEATURES)}",
        )

    _simulation_running = False
    _state  = PipelineState(config.model_dump())
    payload = _state.initialize()
    return {'status': 'ok', 'data': payload}


@app.get('/api/simulate')
async def simulate(request: Request):
    global _simulation_running

    if _state is None:
        async def err():
            yield 'data: {"error": "Call /api/initialize first"}\n\n'
        return StreamingResponse(err(), media_type='text/event-stream')

    async def event_stream():
        global _simulation_running
        _simulation_running = True
        try:
            while True:
                if await request.is_disconnected():
                    break
                if not _simulation_running:
                    yield 'data: {"paused": true}\n\n'
                    await asyncio.sleep(0.2)
                    continue
                payload = _state.tick()
                if payload is None:
                    yield 'data: {"done": true}\n\n'
                    break
                yield f'data: {json.dumps(payload, cls=NumpyEncoder)}\n\n'
                if payload.get('done'):
                    break
                await asyncio.sleep(_state.speed)
        except asyncio.CancelledError:
            pass
        finally:
            _simulation_running = False

    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream',
        headers={
            'Cache-Control':               'no-cache',
            'X-Accel-Buffering':           'no',
            'Access-Control-Allow-Origin': '*',
        }
    )


@app.post('/api/pause')
async def pause():
    global _simulation_running
    _simulation_running = False
    return {'status': 'paused'}


@app.post('/api/resume')
async def resume():
    global _simulation_running
    _simulation_running = True
    return {'status': 'resumed'}


@app.post('/api/reset')
async def reset():
    global _state, _simulation_running
    _simulation_running = False
    _state = None
    return {'status': 'reset'}


@app.get('/api/status')
async def status():
    if _state is None:
        return {
            'initialized': False,
            'running': False,
            'month_idx': None,
            'total': None,
        }

    # Compatibility across legacy month-based and new day-based PipelineState.
    total = None
    if hasattr(_state, 'months'):
        total = len(_state.months)
    elif hasattr(_state, 'days'):
        total = len(_state.days)

    return {
        'initialized': True,
        'running': _simulation_running,
        'month_idx': _state.current_month,
        'total': total,
    }


@app.post('/api/predict')
async def predict(req: PredictRequest):
    """
    Standalone inference endpoint for load testing.
    Accepts 9 energy mix feature values, returns forecast intensity.
    """
    if _state is None or _state.model is None:
        return {'error': 'model not initialized — call /api/initialize first'}
    X    = np.array(req.features).reshape(1, -1)
    pred = float(_state.model.predict(X)[0])
    return {'forecast_intensity': round(pred, 2)}


# ── Agent routes (background batch pipeline runner) ───────────────────────────
@app.get('/api/agent/status')
async def agent_status():
    with _agent_lock:
        return _agent_status_payload()


@app.get('/api/agent/logs')
async def agent_logs():
    with _agent_lock:
        return {
            'job_id': _agent_job['job_id'],
            'running': _agent_job['running'],
            'report_path': _agent_job.get('report_path'),
            'logs': list(_agent_job['logs']),
        }


@app.post('/api/agent/run')
async def agent_run():
    with _agent_lock:
        if not _agent_job['available']:
            raise HTTPException(status_code=503, detail='run_batch_pipeline.py not available on this deployment')
        if _agent_job['running']:
            return {'status': 'already_running', 'job_id': _agent_job['job_id']}

        job_id = str(uuid.uuid4())
        _agent_job['running'] = True
        _agent_job['job_id'] = job_id
        _agent_job['status'] = 'running'
        _agent_job['started_at'] = _utc_now_iso()
        _agent_job['ended_at'] = None
        _agent_job['exit_code'] = None
        _agent_job['error'] = None
        _agent_job['report_path'] = None
        _agent_job['logs'].clear()
        _agent_job['logs'].append(f'[{_agent_job["started_at"]}] agent run started')

    t = threading.Thread(target=_run_agent_job, args=(job_id,), daemon=True)
    t.start()
    return {'status': 'started', 'job_id': job_id}


@app.get('/api/agent/report/latest')
async def agent_report_latest():
    with _agent_lock:
        report_path_raw = _agent_job.get('report_path')

    if not report_path_raw:
        raise HTTPException(status_code=404, detail='No report generated yet')

    report_path = Path(report_path_raw).expanduser().resolve()
    project_root = Path(__file__).resolve().parent
    if project_root not in report_path.parents:
        raise HTTPException(status_code=400, detail='Invalid report path')
    if not report_path.exists() or not report_path.is_file():
        raise HTTPException(status_code=404, detail='Report file not found')

    return FileResponse(
        str(report_path),
        media_type='text/html',
        filename=report_path.name,
    )


# ── Monitoring routes (Page 2 — requires AWS CloudWatch) ─────────────────────
@app.get('/api/cloudwatch/metrics')
async def cw_metrics():
    if not MONITORING_ENABLED:
        return {'error': 'CloudWatch not configured — set APP_RUNNER_SERVICE_ARN env var'}
    return get_current_metrics()


@app.get('/api/cloudwatch/stream')
async def cw_stream(request: Request):
    if not MONITORING_ENABLED:
        async def fallback():
            yield 'data: {"error": "CloudWatch not configured"}\n\n'
        return StreamingResponse(fallback(), media_type='text/event-stream')
    return await cloudwatch_stream(request)