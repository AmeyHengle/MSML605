# main.py
import asyncio
import json
import numpy as np
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pipeline import PipelineState

try:
    from monitoring import get_current_metrics, cloudwatch_stream
    MONITORING_ENABLED = True
except ImportError:
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
_simulation_running: bool       = False


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


# ── Pipeline routes ───────────────────────────────────────────────────────────
@app.post('/api/initialize')
async def initialize(config: InitConfig):
    global _state, _simulation_running
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
    return {
        'initialized': _state is not None,
        'running':     _simulation_running,
        'month_idx':   _state.current_month if _state else None,
        'total':       len(_state.months)   if _state else None,
    }


# ── Monitoring routes (Page 2) ────────────────────────────────────────────────
@app.get('/api/cloudwatch/metrics')
async def cw_metrics():
    if not MONITORING_ENABLED:
        return {'error': 'CloudWatch not configured'}
    return get_current_metrics()


@app.get('/api/cloudwatch/stream')
async def cw_stream(request: Request):
    """SSE stream — polls CloudWatch every 10s, pushes to Page 2."""
    return await cloudwatch_stream(request)


class PredictRequest(BaseModel):
    features: list

@app.post('/api/predict')
async def predict(req: PredictRequest):
    if _state is None or _state.model is None:
        return {'error': 'model not initialized'}
    import numpy as np
    X    = np.array(req.features).reshape(1, -1)
    pred = float(_state.model.predict(X)[0])
    return {'forecast_intensity': round(pred, 2)}