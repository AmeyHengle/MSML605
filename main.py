# main.py
# FastAPI backend — three routes:
#   POST /api/initialize  → train baseline, return initial state
#   GET  /api/simulate    → SSE stream, one event per month
#   POST /api/reset       → clear state

import asyncio
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pipeline import PipelineState
from typing import Optional
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Converts numpy scalar types to native Python before JSON serialization."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

app = FastAPI(title='Carbon Intensity MLOps Pipeline')

# Allow the frontend (served from a different port during dev) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# Serve frontend static files at root
app.mount('/app', StaticFiles(directory='frontend', html=True), name='frontend')

# ── Shared state (single-user prototype) ─────────────────────────────────────
_state: PipelineState = None
_simulation_running: bool    = False


# ── Request / response models ─────────────────────────────────────────────────
class InitConfig(BaseModel):
    feature_x:    str   = 'gas'
    feature_y:    str   = 'forecast_intensity'
    ks_threshold: float = 0.10
    n_init:       int   = 50
    n_monthly:    int   = 5
    speed:        float = 1.0       # seconds between SSE events
    models_dir:   str   = 'models'
    data_path:    str   = 'data/historical_data.csv'


# ── Routes ────────────────────────────────────────────────────────────────────
@app.post('/api/initialize')
async def initialize(config: InitConfig):
    """
    Load data, train the baseline model on month 0, return initial state.
    Must be called before /api/simulate.
    """
    global _state, _simulation_running
    _simulation_running = False
    _state = PipelineState(config.model_dump())
    payload = _state.initialize()
    return {'status': 'ok', 'data': payload}


@app.get('/api/simulate')
async def simulate(request: Request):
    """
    SSE stream — advances the pipeline one month per event.
    Each event is a JSON object (see pipeline.PipelineState.tick).
    The stream ends when all months are processed or the client disconnects.
    """
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
                # Check for client disconnect
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

                # Respect done flag without waiting another sleep cycle
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
            'X-Accel-Buffering':           'no',    # important for nginx proxies
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
