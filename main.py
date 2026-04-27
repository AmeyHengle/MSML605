# main.py
import asyncio
import json
import numpy as np
import os
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
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from pipeline import PipelineState, ENERGY_FEATURES
import urllib.error
import urllib.request
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError
except Exception:  # noqa: BLE001
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

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
_main_report_lock = threading.Lock()
_main_report_job = {
    'status': 'idle',
    'running': False,
    'started_at': None,
    'ended_at': None,
    'generated_at': None,
    'error': None,
    'retrain_found': None,
    'pdf_path': None,
    'html_path': None,
    'pdf_s3_url': None,
    'html_s3_url': None,
    'summary': None,
    'report_data': None,
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


def _generate_main_pipeline_summary(report_data: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return (
            "Groq summary unavailable because GROQ_API_KEY is not configured. "
            "Retrain event detected in main pipeline; review attached metrics and charts in the report."
        )

    body = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 700,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an ML product analyst. Produce a comprehensive summary for product stakeholders. "
                    "Cover drift behavior, retrain trigger rationale, old/new model impact, risk, and next actions."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(report_data)[:14000],
            },
        ],
    }
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=25) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned")
        out = (choices[0].get("message") or {}).get("content", "").strip()
        if not out:
            raise RuntimeError("Empty summary")
        return out
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError):
        return (
            "Groq summary request failed. Retrain event detected in main pipeline; "
            "use the attached metrics and report artifacts for review."
        )


def _write_pdf_report(pdf_path: Path, report_data: dict, summary: str) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    y = height - 40

    def write_line(text: str, *, font="Helvetica", size=10, gap=14):
        nonlocal y
        c.setFont(font, size)
        lines = simpleSplit(text, font, size, width - 72)
        for ln in lines:
            if y < 60:
                c.showPage()
                y = height - 40
                c.setFont(font, size)
            c.drawString(36, y, ln)
            y -= gap

    write_line("Main Pipeline Retrain Report", font="Helvetica-Bold", size=16, gap=20)
    write_line(f"Generated at (UTC): {report_data.get('generated_at')}", font="Helvetica", size=9, gap=12)
    write_line(f"Retrain detected: {bool(report_data.get('retrain_found'))}", font="Helvetica", size=9, gap=12)
    write_line(f"Retrain period: {report_data.get('retrain_period')}", font="Helvetica", size=9, gap=12)
    write_line(f"Model version: {report_data.get('model_version')}", font="Helvetica", size=9, gap=12)
    write_line(f"KS at retrain: {report_data.get('ks_stat')}", font="Helvetica", size=9, gap=12)
    write_line(f"PSI at retrain: {report_data.get('psi')}", font="Helvetica", size=9, gap=12)
    write_line(f"R2 at retrain: {report_data.get('r2')}", font="Helvetica", size=9, gap=12)
    write_line(f"RMSE at retrain: {report_data.get('rmse')}", font="Helvetica", size=9, gap=12)
    y -= 6
    write_line("Groq Comprehensive Summary", font="Helvetica-Bold", size=12, gap=16)
    write_line(summary, font="Helvetica", size=10, gap=14)
    y -= 4
    write_line("Top Drift Features", font="Helvetica-Bold", size=11, gap=16)
    for f in report_data.get("top_drift_features", []):
        write_line(f"- {f}", font="Helvetica", size=10, gap=14)

    c.save()


def _build_main_report_html(report_data: dict, summary: str) -> str:
    payload_json = json.dumps(report_data)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Main Pipeline Retrain Report</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2937; }}
    .meta {{ color: #4b5563; margin-bottom: 8px; }}
    .panel {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; margin-bottom: 14px; }}
    .plot {{ height: 320px; }}
  </style>
</head>
<body>
  <h1>Main Pipeline Retrain Report</h1>
  <div class="meta">Generated at: {report_data.get('generated_at')}</div>
  <div class="meta">Retrain detected: {bool(report_data.get('retrain_found'))}</div>
  <div class="meta">Retrain period: {report_data.get('retrain_period')}</div>
  <div class="meta">Model version: {report_data.get('model_version')}</div>
  <div class="panel">
    <h3>Groq Summary</h3>
    <p>{summary}</p>
  </div>
  <div class="panel">
    <h3>Retrain Metrics</h3>
    <p>KS: <strong>{report_data.get('ks_stat')}</strong> | PSI: <strong>{report_data.get('psi')}</strong></p>
    <p>R2: <strong>{report_data.get('r2')}</strong> | RMSE: <strong>{report_data.get('rmse')}</strong></p>
  </div>
  <div class="panel"><h3>KS and PSI history</h3><div id="plot-drift" class="plot"></div></div>
  <div class="panel"><h3>R2 and RMSE history</h3><div id="plot-performance" class="plot"></div></div>
  <script>
    const p = {payload_json};
    Plotly.newPlot('plot-drift', [
      {{ x: p.history.periods, y: p.history.ks, mode: 'lines+markers', name: 'KS' }},
      {{ x: p.history.periods, y: p.history.psi, mode: 'lines+markers', name: 'PSI' }}
    ], {{ margin: {{ t: 30 }} }}, {{ displayModeBar: false }});
    Plotly.newPlot('plot-performance', [
      {{ x: p.history.periods, y: p.history.r2, mode: 'lines+markers', name: 'R2' }},
      {{ x: p.history.periods, y: p.history.rmse, mode: 'lines+markers', name: 'RMSE', yaxis: 'y2' }}
    ], {{
      margin: {{ t: 30 }},
      yaxis: {{ title: 'R2' }},
      yaxis2: {{ title: 'RMSE', overlaying: 'y', side: 'right' }}
    }}, {{ displayModeBar: false }});
  </script>
</body>
</html>
"""


def _latest_main_report_path(suffix: str) -> Path | None:
    reports_dir = Path(os.getenv('REPORTS_DIR', Path(__file__).resolve().parent / 'reports'))
    if not reports_dir.exists():
        return None
    candidates = sorted(reports_dir.glob(f"main_retrain_*{suffix}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _maybe_upload_report_to_s3(local_path: Path) -> str | None:
    bucket = os.getenv("REPORT_S3_BUCKET", "").strip()
    if not bucket or boto3 is None:
        return None

    prefix = os.getenv("REPORT_S3_PREFIX", "reports").strip().strip("/")
    key = f"{prefix}/{local_path.name}" if prefix else local_path.name
    content_type = "application/pdf" if local_path.suffix.lower() == ".pdf" else "text/html"
    try:
        client = boto3.client("s3")
        client.upload_file(
            str(local_path),
            bucket,
            key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    except (BotoCoreError, ClientError, OSError):
        return None


def _run_main_retrain_report_job() -> dict:
    cfg = InitConfig().model_dump()
    state = PipelineState(cfg)
    init_payload = state.initialize()

    periods = []
    ks_hist = []
    psi_hist = []
    r2_hist = []
    rmse_hist = []
    retrain_payload = None
    last_payload = None
    for _ in range(len(state.days)):
        tick = state.tick()
        if tick is None:
            break
        last_payload = tick
        periods.append(tick.get('month'))
        ks_hist.append(tick.get('ks_stat'))
        psi_hist.append(tick.get('psi'))
        r2_hist.append(tick.get('r2'))
        rmse_hist.append(tick.get('rmse'))
        if tick.get('retrained'):
            retrain_payload = tick
            break

    anchor = retrain_payload or last_payload
    if anchor is None:
        raise RuntimeError("Pipeline produced no simulation payload")

    retrain_found = retrain_payload is not None
    top_drift = sorted(
        (f for f, sev in (anchor.get('drift_pills') or {}).items() if sev in {'high', 'critical'}),
        key=lambda x: x,
    )[:6]
    report_data = {
        'generated_at': _utc_now_iso(),
        'retrain_found': retrain_found,
        'init': {
            'feature_x': init_payload.get('feature_x'),
            'feature_y': init_payload.get('feature_y'),
            'total_periods': init_payload.get('total_months'),
        },
        'retrain_period': anchor.get('month'),
        'model_version': anchor.get('model_version'),
        'ks_stat': anchor.get('ks_stat'),
        'psi': anchor.get('psi'),
        'r2': anchor.get('r2'),
        'rmse': anchor.get('rmse'),
        'top_drift_features': top_drift,
        'ui_snapshot': {
            'pca_x': anchor.get('pca_x', []),
            'pca_y': anchor.get('pca_y', []),
            'kde_ref_x': anchor.get('kde_ref_x', []),
            'kde_ref_y': anchor.get('kde_ref_y', []),
            'kde_cur_x': anchor.get('kde_cur_x', []),
            'kde_cur_y': anchor.get('kde_cur_y', []),
            'line_pc1': anchor.get('new_line', {}).get('line_pc1') if anchor.get('new_line') else [],
            'line_y': anchor.get('new_line', {}).get('line_y') if anchor.get('new_line') else [],
        },
        'history': {
            'periods': periods,
            'ks': ks_hist,
            'psi': psi_hist,
            'r2': r2_hist,
            'rmse': rmse_hist,
        },
    }
    summary = _generate_main_pipeline_summary(report_data)

    reports_dir = Path(os.getenv('REPORTS_DIR', Path(__file__).resolve().parent / 'reports'))
    stem = f"main_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    html_path = reports_dir / f"{stem}.html"
    pdf_path = reports_dir / f"{stem}.pdf"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_build_main_report_html(report_data, summary), encoding='utf-8')
    _write_pdf_report(pdf_path, report_data, summary)

    return {
        'status': 'succeeded',
        'generated_at': report_data['generated_at'],
        'summary': summary,
        'report_data': report_data,
        'pdf_path': str(pdf_path.resolve()),
        'html_path': str(html_path.resolve()),
        'pdf_s3_url': _maybe_upload_report_to_s3(pdf_path),
        'html_s3_url': _maybe_upload_report_to_s3(html_path),
    }


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


@app.post('/api/retrain-report/run')
async def retrain_report_run():
    with _main_report_lock:
        if _main_report_job.get('running'):
            raise HTTPException(status_code=409, detail='Main retrain report job already running')
        _main_report_job['status'] = 'running'
        _main_report_job['running'] = True
        _main_report_job['started_at'] = _utc_now_iso()
        _main_report_job['ended_at'] = None
        _main_report_job['generated_at'] = None
        _main_report_job['error'] = None
        _main_report_job['retrain_found'] = None

    try:
        result = _run_main_retrain_report_job()
        report_data = result['report_data']
        with _main_report_lock:
            _main_report_job['status'] = 'succeeded'
            _main_report_job['running'] = False
            _main_report_job['ended_at'] = _utc_now_iso()
            _main_report_job['generated_at'] = report_data['generated_at']
            _main_report_job['error'] = None
            _main_report_job['retrain_found'] = bool(report_data.get('retrain_found'))
            _main_report_job['pdf_path'] = result['pdf_path']
            _main_report_job['html_path'] = result['html_path']
            _main_report_job['pdf_s3_url'] = result.get('pdf_s3_url')
            _main_report_job['html_s3_url'] = result.get('html_s3_url')
            _main_report_job['summary'] = result['summary']
            _main_report_job['report_data'] = report_data
    except Exception as exc:  # noqa: BLE001
        with _main_report_lock:
            _main_report_job['status'] = 'failed'
            _main_report_job['running'] = False
            _main_report_job['ended_at'] = _utc_now_iso()
            _main_report_job['generated_at'] = _utc_now_iso()
            _main_report_job['error'] = str(exc)
        raise HTTPException(status_code=500, detail=f"Failed to generate main retrain report: {exc}") from exc

    return {
        'status': 'succeeded',
        'generated_at': report_data['generated_at'],
        'report_url_pdf': '/api/retrain-report/latest/pdf',
        'report_url_html': '/api/retrain-report/latest/html',
        'report_s3_pdf_url': result.get('pdf_s3_url'),
        'report_s3_html_url': result.get('html_s3_url'),
        'summary': result['summary'],
        'metrics': {
            'ks_stat': report_data['ks_stat'],
            'psi': report_data['psi'],
            'r2': report_data['r2'],
            'rmse': report_data['rmse'],
            'top_drift_features': report_data['top_drift_features'],
        },
        'retrain_found': bool(report_data.get('retrain_found')),
    }


@app.get('/api/retrain-report/status')
async def retrain_report_status():
    with _main_report_lock:
        pdf_path = _main_report_job.get('pdf_path')
        html_path = _main_report_job.get('html_path')
        pdf_exists = bool(pdf_path and Path(pdf_path).exists())
        html_exists = bool(html_path and Path(html_path).exists())
        if not pdf_exists and _latest_main_report_path(".pdf") is not None:
            pdf_exists = True
        if not html_exists and _latest_main_report_path(".html") is not None:
            html_exists = True
        return {
            'status': _main_report_job.get('status', 'idle'),
            'running': bool(_main_report_job.get('running', False)),
            'started_at': _main_report_job.get('started_at'),
            'ended_at': _main_report_job.get('ended_at'),
            'generated_at': _main_report_job.get('generated_at'),
            'error': _main_report_job.get('error'),
            'retrain_found': _main_report_job.get('retrain_found'),
            'pdf_available': pdf_exists,
            'html_available': html_exists,
            'pdf_url': '/api/retrain-report/latest/pdf' if pdf_exists else None,
            'html_url': '/api/retrain-report/latest/html' if html_exists else None,
            'pdf_s3_url': _main_report_job.get('pdf_s3_url'),
            'html_s3_url': _main_report_job.get('html_s3_url'),
        }


@app.get('/api/retrain-report/latest/pdf')
async def retrain_report_latest_pdf():
    with _main_report_lock:
        pdf_path_raw = _main_report_job.get('pdf_path')
    if pdf_path_raw:
        pdf_path = Path(pdf_path_raw).resolve()
    else:
        found = _latest_main_report_path(".pdf")
        if found is None:
            raise HTTPException(status_code=404, detail='No main-pipeline retrain PDF report generated yet')
        pdf_path = found.resolve()
        with _main_report_lock:
            _main_report_job['pdf_path'] = str(pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail='PDF report file not found')
    return FileResponse(str(pdf_path), media_type='application/pdf', filename=pdf_path.name)


@app.get('/api/retrain-report/latest/html')
async def retrain_report_latest_html():
    with _main_report_lock:
        html_path_raw = _main_report_job.get('html_path')
    if html_path_raw:
        html_path = Path(html_path_raw).resolve()
    else:
        found = _latest_main_report_path(".html")
        if found is None:
            raise HTTPException(status_code=404, detail='No main-pipeline retrain HTML report generated yet')
        html_path = found.resolve()
        with _main_report_lock:
            _main_report_job['html_path'] = str(html_path)
    if not html_path.exists():
        raise HTTPException(status_code=404, detail='HTML report file not found')
    return FileResponse(str(html_path), media_type='text/html', filename=html_path.name)


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