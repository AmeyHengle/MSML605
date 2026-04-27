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
_report_events = deque(maxlen=300)
_report_notify_lock = threading.Lock()
_last_notified_model_version = 0
_sim_history = {
    'periods': [],
    'ks': [],
    'psi': [],
    'r2': [],
    'rmse': [],
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_report_event(level: str, message: str, **meta) -> None:
    _report_events.append({
        'ts': _utc_now_iso(),
        'level': level,
        'message': message,
        'meta': meta,
    })


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


def _deterministic_comprehensive_summary(report_data: dict) -> str:
    retrain_found = bool(report_data.get("retrain_found"))
    retrain_period = report_data.get("retrain_period", "n/a")
    model_version = report_data.get("model_version", "n/a")
    ks_stat = report_data.get("ks_stat", "n/a")
    psi = report_data.get("psi", "n/a")
    r2 = report_data.get("r2", "n/a")
    rmse = report_data.get("rmse", "n/a")
    top_feats = report_data.get("top_drift_features", [])[:6]
    top_feats_text = ", ".join(top_feats) if top_feats else "none"

    history = report_data.get("history") or {}
    ks_hist = history.get("ks") or []
    psi_hist = history.get("psi") or []
    r2_hist = history.get("r2") or []
    rmse_hist = history.get("rmse") or []

    ks_start = ks_hist[0] if ks_hist else "n/a"
    ks_end = ks_hist[-1] if ks_hist else "n/a"
    psi_start = psi_hist[0] if psi_hist else "n/a"
    psi_end = psi_hist[-1] if psi_hist else "n/a"
    r2_start = r2_hist[0] if r2_hist else "n/a"
    r2_end = r2_hist[-1] if r2_hist else "n/a"
    rmse_start = rmse_hist[0] if rmse_hist else "n/a"
    rmse_end = rmse_hist[-1] if rmse_hist else "n/a"

    status_line = (
        "Retraining was triggered and a new model was promoted."
        if retrain_found
        else "No retraining trigger was observed; monitoring continued."
    )
    risk_line = (
        "Risk is elevated due to large distribution shift (high KS/PSI), so continued monitoring without retraining may degrade forecast reliability."
        if retrain_found
        else "Risk remains moderate; continue monitoring for sustained drift trends before promoting a new model."
    )
    action_line = (
        "Recommended action: keep the promoted model under close observation for the next few windows, and validate error stability before declaring steady-state."
        if retrain_found
        else "Recommended action: keep the current model, gather more windows, and retrain only if drift and error trends persist."
    )

    return (
        "Executive summary: "
        f"{status_line} At period {retrain_period}, model version {model_version} recorded KS={ks_stat}, PSI={psi}, R2={r2}, and RMSE={rmse}. "
        f"Top drift-contributing features were {top_feats_text}. "
        "Trend analysis: "
        f"KS moved from {ks_start} to {ks_end}, PSI moved from {psi_start} to {psi_end}, R2 moved from {r2_start} to {r2_end}, and RMSE moved from {rmse_start} to {rmse_end} across the observed window. "
        "Operational interpretation: The drift profile indicates a material input distribution change that can alter forecast behavior and confidence bands. "
        f"{risk_line} {action_line}"
    )


def _generate_main_pipeline_summary(report_data: dict) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        _log_report_event("warn", "groq_api_key_missing")
        return _deterministic_comprehensive_summary(report_data)

    analysis_prompt = (
        "Write a comprehensive ML monitoring report in plain text (no markdown headings). "
        "Use 4 sections with labels exactly as:\n"
        "1) Executive Summary:\n"
        "2) Drift and Data Shift Analysis:\n"
        "3) Model Performance and Retrain Impact:\n"
        "4) Risks and Recommended Actions:\n"
        "Requirements: 220-320 words total, include numeric evidence (KS, PSI, R2, RMSE), "
        "mention top drifted features, describe trend direction from history arrays, "
        "and end with concrete next actions for product and ML teams."
    )

    model_candidates = [
        os.getenv("GROQ_MODEL", "").strip(),
        "llama-3.1-8b-instant",
        "llama-3.1-70b-versatile",
    ]
    model_candidates = [m for m in model_candidates if m]
    for model_name in model_candidates:
        body = {
            "model": model_name,
            "temperature": 0.2,
            "max_tokens": 700,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an ML product analyst writing for product leadership and MLOps engineers. "
                        "Be specific, data-driven, and actionable."
                    ),
                },
                {
                    "role": "user",
                    "content": analysis_prompt + "\n\nInput metrics JSON:\n" + json.dumps(report_data)[:14000],
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
            _log_report_event(
                "info",
                "groq_request_started",
                retrain_found=bool(report_data.get("retrain_found")),
                model=model_name,
            )
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError("No choices returned")
            out = (choices[0].get("message") or {}).get("content", "").strip()
            if not out:
                raise RuntimeError("Empty summary")
            if len(out) < 350:
                raise RuntimeError("Summary too short")
            _log_report_event("info", "groq_request_succeeded", summary_chars=len(out), model=model_name)
            return out
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            _log_report_event("error", "groq_request_failed", error=str(exc), model=model_name)

    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    if gemini_key:
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip() or "gemini-1.5-flash"
        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{gemini_model}:generateContent?key={gemini_key}"
        )
        gemini_body = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are an ML product analyst writing for product leadership and MLOps engineers. "
                            "Be specific, data-driven, and actionable."
                        )
                    }
                ]
            },
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1100,
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                analysis_prompt
                                + "\n\nInput metrics JSON:\n"
                                + json.dumps(report_data)[:14000]
                            )
                        }
                    ],
                }
            ],
        }
        gemini_req = urllib.request.Request(
            gemini_url,
            data=json.dumps(gemini_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            _log_report_event("info", "gemini_request_started", model=gemini_model)
            with urllib.request.urlopen(gemini_req, timeout=25) as resp:
                gemini_payload = json.loads(resp.read().decode("utf-8"))
            candidates = gemini_payload.get("candidates") or []
            if not candidates:
                raise RuntimeError("No candidates returned")
            parts = (((candidates[0].get("content") or {}).get("parts")) or [])
            text_out = "\n".join(
                (p.get("text", "") or "").strip()
                for p in parts
                if isinstance(p, dict)
            ).strip()
            if not text_out:
                raise RuntimeError("Empty Gemini summary")
            if len(text_out) < 350:
                raise RuntimeError("Gemini summary too short")
            _log_report_event("info", "gemini_request_succeeded", summary_chars=len(text_out), model=gemini_model)
            return text_out
        except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            _log_report_event("error", "gemini_request_failed", error=str(exc), model=gemini_model)
    else:
        _log_report_event("warn", "gemini_api_key_missing")

    return (
        _deterministic_comprehensive_summary(report_data)
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
    <h3>LLM Comprehensive Summary</h3>
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


def _save_report_artifacts(report_data: dict, summary: str) -> dict:
    reports_dir = Path(os.getenv('REPORTS_DIR', Path(__file__).resolve().parent / 'reports'))
    stem = f"main_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    html_path = reports_dir / f"{stem}.html"
    pdf_path = reports_dir / f"{stem}.pdf"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(_build_main_report_html(report_data, summary), encoding='utf-8')
    _write_pdf_report(pdf_path, report_data, summary)
    _log_report_event("info", "report_artifacts_written", pdf=str(pdf_path), html=str(html_path))
    return {
        'pdf_path': str(pdf_path.resolve()),
        'html_path': str(html_path.resolve()),
        'pdf_s3_url': _maybe_upload_report_to_s3(pdf_path),
        'html_s3_url': _maybe_upload_report_to_s3(html_path),
    }


def _send_slack_report_notification(payload: dict) -> None:
    webhook = os.getenv("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        _log_report_event("warn", "slack_webhook_missing")
        return
    req = urllib.request.Request(
        webhook,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        _log_report_event("info", "slack_webhook_succeeded", response=body[:120])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        _log_report_event("error", "slack_webhook_http_error", code=exc.code, body=body[:400])
    except Exception as exc:  # noqa: BLE001
        _log_report_event("error", "slack_webhook_failed", error=str(exc))


def _notify_live_retrain(payload: dict) -> None:
    try:
        global _last_notified_model_version
        mv = int(payload.get("model_version", 0) or 0)
        with _report_notify_lock:
            if mv <= _last_notified_model_version:
                _log_report_event("info", "live_retrain_duplicate_skipped", model_version=mv)
                return
            _last_notified_model_version = mv

        with _main_report_lock:
            hist = {
                'periods': list(_sim_history['periods']),
                'ks': list(_sim_history['ks']),
                'psi': list(_sim_history['psi']),
                'r2': list(_sim_history['r2']),
                'rmse': list(_sim_history['rmse']),
            }

        top_drift = sorted(
            (f for f, sev in (payload.get('drift_pills') or {}).items() if sev in {'high', 'critical'}),
            key=lambda x: x,
        )[:6]
        report_data = {
            'generated_at': _utc_now_iso(),
            'retrain_found': True,
            'init': {
                'feature_x': getattr(_state, 'feature_x', 'gas') if _state is not None else 'gas',
                'feature_y': getattr(_state, 'feature_y', 'forecast_intensity') if _state is not None else 'forecast_intensity',
                'total_periods': int(payload.get('total_months', 0) or 0),
            },
            'retrain_period': payload.get('month'),
            'model_version': payload.get('model_version'),
            'ks_stat': payload.get('ks_stat'),
            'psi': payload.get('psi'),
            'r2': payload.get('r2'),
            'rmse': payload.get('rmse'),
            'top_drift_features': top_drift,
            'ui_snapshot': {
                'pca_x': payload.get('pca_x', []),
                'pca_y': payload.get('pca_y', []),
                'kde_ref_x': payload.get('kde_ref_x', []),
                'kde_ref_y': payload.get('kde_ref_y', []),
                'kde_cur_x': payload.get('kde_cur_x', []),
                'kde_cur_y': payload.get('kde_cur_y', []),
                'line_pc1': payload.get('new_line', {}).get('line_pc1') if payload.get('new_line') else [],
                'line_y': payload.get('new_line', {}).get('line_y') if payload.get('new_line') else [],
            },
            'history': hist,
        }
        summary = _generate_main_pipeline_summary(report_data)
        artifacts = _save_report_artifacts(report_data, summary)

        with _main_report_lock:
            _main_report_job['status'] = 'succeeded'
            _main_report_job['running'] = False
            _main_report_job['ended_at'] = _utc_now_iso()
            _main_report_job['generated_at'] = report_data['generated_at']
            _main_report_job['error'] = None
            _main_report_job['retrain_found'] = True
            _main_report_job['pdf_path'] = artifacts['pdf_path']
            _main_report_job['html_path'] = artifacts['html_path']
            _main_report_job['pdf_s3_url'] = artifacts.get('pdf_s3_url')
            _main_report_job['html_s3_url'] = artifacts.get('html_s3_url')
            _main_report_job['summary'] = summary
            _main_report_job['report_data'] = report_data

        base = os.getenv("RENDER_URL", "").rstrip("/")
        pdf_url = artifacts.get('pdf_s3_url') or (f"{base}/api/retrain-report/latest/pdf" if base else "/api/retrain-report/latest/pdf")
        html_url = artifacts.get('html_s3_url') or (f"{base}/api/retrain-report/latest/html" if base else "/api/retrain-report/latest/html")
        slack_payload = {
            "attachments": [{
                "color": "#2eb67d",
                "blocks": [
                    {"type": "header", "text": {"type": "plain_text", "text": ":bar_chart: Live retrain report", "emoji": True}},
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Period*\n{payload.get('month', 'n/a')}"},
                            {"type": "mrkdwn", "text": f"*Model version*\n{payload.get('model_version', 'n/a')}"},
                            {"type": "mrkdwn", "text": f"*KS*\n{payload.get('ks_stat', 'n/a')}"},
                            {"type": "mrkdwn", "text": f"*PSI*\n{payload.get('psi', 'n/a')}"},
                            {"type": "mrkdwn", "text": f"*R2*\n{payload.get('r2', 'n/a')}"},
                            {"type": "mrkdwn", "text": f"*RMSE*\n{payload.get('rmse', 'n/a')}"},
                        ],
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*PDF report*\n{pdf_url}"},
                            {"type": "mrkdwn", "text": f"*HTML report*\n{html_url}"},
                        ],
                    },
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*LLM summary*\n{summary[:1000]}"}},
                ],
            }]
        }
        _send_slack_report_notification(slack_payload)
    except Exception as exc:  # noqa: BLE001
        _log_report_event("error", "live_retrain_notify_failed", error=str(exc))


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

    artifacts = _save_report_artifacts(report_data, summary)

    return {
        'status': 'succeeded',
        'generated_at': report_data['generated_at'],
        'summary': summary,
        'report_data': report_data,
        'pdf_path': artifacts['pdf_path'],
        'html_path': artifacts['html_path'],
        'pdf_s3_url': artifacts.get('pdf_s3_url'),
        'html_s3_url': artifacts.get('html_s3_url'),
    }


# ── Pipeline routes ───────────────────────────────────────────────────────────
@app.post('/api/initialize')
async def initialize(config: InitConfig):
    global _state, _simulation_running, _last_notified_model_version

    # Validate early so bad input returns 4xx instead of internal KeyError.
    if config.feature_x not in ENERGY_FEATURES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid feature_x '{config.feature_x}'. Must be one of: {', '.join(ENERGY_FEATURES)}",
        )

    _simulation_running = False
    _state  = PipelineState(config.model_dump())
    with _main_report_lock:
        _sim_history['periods'].clear()
        _sim_history['ks'].clear()
        _sim_history['psi'].clear()
        _sim_history['r2'].clear()
        _sim_history['rmse'].clear()
    _last_notified_model_version = 0
    _log_report_event("info", "pipeline_initialized", feature_x=config.feature_x, feature_y=config.feature_y)
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
                with _main_report_lock:
                    _sim_history['periods'].append(payload.get('month'))
                    _sim_history['ks'].append(payload.get('ks_stat'))
                    _sim_history['psi'].append(payload.get('psi'))
                    _sim_history['r2'].append(payload.get('r2'))
                    _sim_history['rmse'].append(payload.get('rmse'))
                if payload.get('retrained'):
                    _log_report_event(
                        "info",
                        "live_retrain_detected",
                        period=payload.get('month'),
                        model_version=payload.get('model_version'),
                    )
                    threading.Thread(target=_notify_live_retrain, args=(payload,), daemon=True).start()
                    payload['done'] = True
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
    global _state, _simulation_running, _last_notified_model_version
    _simulation_running = False
    _state = None
    with _main_report_lock:
        _sim_history['periods'].clear()
        _sim_history['ks'].clear()
        _sim_history['psi'].clear()
        _sim_history['r2'].clear()
        _sim_history['rmse'].clear()
    _last_notified_model_version = 0
    _log_report_event("info", "pipeline_reset")
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
    _log_report_event("info", "manual_report_run_started")

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
        _log_report_event("info", "manual_report_run_succeeded", retrain_found=bool(report_data.get('retrain_found')))
    except Exception as exc:  # noqa: BLE001
        with _main_report_lock:
            _main_report_job['status'] = 'failed'
            _main_report_job['running'] = False
            _main_report_job['ended_at'] = _utc_now_iso()
            _main_report_job['generated_at'] = _utc_now_iso()
            _main_report_job['error'] = str(exc)
        _log_report_event("error", "manual_report_run_failed", error=str(exc))
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


@app.get('/api/retrain-report/debug')
async def retrain_report_debug():
    with _main_report_lock:
        return {
            'env': {
                'groq_api_key_set': bool(os.getenv("GROQ_API_KEY", "").strip()),
                'slack_webhook_set': bool(os.getenv("SLACK_WEBHOOK_URL", "").strip()),
                'render_url_set': bool(os.getenv("RENDER_URL", "").strip()),
                'reports_dir': str(Path(os.getenv('REPORTS_DIR', Path(__file__).resolve().parent / 'reports')).resolve()),
            },
            'main_report_status': {
                'status': _main_report_job.get('status'),
                'running': _main_report_job.get('running'),
                'started_at': _main_report_job.get('started_at'),
                'ended_at': _main_report_job.get('ended_at'),
                'error': _main_report_job.get('error'),
                'retrain_found': _main_report_job.get('retrain_found'),
            },
            'recent_events': list(_report_events),
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