// app.js — CarbonWatch MLOps pipeline page (index.html only)
// monitoring.html has its own self-contained script and does NOT load this file.

const API = 'https://msml605.onrender.com';

// ── State ─────────────────────────────────────────────────────────────────────
let evtSource   = null;
let initialized = false;
let simRunning  = false;
let logCount    = 0;

let kdeStore   = {};
let kdeFeature = 'coal';
let driftLabels = [];
let ksHistory   = [];
let psiHistory  = [];
let retrainPeaks = [];
let pcaTrainingX = [];
let pcaTrainingY = [];
let pcaIncomingX = [];
let pcaIncomingY = [];

let PC1_RANGE       = null;
let INTENSITY_RANGE = null;
let KS_THRESHOLD    = 0.10;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const btnInit     = document.getElementById('btn-init');
const btnSim      = document.getElementById('btn-simulate');
const btnPause    = document.getElementById('btn-pause');
const btnReset    = document.getElementById('btn-reset');
const btnAgentRun = document.getElementById('btn-agent-run');
const btnAgentToggle = document.getElementById('btn-agent-toggle');
const btnAgentRefresh = document.getElementById('btn-agent-refresh');
const cfgSpeed    = document.getElementById('cfg-speed');
const cfgSpeedVal = document.getElementById('cfg-speed-val');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon   = document.getElementById('theme-icon');
const agentStatusEl = document.getElementById('agent-status');
const agentBadge = document.getElementById('agent-badge');
const agentLogBody = document.getElementById('agent-log-body');

let agentPollTimer = null;
let agentLogsVisible = true;
let lastAgentLogCount = 0;

cfgSpeed.addEventListener('input', () => {
  cfgSpeedVal.textContent = parseFloat(cfgSpeed.value).toFixed(1) + 's';
});

// ── Theme ─────────────────────────────────────────────────────────────────────
themeToggle.addEventListener('click', () => {
  const html   = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  themeIcon.textContent = isDark ? '☽' : '☀';
  reapplyPlotlyTheme();
});

function plotlyTheme() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    paper: 'rgba(0,0,0,0)',
    plot:  'rgba(0,0,0,0)',
    grid:  dark ? '#1e2233' : '#e5e7ef',
    text:  dark ? '#6b7394' : '#8890aa',
  };
}

function reapplyPlotlyTheme() {
  const t = plotlyTheme();
  const u = {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot, 'font.color': t.text,
    'xaxis.gridcolor': t.grid, 'xaxis.color': t.text,
    'yaxis.gridcolor': t.grid, 'yaxis.color': t.text,
  };
  ['plot-pca', 'plot-kde', 'plot-drift'].forEach(id => {
    try { Plotly.relayout(id, u); } catch (_) {}
  });
}

// ── Plotly layout factory ─────────────────────────────────────────────────────
function makeLayout(xLabel, yLabel, xrange, yrange) {
  const t = plotlyTheme();
  return {
    paper_bgcolor: t.paper,
    plot_bgcolor:  t.plot,
    font:   { family: "'DM Mono', monospace", color: t.text, size: 10 },
    xaxis:  { title: { text: xLabel, font: { size: 10 } },
               gridcolor: t.grid, zeroline: false, color: t.text,
               range: xrange || null },
    yaxis:  { title: { text: yLabel, font: { size: 10 } },
               gridcolor: t.grid, zeroline: false, color: t.text,
               range: yrange || null },
    margin: { l: 52, r: 16, t: 12, b: 44 },
    showlegend: true,
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 }, x: 0, y: 1 },
  };
}

const CFG = { displayModeBar: false, responsive: true };

// ── PCA scatter ───────────────────────────────────────────────────────────────
function initPcaPlot(pca_x, pca_y, line_pc1, line_y) {
  pcaTrainingX = [...pca_x];
  pcaTrainingY = [...pca_y];
  pcaIncomingX = [];
  pcaIncomingY = [];
  Plotly.newPlot('plot-pca', [
    {
      x: pca_x, y: pca_y, mode: 'markers', type: 'scatter', name: 'Model training data',
      marker: { color: '#4C72B0', size: 5, opacity: 0.75,
                line: { color: '#fff', width: 0.4 } },
    },
    {
      x: [], y: [], mode: 'markers', type: 'scatter', name: 'New incoming data',
      marker: { color: '#f5a623', size: 5, opacity: 0.82,
                line: { color: '#fff', width: 0.4 } },
    },
    {
      x: line_pc1, y: line_y, mode: 'lines', type: 'scatter', name: 'Model fit',
      line: { color: '#e05252', width: 2.2 },
    },
  ],
  makeLayout('PC1 (energy mix)', 'Intensity (gCO₂/kWh)', PC1_RANGE, INTENSITY_RANGE),
  CFG);
}

function updatePcaPoints(pca_x, pca_y, retrained) {
  if (retrained) {
    // On retrain, promote all incoming points plus current tick to "training data"
    // and reset incoming trace to show post-retrain production drift clearly.
    pcaTrainingX = pcaTrainingX.concat(pcaIncomingX, pca_x);
    pcaTrainingY = pcaTrainingY.concat(pcaIncomingY, pca_y);
    pcaIncomingX = [];
    pcaIncomingY = [];
    Plotly.restyle('plot-pca', { x: [pcaTrainingX], y: [pcaTrainingY] }, [0]);
    Plotly.restyle('plot-pca', { x: [[]], y: [[]] }, [1]);
    return;
  }
  pcaIncomingX = pcaIncomingX.concat(pca_x);
  pcaIncomingY = pcaIncomingY.concat(pca_y);
  Plotly.extendTraces('plot-pca', { x: [pca_x], y: [pca_y] }, [1]);
}

function updatePcaLine(line_pc1, line_y) {
  Plotly.restyle('plot-pca', { x: [line_pc1], y: [line_y] }, [2]);
}

// ── KDE plot ──────────────────────────────────────────────────────────────────
function initKdePlot() {
  Plotly.newPlot('plot-kde', [
    {
      x: [], y: [], mode: 'lines', type: 'scatter', name: 'Reference',
      fill: 'tozeroy', fillcolor: 'rgba(91,143,255,0.15)',
      line: { color: '#5b8fff', width: 2 },
    },
    {
      x: [], y: [], mode: 'lines', type: 'scatter', name: 'Current',
      fill: 'tozeroy', fillcolor: 'rgba(224,82,82,0.12)',
      line: { color: '#e05252', width: 2 },
    },
  ],
  makeLayout('Feature value (%)', 'Density', null, null),
  CFG);
}

function updateKdePlot(feat) {
  const d = kdeStore[feat];
  if (!d) return;
  Plotly.restyle('plot-kde', { x: [d.ref_x, d.cur_x], y: [d.ref_y, d.cur_y] }, [0, 1]);
}

document.getElementById('kde-feature-select').addEventListener('change', e => {
  kdeFeature = e.target.value;
  updateKdePlot(kdeFeature);
});

// ── Drift metrics history (KS + PSI) ─────────────────────────────────────────
function initDriftPlot() {
  const t = plotlyTheme();
  Plotly.newPlot('plot-drift', [
    {
      x: [], y: [], mode: 'lines+markers', type: 'scatter',
      name: 'KS',
      line: { color: '#5b8fff', width: 1.5 }, marker: { size: 4, color: '#5b8fff' },
    },
    {
      x: [], y: [], mode: 'lines+markers', type: 'scatter',
      name: 'PSI',
      line: { color: '#f5a623', width: 1.5 }, marker: { size: 4, color: '#f5a623' },
    },
    {
      x: [], y: [], mode: 'markers', type: 'scatter',
      name: 'Retrain trigger',
      marker: { size: 8, color: '#e05252', symbol: 'diamond' },
    },
    {
      x: [], y: [], mode: 'lines', type: 'scatter',
      name: 'KS threshold',
      line: { color: '#e05252', width: 1, dash: 'dot' },
    },
    {
      x: [], y: [], mode: 'lines', type: 'scatter',
      name: 'PSI threshold',
      line: { color: '#ba7517', width: 1, dash: 'dot' },
    },
  ], {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot,
    font: { family: "'DM Mono', monospace", color: t.text, size: 9 },
    xaxis: { gridcolor: t.grid, zeroline: false, color: t.text },
    yaxis: { gridcolor: t.grid, zeroline: false, color: t.text },
    margin: { l: 36, r: 8, t: 6, b: 28 },
    showlegend: true,
    legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 8 }, x: 0, y: 1 },
  }, CFG);
}

function updateDriftHistory(periodLabel, ksVal, psiVal, retrained) {
  driftLabels.push(periodLabel);
  ksHistory.push(ksVal);
  psiHistory.push(psiVal);

  if (retrained) {
    retrainPeaks.push(Math.max(ksVal, psiVal));
  } else {
    retrainPeaks.push(null);
  }

  Plotly.restyle('plot-drift', { x: [driftLabels], y: [ksHistory] }, [0]);
  Plotly.restyle('plot-drift', { x: [driftLabels], y: [psiHistory] }, [1]);
  Plotly.restyle('plot-drift', { x: [driftLabels], y: [retrainPeaks] }, [2]);
  Plotly.restyle('plot-drift', {
    x: [driftLabels],
    y: [driftLabels.map(() => KS_THRESHOLD)],
  }, [3]);
  Plotly.restyle('plot-drift', {
    x: [driftLabels],
    y: [driftLabels.map(() => 0.25)],
  }, [4]);
}

// ── Drift pills ───────────────────────────────────────────────────────────────
function updatePills(pills) {
  document.querySelectorAll('.feat-pill').forEach(el => {
    el.className = `feat-pill ${pills[el.dataset.feat] || 'none'}`;
  });
}

// ── Log ───────────────────────────────────────────────────────────────────────
function addLog(msg, type) {
  logCount++;
  const body  = document.getElementById('log-body');
  const entry = document.createElement('div');
  entry.className = `log-entry log-entry--${type || 'ok'}`;
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  entry.innerHTML = `<span class="log-ts">${ts}</span>${msg}`;
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;
  document.getElementById('log-count').textContent = `${logCount} events`;
}

// ── Header / dashboard ────────────────────────────────────────────────────────
function updateHeader(month, idx, total, status) {
  document.getElementById('hdr-month').textContent    = month;
  document.getElementById('hdr-progress').textContent = `${idx} / ${total}`;
  document.getElementById('hdr-status').textContent   = status;
}

function updateDash(d) {
  const set = (id, v) => { document.getElementById(id).textContent = v ?? '—'; };
  set('m-r2',   d.r2    != null ? d.r2.toFixed(4)    : null);
  set('m-rmse', d.rmse  != null ? d.rmse.toFixed(2)  : null);
  set('m-ks',   d.ks_stat != null ? d.ks_stat.toFixed(4) : null);
  set('m-psi',  d.psi   != null ? d.psi.toFixed(4)   : null);
  set('m-version', d.model_version ? `v${d.model_version}` : null);
  if (d.retrained || d.month_idx === 0) {
    set('m-retrain', d.month);
    document.getElementById('version-badge').textContent = `v ${d.model_version}`;
  }
}

// ── Drift flash ───────────────────────────────────────────────────────────────
function flashDrift(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('drift-flash');
  setTimeout(() => el.classList.remove('drift-flash'), 2200);
}

function renderAgentStatus(s) {
  if (!agentStatusEl || !s) return;
  if (agentBadge) {
    agentBadge.className = 'pill';
    agentBadge.textContent = (s.status || 'idle').toUpperCase();
  }
  if (!s.available) {
    agentStatusEl.textContent = 'Agent: unavailable';
    btnAgentRun.disabled = true;
    if (agentBadge) {
      agentBadge.className = 'pill';
      agentBadge.textContent = 'UNAVAILABLE';
    }
    return;
  }

  if (s.running) {
    agentStatusEl.textContent = `Agent: running (${s.job_id?.slice(0, 8) || 'n/a'})`;
    btnAgentRun.disabled = true;
    if (agentBadge) {
      agentBadge.className = 'pill drifting';
      agentBadge.textContent = 'RUNNING';
    }
    return;
  }

  btnAgentRun.disabled = false;
  if (s.status === 'succeeded') {
    agentStatusEl.textContent = `Agent: succeeded (exit ${s.exit_code})`;
    if (agentBadge) {
      agentBadge.className = 'pill ok';
      agentBadge.textContent = 'SUCCEEDED';
    }
  } else if (s.status === 'failed') {
    agentStatusEl.textContent = `Agent: failed (${s.error || 'check logs'})`;
    if (agentBadge) {
      agentBadge.className = 'pill drifting';
      agentBadge.textContent = 'FAILED';
    }
  } else {
    agentStatusEl.textContent = 'Agent: idle';
    if (agentBadge) {
      agentBadge.className = 'pill';
      agentBadge.textContent = 'IDLE';
    }
  }
}

async function fetchAgentStatus() {
  try {
    const res = await fetch(`${API}/api/agent/status`);
    if (!res.ok) return;
    const status = await res.json();
    renderAgentStatus(status);
    if (status.running || status.log_lines !== lastAgentLogCount) {
      fetchAgentLogs();
    }
  } catch (_) {}
}

function renderAgentLogs(payload) {
  if (!agentLogBody || !payload) return;
  const logs = payload.logs || [];
  lastAgentLogCount = logs.length;

  if (logs.length === 0) {
    agentLogBody.innerHTML = '<div class="log-entry log-entry--idle">No agent logs yet.</div>';
    return;
  }

  agentLogBody.innerHTML = logs.map(line =>
    `<div class="log-entry log-entry--ok">${line.replaceAll('<', '&lt;').replaceAll('>', '&gt;')}</div>`
  ).join('');
  agentLogBody.scrollTop = agentLogBody.scrollHeight;
}

async function fetchAgentLogs() {
  try {
    const res = await fetch(`${API}/api/agent/logs`);
    if (!res.ok) return;
    const payload = await res.json();
    renderAgentLogs(payload);
  } catch (_) {}
}

// ── INITIALIZE ────────────────────────────────────────────────────────────────
btnInit.addEventListener('click', async () => {
  btnInit.disabled = true;
  addLog('Initializing pipeline…', 'idle');
  updateHeader('—', 0, '—', 'INITIALIZING');

  KS_THRESHOLD = parseFloat(document.getElementById('cfg-threshold').value);

  const config = {
    feature_x:    document.getElementById('cfg-feature-x').value,
    feature_y:    'forecast_intensity',
    ks_threshold: KS_THRESHOLD,
    n_init:       parseInt(document.getElementById('cfg-n-init').value),
    n_monthly:    parseInt(document.getElementById('cfg-n-monthly').value),
    speed:        parseFloat(document.getElementById('cfg-speed').value),
  };

  // AbortController for broad browser compatibility
  const controller = new AbortController();
  const tid = setTimeout(() => controller.abort(), 120000);

  try {
    const res = await fetch(`${API}/api/initialize`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(config),
      signal:  controller.signal,
    });
    clearTimeout(tid);

    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const json = await res.json();
    const d    = json.data;

    if (!d || !d.pc1_range || !d.intensity_range) {
      throw new Error(
        `Backend missing required fields. Got: ${Object.keys(d || {}).join(', ')}`
      );
    }

    PC1_RANGE       = d.pc1_range;
    INTENSITY_RANGE = d.intensity_range;

    initPcaPlot(d.pca_x, d.pca_y, d.line_pc1, d.line_y);
    initKdePlot();
    initDriftPlot();

    kdeStore[config.feature_x] = {
      ref_x: d.kde_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_x, cur_y: d.kde_ref_y,
    };
    updateKdePlot(kdeFeature);

    updateDash({ ...d, retrained: false, month_idx: 0 });
    updateHeader(d.month, 1, d.total_months, 'READY');

    addLog(
      `Baseline model trained on ${d.month}  |  R²=${d.r2}  RMSE=${d.rmse}`,
      'retrain'
    );

    document.querySelectorAll('.ctrl-group select, .ctrl-group input')
            .forEach(el => { el.disabled = true; });

    initialized       = true;
    btnSim.disabled   = false;
    btnReset.disabled = false;

  } catch (err) {
    clearTimeout(tid);
    const msg = err.name === 'AbortError' ? 'Request timed out after 120s' : err.message;
    addLog(`Init failed: ${msg}`, 'drift');
    btnInit.disabled = false;
  }
});

// ── SIMULATE ──────────────────────────────────────────────────────────────────
btnSim.addEventListener('click', () => {
  if (!initialized) return;
  btnSim.disabled   = true;
  btnPause.disabled = false;
  simRunning        = true;
  updateHeader('—', '—', '—', 'RUNNING');

  evtSource = new EventSource(`${API}/api/simulate`);

  evtSource.onmessage = (e) => {
    const d = JSON.parse(e.data);
    if (d.error)  { addLog(d.error, 'drift'); evtSource.close(); return; }
    if (d.paused) return;
    if (d.done && !d.month) {
      addLog('Simulation complete.', 'done');
      updateHeader('—', '—', '—', 'DONE');
      evtSource.close(); btnPause.disabled = true; return;
    }

    updateHeader(d.month, d.month_idx + 1, d.total_months, 'RUNNING');

    updatePcaPoints(d.pca_x, d.pca_y, d.retrained);

    if (d.retrained && d.new_line) {
      updatePcaLine(d.new_line.line_pc1, d.new_line.line_y);
    }

    const feat = document.getElementById('cfg-feature-x').value;
    kdeStore[feat] = {
      ref_x: d.kde_ref_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_cur_x, cur_y: d.kde_cur_y,
    };
    if (kdeFeature === feat) updateKdePlot(kdeFeature);

    updateDriftHistory(d.month, d.ks_stat, d.psi, d.retrained);
    if (d.drift_pills) updatePills(d.drift_pills);
    updateDash(d);

    if (d.drift_detected) {
      flashDrift('panel-pca');
      flashDrift('panel-kde');
      flashDrift('panel-pred');
      document.getElementById('pill-drift').className   = 'pill drifting';
      document.getElementById('pill-drift').textContent = `DRIFT: ${d.ks_stat}`;
    } else {
      document.getElementById('pill-drift').className   = 'pill ok';
      document.getElementById('pill-drift').textContent = `KS: ${d.ks_stat}`;
    }

    const logType = d.retrained ? 'retrain' : d.drift_detected ? 'drift' : 'ok';
    addLog(`[${d.month}]  ${d.log}`, logType);

    if (d.done) {
      addLog('All months processed — simulation complete.', 'done');
      updateHeader(d.month, d.total_months, d.total_months, 'DONE');
      evtSource.close(); btnPause.disabled = true;
    }
  };

  evtSource.onerror = () => {
    addLog('SSE connection lost.', 'drift');
    evtSource.close(); btnPause.disabled = true;
    updateHeader('—', '—', '—', 'ERROR');
  };
});

// ── PAUSE / RESUME ────────────────────────────────────────────────────────────
btnPause.addEventListener('click', async () => {
  if (simRunning) {
    await fetch(`${API}/api/pause`, { method: 'POST' });
    btnPause.textContent = '▶ Resume';
    simRunning = false;
    document.getElementById('hdr-status').textContent = 'PAUSED';
    addLog('Simulation paused.', 'idle');
  } else {
    await fetch(`${API}/api/resume`, { method: 'POST' });
    btnPause.textContent = '⏸ Pause';
    simRunning = true;
    document.getElementById('hdr-status').textContent = 'RUNNING';
    addLog('Simulation resumed.', 'ok');
  }
});

// ── RESET ─────────────────────────────────────────────────────────────────────
btnReset.addEventListener('click', async () => {
  if (evtSource) { evtSource.close(); evtSource = null; }
  await fetch(`${API}/api/reset`, { method: 'POST' });

  initialized = false; simRunning = false; logCount = 0;
  driftLabels = []; ksHistory = []; psiHistory = []; retrainPeaks = []; kdeStore = {};
  pcaTrainingX = []; pcaTrainingY = []; pcaIncomingX = []; pcaIncomingY = [];
  PC1_RANGE = null; INTENSITY_RANGE = null;

  ['plot-pca', 'plot-kde', 'plot-drift'].forEach(id => Plotly.purge(id));

  document.getElementById('log-body').innerHTML =
    '<div class="log-entry log-entry--idle">Awaiting initialization…</div>';
  document.getElementById('log-count').textContent = '0 events';
  document.querySelectorAll('.feat-pill').forEach(el => el.className = 'feat-pill none');
  updateHeader('—', 0, '—', 'IDLE');
  document.getElementById('pill-drift').className   = 'pill';
  document.getElementById('pill-drift').textContent = 'DRIFT: —';
  document.getElementById('version-badge').textContent = 'v —';
  ['m-r2','m-rmse','m-ks','m-psi','m-retrain','m-version'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });

  document.querySelectorAll('.ctrl-group select, .ctrl-group input')
          .forEach(el => { el.disabled = false; });

  btnInit.disabled  = false;
  btnSim.disabled   = true;
  btnPause.disabled = true;
  btnReset.disabled = true;
  btnPause.textContent = '⏸ Pause';
  addLog('Pipeline reset.', 'idle');
});

btnAgentRun.addEventListener('click', async () => {
  btnAgentRun.disabled = true;
  try {
    const res = await fetch(`${API}/api/agent/run`, { method: 'POST' });
    const out = await res.json();
    if (out.status === 'started') {
      addLog(`Agent run started (${out.job_id.slice(0, 8)})`, 'ok');
    } else if (out.status === 'already_running') {
      addLog(`Agent already running (${out.job_id.slice(0, 8)})`, 'idle');
    } else {
      addLog('Agent run request sent.', 'idle');
    }
  } catch (err) {
    addLog(`Agent run failed: ${err.message}`, 'drift');
  } finally {
    fetchAgentStatus();
  }
});

btnAgentToggle.addEventListener('click', () => {
  agentLogsVisible = !agentLogsVisible;
  agentLogBody.style.display = agentLogsVisible ? 'block' : 'none';
  btnAgentToggle.textContent = agentLogsVisible ? 'Hide' : 'Show';
});

btnAgentRefresh.addEventListener('click', () => {
  fetchAgentLogs();
});

fetchAgentStatus();
fetchAgentLogs();
agentPollTimer = setInterval(fetchAgentStatus, 3000);