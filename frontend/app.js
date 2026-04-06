// app.js — CarbonWatch MLOps
const API = 'https://msml605.onrender.com';

// ── Global state ──────────────────────────────────────────────────────────────
let evtSource   = null;
let initialized = false;
let simRunning  = false;
let logCount    = 0;

let kdeStore   = {};
let kdeFeature = 'gas';
let ksHistory  = [];
let ksMonths   = [];

// Fixed axis ranges — set once from /api/initialize, never changed again
let PC1_RANGE       = null;
let INTENSITY_RANGE = null;
let KS_THRESHOLD    = 0.10;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const btnInit     = document.getElementById('btn-init');
const btnSim      = document.getElementById('btn-simulate');
const btnPause    = document.getElementById('btn-pause');
const btnReset    = document.getElementById('btn-reset');
const cfgSpeed    = document.getElementById('cfg-speed');
const cfgSpeedVal = document.getElementById('cfg-speed-val');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon   = document.getElementById('theme-icon');

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
  ['plot-pca', 'plot-pred', 'plot-kde', 'plot-ks'].forEach(id => {
    try { Plotly.relayout(id, u); } catch (_) {}
  });
}

// ── Plotly layout helper ──────────────────────────────────────────────────────
function makeLayout(xLabel, yLabel, xrange, yrange, extra) {
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
    ...(extra || {}),
  };
}

const CFG = { displayModeBar: false, responsive: true };

// ── PCA scatter ───────────────────────────────────────────────────────────────
function initPcaPlot(pca_x, pca_y, line_pc1, line_y) {
  Plotly.newPlot('plot-pca', [
    {
      x: pca_x, y: pca_y, mode: 'markers', type: 'scatter', name: 'Samples',
      marker: { color: '#4C72B0', size: 5, opacity: 0.75,
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

function updatePcaPoints(pca_x, pca_y) {
  Plotly.extendTraces('plot-pca', { x: [pca_x], y: [pca_y] }, [0]);
}

function updatePcaLine(line_pc1, line_y) {
  Plotly.restyle('plot-pca', { x: [line_pc1], y: [line_y] }, [1]);
}

// ── Predicted vs Actual ───────────────────────────────────────────────────────
function initPredPlot(pred_x, actual_y) {
  const lo = INTENSITY_RANGE[0], hi = INTENSITY_RANGE[1];
  Plotly.newPlot('plot-pred', [
    {
      x: pred_x, y: actual_y, mode: 'markers', type: 'scatter', name: 'Samples',
      marker: { color: '#5b8fff', size: 5, opacity: 0.75,
                line: { color: '#fff', width: 0.4 } },
    },
    {
      x: [lo, hi], y: [lo, hi], mode: 'lines', type: 'scatter',
      name: 'Ideal (y = x)', line: { color: '#3ecf8e', width: 1.8, dash: 'dot' },
    },
  ],
  makeLayout('Predicted (gCO₂/kWh)', 'Actual (gCO₂/kWh)', INTENSITY_RANGE, INTENSITY_RANGE),
  CFG);
}

function updatePredPoints(pred_x, actual_y) {
  Plotly.extendTraces('plot-pred', { x: [pred_x], y: [actual_y] }, [0]);
}

// ── KDE ───────────────────────────────────────────────────────────────────────
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

// ── KS sparkline ──────────────────────────────────────────────────────────────
function initKsPlot() {
  const t = plotlyTheme();
  Plotly.newPlot('plot-ks', [
    {
      x: [], y: [], mode: 'lines+markers', type: 'scatter',
      line: { color: '#5b8fff', width: 1.5 }, marker: { size: 4, color: '#5b8fff' },
    },
    {
      x: [], y: [KS_THRESHOLD, KS_THRESHOLD], mode: 'lines', type: 'scatter',
      line: { color: '#e05252', width: 1, dash: 'dot' },
    },
  ], {
    paper_bgcolor: t.paper, plot_bgcolor: t.plot,
    font: { family: "'DM Mono', monospace", color: t.text, size: 9 },
    xaxis: { gridcolor: t.grid, zeroline: false, color: t.text },
    yaxis: { gridcolor: t.grid, zeroline: false, color: t.text },
    margin: { l: 36, r: 8, t: 6, b: 28 },
    showlegend: false,
  }, CFG);
}

function updateKsSpark(month, ksVal) {
  ksMonths.push(month);
  ksHistory.push(ksVal);
  Plotly.restyle('plot-ks', { x: [ksMonths], y: [ksHistory] }, [0]);
  Plotly.restyle('plot-ks', {
    x: [[ksMonths[0], ksMonths[ksMonths.length - 1]]],
    y: [[KS_THRESHOLD, KS_THRESHOLD]],
  }, [1]);
}

// ── Pills ─────────────────────────────────────────────────────────────────────
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
  set('m-r2',   d.r2   != null ? d.r2.toFixed(4)   : null);
  set('m-rmse', d.rmse != null ? d.rmse.toFixed(2)  : null);
  set('m-ks',   d.ks_stat != null ? d.ks_stat.toFixed(4) : null);
  set('m-psi',  d.psi  != null ? d.psi.toFixed(4)   : null);
  set('m-version', d.model_version ? `v${d.model_version}` : null);
  if (d.retrained || d.month_idx === 0) {
    set('m-retrain', d.month);
    document.getElementById('version-badge').textContent = `v ${d.model_version}`;
  }
}

// ── Flash ─────────────────────────────────────────────────────────────────────
function flashDrift(id) {
  const el = document.getElementById(id);
  if (!el) return;
  el.classList.add('drift-flash');
  setTimeout(() => el.classList.remove('drift-flash'), 2200);
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

  try {
    const res  = await fetch(`${API}/api/initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!res.ok) {
      throw new Error(`Server returned ${res.status}: ${await res.text()}`);
    }

    const json = await res.json();
    const d    = json.data;

    // Validate that the backend returned the expected fields
    if (!d || !d.pc1_range || !d.intensity_range) {
      throw new Error(
        `Backend missing required fields. Got keys: ${Object.keys(d || {}).join(', ')}`
      );
    }

    // Lock in global axis ranges
    PC1_RANGE       = d.pc1_range;
    INTENSITY_RANGE = d.intensity_range;

    // Boot all four plots
    initPcaPlot(d.pca_x, d.pca_y, d.line_pc1, d.line_y);
    initPredPlot(d.pred_x, d.actual_y);
    initKdePlot();
    initKsPlot();

    // Seed KDE store
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

    // Lock controls during simulation
    document.querySelectorAll('.ctrl-group select, .ctrl-group input')
            .forEach(el => { el.disabled = true; });

    initialized       = true;
    btnSim.disabled   = false;
    btnReset.disabled = false;

  } catch (err) {
    addLog(`Init failed: ${err.message}`, 'drift');
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

    // Header
    updateHeader(d.month, d.month_idx + 1, d.total_months, 'RUNNING');

    // Scatter: add new points every tick
    updatePcaPoints(d.pca_x, d.pca_y);
    updatePredPoints(d.pred_x, d.actual_y);

    // Regression line: only update on retrain
    if (d.retrained && d.new_line) {
      updatePcaLine(d.new_line.line_pc1, d.new_line.line_y);
    }

    // KDE
    const feat = document.getElementById('cfg-feature-x').value;
    kdeStore[feat] = {
      ref_x: d.kde_ref_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_cur_x, cur_y: d.kde_cur_y,
    };
    if (kdeFeature === feat) updateKdePlot(kdeFeature);

    // KS sparkline
    updateKsSpark(d.month, d.ks_stat);

    // Pills + dashboard
    if (d.drift_pills) updatePills(d.drift_pills);
    updateDash(d);

    // Drift indicators
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

    // Log
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
  ksHistory = []; ksMonths = []; kdeStore = {};
  PC1_RANGE = null; INTENSITY_RANGE = null;

  ['plot-pca', 'plot-pred', 'plot-kde', 'plot-ks'].forEach(id => Plotly.purge(id));

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