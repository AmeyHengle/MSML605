// app.js — CarbonWatch MLOps frontend
// Communicates with FastAPI backend via REST + SSE (EventSource).

// const API = 'http://localhost:8000';
const API = 'https://msml605.onrender.com';

// ── State ─────────────────────────────────────────────────────────────────────
let evtSource   = null;
let initialized = false;
let simRunning  = false;
let logCount    = 0;

let scatterPC1 = [], scatterY = [];
let ksHistory  = [], ksMonths  = [];
let kdeStore   = {};
let kdeFeature = 'gas';
let axisRanges = null;
let threshold  = 0.10;

// ── DOM ───────────────────────────────────────────────────────────────────────
const btnInit  = document.getElementById('btn-init');
const btnSim   = document.getElementById('btn-simulate');
const btnPause = document.getElementById('btn-pause');
const btnReset = document.getElementById('btn-reset');
const cfgSpeed = document.getElementById('cfg-speed');

cfgSpeed.addEventListener('input', () => {
  document.getElementById('cfg-speed-val').textContent =
    parseFloat(cfgSpeed.value).toFixed(1) + 's';
});

// ── Theme toggle ──────────────────────────────────────────────────────────────
document.getElementById('theme-toggle').addEventListener('click', () => {
  const html   = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  html.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('theme-icon').textContent = isDark ? '☾' : '☀';
  if (initialized) relayoutAllPlots();
});

function getPlotColors() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    grid: dark ? '#1e2233' : '#e5e7ef',
    text: dark ? '#6b7394' : '#8890aa',
  };
}

function relayoutAllPlots() {
  const c      = getPlotColors();
  const layout = {
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    'xaxis.gridcolor': c.grid, 'xaxis.color': c.text,
    'yaxis.gridcolor': c.grid, 'yaxis.color': c.text,
    'font.color': c.text,
  };
  ['plot-scatter', 'plot-pva', 'plot-kde', 'plot-ks'].forEach(id => {
    try { Plotly.relayout(id, layout); } catch(e) {}
  });
}

// ── Plotly layout factory ─────────────────────────────────────────────────────
function mkLayout(xLabel, yLabel, extra = {}) {
  const c = getPlotColors();
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    font:   { family: "'DM Mono', monospace", color: c.text, size: 10 },
    xaxis:  { title: { text: xLabel, font: { size: 10 } },
               gridcolor: c.grid, zeroline: false, color: c.text },
    yaxis:  { title: { text: yLabel, font: { size: 10 } },
               gridcolor: c.grid, zeroline: false, color: c.text },
    margin:     { l: 50, r: 12, t: 10, b: 40 },
    showlegend: true,
    legend:     { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 }, x: 0, y: 1 },
    ...extra
  };
}

const plotCfg = { displayModeBar: false, responsive: true };

// ── PCA scatter ───────────────────────────────────────────────────────────────
function initScatterPlot(initPC1, initY, linePC1, lineY) {
  scatterPC1 = [...initPC1];
  scatterY   = [...initY];
  Plotly.newPlot('plot-scatter', [
    { x: scatterPC1, y: scatterY,
      mode: 'markers', type: 'scatter', name: 'Samples',
      marker: { color: '#4C72B0', size: 5, opacity: 0.75,
                line: { color: '#fff', width: 0.4 } } },
    { x: linePC1, y: lineY,
      mode: 'lines', type: 'scatter', name: 'Fit (v1)',
      line: { color: '#e05252', width: 2.2 } },
  ], mkLayout('PC1 — grid energy mix', 'Forecast intensity (gCO₂/kWh)', {
    xaxis: { range: [axisRanges.pc1_min, axisRanges.pc1_max],
             gridcolor: getPlotColors().grid, zeroline: false,
             color: getPlotColors().text,
             title: { text: 'PC1 — grid energy mix', font: { size: 10 } } },
    yaxis: { range: [axisRanges.y_min, axisRanges.y_max],
             gridcolor: getPlotColors().grid, zeroline: false,
             color: getPlotColors().text,
             title: { text: 'Forecast intensity (gCO₂/kWh)', font: { size: 10 } } },
  }), plotCfg);
}

function updateScatterPoints(newPC1, newY) {
  scatterPC1 = scatterPC1.concat(newPC1);
  scatterY   = scatterY.concat(newY);
  Plotly.extendTraces('plot-scatter', { x: [newPC1], y: [newY] }, [0]);
}

function updateRegressionLine(linePC1, lineY, version) {
  Plotly.restyle('plot-scatter',
    { x: [linePC1], y: [lineY], name: [`Fit (v${version})`] }, [1]);
}

// ── Predicted vs Actual ───────────────────────────────────────────────────────
function initPvaPlot(predY, actualY) {
  const diag = [axisRanges.y_min, axisRanges.y_max];
  Plotly.newPlot('plot-pva', [
    { x: diag, y: diag,
      mode: 'lines', type: 'scatter', name: 'Ideal (perfect model)',
      line: { color: '#3ecf8e', width: 1.5, dash: 'dot' } },
    { x: predY, y: actualY,
      mode: 'markers', type: 'scatter', name: 'Current period',
      marker: { color: '#F0997B', size: 4, opacity: 0.65,
                line: { color: '#D85A30', width: 0.4 } } },
  ], mkLayout('Predicted (gCO₂/kWh)', 'Actual (gCO₂/kWh)', {
    xaxis: { range: diag, gridcolor: getPlotColors().grid,
             zeroline: false, color: getPlotColors().text,
             title: { text: 'Predicted (gCO₂/kWh)', font: { size: 10 } } },
    yaxis: { range: diag, gridcolor: getPlotColors().grid,
             zeroline: false, color: getPlotColors().text,
             title: { text: 'Actual (gCO₂/kWh)', font: { size: 10 } } },
  }), plotCfg);
}

function updatePvaPoints(predY, actualY) {
  Plotly.restyle('plot-pva', { x: [predY], y: [actualY] }, [1]);
}

// ── KDE ───────────────────────────────────────────────────────────────────────
function initKdePlot() {
  Plotly.newPlot('plot-kde', [
    { x: [], y: [], mode: 'lines', type: 'scatter', name: 'Reference',
      fill: 'tozeroy', fillcolor: 'rgba(91,143,255,0.15)',
      line: { color: '#5b8fff', width: 2 } },
    { x: [], y: [], mode: 'lines', type: 'scatter', name: 'Current',
      fill: 'tozeroy', fillcolor: 'rgba(224,82,82,0.12)',
      line: { color: '#e05252', width: 2 } },
  ], mkLayout('Feature value (%)', 'Density'), plotCfg);
}

function updateKdePlot(feat) {
  const d = kdeStore[feat];
  if (!d) return;
  Plotly.restyle('plot-kde', {
    x: [d.ref_x, d.cur_x], y: [d.ref_y, d.cur_y],
  }, [0, 1]);
}

document.getElementById('kde-feature-select').addEventListener('change', e => {
  kdeFeature = e.target.value;
  updateKdePlot(kdeFeature);
});

// ── KS sparkline ──────────────────────────────────────────────────────────────
function initKsPlot() {
  Plotly.newPlot('plot-ks', [
    { x: [], y: [], mode: 'lines+markers', type: 'scatter', name: 'KS',
      line: { color: '#5b8fff', width: 1.5 },
      marker: { size: 4, color: '#5b8fff' } },
    { x: [], y: [threshold, threshold], mode: 'lines', type: 'scatter',
      name: 'Threshold', line: { color: '#e05252', width: 1, dash: 'dot' } },
  ], { ...mkLayout('Month', 'KS'), showlegend: false,
       margin: { l: 36, r: 8, t: 6, b: 28 } }, plotCfg);
}

function updateKsSpark(month, ksVal) {
  ksMonths.push(month);
  ksHistory.push(ksVal);
  Plotly.restyle('plot-ks', { x: [ksMonths], y: [ksHistory] }, [0]);
  Plotly.restyle('plot-ks', {
    x: [[ksMonths[0], ksMonths[ksMonths.length - 1]]],
    y: [[threshold, threshold]],
  }, [1]);
}

// ── Pills ─────────────────────────────────────────────────────────────────────
function updatePills(pills) {
  document.querySelectorAll('.feat-pill').forEach(el => {
    el.className = `feat-pill ${pills[el.dataset.feat] || 'none'}`;
  });
}

// ── Log ───────────────────────────────────────────────────────────────────────
function addLog(msg, type = 'ok') {
  logCount++;
  const body  = document.getElementById('log-body');
  const entry = document.createElement('div');
  entry.className = `log-entry log-entry--${type}`;
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  entry.innerHTML = `<span class="log-ts">${ts}</span>${msg}`;
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;
  document.getElementById('log-count').textContent = `${logCount} events`;
}

// ── Header & dashboard ────────────────────────────────────────────────────────
function updateHeader(month, idx, total, status) {
  document.getElementById('hdr-month').textContent    = month;
  document.getElementById('hdr-progress').textContent = `${idx} / ${total}`;
  document.getElementById('hdr-status').textContent   = status;
}

function updateDash(d) {
  document.getElementById('m-r2').textContent   = d.r2    ?? '—';
  document.getElementById('m-rmse').textContent = d.rmse  ?? '—';
  document.getElementById('m-ks').textContent   = d.ks_stat ?? '—';
  document.getElementById('m-psi').textContent  = d.psi   ?? '—';
  document.getElementById('m-version').textContent = d.model_version ?? '—';
  if (d.retrained || d.model_version === 1) {
    document.getElementById('m-retrain').textContent = d.month ?? '—';
    document.getElementById('version-badge').textContent = `v ${d.model_version}`;
  }
}

function flashDrift(panelId) {
  const el = document.getElementById(panelId);
  el.classList.add('drift-flash');
  setTimeout(() => el.classList.remove('drift-flash'), 2200);
}

// ── Initialize ────────────────────────────────────────────────────────────────
btnInit.addEventListener('click', async () => {
  btnInit.disabled = true;
  addLog('Initializing pipeline…', 'idle');
  updateHeader('—', 0, '—', 'INITIALIZING');
  threshold = parseFloat(document.getElementById('cfg-threshold').value);

  const config = {
    feature_x:    document.getElementById('cfg-feature-x').value,
    feature_y:    'forecast_intensity',
    ks_threshold: threshold,
    n_init:       parseInt(document.getElementById('cfg-n-init').value),
    n_monthly:    parseInt(document.getElementById('cfg-n-monthly').value),
    speed:        parseFloat(cfgSpeed.value),
  };

  try {
    const res  = await fetch(`${API}/api/initialize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
    const json = await res.json();
    const d    = json.data;

    axisRanges = d.axis_ranges;

    initScatterPlot(d.scatter_pc1, d.scatter_y, d.line_pc1, d.line_y);
    initPvaPlot(d.pred_y, d.actual_y);
    initKdePlot();
    initKsPlot();

    kdeStore[config.feature_x] = {
      ref_x: d.kde_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_x, cur_y: d.kde_ref_y,
    };
    updateKdePlot(kdeFeature);

    updateDash({ ...d, retrained: true });
    updateHeader(d.month, 1, d.total_months, 'READY');
    addLog(`Baseline model trained  |  ${d.month}  |  R²=${d.r2}  RMSE=${d.rmse}`, 'retrain');

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

// ── Simulate ──────────────────────────────────────────────────────────────────
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
    updateScatterPoints(d.new_scatter_pc1, d.new_scatter_y);

    if (d.retrained && d.new_line)
      updateRegressionLine(d.new_line.line_pc1, d.new_line.line_y, d.model_version);

    updatePvaPoints(d.pred_y, d.actual_y);

    const primaryFeat = document.getElementById('cfg-feature-x').value;
    kdeStore[primaryFeat] = {
      ref_x: d.kde_ref_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_cur_x, cur_y: d.kde_cur_y,
    };
    if (kdeFeature === primaryFeat) updateKdePlot(kdeFeature);

    updateKsSpark(d.month, d.ks_stat);
    if (d.drift_pills) updatePills(d.drift_pills);

    const pillEl = document.getElementById('pill-drift');
    if (d.drift_detected) {
      flashDrift('panel-scatter'); flashDrift('panel-pva'); flashDrift('panel-kde');
      pillEl.className   = 'pill drifting';
      pillEl.textContent = `DRIFT: ${d.ks_stat}`;
    } else {
      pillEl.className   = 'pill ok';
      pillEl.textContent = `KS: ${d.ks_stat}`;
    }

    updateDash(d);
    addLog(`[${d.month}]  ${d.log}`,
           d.retrained ? 'retrain' : d.drift_detected ? 'drift' : 'ok');

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

// ── Pause / Resume ────────────────────────────────────────────────────────────
btnPause.addEventListener('click', async () => {
  if (simRunning) {
    await fetch(`${API}/api/pause`, { method: 'POST' });
    btnPause.textContent = '▶ Resume';
    simRunning = false;
    addLog('Paused.', 'idle');
  } else {
    await fetch(`${API}/api/resume`, { method: 'POST' });
    btnPause.textContent = '⏸ Pause';
    simRunning = true;
    addLog('Resumed.', 'ok');
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────────
btnReset.addEventListener('click', async () => {
  if (evtSource) { evtSource.close(); evtSource = null; }
  await fetch(`${API}/api/reset`, { method: 'POST' });

  initialized = false; simRunning = false; logCount = 0;
  scatterPC1 = []; scatterY = []; ksHistory = []; ksMonths = [];
  kdeStore = {}; axisRanges = null;

  ['plot-scatter', 'plot-pva', 'plot-kde', 'plot-ks'].forEach(id => {
    try { Plotly.purge(id); } catch(e) {}
  });

  document.getElementById('log-body').innerHTML =
    '<div class="log-entry log-entry--idle">Awaiting initialization…</div>';
  document.getElementById('log-count').textContent = '0 events';
  updateHeader('—', 0, '—', 'IDLE');

  document.getElementById('pill-drift').className   = 'pill';
  document.getElementById('pill-drift').textContent = 'DRIFT: —';
  document.getElementById('version-badge').textContent = 'v —';
  ['m-r2','m-rmse','m-ks','m-psi','m-retrain','m-version'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });
  document.querySelectorAll('.feat-pill').forEach(el => { el.className = 'feat-pill none'; });
  document.querySelectorAll('.ctrl-group select, .ctrl-group input').forEach(el => {
    el.disabled = false;
  });

  btnInit.disabled = false; btnSim.disabled = true;
  btnPause.disabled = true; btnReset.disabled = true;
  btnPause.textContent = '⏸ Pause';
  addLog('Pipeline reset.', 'idle');
});