// app.js — CarbonWatch MLOps pipeline page (simulation.html only)
// index.html (dashboard) and monitoring.html have their own scripts.

const API = 'https://msml605.onrender.com';

// ── State ─────────────────────────────────────────────────────────────────────
let evtSource   = null;
let initialized = false;
let simRunning  = false;
let logCount    = 0;

let kdeStore   = {};
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
const KS_THRESHOLD  = 0.30;
const FIXED_SPEED_S = 1.0;
const KDE_FEATURES  = ['coal', 'gas', 'wind'];

// ── DOM refs ──────────────────────────────────────────────────────────────────
const btnInit     = document.getElementById('btn-init');
const btnSim      = document.getElementById('btn-simulate');
const btnPause    = document.getElementById('btn-pause');
const btnReset    = document.getElementById('btn-reset');
const themeToggle = document.getElementById('theme-toggle');
const themeIcon   = document.getElementById('theme-icon');

// ── Theme ─────────────────────────────────────────────────────────────────────
function applyStoredTheme() {
  const saved = localStorage.getItem('cw-theme');
  const theme = saved === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
  themeIcon.textContent = theme === 'dark' ? '☀' : '☽';
}

themeToggle.addEventListener('click', () => {
  const html   = document.documentElement;
  const isDark = html.getAttribute('data-theme') === 'dark';
  const next = isDark ? 'light' : 'dark';
  html.setAttribute('data-theme', next);
  localStorage.setItem('cw-theme', next);
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
    'xaxis2.gridcolor': t.grid, 'xaxis2.color': t.text,
    'yaxis2.gridcolor': t.grid, 'yaxis2.color': t.text,
    'xaxis3.gridcolor': t.grid, 'xaxis3.color': t.text,
    'yaxis3.gridcolor': t.grid, 'yaxis3.color': t.text,
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
      marker: { color: '#4C72B0', size: 8, opacity: 0.88,
                line: { color: '#fff', width: 0.8 } },
    },
    {
      x: [], y: [], mode: 'markers', type: 'scatter', name: 'New incoming data',
      marker: { color: '#f5a623', size: 9, opacity: 0.95,
                line: { color: '#fff', width: 0.9 } },
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
  const traces = [];
  KDE_FEATURES.forEach((feat, idx) => {
    traces.push({
      x: [], y: [], mode: 'lines', type: 'scatter',
      name: `${feat} (reference)`,
      line: { width: 1.9, color: ['#5b8fff', '#6fc27d', '#8b7cff'][idx] },
      xaxis: `x${idx + 1}`,
      yaxis: `y${idx + 1}`,
    });
    traces.push({
      x: [], y: [], mode: 'lines', type: 'scatter',
      name: `${feat} (current)`,
      line: { width: 1.9, color: ['#e05252', '#d97706', '#ef4444'][idx], dash: 'dot' },
      xaxis: `x${idx + 1}`,
      yaxis: `y${idx + 1}`,
    });
  });
  const t = plotlyTheme();
  Plotly.newPlot('plot-kde', traces, {
    paper_bgcolor: t.paper,
    plot_bgcolor: t.plot,
    font: { family: "'DM Mono', monospace", color: t.text, size: 9 },
    margin: { l: 42, r: 10, t: 10, b: 34 },
    showlegend: true,
    legend: { orientation: 'h', y: 1.18, x: 0, bgcolor: 'rgba(0,0,0,0)', font: { size: 8 } },
    grid: { rows: 1, columns: 3, pattern: 'independent' },
    xaxis: { title: { text: 'coal', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
    yaxis: { title: { text: 'density', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
    xaxis2: { title: { text: 'gas', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
    yaxis2: { title: { text: 'density', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
    xaxis3: { title: { text: 'wind', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
    yaxis3: { title: { text: 'density', font: { size: 9 } }, gridcolor: t.grid, color: t.text },
  }, CFG);
}

function updateKdePlot() {
  const xVals = [];
  const yVals = [];
  KDE_FEATURES.forEach((feat) => {
    const d = kdeStore[feat] || { ref_x: [], ref_y: [], cur_x: [], cur_y: [] };
    xVals.push(d.ref_x, d.cur_x);
    yVals.push(d.ref_y, d.cur_y);
  });
  Plotly.restyle('plot-kde', { x: xVals, y: yVals }, [0, 1, 2, 3, 4, 5]);
}

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

// ── INITIALIZE ────────────────────────────────────────────────────────────────
btnInit.addEventListener('click', async () => {
  btnInit.disabled = true;
  addLog('Initializing pipeline…', 'idle');
  updateHeader('—', 0, '—', 'INITIALIZING');

  const config = {
    feature_x:    'coal',
    feature_y:    'forecast_intensity',
    ks_threshold: KS_THRESHOLD,
    n_init:       parseInt(document.getElementById('cfg-n-init').value),
    n_monthly:    parseInt(document.getElementById('cfg-n-monthly').value),
    speed:        FIXED_SPEED_S,
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

    kdeStore = d.kde_multi || {};
    if (!kdeStore.coal) {
      kdeStore.coal = {
        ref_x: d.kde_x,
        ref_y: d.kde_ref_y,
        cur_x: d.kde_x,
        cur_y: d.kde_ref_y,
      };
    }
    updateKdePlot();

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

    if (d.kde_multi) {
      kdeStore = d.kde_multi;
    } else {
      kdeStore.coal = {
        ref_x: d.kde_ref_x, ref_y: d.kde_ref_y,
        cur_x: d.kde_cur_x, cur_y: d.kde_cur_y,
      };
    }
    updateKdePlot();

    updateDriftHistory(d.month, d.ks_stat, d.psi, d.retrained);
    if (d.drift_pills) updatePills(d.drift_pills);
    updateDash(d);

    if (d.drift_detected) {
      flashDrift('panel-pca');
      flashDrift('panel-kde');
      document.getElementById('pill-drift').className   = 'pill drifting';
      document.getElementById('pill-drift').textContent = `DRIFT: ${d.ks_stat}`;
    } else {
      document.getElementById('pill-drift').className   = 'pill ok';
      document.getElementById('pill-drift').textContent = `KS: ${d.ks_stat}`;
    }

    const logType = d.retrained ? 'retrain' : d.drift_detected ? 'drift' : 'ok';
    addLog(`[${d.month}]  ${d.log}`, logType);

    if (d.done) {
      if (d.retrained) {
        addLog('First retrain reached — simulation auto-stopped for demo.', 'done');
      } else {
        addLog('All months processed — simulation complete.', 'done');
      }
      updateHeader(d.month, d.month_idx + 1, d.total_months, 'DONE');
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

applyStoredTheme();