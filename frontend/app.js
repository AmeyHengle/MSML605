// app.js — CarbonWatch MLOps frontend
// Communicates with FastAPI backend via REST + SSE (EventSource).

// const API = 'http://localhost:8000';
// const API = 'https://msml605-backend.onrender.com';
const API = 'https://msml605.onrender.com';

// ── State ────────────────────────────────────────────────────────────────────
let evtSource    = null;
let initialized  = false;
let simRunning   = false;
let logCount     = 0;

// Accumulated scatter data (all representative points, entire timeline)
let scatterX = [], scatterY = [];

// KS history for sparkline
let ksHistory = [], ksMonths = [];

// Current KDE data per feature (updated on each tick)
let kdeStore = {};   // { feature: { ref_x, ref_y, cur_x, cur_y } }

// Current active feature in KDE dropdown
let kdeFeature = 'gas';

// Current active line {x0,x1,y0,y1}
let activeLine = null;

// ── DOM refs ─────────────────────────────────────────────────────────────────
const btnInit     = document.getElementById('btn-init');
const btnSim      = document.getElementById('btn-simulate');
const btnPause    = document.getElementById('btn-pause');
const btnReset    = document.getElementById('btn-reset');
const cfgSpeed    = document.getElementById('cfg-speed');
const cfgSpeedVal = document.getElementById('cfg-speed-val');

cfgSpeed.addEventListener('input', () => {
  cfgSpeedVal.textContent = parseFloat(cfgSpeed.value).toFixed(1) + 's';
});

// ── Plotly layout helpers ─────────────────────────────────────────────────────
const darkLayout = (xLabel, yLabel, extra = {}) => ({
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor:  'rgba(0,0,0,0)',
  font:   { family: "'DM Mono', monospace", color: '#6b7394', size: 10 },
  xaxis:  { title: { text: xLabel, font: { size: 10 } }, gridcolor: '#1e2233',
             zeroline: false, color: '#6b7394' },
  yaxis:  { title: { text: yLabel, font: { size: 10 } }, gridcolor: '#1e2233',
             zeroline: false, color: '#6b7394' },
  margin:   { l: 46, r: 12, t: 10, b: 38 },
  showlegend: true,
  legend: { bgcolor: 'rgba(0,0,0,0)', font: { size: 9 }, x: 0, y: 1 },
  ...extra
});

const plotCfg = { displayModeBar: false, responsive: true };

// ── Scatter + regression line ─────────────────────────────────────────────────
function initScatterPlot(featureX, featureY, initX, initY, lineX, lineY) {
  scatterX = [...initX];
  scatterY = [...initY];
  activeLine = { x: lineX, y: lineY };

  const traces = [
    {
      x: scatterX, y: scatterY,
      mode: 'markers',
      type: 'scatter',
      name: 'Samples',
      marker: { color: '#4C72B0', size: 6, opacity: 0.78,
                line: { color: '#fff', width: 0.5 } },
    },
    {
      x: lineX, y: lineY,
      mode: 'lines',
      type: 'scatter',
      name: 'Fit (v1)',
      line: { color: '#e05252', width: 2.5 },
    }
  ];

  Plotly.newPlot('plot-scatter', traces,
    darkLayout(`${featureX} (%)`, 'Forecast Intensity (gCO₂/kWh)'),
    plotCfg
  );

  document.getElementById('scatter-subtitle').textContent =
    `${featureX} → ${featureY}`;
}

function updateScatterPoints(newX, newY) {
  scatterX = scatterX.concat(newX);
  scatterY = scatterY.concat(newY);

  // Update trace 0 (scatter points) in-place — much faster than full re-render
  Plotly.extendTraces('plot-scatter', { x: [newX], y: [newY] }, [0]);
}

function updateRegressionLine(lineX, lineY, version) {
  Plotly.restyle('plot-scatter', {
    x: [lineX],
    y: [lineY],
    name: [`Fit (v${version})`],
  }, [1]);
}

// ── KDE plot ──────────────────────────────────────────────────────────────────
function initKdePlot() {
  const traces = [
    { x: [], y: [], mode: 'lines', type: 'scatter', name: 'Reference',
      fill: 'tozeroy', fillcolor: 'rgba(91,143,255,0.15)',
      line: { color: '#5b8fff', width: 2 } },
    { x: [], y: [], mode: 'lines', type: 'scatter', name: 'Current',
      fill: 'tozeroy', fillcolor: 'rgba(224,82,82,0.12)',
      line: { color: '#e05252', width: 2 } },
  ];
  Plotly.newPlot('plot-kde', traces,
    darkLayout('Feature value (%)', 'Density'),
    plotCfg
  );
}

function updateKdePlot(feat) {
  const d = kdeStore[feat];
  if (!d) return;
  Plotly.restyle('plot-kde', {
    x: [d.ref_x, d.cur_x],
    y: [d.ref_y, d.cur_y],
  }, [0, 1]);
}

document.getElementById('kde-feature-select').addEventListener('change', e => {
  kdeFeature = e.target.value;
  updateKdePlot(kdeFeature);
});

// ── KS sparkline ─────────────────────────────────────────────────────────────
function initKsPlot(threshold) {
  Plotly.newPlot('plot-ks',
    [
      { x: [], y: [], mode: 'lines+markers', type: 'scatter',
        name: 'KS', line: { color: '#5b8fff', width: 1.5 },
        marker: { size: 4, color: '#5b8fff' } },
      { x: [], y: [threshold, threshold],
        mode: 'lines', type: 'scatter',
        name: 'Threshold', line: { color: '#e05252', width: 1, dash: 'dot' } },
    ],
    { ...darkLayout('Month', 'KS'), showlegend: false,
      margin: { l: 36, r: 8, t: 6, b: 28 },
      shapes: [] },
    plotCfg
  );
}

function updateKsSpark(month, ksVal, drifted, threshold) {
  ksMonths.push(month);
  ksHistory.push(ksVal);

  Plotly.restyle('plot-ks', {
    x: [ksMonths],
    y: [ksHistory],
  }, [0]);

  // Keep threshold line spanning full x range
  Plotly.restyle('plot-ks', {
    x: [[ksMonths[0], ksMonths[ksMonths.length - 1]]],
    y: [[threshold, threshold]],
  }, [1]);

  // Flash a red shape marker if drifted
  if (drifted) {
    Plotly.relayout('plot-ks', {
      shapes: [{
        type: 'line', xref: 'x', yref: 'paper',
        x0: month, x1: month, y0: 0, y1: 1,
        line: { color: '#e05252', width: 1, dash: 'dot' }
      }]
    });
  }
}

// ── Drift pills ───────────────────────────────────────────────────────────────
function updatePills(pills) {
  document.querySelectorAll('.feat-pill').forEach(el => {
    const feat = el.dataset.feat;
    el.className = `feat-pill ${pills[feat] || 'none'}`;
  });
}

// ── Log ───────────────────────────────────────────────────────────────────────
function addLog(msg, type = 'ok') {
  logCount++;
  const body = document.getElementById('log-body');
  const entry = document.createElement('div');
  entry.className = `log-entry log-entry--${type}`;
  const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
  entry.innerHTML = `<span class="log-ts">${ts}</span>${msg}`;
  body.appendChild(entry);
  body.scrollTop = body.scrollHeight;
  document.getElementById('log-count').textContent = `${logCount} events`;
}

// ── Header / metrics update ───────────────────────────────────────────────────
function updateHeader(month, idx, total, status) {
  document.getElementById('hdr-month').textContent    = month;
  document.getElementById('hdr-progress').textContent = `${idx} / ${total}`;
  document.getElementById('hdr-status').textContent   = status;
}

function updateModelDash(data) {
  document.getElementById('m-r2').textContent      = data.r2 ?? '—';
  document.getElementById('m-slope').textContent   = data.slope ?? data.new_line?.slope ?? '—';
  document.getElementById('m-intercept').textContent = data.intercept ?? data.new_line?.intercept ?? '—';
  document.getElementById('m-ks').textContent      = data.ks_stat ?? '—';
  document.getElementById('m-npts').textContent    = data.n_train?.toLocaleString() ?? '—';
  if (data.retrained) {
    document.getElementById('m-retrain').textContent = data.month;
    document.getElementById('version-badge').textContent = `v ${data.model_version}`;
  }
}

// ── Drift flash ───────────────────────────────────────────────────────────────
function flashDrift(panel) {
  const el = document.getElementById(panel);
  el.classList.add('drift-flash');
  setTimeout(() => el.classList.remove('drift-flash'), 2200);
}

// ── Initialize ────────────────────────────────────────────────────────────────
btnInit.addEventListener('click', async () => {
  btnInit.disabled = true;
  addLog('Initializing pipeline…', 'idle');
  updateHeader('—', 0, '—', 'INITIALIZING');

  const config = {
    feature_x:    document.getElementById('cfg-feature-x').value,
    feature_y:    'forecast_intensity',
    ks_threshold: parseFloat(document.getElementById('cfg-threshold').value),
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
    const json = await res.json();
    const d    = json.data;

    // Boot plots
    initScatterPlot(d.feature_x, d.feature_y,
                    d.scatter_x, d.scatter_y,
                    d.line_x, d.line_y);
    initKdePlot();
    initKsPlot(config.ks_threshold);

    // Store initial KDE data for primary feature
    kdeStore[config.feature_x] = {
      ref_x: d.kde_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_x, cur_y: d.kde_ref_y,
    };
    updateKdePlot(kdeFeature);

    // Dashboard
    updateModelDash({
      r2: d.r2, slope: d.slope, intercept: d.intercept,
      n_train: d.n_train, retrained: true,
      month: d.month, model_version: 1,
    });
    updateHeader(d.month, 1, d.total_months, 'READY');

    document.getElementById('m-retrain').textContent = d.month;
    document.getElementById('version-badge').textContent = 'v 1';

    addLog(`Baseline model trained on ${d.month}  |  R²=${d.r2}  |  n=${d.n_train.toLocaleString()}`, 'retrain');

    // Lock controls
    document.querySelectorAll('.ctrl-group select, .ctrl-group input').forEach(el => {
      el.disabled = true;
    });

    initialized = true;
    btnSim.disabled   = false;
    btnReset.disabled = false;

  } catch (err) {
    addLog(`Initialization failed: ${err.message}`, 'drift');
    btnInit.disabled = false;
  }
});

// ── Simulate ──────────────────────────────────────────────────────────────────
btnSim.addEventListener('click', () => {
  if (!initialized) return;

  btnSim.disabled   = true;
  btnPause.disabled = false;
  simRunning        = true;

  updateHeader(
    document.getElementById('hdr-month').textContent,
    document.getElementById('hdr-progress').textContent.split(' / ')[0],
    document.getElementById('hdr-progress').textContent.split(' / ')[1],
    'RUNNING'
  );

  const threshold = parseFloat(document.getElementById('cfg-threshold').value);

  evtSource = new EventSource(`${API}/api/simulate`);

  evtSource.onmessage = (e) => {
    const d = JSON.parse(e.data);

    if (d.error) { addLog(d.error, 'drift'); evtSource.close(); return; }
    if (d.done && !d.month) {
      addLog('Simulation complete — all months processed.', 'done');
      updateHeader('—', '—', '—', 'DONE');
      evtSource.close();
      btnPause.disabled = true;
      return;
    }
    if (d.paused) return;

    // ── Header ──────────────────────────────────────────────────
    updateHeader(d.month, d.month_idx + 1, d.total_months, 'RUNNING');

    // ── Scatter: add new points every tick ──────────────────────
    updateScatterPoints(d.new_scatter_x, d.new_scatter_y);

    // ── Regression line: only on retrain ────────────────────────
    if (d.retrained && d.new_line) {
      updateRegressionLine(d.new_line.line_x, d.new_line.line_y, d.model_version);
    }

    // ── KDE: store for all features, render selected ─────────────
    // Store primary feature KDE from SSE (gas by default)
    // For the selected KDE feature we always render what we got
    const primaryFeat = document.getElementById('cfg-feature-x').value;
    kdeStore[primaryFeat] = {
      ref_x: d.kde_ref_x, ref_y: d.kde_ref_y,
      cur_x: d.kde_cur_x, cur_y: d.kde_cur_y,
    };
    if (kdeFeature === primaryFeat) updateKdePlot(kdeFeature);

    // ── KS sparkline ─────────────────────────────────────────────
    updateKsSpark(d.month, d.ks_stat, d.drift_detected, threshold);

    // ── Drift pills ───────────────────────────────────────────────
    if (d.drift_pills) updatePills(d.drift_pills);

    // ── Drift flash ───────────────────────────────────────────────
    if (d.drift_detected) {
      flashDrift('panel-scatter');
      flashDrift('panel-kde');
      document.getElementById('pill-drift').className = 'pill drifting';
      document.getElementById('pill-drift').textContent = `DRIFT: ${d.ks_stat}`;
    } else {
      document.getElementById('pill-drift').className = 'pill ok';
      document.getElementById('pill-drift').textContent = `KS: ${d.ks_stat}`;
    }

    // ── Model dashboard ───────────────────────────────────────────
    updateModelDash(d);

    // ── Log ────────────────────────────────────────────────────────
    const logType = d.retrained ? 'retrain' : d.drift_detected ? 'drift' : 'ok';
    addLog(`[${d.month}]  ${d.log}`, logType);

    // ── Done ───────────────────────────────────────────────────────
    if (d.done) {
      addLog('All months processed — simulation complete.', 'done');
      updateHeader(d.month, d.total_months, d.total_months, 'DONE');
      evtSource.close();
      btnPause.disabled = true;
    }
  };

  evtSource.onerror = () => {
    addLog('SSE connection lost.', 'drift');
    evtSource.close();
    btnPause.disabled = true;
    updateHeader('—', '—', '—', 'ERROR');
  };
});

// ── Pause / Resume ────────────────────────────────────────────────────────────
btnPause.addEventListener('click', async () => {
  if (simRunning) {
    await fetch(`${API}/api/pause`, { method: 'POST' });
    btnPause.textContent = '▶ Resume';
    simRunning = false;
    updateHeader(
      document.getElementById('hdr-month').textContent,
      document.getElementById('hdr-progress').textContent.split(' / ')[0],
      document.getElementById('hdr-progress').textContent.split(' / ')[1],
      'PAUSED'
    );
    addLog('Simulation paused.', 'idle');
  } else {
    await fetch(`${API}/api/resume`, { method: 'POST' });
    btnPause.textContent = '⏸ Pause';
    simRunning = true;
    updateHeader(
      document.getElementById('hdr-month').textContent,
      document.getElementById('hdr-progress').textContent.split(' / ')[0],
      document.getElementById('hdr-progress').textContent.split(' / ')[1],
      'RUNNING'
    );
    addLog('Simulation resumed.', 'ok');
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────────
btnReset.addEventListener('click', async () => {
  if (evtSource) { evtSource.close(); evtSource = null; }
  await fetch(`${API}/api/reset`, { method: 'POST' });

  initialized = false; simRunning = false; logCount = 0;
  scatterX = []; scatterY = []; ksHistory = []; ksMonths = [];
  kdeStore = {}; activeLine = null;

  // Clear plots
  Plotly.purge('plot-scatter');
  Plotly.purge('plot-kde');
  Plotly.purge('plot-ks');

  // Clear log
  document.getElementById('log-body').innerHTML =
    '<div class="log-entry log-entry--idle">Awaiting initialization…</div>';
  document.getElementById('log-count').textContent = '0 events';

  // Reset header
  updateHeader('—', 0, '—', 'IDLE');
  document.getElementById('hdr-status').textContent = 'IDLE';
  document.getElementById('pill-drift').className   = 'pill';
  document.getElementById('pill-drift').textContent = 'DRIFT: —';
  document.getElementById('version-badge').textContent = 'v —';
  ['m-r2','m-slope','m-intercept','m-ks','m-npts','m-retrain'].forEach(id => {
    document.getElementById(id).textContent = '—';
  });

  // Unlock controls
  document.querySelectorAll('.ctrl-group select, .ctrl-group input').forEach(el => {
    el.disabled = false;
  });
  document.querySelectorAll('.feat-pill').forEach(el => {
    el.className = 'feat-pill none';
  });

  btnInit.disabled  = false;
  btnSim.disabled   = true;
  btnPause.disabled = true;
  btnReset.disabled = true;
  btnPause.textContent = '⏸ Pause';
  addLog('Pipeline reset.', 'idle');
});
