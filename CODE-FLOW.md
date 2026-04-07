# CODE-FLOW.md — ml605-project Execution Paths and Internal Call Chains

This document traces every entry point, internal call chain, and state transition in the ml605 agentic MLOps pipeline. All content is derived directly from the source files. File and function references are precise.

---

## 1. Entry Points

### 1.1 `run_pipeline.py` — Daily window pipeline (non-agentic)

**What it does:** Fetches the last N hours of carbon intensity data, engineers features, runs drift detection against `historical_data.csv`, and — if drift is detected or no reference data exists — retrains via AutoML and promotes the winner to Production in MLflow.

**Call chain:**

```
__main__
  └── main()
        ├── setup_run_logging("run_pipeline")          [log_tracking.py]
        ├── load_config_from_env()                     [config.py] → PipelineConfig
        ├── mlflow.set_experiment(cfg.mlflow_experiment)
        ├── mlflow.sklearn.autolog(...)
        ├── mlflow.start_run(run_name=...)
        │     ├── fetch_window_dataframe(start_dt, end_dt)  [data.py]
        │     │     ├── _get_json(session, intensity_url, retries=3)
        │     │     ├── _get_json(session, generation_url, retries=3)
        │     │     ├── merge rows on "from" field (generation_by_from dict)
        │     │     ├── pd.DataFrame(rows) + timestamp parse + sort
        │     │     └── fetch_intensity_factors(session) → WindowFetchResult
        │     ├── add_time_features(df)                [features.py]
        │     ├── apply_factor_columns(df, result.factors)  [features.py]
        │     ├── one_hot_intensity_index(df)           [features.py]
        │     ├── df.to_csv(cfg.output_csv)            [side-effect: data/ dir]
        │     ├── mlflow.log_artifact(out_csv)
        │     ├── load_feature_list(features_path)     [features.py]
        │     ├── ensure_feature_columns(df, feature_cols)  [features.py]
        │     ├── [if historical_data.csv exists]:
        │     │     ├── pd.read_csv("historical_data.csv")
        │     │     ├── add_time_features(ref_df)
        │     │     ├── apply_factor_columns(ref_df, ...)
        │     │     ├── one_hot_intensity_index(ref_df)
        │     │     └── detect_drift(ref_df, df, numeric_feature_cols)  [drift.py]
        │     │           └── returns DriftReport with overall_drift bool
        │     ├── [if should_retrain]:
        │     │     ├── time_split(df, feature_cols)   [modeling.py] → X_train/test, y_train/test
        │     │     ├── run_automl(X_train, y_train, X_test, y_test)  [automl.py]
        │     │     │     └── for each CANDIDATE_MODEL:
        │     │     │           ├── mlflow.start_run(run_name=name, nested=True)
        │     │     │           ├── _evaluate_candidate(...) → ModelCandidate
        │     │     │           └── mlflow.sklearn.log_model(model, "model")
        │     │     ├── register_model(run_id=automl_result.best.run_id)  [registry.py]
        │     │     └── transition_model_stage(version, "Production")  [registry.py]
        └── [end mlflow run]
```

**Env vars read:**
- `PIPELINE_WINDOW_HOURS` (default `"12"`)
- `PIPELINE_INTERVAL_SECONDS` (default `"30"`)
- `MLFLOW_EXPERIMENT` (default `"daily-intensity-pipeline"`)

**Side effects:** Writes timestamped CSV to `data/`, writes MLflow run to `daily-intensity-pipeline` experiment, optionally registers/promotes model.

---

### 1.2 `fetch_historical_data.py` — One-time historical data fetch

**What it does:** Fetches 6 years of UK carbon intensity data in monthly chunks (2020-03-24 to 2026-03-24), writes `historical_data.csv` and `intensity_factors.json`, and logs the run to MLflow.

**Call chain:**

```
__main__
  └── main()
        ├── setup_run_logging("fetch_historical_data")
        ├── mlflow.set_experiment("historical-data-pipeline")
        ├── mlflow.start_run(run_name="monthly_historical_data_pull")
        │     ├── fetch_intensity_factors()            [local function, not data.py]
        │     │     └── GET https://api.carbonintensity.org.uk/intensity/factors
        │     ├── FACTORS_JSON.write_text(json.dumps(factors)) → intensity_factors.json
        │     ├── mlflow.log_artifact(FACTORS_JSON)
        │     ├── [monthly loop]: while current_chunk_start_date <= END_DATE:
        │     │     └── fetch_data_for_range(start_dt, end_dt)   [local function]
        │     │           ├── GET INTENSITY_API_BASE/{start}/{end}
        │     │           └── GET GENERATION_API_BASE/{start}/{end}
        │     │               merge on "from" field → list[dict]
        │     ├── pd.DataFrame(all_historical_data) → parse timestamps → sort
        │     ├── apply factor columns inline (not via features.py)
        │     ├── df.to_csv("historical_data.csv")    [side-effect]
        │     ├── mlflow.log_metric("rows_fetched", ...)
        │     └── mlflow.log_artifact("historical_data.csv")
        └── [end mlflow run]
```

**Env vars read:** None (dates and output path are hard-coded constants).

**Side effects:** Writes `historical_data.csv` and `intensity_factors.json`.

---

### 1.3 `train_with_mlflow.py` — Standalone baseline model training

**What it does:** Reads `historical_data.csv` and `features_used.txt`, trains a single `RandomForestRegressor(n_estimators=300, max_depth=14)`, logs all metrics and the model to MLflow, and logs feature importances.

**Call chain:**

```
__main__
  └── main()
        ├── setup_run_logging("train_with_mlflow")
        ├── mlflow.set_experiment("intensity-model-training")
        ├── mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)
        ├── pd.read_csv("historical_data.csv") → parse timestamps → dropna → sort
        ├── add time features inline (hour, day_of_week, month, day_of_year, is_weekend)
        ├── pd.get_dummies(df["intensity_index"], prefix="intensity_index")  [inline, not features.py]
        ├── load feature list from features_used.txt
        ├── ensure missing feature cols filled with 0.0 (inline)
        ├── time_split (inline, split_idx = int(len(df)*0.8))
        ├── mlflow.start_run(run_name="random_forest_baseline"):
        │     ├── [optional] baseline_rmse from forecast_intensity
        │     ├── RandomForestRegressor.fit(X_train, y_train)
        │     ├── model.predict(X_test), model.predict(X_train)
        │     ├── compute rmse, mae, r2, mape, rmse_train, r2_train, oob_score
        │     ├── mlflow.log_metric(k, v) for each metric
        │     ├── feature_importances → mlflow.log_dict(..., "top_10_feature_importance.json")
        │     └── mlflow.log_artifact("features_used.txt")
        └── [end mlflow run]
```

**Note:** This entry point does NOT use `automl.py`, `modeling.py`, or `registry.py`. It trains a single model and does not register it in the MLflow Model Registry.

**Env vars read:** None.

---

### 1.4 `src/ml605_agent/__main__.py` — Agentic LangGraph pipeline

**What it does:** Spawns the MCP server if it is not already running, builds the LangGraph state graph, starts an MLflow parent run, invokes the full pipeline graph, and prints a results summary.

**Call chain:**

```
__main__
  └── main()
        ├── load_config_from_env()                     [config.py]
        ├── ensure_mcp_server()
        │     ├── httpx.get("http://localhost:8000/health")  [liveness check]
        │     └── [if not running]:
        │           ├── subprocess.Popen(["uv", "run", "python", "src/ml605_mcp/server.py"])
        │           ├── atexit.register(_cleanup_mcp_server)
        │           └── poll /health for up to 15 seconds (30 × 0.5s)
        ├── build_graph()                              [graph.py] → CompiledGraph
        ├── mlflow.set_experiment("agentic-pipeline")
        ├── mlflow.start_run(run_name="agent-pipeline-run")
        │     ├── initial_state = {window_hours, mlflow_run_id, status="running", retrain_done=False}
        │     └── graph.invoke(initial_state, config={"recursion_limit": 25})
        │           → executes the full graph topology (see Section 3)
        └── print status, error, eval metrics, drift flag, new model version
```

**Env vars read:**
- `PIPELINE_WINDOW_HOURS` (default `"12"`)
- `PIPELINE_INTERVAL_SECONDS` (default `"30"`)
- `MLFLOW_EXPERIMENT` (default `"daily-intensity-pipeline"`)
- `GROQ_API_KEY` (consumed inside `report_worker` via `_generate_llm_summary`)

---

## 2. MCP Server Tools

The MCP server (`src/ml605_mcp/server.py`) is a `FastMCP("carbon-intensity")` application running on **Streamable HTTP transport, port 8000**. The server path injected into `sys.path` is `src/` so `ml605_pipeline` imports resolve.

### 2.1 `fetch_intensity`

**Tool registration:** `@mcp.tool` decorator on `fetch_intensity()` function.

**Underlying function called:** `fetch_window_dataframe(start, end)` from `ml605_pipeline.data`.

**Input parameters:**
- `hours_back: int = 12` — hours before now to include; ignored when both datetime params are provided
- `start_dt: str | None = None` — ISO8601 start (e.g., `"2026-03-01T00:00Z"`)
- `end_dt: str | None = None` — ISO8601 end (e.g., `"2026-03-01T12:00Z"`)

**Return shape:**
```python
{
  "readings": list[dict],    # one record per 30-min interval: timestamp, interval_end,
                             # actual_intensity, forecast_intensity, intensity_index,
                             # plus one key per fuel type (perc values)
  "start": str,              # ISO8601
  "end": str,                # ISO8601
  "count": int,
  "factors": dict[str, float]   # e.g. {"Gas (Combined Cycle)": 394.0, ...}
}
```

**Timestamp handling:** `_df_to_records()` converts pandas Timestamps to ISO strings via `dt.strftime("%Y-%m-%dT%H:%M:%SZ")` and replaces `float NaN` with `None`.

**Error / retry handling:** Delegated to `_get_json()` in `data.py` which retries up to 3 times with `time.sleep(1.5 * (attempt + 1))` backoff. Raises `RuntimeError` after all retries exhausted.

---

### 2.2 `fetch_generation_mix`

**Tool registration:** `@mcp.tool` decorator on `fetch_generation_mix()` function.

**Underlying function called:** `fetch_window_dataframe(start, end)` from `ml605_pipeline.data` (same as `fetch_intensity`).

**Input parameters:** Same as `fetch_intensity` (`hours_back`, `start_dt`, `end_dt`).

**Return shape:**
```python
{
  "readings": list[dict],    # only timestamp + interval_end + fuel-type perc columns
                             # excludes: actual_intensity, forecast_intensity, intensity_index
  "start": str,
  "end": str,
  "count": int
}
```

**Note:** The server strips `intensity_cols = {"timestamp", "interval_end", "actual_intensity", "forecast_intensity", "intensity_index"}` and returns only `timestamp`, `interval_end`, and fuel-type columns.

**Error / retry handling:** Same as `fetch_intensity`.

---

### 2.3 `/health` custom route

**Registration:** `@mcp.custom_route("/health", methods=["GET"])` on async `health_check()`.

**Purpose:** Liveness probe used by `ensure_mcp_server()` in `__main__.py` to determine whether the server subprocess has started.

---

## 3. LangGraph Agent Graph

### 3.1 Graph topology

Built in `graph.py::build_graph()` using `StateGraph(PipelineState)`. No `MemorySaver` checkpointer (removed in Phase 2 — stateless execution).

**Node registrations:**

| Node name | Worker function | File |
|---|---|---|
| `fetch_worker` | `fetch_worker(state)` | `workers.py` |
| `feature_worker` | `feature_worker(state)` | `workers.py` |
| `test_worker` | `test_worker(state)` | `workers.py` |
| `drift_worker` | `drift_worker(state)` | `workers.py` |
| `retrain_worker` | `retrain_worker(state)` | `workers.py` |
| `report_worker` | `report_worker(state)` | `graph.py` |
| `alert_worker` | `alert_worker(state)` | `graph.py` |
| `error_handler` | `error_handler(state)` | `graph.py` |

**Edges:**

```
START
  └─(conditional, lambda always "fetch_worker")─► fetch_worker
        │
        └─(route_after_fetch)────────────────────► feature_worker
                                                   error_handler
              │
              └─(route_after_feature)────────────► test_worker
                                                   error_handler
                    │
                    └─(route_after_test)──────────► drift_worker
                                                    error_handler
                          │
                          └─(route_after_drift)───► retrain_worker
                                                    report_worker
                                                    error_handler
                                │
                                └─(route_after_retrain)─► test_worker  [back-edge]
                                                           error_handler

report_worker ──(edge)──► alert_worker ──(edge)──► END
error_handler ──(edge)──► END
```

---

### 3.2 Conditional edge routing functions

All routing functions are defined in `graph.py` (after workers.py is imported).

#### `route_after_fetch(state) -> str`
```
if state.get("status") == "error"  →  "error_handler"
else                               →  "feature_worker"
```

#### `route_after_feature(state) -> str`
```
if state.get("status") == "error"  →  "error_handler"
else                               →  "test_worker"
```

#### `route_after_test(state) -> str`
```
if state.get("status") == "error"  →  "error_handler"
else                               →  "drift_worker"
```

#### `route_after_drift(state) -> str`
```
if state.get("status") == "error"               →  "error_handler"
if state.get("overall_drift")
   and not state.get("retrain_done", False)      →  "retrain_worker"
else                                             →  "report_worker"
```

The `retrain_done` guard prevents a second retrain cycle after `retrain_worker` returns and `test_worker` re-evaluates the new model, which would re-enter `drift_worker` with `overall_drift` still potentially True.

#### `route_after_retrain(state) -> str`
```
if state.get("status") == "error"  →  "error_handler"
else                               →  "test_worker"   [back-edge to re-evaluate new model]
```

---

### 3.3 PipelineState — what each worker reads and writes

`PipelineState` is defined in `src/ml605_agent/state.py` as `TypedDict, total=False`. Workers return partial dicts (only their owned keys). This is possible because `total=False` means all keys are optional and LangGraph merges the returned partial dict into the running state.

| Worker | Reads | Writes |
|---|---|---|
| `fetch_worker` | `window_hours`, `mlflow_run_id` | `df`, `factors`, `rows_fetched` or `status`+`error` |
| `feature_worker` | `df`, `factors`, `mlflow_run_id` | `df_featured`, `feature_cols` or `status`+`error` |
| `test_worker` | `df_featured`, `feature_cols`, `mlflow_run_id` | `eval_result` or `status`+`error` |
| `drift_worker` | `feature_cols`, `factors`, `df_featured`, `eval_result`, `mlflow_run_id` | `drift_report`, `overall_drift`, `rmse_degradation_pct`, `rmse_degradation_fired`, `production_rmse` or `status`+`error` |
| `retrain_worker` | `df_featured`, `feature_cols`, `mlflow_run_id` | `new_model_version`, `retrain_done=True` or `status`+`error` |
| `report_worker` | `df_featured`, `feature_cols`, `eval_result`, `drift_report`, `overall_drift`, `rmse_degradation_pct`, `new_model_version`, `retrain_done`, `mlflow_run_id`, `status` | `report_path`, `shap_top_features` or `status`+`error` |
| `alert_worker` | `mlflow_run_id` | `alert_sent=False` |
| `error_handler` | `mlflow_run_id`, `error` | `status="error"` |

---

### 3.4 MLflow nested runs

The `mlflow_run_id` field carries the parent run ID (started in `__main__.py`). Workers open **nested** MLflow runs using `mlflow.start_run(run_name="<worker_name>", nested=True)` when `mlflow_run_id` is present in state. The outer `mlflow.start_run(run_id=...)` wrapper was removed from all workers in Phase 2 — workers use `nested=True` directly while the parent context from `__main__.py` is active.

Workers that open nested runs: `fetch_worker`, `feature_worker`, `test_worker`, `drift_worker`, `retrain_worker`, `report_worker`, `alert_worker`, `error_handler`.

---

## 4. Data Transformation Pipeline

### Step 1: API response shape

The National Grid ESO API (`carbonintensity.org.uk`) returns two payloads:

**Intensity endpoint** (`GET /intensity/{from}/{to}`)  — each `data[]` item:
```json
{
  "from": "2026-03-01T00:00Z",
  "to": "2026-03-01T00:30Z",
  "intensity": {
    "forecast": 211,
    "actual": 213,
    "index": "moderate"
  }
}
```

**Generation mix endpoint** (`GET /generation/{from}/{to}`)  — each `data[]` item:
```json
{
  "from": "2026-03-01T00:00Z",
  "to": "2026-03-01T00:30Z",
  "generationmix": [
    {"fuel": "gas", "perc": 35.4},
    {"fuel": "wind", "perc": 22.1},
    ...
  ]
}
```

**Factors endpoint** (`GET /intensity/factors`) — returns one dict of fuel → gCO2eq/kWh constants, e.g.:
```json
{"Gas (Combined Cycle)": 394, "Wind": 0, "Nuclear": 0, ...}
```

---

### Step 2: Merge logic (`data.py::fetch_window_dataframe`)

```python
generation_by_from = {item.get("from"): item for item in generation_payload}

for item in intensity_payload:
    row = {
        "timestamp": item["from"],
        "interval_end": item["to"],
        "actual_intensity": item["intensity"]["actual"],
        "forecast_intensity": item["intensity"]["forecast"],
        "intensity_index": item["intensity"]["index"],
    }
    generation_item = generation_by_from.get(item["from"], {})
    for fuel_item in generation_item.get("generationmix", []):
        row[fuel_item["fuel"]] = fuel_item["perc"]
    rows.append(row)
```

**Join key:** `"from"` timestamp string (dict lookup, not SQL join). Intensity intervals with no matching generation entry get `NaN` for all fuel columns.

**Column names from merge:** `timestamp`, `interval_end`, `actual_intensity`, `forecast_intensity`, `intensity_index`, `gas`, `wind`, `nuclear`, `coal`, `biomass`, `imports`, `other`, `hydro`, `solar` (names match API `fuel` field values).

After DataFrame construction, `timestamp` and `interval_end` are parsed to `pd.Timestamp` (UTC), rows with null `timestamp` are dropped, and the result is sorted ascending.

---

### Step 3: `add_time_features(df)` — `features.py`

Adds 5 columns derived from `df["timestamp"]`:

| Column | Derivation |
|---|---|
| `hour` | `df["timestamp"].dt.hour` — 0 to 23 |
| `day_of_week` | `df["timestamp"].dt.dayofweek` — 0=Monday, 6=Sunday |
| `month` | `df["timestamp"].dt.month` — 1 to 12 |
| `day_of_year` | `df["timestamp"].dt.dayofyear` — 1 to 366 |
| `is_weekend` | `(day_of_week >= 5).astype(int)` — 0 or 1 |

Does nothing if `df.empty`. Returns a copy (does not mutate in place).

---

### Step 4: `apply_factor_columns(df, factors)` — `features.py`

For each key-value pair in the `factors` dict returned by the API:
1. Normalizes the key via `normalize_factor_name(name)`: lowercases, replaces non-alphanumeric characters with `_`, strips leading/trailing `_`, prepends `factor_` prefix. Example: `"Gas (Combined Cycle)"` → `"factor_gas_combined_cycle"`.
2. Assigns the scalar float value as a constant column on every row.

All factor columns have a single value (the CO2 emission factor for that fuel type) repeated for every row in the DataFrame.

---

### Step 5: `one_hot_intensity_index(df)` — `features.py`

Categories defined: `["very low", "low", "moderate", "high", "very high"]`.

Process:
1. Casts `intensity_index` to `pd.CategoricalDtype(categories=_INTENSITY_CATEGORIES, ordered=False)`
2. `pd.get_dummies(df["intensity_index"].astype(cat), prefix="intensity_index")` — produces exactly 5 columns regardless of which categories appear in the data
3. Drops `intensity_index` and concatenates the 5 dummy columns

**Resulting columns:**
- `intensity_index_very low`
- `intensity_index_low`
- `intensity_index_moderate`
- `intensity_index_high`
- `intensity_index_very high`

**Note:** In `run_pipeline.py`, `one_hot_intensity_index` is called explicitly. In the agentic worker path (`feature_worker`), it is NOT called — only `add_time_features`, `apply_factor_columns`, and `ensure_feature_columns` are applied. The agent path relies on `ensure_feature_columns` to pad any missing `intensity_index_*` columns with `0.0`.

---

### Step 6: `ensure_feature_columns(df, feature_cols)` — `features.py`

Iterates over `feature_cols` (loaded from `features_used.txt`). For any column in the list that is missing from `df`, adds it as a constant `0.0` column. Returns a copy. This keeps inference aligned with training when the live window lacks a category that appeared in training data.

---

### Final feature vector

Columns fed to `RandomForestRegressor.predict()` are exactly those listed in `features_used.txt`, loaded by `load_feature_list(Path("features_used.txt"))`. Typical contents:

- Time features: `hour`, `day_of_week`, `month`, `day_of_year`, `is_weekend`
- Fuel mix: `gas`, `wind`, `nuclear`, `coal`, `biomass`, `imports`, `other`, `hydro`, `solar`
- Factor columns: `factor_gas_combined_cycle`, `factor_gas_open_cycle`, `factor_coal`, `factor_wind`, etc.
- One-hot intensity: `intensity_index_low`, `intensity_index_moderate`, `intensity_index_high`, `intensity_index_very_high`, `intensity_index_very_low`
- `forecast_intensity` (if included in features list)

---

## 5. Drift Detection Logic

Implemented in `src/ml605_pipeline/drift.py` and called by `drift_worker` in `workers.py`.

### Signal 1: Population Stability Index (PSI)

**Function:** `_compute_psi(reference, current, bins=10)` — private helper inside `drift.py`.

**Algorithm:**
1. Compute percentile breakpoints from `reference` array using `np.linspace(0, 100, bins+1)` → `np.unique(breakpoints)`
2. Histogram both arrays against the same breakpoints
3. Regularize with `eps=1e-8` to prevent `log(0)` on empty bins
4. `PSI = Σ (cur_pct - ref_pct) * log(cur_pct / ref_pct)` over all bins

**Threshold constants (hard-coded in `drift.py`):**
```python
PSI_LOW = 0.1    # Below: no significant change
PSI_HIGH = 0.25  # At or above: significant drift — triggers retrain
```

The public `detect_drift()` function uses `psi_threshold=PSI_HIGH` (0.25) as default.

### Signal 2: Kolmogorov-Smirnov (KS) test

**Function:** `stats.ks_2samp(ref_vals, cur_vals)` from `scipy.stats` — called inside `detect_drift()`.

**Threshold:** `ks_alpha=0.05` (default). A feature is flagged if `ks_p_value < 0.05`.

**Minimum sample requirement:** Skips features with fewer than 5 samples in either reference or current array.

### Signal 3: RMSE Degradation

**Function:** `_get_production_rmse()` — private helper inside `workers.py`.

**Algorithm:**
1. Queries MLflow Registry for the current Production model run via `MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])`
2. Reads the `rmse` metric from that run via `client.get_run(run_id).data.metrics.get("rmse")`
3. `rmse_degradation_pct = (current_rmse - production_rmse) / production_rmse * 100.0`
4. `rmse_degradation_fired = current_rmse > production_rmse * 1.20` — fires at **20% degradation threshold** (hard-coded in `drift_worker`)

### Combining the three signals

```python
# Per-feature verdict in detect_drift():
drift = psi >= psi_threshold or ks_p < ks_alpha

# DriftReport.overall_drift:
overall_drift = len([r for r in results if r.drift_detected]) > 0

# drift_worker combines PSI+KS verdict with RMSE signal:
overall_drift = drift_report.overall_drift or rmse_degradation_fired
```

**Single drift boolean:** `overall_drift` in `PipelineState` is `True` if **any** of the three signals fires. Routing function `route_after_drift` checks this boolean to decide whether to branch to `retrain_worker`.

---

## 6. SHAP and Report Generation

### SHAP computation — `_compute_shap(model, X, max_display=10)` in `graph.py`

**Explainer:** `shap.Explainer(model)` — uses the generic SHAP Explainer which auto-selects `TreeExplainer` for RandomForest/GradientBoosting models.

**Data:** `X = df[feature_cols]` — the full featured DataFrame from the current pipeline window.

**Output:** `(shap_values Explanation, top_features list)`. `top_features` is derived by sorting `np.abs(shap_values.values).mean(axis=0)` descending and taking the top `max_display` indices.

**Fallback:** On exception (e.g., Ridge model from AutoML not supported by TreeExplainer), logs the error as `mlflow.log_param("shap_warning", ...)` and returns `(None, [])`.

### Plots generated

1. **SHAP bar chart** — `_shap_bar_to_base64(shap_values, max_display=10)`: calls `shap.plots.bar()`, captures the current matplotlib figure, encodes as base64 PNG string using `io.BytesIO`.

2. **Forecast vs. actual chart** — `_forecast_chart_to_base64(df, feature_cols, model)`: uses last 50 rows of `df`, calls `model.predict(X)`, creates a dual-line `matplotlib` plot (actual in `#2c5f8f`, forecast in `#e8963e`), encodes as base64 PNG string.

### Jinja2 template rendering — `_render_report(context)` in `graph.py`

**Template location:** `src/ml605_agent/templates/report.html.j2`

**Jinja2 Environment settings:**
```python
Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=False    # base64 data URIs must not be HTML-escaped
)
```

**Variables passed to `template.render(**context)`:**

| Variable | Source |
|---|---|
| `run_timestamp` | `datetime.now().strftime("%Y-%m-%d %H:%M:%S")` |
| `mlflow_run_id` | `state.get("mlflow_run_id", "N/A")` |
| `status` | `state.get("status", "complete")` |
| `metrics` | dict with `rmse`, `mae`, `r2`, `mape` from `eval_result` |
| `forecast_chart_b64` | base64 PNG from `_forecast_chart_to_base64()` |
| `shap_available` | bool — whether SHAP computation succeeded |
| `shap_chart_b64` | base64 PNG from `_shap_bar_to_base64()` |
| `overall_drift` | `state.get("overall_drift", False)` |
| `rmse_degradation_pct` | `state.get("rmse_degradation_pct")` |
| `drift_features` | list of dicts: `{feature, psi, ks_p_value, drift_detected}` |
| `model_comparison` | dict from `_build_model_comparison()` — current vs. production metrics |
| `llm_summary` | HTML string from `_generate_llm_summary()` |

### Groq LLM summary — `_generate_llm_summary(context_payload)` in `graph.py`

**Condition check:** Reads `GROQ_API_KEY` via `os.getenv("GROQ_API_KEY")` at call time (not at module load time — enables test patching).

**Model used:** `"llama-3.3-70b-versatile"` via `groq.Groq(api_key=api_key).chat.completions.create(temperature=0.3, max_completion_tokens=512)`.

**Prompt structure:** Instructs the model to produce exactly 2-3 paragraphs covering: (1) drift diagnosis, (2) top drifted/important features, (3) retrain recommendation. Sends `json.dumps(context_payload)` as user message.

**Fallback — `_fallback_summary(ctx)`:** Returns a formatted HTML string with drift status, current RMSE, retrain flag, and top SHAP features. Invoked when `GROQ_API_KEY` is absent or on any `groq` SDK exception. Exception is logged to MLflow via `mlflow.log_param("llm_error", str(exc)[:200])`.

### MLflow artifacts logged by `report_worker`

| Artifact | Path logged |
|---|---|
| SHAP top features JSON | `artifact_path="shap"`, temp file `*_shap_top_features.json` |
| HTML report file | `artifact_path="reports"`, local path `reports/pipeline_report_{timestamp}.html` |
| `report_path` param | `mlflow.log_param("report_path", report_path)` |

Report is also saved to disk at `reports/pipeline_report_{timestamp}.html` and its path is written to `PipelineState["report_path"]`.

### Model comparison table — `_build_model_comparison(eval_result, new_model_version)` in `graph.py`

Queries `MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])` and fetches `rmse` and `mae` from that run. Returns `None` if no Production model exists. Exposed to Jinja2 template as `model_comparison` dict.

---

## 7. Planned Slack Integration (Phase 4 — Not Yet Implemented)

Based on `ROADMAP.md` Phase 4 and plan outlines. No code exists for this phase yet.

### New nodes to be added to the LangGraph graph

| New node | Plan | Purpose |
|---|---|---|
| `hitl_decision_node` | 04-03 | Calls LangGraph `interrupt()` to pause execution pending human Slack response |
| `alert_worker` (replacement) | 04-03 | Replaces the current stub; posts real Slack Block Kit message |

The `hitl_decision_node` will be inserted **between `drift_worker` and `retrain_worker`** in the graph topology. Current routing: `drift_worker → retrain_worker` (when drift=True). Phase 4 routing: `drift_worker → hitl_decision_node → [approved: retrain_worker | rejected: report_worker]`.

### HITL interrupt() flow

`interrupt()` is a LangGraph primitive that:
1. **Pauses** graph execution and serializes current `PipelineState` to a checkpoint
2. Returns control to the caller with a `GraphInterrupt` exception
3. **Resumes** when the orchestrator calls `graph.invoke(None, config={"thread_id": ...})` with updated state

**Pause point:** After `drift_worker` confirms drift and before retraining starts.  
**Resume condition:** A Slack button handler (in `src/ml605_slack/bot.py`, plan 04-02) receives an `approve` or `reject` action from the interactive Slack message, calls `graph.invoke` with `hitl_decision = "approved"` or `"rejected"` in state.

For `interrupt()` to work, `MemorySaver` (or an equivalent checkpointer) will need to be reinstated in `build_graph()` — it was removed in Phase 2 since stateless execution was sufficient.

### Block Kit messages

| Message | When sent | Contents |
|---|---|---|
| Drift alert + HITL | When `overall_drift=True` | Drift verdict, top drifted features, SHAP top-3, link to HTML report, `Approve Retrain` / `Reject` buttons |
| Pipeline completion | When `overall_drift=False` | Summary metrics (RMSE, forecast summary, model version) |
| Approval confirmed | After human clicks Approve | Confirms retrain triggered, new model version when registered |
| Rejection confirmed | After human clicks Reject | Confirms retrain skipped |

### Slash commands

Per ROADMAP plan 04-02:

| Command | Handler | Action |
|---|---|---|
| `/pipeline run` | `bot.py` | Triggers full `graph.invoke(initial_state)` in background, responds with run-in-progress message |
| `/model status` | `bot.py` | Queries `MlflowClient` for Production model version, RMSE, last run timestamp — returns structured Block Kit message |
| `/model promote` | `bot.py` | Calls `transition_model_stage(version, "Production")` for a named model version |

### PipelineState additions for Phase 4

Plan 04-01 will add:
- `shap_top_features: list[str]` — already written by `report_worker` in Phase 3 (field name `shap_top_features`)
- `hitl_decision: str | None` — `"approved"` or `"rejected"` written by the Slack button handler and read by `hitl_decision_node`

### HITL timing tracking

Human approval/rejection timestamp will be logged to MLflow via `mlflow.log_metric("hitl_response_time_seconds", ...)` to support Mean Time to Acknowledge (MTTA) tracking per ROADMAP requirement HITL-03.
