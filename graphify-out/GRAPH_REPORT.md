# Graph Report - .  (2026-04-23)

## Corpus Check
- 62 files · ~95,853 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 892 nodes · 1583 edges · 38 communities detected
- Extraction: 63% EXTRACTED · 37% INFERRED · 0% AMBIGUOUS · INFERRED: 590 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]

## God Nodes (most connected - your core abstractions)
1. `PipelineState` - 84 edges
2. `EvalResult` - 43 edges
3. `DriftReport` - 41 edges
4. `initialize()` - 29 edges
5. `FeatureDriftResult` - 23 edges
6. `PipelineState` - 22 edges
7. `report_worker()` - 18 edges
8. `handle_ml605_command()` - 15 edges
9. `TestInitialize` - 15 edges
10. `main()` - 14 edges

## Surprising Connections (you probably didn't know these)
- `LangGraph topology built in build_graph()` --semantically_similar_to--> `Current active graph topology (no HITL, report_worker -> route_after_report)`  [INFERRED] [semantically similar]
  assets/CODE-FLOW.md → docs/SLACK_HITL_ROADMAP.md
- `initialize()` --calls--> `PipelineState`  [INFERRED]
  main.py → src\ml605_agent\state.py
- `Standalone inference endpoint for load testing.     Accepts 9 energy mix featur` --uses--> `PipelineState`  [INFERRED]
  main.py → pipeline.py
- `Generate 2-3 paragraph plain-English summary via Groq. Falls back to template on` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py
- `Render the Jinja2 HTML template with the given context dict.` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py

## Hyperedges (group relationships)
- **All LangGraph workers participating in the agentic pipeline flow** — architecture_fetch_worker, architecture_feature_worker, architecture_test_worker, architecture_drift_worker, architecture_retrain_worker, architecture_report_worker, architecture_alert_worker, architecture_error_handler [EXTRACTED 0.95]
- **Three combined drift signals forming overall_drift verdict** — concept_psi_drift_signal, concept_ks_drift_signal, concept_rmse_degradation_signal [EXTRACTED 0.95]
- **Steps required to restore Slack+HITL integration** — slack_rewire_graph_step, slack_staging_first_option, slack_memorysaver_required, architecture_hitl_decision_node, architecture_alert_worker [EXTRACTED 0.90]
- **MLOps Code Flow Map: Carbon Intensity Model Lifecycle** — architecture_historical_csv, architecture_live_api, architecture_train_automl_py, architecture_detect_drift_py, architecture_drift_decision, architecture_mlflow_model_registry_staging, architecture_mlflow_model_registry_production, architecture_run_pipeline_py [EXTRACTED 0.90]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (91): _compute_psi(), detect_drift(), DriftReport, FeatureDriftResult, Population Stability Index.     PSI < 0.1: no change. 0.1-0.25: moderate. >= 0., Compare current data distribution against reference (training) distribution., EvalResult, _generate_llm_summary() (+83 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (84): _build_model_comparison(), _compute_shap(), error_handler(), _fallback_summary(), _fig_to_base64(), _forecast_chart_to_base64(), hitl_decision_node(), LangGraph pipeline graph assembly for the ml605 agent.  Current topology (Slac (+76 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (59): A single-threaded version of fetch_window_dataframe().      It deliberately pe, _sequential_fetch_window_dataframe(), create_app(), _get_or_create_graph(), _handle_retrain_background(), Run the full LangGraph pipeline in a background thread (per D-03).      Posts, Force retrain in background thread (per D-05)., Create and configure the Slack Bolt App with all handlers.      Requires env v (+51 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (22): compute_psi(), compute_rmse(), drift_pills(), kde_curve(), ks_severity(), pca_line_data(), PipelineState, quantile_sample_idx() (+14 more)

### Community 4 - "Community 4"
Cohesion: 0.04
Nodes (53): metrics_improvements scripts must remain standalone (no MLflow), Learned user preferences (uv, PowerShell, replay-back), Agent layer (LangGraph StateGraph in src/ml605_agent), Rationale: features.py chain is composable DataFrame->DataFrame, ARCHITECTURE Diagram 3 — data flow, Rationale: MCP decouples agent from raw API for swap/mock, MCP layer (FastMCP carbon-intensity, Streamable HTTP :8000), ML layer (ml605_pipeline modules data/features/modeling/drift/automl) (+45 more)

### Community 5 - "Community 5"
Cohesion: 0.06
Nodes (41): _build_history_blocks(), _build_status_blocks(), _get_latest_report_path(), _get_recent_runs(), handle_ml605_command(), _handle_promote(), _handle_report(), Slack Bolt App with Socket Mode, slash commands, and action handlers.  Per D-0 (+33 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (39): main(), add_time_features(), apply_factor_columns(), ensure_feature_columns(), load_feature_list(), normalize_factor_name(), one_hot_intensity_index(), Per-run file logging: each script execution gets a new file under logs/. (+31 more)

### Community 7 - "Community 7"
Cohesion: 0.12
Nodes (10): clean_state(), initialize(), Reset server state before each test so tests are independent., reset(), status(), test_required_key_present(), TestInitialize, TestPauseResume (+2 more)

### Community 8 - "Community 8"
Cohesion: 0.09
Nodes (35): _build_conclusion(), _df_to_markdown(), _fmt_metric(), _fmt_percent(), generate_tuning_report(), main(), parse_args(), Render a DataFrame as a GitHub-flavored markdown table.      Prefers pandas.to (+27 more)

### Community 9 - "Community 9"
Cohesion: 0.1
Nodes (28): HttpUser, check_status(), on_quit(), predict(), PredictUser, Simulates a user hitting the prediction endpoint.     wait_time=between(0.05, 0, Initialize the model before sending predictions., Print summary when load test ends. (+20 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (34): Learned workspace facts (layout, historical_data.csv, env vars), alert_worker node (stub — Phase 4 Slack), drift_worker node, Rationale: ensure_feature_columns guarantees inference-time column alignment, error_handler terminal node, feature_worker node, fetch_worker node, hitl_decision_node (LangGraph interrupt-based gate) (+26 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (22): AutoMLResult, _evaluate_candidate(), ModelCandidate, Train all candidates concurrently using a thread pool.      Threading is safe, Train all CANDIDATE_MODELS, log each as a nested MLflow run.     Returns AutoML, Train one model and return its evaluation metrics. Does NOT start an MLflow run., Train all candidates one after another (legacy path, used for benchmarking)., run_automl() (+14 more)

### Community 12 - "Community 12"
Cohesion: 0.09
Nodes (18): build_drift_alert_blocks(), build_no_drift_blocks(), Block Kit message builder functions (pure, testable).  All functions return li, Build Block Kit blocks for a no-drift pipeline summary (per D-07).      Return, Build Block Kit blocks for a drift-detected alert (per D-06).      Returns a l, Unit tests for Block Kit message builders (blocks.py)., SLACK-02: No-drift pipeline summary., RMSE and MAE metrics appear in message. (+10 more)

### Community 13 - "Community 13"
Cohesion: 0.09
Nodes (19): BaseModel, cw_metrics(), cw_stream(), InitConfig, initialize(), NumpyEncoder, predict(), PredictRequest (+11 more)

### Community 14 - "Community 14"
Cohesion: 0.1
Nodes (25): Route after drift_worker.      - status=error → error_handler     - all other, Route after report_worker (replaces the old HITL decision node).      - status, Route after retrain_worker: back-edge to test_worker to re-verify new model., route_after_drift(), route_after_report(), route_after_retrain(), Tests for ml605_agent graph topology and routing.  Graph topology tests, routi, status=error → error_handler. (+17 more)

### Community 15 - "Community 15"
Cohesion: 0.11
Nodes (19): alert_worker(), Post pipeline results to Slack (per D-06, D-07, D-10).      Reads SLACK_BOT_TO, _make_drift_state(), _make_no_drift_state(), Tests for alert_worker Slack posting.  Tests that alert_worker posts drift ale, alert_worker posts no-drift summary., chat_postMessage called with no-drift blocks when overall_drift=False., files_upload_v2 called with report_path for no-drift run. (+11 more)

### Community 16 - "Community 16"
Cohesion: 0.11
Nodes (20): compute_metrics(), Compute regression metrics. Zero targets are excluded from MAPE., _df_to_records(), fetch_generation_mix(), fetch_intensity(), health_check(), FastMCP server exposing National Grid ESO carbon intensity API as MCP tools., Fetch generation mix (fuel-type percentages) for a time window.      Returns o (+12 more)

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (7): initKdePlot(), initKsPlot(), initPcaPlot(), initPredPlot(), makeLayout(), plotlyTheme(), reapplyPlotlyTheme()

### Community 18 - "Community 18"
Cohesion: 0.11
Nodes (17): mock_fetch_result(), Tests for the ml605_mcp FastMCP server.  Requirements covered:   MCP-01: Fast, MCP-02: fetch_generation_mix returns correct schema keys (no 'factors' key)., MCP-02: Both tools accept hours_back and start_dt/end_dt parameters., Start the MCP server as a subprocess and wait for /health to respond., MCP-04: Server starts as standalone process and /health responds 200 OK., MCP-03: MultiServerMCPClient can discover tools from running server via HTTP., Return a mock WindowFetchResult with minimal plausible data. (+9 more)

### Community 19 - "Community 19"
Cohesion: 0.29
Nodes (2): Open SSE stream and collect first n events, then close., TestSimulate

### Community 20 - "Community 20"
Cohesion: 0.42
Nodes (8): compare_single_run_runtime(), compare_time_to_target(), compute_time_to_target(), load_runs(), main(), metric_hits_target(), parse_args(), run_command()

### Community 21 - "Community 21"
Cohesion: 0.33
Nodes (5): BenchmarkResult, main(), _print_metrics(), _run_benchmark(), _save_plots()

### Community 22 - "Community 22"
Cohesion: 0.29
Nodes (8): detect_drift.py (MLflow drift run), drift? (decision), Historical CSV, Live API, MLflow Model Registry Production (manual promote), MLflow Model Registry 'carbon-intensity-model' Staging, run_pipeline.py (uses existing Production or retrains inline), train_automl.py (AutoML: 5 models compared, nested MLflow runs per model)

### Community 23 - "Community 23"
Cohesion: 0.67
Nodes (3): SHAP computed on aligned feature matrix (post ensure_feature_columns), SHAP fallback when TreeExplainer unsupported (e.g. Ridge), Concept: SHAP-based explainability in HTML report

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (2): Rationale: PipelineState TypedDict total=False allows partial-dict merges, Concept: PipelineState TypedDict total=False partial-merge

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Main task — hit /api/predict with a random sample.

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Occasional status check — background health polling.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Maximum PSI across all features. 0.0 if no features were tested.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (0): 

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Fetch carbon intensity readings with emission factors for a time window.

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Fetch generation mix (fuel-type percentages) for a time window.      Returns o

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Simple liveness probe for the MCP server.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Fetch intensity + generationmix for [start_dt, end_dt] and merge on interval sta

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): MLOps Principles: Continuous Training, Experiment Tracking, Drift Detection

## Knowledge Gaps
- **201 isolated node(s):** `Benchmark: parallel vs sequential AutoML candidate training.  Measures the wal`, `A single-threaded version of fetch_window_dataframe().      It deliberately pe`, `Per-run file logging: each script execution gets a new file under logs/.`, `Create logs/ if needed and attach a unique log file for this process run.`, `Extract service name from App Runner ARN.` (+196 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 24`** (2 nodes): `Rationale: PipelineState TypedDict total=False allows partial-dict merges`, `Concept: PipelineState TypedDict total=False partial-merge`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Main task — hit /api/predict with a random sample.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Occasional status check — background health polling.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Maximum PSI across all features. 0.0 if no features were tested.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Fetch carbon intensity readings with emission factors for a time window.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Fetch generation mix (fuel-type percentages) for a time window.      Returns o`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Simple liveness probe for the MCP server.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Fetch intensity + generationmix for [start_dt, end_dt] and merge on interval sta`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `MLOps Principles: Continuous Training, Experiment Tracking, Drift Detection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PipelineState` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 13`, `Community 14`, `Community 15`?**
  _High betweenness centrality (0.143) - this node is a cross-community bridge._
- **Why does `predict()` connect `Community 9` to `Community 0`, `Community 1`, `Community 3`, `Community 6`, `Community 7`, `Community 11`, `Community 16`?**
  _High betweenness centrality (0.061) - this node is a cross-community bridge._
- **Why does `PipelineState` connect `Community 3` to `Community 13`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 82 inferred relationships involving `PipelineState` (e.g. with `LangGraph pipeline graph assembly for the ml605 agent.  Current topology (Slac` and `Compute SHAP values using shap.Explainer.      Returns (shap_values Explanatio`) actually correct?**
  _`PipelineState` has 82 INFERRED edges - model-reasoned connections that need verification._
- **Are the 40 inferred relationships involving `EvalResult` (e.g. with `PipelineState` and `Shared PipelineState TypedDict contract for the ml605 LangGraph agent.  This i`) actually correct?**
  _`EvalResult` has 40 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `DriftReport` (e.g. with `PipelineState` and `Shared PipelineState TypedDict contract for the ml605 LangGraph agent.  This i`) actually correct?**
  _`DriftReport` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `initialize()` (e.g. with `state()` and `.test_initialize_returns_required_keys()`) actually correct?**
  _`initialize()` has 5 INFERRED edges - model-reasoned connections that need verification._