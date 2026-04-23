# Graph Report - .  (2026-04-22)

## Corpus Check
- Large corpus: 466 files · ~87,225 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 678 nodes · 1206 edges · 29 communities detected
- Extraction: 62% EXTRACTED · 38% INFERRED · 0% AMBIGUOUS · INFERRED: 458 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Report Orchestration|Report Orchestration]]
- [[_COMMUNITY_Drift & Evaluation Metrics|Drift & Evaluation Metrics]]
- [[_COMMUNITY_Slack Bot Handlers|Slack Bot Handlers]]
- [[_COMMUNITY_Architecture Rationale Notes|Architecture Rationale Notes]]
- [[_COMMUNITY_Tuning Report Builders|Tuning Report Builders]]
- [[_COMMUNITY_Feature Engineering Entrypoints|Feature Engineering Entrypoints]]
- [[_COMMUNITY_Slack Slash Commands|Slack Slash Commands]]
- [[_COMMUNITY_Data Fetch & History|Data Fetch & History]]
- [[_COMMUNITY_AutoML Training|AutoML Training]]
- [[_COMMUNITY_Slack Block Kit Messages|Slack Block Kit Messages]]
- [[_COMMUNITY_Hyperparameter Tuning Benchmark|Hyperparameter Tuning Benchmark]]
- [[_COMMUNITY_Alert Worker & Tests|Alert Worker & Tests]]
- [[_COMMUNITY_LangGraph Worker Nodes|LangGraph Worker Nodes]]
- [[_COMMUNITY_MLflow Model Registry|MLflow Model Registry]]
- [[_COMMUNITY_MCP Server Tests|MCP Server Tests]]
- [[_COMMUNITY_FastMCP Server|FastMCP Server]]
- [[_COMMUNITY_Drift PSI Calculation|Drift PSI Calculation]]
- [[_COMMUNITY_Runtime Comparison Analysis|Runtime Comparison Analysis]]
- [[_COMMUNITY_Multithreading Benchmark|Multithreading Benchmark]]
- [[_COMMUNITY_Pipeline Data Flow|Pipeline Data Flow]]
- [[_COMMUNITY_SHAP Explainability|SHAP Explainability]]
- [[_COMMUNITY_PipelineState Design|PipelineState Design]]
- [[_COMMUNITY_Agent Package Init|Agent Package Init]]
- [[_COMMUNITY_Pipeline Package Init|Pipeline Package Init]]
- [[_COMMUNITY_Overall PSI Metric|Overall PSI Metric]]
- [[_COMMUNITY_Slack Package Init|Slack Package Init]]
- [[_COMMUNITY_MCP Package Init|MCP Package Init]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_MLOps Principles|MLOps Principles]]

## God Nodes (most connected - your core abstractions)
1. `PipelineState` - 56 edges
2. `EvalResult` - 40 edges
3. `DriftReport` - 38 edges
4. `report_worker()` - 18 edges
5. `FeatureDriftResult` - 18 edges
6. `handle_ml605_command()` - 15 edges
7. `main()` - 14 edges
8. `alert_worker()` - 13 edges
9. `drift_worker()` - 13 edges
10. `fetch_window_dataframe()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Render the Jinja2 HTML template with the given context dict.` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py
- `Generate SHAP explainability, HTML report, LLM summary, and log to MLflow.` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py
- `Post pipeline results to Slack (per D-06, D-07, D-10).      Reads SLACK_BOT_TO` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py
- `Route after drift_worker.      - status=error → error_handler     - all other` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py
- `Route after report_worker (replaces the old HITL decision node).      - status` --uses--> `PipelineState`  [INFERRED]
  src\ml605_agent\graph.py → src\ml605_agent\state.py

## Hyperedges (group relationships)
- **All LangGraph workers participating in the agentic pipeline flow** — architecture_fetch_worker, architecture_feature_worker, architecture_test_worker, architecture_drift_worker, architecture_retrain_worker, architecture_report_worker, architecture_alert_worker, architecture_error_handler [EXTRACTED 0.95]
- **Three combined drift signals forming overall_drift verdict** — concept_psi_drift_signal, concept_ks_drift_signal, concept_rmse_degradation_signal [EXTRACTED 0.95]
- **Steps required to restore Slack+HITL integration** — slack_rewire_graph_step, slack_staging_first_option, slack_memorysaver_required, architecture_hitl_decision_node, architecture_alert_worker [EXTRACTED 0.90]
- **MLOps Code Flow Map: Carbon Intensity Model Lifecycle** — architecture_historical_csv, architecture_live_api, architecture_train_automl_py, architecture_detect_drift_py, architecture_drift_decision, architecture_mlflow_model_registry_staging, architecture_mlflow_model_registry_production, architecture_run_pipeline_py [EXTRACTED 0.90]

## Communities

### Community 0 - "Report Orchestration"
Cohesion: 0.04
Nodes (73): _build_model_comparison(), _compute_shap(), error_handler(), _fallback_summary(), _fig_to_base64(), _forecast_chart_to_base64(), _generate_llm_summary(), hitl_decision_node() (+65 more)

### Community 1 - "Drift & Evaluation Metrics"
Cohesion: 0.06
Nodes (72): DriftReport, FeatureDriftResult, compute_metrics(), EvalResult, Compute regression metrics. Zero targets are excluded from MAPE., Render the Jinja2 HTML template with the given context dict., Generate SHAP explainability, HTML report, LLM summary, and log to MLflow., _render_report() (+64 more)

### Community 2 - "Slack Bot Handlers"
Cohesion: 0.03
Nodes (65): _build_status_blocks(), create_app(), _get_latest_report_path(), _get_or_create_graph(), _handle_promote(), _handle_report(), _handle_retrain_background(), Slack Bolt App with Socket Mode, slash commands, and action handlers.  Per D-0 (+57 more)

### Community 3 - "Architecture Rationale Notes"
Cohesion: 0.04
Nodes (61): metrics_improvements scripts must remain standalone (no MLflow), Learned user preferences (uv, PowerShell, replay-back), Learned workspace facts (layout, historical_data.csv, env vars), Agent layer (LangGraph StateGraph in src/ml605_agent), Rationale: features.py chain is composable DataFrame->DataFrame, ARCHITECTURE Diagram 3 — data flow, Rationale: ensure_feature_columns guarantees inference-time column alignment, Rationale: MCP decouples agent from raw API for swap/mock (+53 more)

### Community 4 - "Tuning Report Builders"
Cohesion: 0.09
Nodes (35): _build_conclusion(), _df_to_markdown(), _fmt_metric(), _fmt_percent(), generate_tuning_report(), main(), parse_args(), Render a DataFrame as a GitHub-flavored markdown table.      Prefers pandas.to (+27 more)

### Community 5 - "Feature Engineering Entrypoints"
Cohesion: 0.09
Nodes (28): main(), add_time_features(), apply_factor_columns(), ensure_feature_columns(), load_feature_list(), normalize_factor_name(), one_hot_intensity_index(), Per-run file logging: each script execution gets a new file under logs/. (+20 more)

### Community 6 - "Slack Slash Commands"
Cohesion: 0.08
Nodes (26): handle_ml605_command(), Single slash command dispatcher for /ml605.      This is a module-level functi, _make_mocks(), Unit tests for slash command handlers (bot.py).  Tests use unittest.mock to is, SLACK-03: /ml605 retrain forces retraining., ack() is called with text containing 'retrain'., SLACK-03: /ml605 report uploads HTML., files_upload_v2 called with a .html file path. (+18 more)

### Community 7 - "Data Fetch & History"
Cohesion: 0.13
Nodes (23): A single-threaded version of fetch_window_dataframe().      It deliberately pe, _sequential_fetch_window_dataframe(), _build_history_blocks(), _get_recent_runs(), Build Block Kit blocks for /ml605 history (per D-11)., Query MLflow for last N runs from the given experiment (per D-11)., fetch_intensity_factors(), fetch_window_dataframe() (+15 more)

### Community 8 - "AutoML Training"
Cohesion: 0.11
Nodes (22): AutoMLResult, _evaluate_candidate(), ModelCandidate, Train all candidates concurrently using a thread pool.      Threading is safe, Train all CANDIDATE_MODELS, log each as a nested MLflow run.     Returns AutoML, Train one model and return its evaluation metrics. Does NOT start an MLflow run., Train all candidates one after another (legacy path, used for benchmarking)., run_automl() (+14 more)

### Community 9 - "Slack Block Kit Messages"
Cohesion: 0.09
Nodes (18): build_drift_alert_blocks(), build_no_drift_blocks(), Block Kit message builder functions (pure, testable).  All functions return li, Build Block Kit blocks for a no-drift pipeline summary (per D-07).      Return, Build Block Kit blocks for a drift-detected alert (per D-06).      Returns a l, Unit tests for Block Kit message builders (blocks.py)., SLACK-02: No-drift pipeline summary., RMSE and MAE metrics appear in message. (+10 more)

### Community 10 - "Hyperparameter Tuning Benchmark"
Cohesion: 0.15
Nodes (20): ApproachSummary, ComboResult, _fmt_seconds(), _iter_grid(), main(), parse_args(), Manual hyperparameter grid search vs AutoML - wall-clock comparison.  This scr, run_automl_candidates() (+12 more)

### Community 11 - "Alert Worker & Tests"
Cohesion: 0.11
Nodes (19): alert_worker(), Post pipeline results to Slack (per D-06, D-07, D-10).      Reads SLACK_BOT_TO, _make_drift_state(), _make_no_drift_state(), Tests for alert_worker Slack posting.  Tests that alert_worker posts drift ale, alert_worker posts no-drift summary., chat_postMessage called with no-drift blocks when overall_drift=False., files_upload_v2 called with report_path for no-drift run. (+11 more)

### Community 12 - "LangGraph Worker Nodes"
Cohesion: 0.09
Nodes (26): alert_worker node (stub — Phase 4 Slack), drift_worker node, error_handler terminal node, feature_worker node, fetch_worker node, hitl_decision_node (LangGraph interrupt-based gate), report_worker node (SHAP + HTML + LLM summary), retrain_done guard prevents back-edge infinite loop (+18 more)

### Community 13 - "MLflow Model Registry"
Cohesion: 0.15
Nodes (16): get_production_model_uri(), load_production_model(), Register a trained model in MLflow Model Registry. Returns version string., Transition model version to 'Staging' or 'Production'., Return URI of the current Production model, or None if none registered., Load and return the current Production model. Returns None if none registered., register_model(), transition_model_stage() (+8 more)

### Community 14 - "MCP Server Tests"
Cohesion: 0.11
Nodes (17): mock_fetch_result(), Tests for the ml605_mcp FastMCP server.  Requirements covered:   MCP-01: Fast, MCP-02: fetch_generation_mix returns correct schema keys (no 'factors' key)., MCP-02: Both tools accept hours_back and start_dt/end_dt parameters., Start the MCP server as a subprocess and wait for /health to respond., MCP-04: Server starts as standalone process and /health responds 200 OK., MCP-03: MultiServerMCPClient can discover tools from running server via HTTP., Return a mock WindowFetchResult with minimal plausible data. (+9 more)

### Community 15 - "FastMCP Server"
Cohesion: 0.23
Nodes (11): _df_to_records(), fetch_generation_mix(), fetch_intensity(), health_check(), FastMCP server exposing National Grid ESO carbon intensity API as MCP tools., Fetch generation mix (fuel-type percentages) for a time window.      Returns o, Simple liveness probe for the MCP server., Return (start, end) UTC datetimes for the requested window.      If both start (+3 more)

### Community 16 - "Drift PSI Calculation"
Cohesion: 0.27
Nodes (10): _compute_psi(), detect_drift(), Population Stability Index.     PSI < 0.1: no change. 0.1-0.25: moderate. >= 0., Compare current data distribution against reference (training) distribution., test_detect_drift_detects_large_shift(), test_detect_drift_no_drift_on_same_data(), test_detect_drift_skips_missing_columns(), test_feature_drift_result_fields() (+2 more)

### Community 17 - "Runtime Comparison Analysis"
Cohesion: 0.42
Nodes (8): compare_single_run_runtime(), compare_time_to_target(), compute_time_to_target(), load_runs(), main(), metric_hits_target(), parse_args(), run_command()

### Community 18 - "Multithreading Benchmark"
Cohesion: 0.33
Nodes (5): BenchmarkResult, main(), _print_metrics(), _run_benchmark(), _save_plots()

### Community 19 - "Pipeline Data Flow"
Cohesion: 0.29
Nodes (8): detect_drift.py (MLflow drift run), drift? (decision), Historical CSV, Live API, MLflow Model Registry Production (manual promote), MLflow Model Registry 'carbon-intensity-model' Staging, run_pipeline.py (uses existing Production or retrains inline), train_automl.py (AutoML: 5 models compared, nested MLflow runs per model)

### Community 20 - "SHAP Explainability"
Cohesion: 0.67
Nodes (3): SHAP computed on aligned feature matrix (post ensure_feature_columns), SHAP fallback when TreeExplainer unsupported (e.g. Ridge), Concept: SHAP-based explainability in HTML report

### Community 21 - "PipelineState Design"
Cohesion: 1.0
Nodes (2): Rationale: PipelineState TypedDict total=False allows partial-dict merges, Concept: PipelineState TypedDict total=False partial-merge

### Community 22 - "Agent Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Pipeline Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Overall PSI Metric"
Cohesion: 1.0
Nodes (1): Maximum PSI across all features. 0.0 if no features were tested.

### Community 25 - "Slack Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "MCP Package Init"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Test Fixtures"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "MLOps Principles"
Cohesion: 1.0
Nodes (1): MLOps Principles: Continuous Training, Experiment Tracking, Drift Detection

## Knowledge Gaps
- **157 isolated node(s):** `Benchmark: parallel vs sequential AutoML candidate training.  Measures the wal`, `A single-threaded version of fetch_window_dataframe().      It deliberately pe`, `Per-run file logging: each script execution gets a new file under logs/.`, `Create logs/ if needed and attach a unique log file for this process run.`, `Render a DataFrame as a GitHub-flavored markdown table.      Prefers pandas.to` (+152 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `PipelineState Design`** (2 nodes): `Rationale: PipelineState TypedDict total=False allows partial-dict merges`, `Concept: PipelineState TypedDict total=False partial-merge`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Agent Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Pipeline Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Overall PSI Metric`** (1 nodes): `Maximum PSI across all features. 0.0 if no features were tested.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Slack Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `MCP Package Init`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Test Fixtures`** (1 nodes): `conftest.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `MLOps Principles`** (1 nodes): `MLOps Principles: Continuous Training, Experiment Tracking, Drift Detection`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `PipelineState` connect `Report Orchestration` to `Drift & Evaluation Metrics`, `Slack Bot Handlers`, `Feature Engineering Entrypoints`, `Alert Worker & Tests`, `MLflow Model Registry`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `main()` connect `Feature Engineering Entrypoints` to `AutoML Training`, `MLflow Model Registry`, `Data Fetch & History`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Why does `load_data_split()` connect `Feature Engineering Entrypoints` to `Hyperparameter Tuning Benchmark`?**
  _High betweenness centrality (0.058) - this node is a cross-community bridge._
- **Are the 54 inferred relationships involving `PipelineState` (e.g. with `LangGraph pipeline graph assembly for the ml605 agent.  Current topology (Slac` and `Compute SHAP values using shap.Explainer.      Returns (shap_values Explanatio`) actually correct?**
  _`PipelineState` has 54 INFERRED edges - model-reasoned connections that need verification._
- **Are the 37 inferred relationships involving `EvalResult` (e.g. with `PipelineState` and `Shared PipelineState TypedDict contract for the ml605 LangGraph agent.  This i`) actually correct?**
  _`EvalResult` has 37 INFERRED edges - model-reasoned connections that need verification._
- **Are the 36 inferred relationships involving `DriftReport` (e.g. with `PipelineState` and `Shared PipelineState TypedDict contract for the ml605 LangGraph agent.  This i`) actually correct?**
  _`DriftReport` has 36 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `report_worker()` (e.g. with `load_production_model()` and `.get()`) actually correct?**
  _`report_worker()` has 9 INFERRED edges - model-reasoned connections that need verification._