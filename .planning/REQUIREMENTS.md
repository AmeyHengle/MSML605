# Requirements: ML605 Agentic MLOps Pipeline

**Defined:** 2026-03-27
**Core Value:** Compress drift detection from days to hours — agents detect, diagnose, report, and alert; humans act via Slack

## v1 Requirements

### Agent Infrastructure

- [x] **AGENT-01**: System has a LangGraph supervisor agent that orchestrates all worker agents via a shared state graph
- [x] **AGENT-02**: Data-fetch worker agent retrieves live data via MCP server tools
- [x] **AGENT-03**: Model-test worker agent runs inference on fetched data and computes performance metrics
- [x] **AGENT-04**: Drift-check worker agent runs multi-test validation (PSI + KS + performance degradation) and returns a drift verdict
- [x] **AGENT-05**: Report-gen worker agent assembles the full HTML analysis report (metrics, SHAP, drift diagnosis, model comparison, LLM summary)
- [x] **AGENT-06**: Alert worker agent sends Slack notifications with drift summary and report link
- [x] **AGENT-07**: Retrain worker agent triggers AutoML retraining and registers the new model in MLflow when drift is confirmed
- [x] **AGENT-08**: Agents operate on a well-defined shared state schema (Pydantic model) passed through the LangGraph graph

### MCP Server

- [x] **MCP-01**: FastMCP server exposes National Grid ESO API as callable tools for agents
- [x] **MCP-02**: MCP server exposes at minimum two tools: `fetch_intensity` and `fetch_generation_mix`
- [x] **MCP-03**: MCP server is reachable by LangGraph agents via `langchain-mcp-adapters`
- [x] **MCP-04**: MCP server runs as a separate process/container from the agent pipeline

### Analysis & Explainability

- [x] **ANALYSIS-01**: SHAP `TreeExplainer` computes feature importance on the current model for each pipeline run
- [x] **ANALYSIS-02**: SHAP summary plot and top-N feature contributions are logged as MLflow artifacts
- [x] **ANALYSIS-03**: Drift validation uses at least three signals before recommending retrain: PSI threshold, KS test p-value, and model RMSE degradation percentage
- [x] **ANALYSIS-04**: HTML report is generated containing: performance metrics table, forecast vs. actual chart, SHAP summary plot, drift verdict with per-feature breakdown, model comparison (current vs. previous production model)
- [x] **ANALYSIS-05**: LLM generates a plain-English 2-3 paragraph summary of findings (drift diagnosis, top drifted features, retrain recommendation) included in the HTML report
- [x] **ANALYSIS-06**: HTML report is saved as a file artifact and logged to MLflow; Slack notification includes a direct link

### Slack Integration

- [ ] **SLACK-01**: Slack bot sends a structured Block Kit message when drift is detected, containing: drift verdict, top drifted features, SHAP top-3 features, and a link to the full HTML report
- [ ] **SLACK-02**: Slack bot sends a pipeline completion summary (no drift case): metrics, model RMSE, forecast summary
- [ ] **SLACK-03**: User can trigger a pipeline run from Slack via slash command or @-mention
- [ ] **SLACK-04**: User can query the current model's status (RMSE, last run timestamp, model version) from Slack
- [ ] **SLACK-05**: User can trigger model deployment (promote retrained model to production) from Slack

### Deployment

- [ ] **DEPLOY-01**: Agent pipeline and MCP server are containerized via Docker (multi-stage build, `python:3.13-slim` base)
- [ ] **DEPLOY-02**: `docker-compose.yml` runs the full system locally (agent pipeline + MCP server + environment config)
- [ ] **DEPLOY-03**: GitHub Actions workflow builds Docker image, pushes to AWS ECR, and deploys to ECS Fargate
- [ ] **DEPLOY-04**: All secrets (Slack token, API keys, LLM keys) are managed via AWS Secrets Manager in production and `.env` file locally
- [ ] **DEPLOY-05**: Existing 29 tests continue to pass; new agent components have their own test coverage

## v2 Requirements

### Human-in-the-Loop Approval

- **HITL-01**: Slack sends interactive Block Kit message with Approve/Reject buttons when retrain is recommended
- **HITL-02**: LangGraph `interrupt()` pauses graph execution pending human Slack response before retraining starts
- **HITL-03**: Human approval/rejection is logged to MLflow with timestamp (MTTA tracking)

### Smart Retraining

- **SMART-01**: Retrain worker uses smart data sampling — oversample data points with highest model error and drift contribution
- **SMART-02**: Sampling strategy and selected data window are logged to MLflow

### Observability Metrics

- **OBS-01**: Pipeline logs MTTD (time from data arrival to drift detection) per run
- **OBS-02**: Pipeline logs MTTD (time from drift detection to alert sent) per run

### LLM Abstraction

- **LLM-01**: LLM backend is abstracted behind a provider interface swappable via `LLM_PROVIDER` env var (Claude / GPT-4o)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Prometheus + Grafana | MLflow covers observability for course demo; Grafana is future milestone |
| Web dashboard (Streamlit/React) | Slack IS the dashboard; no UI framework needed |
| Real-time streaming (Kafka) | National Grid API updates every 30 min — scheduled batch is correct architecture |
| LLM fine-tuning | Prompt engineering on structured data is sufficient |
| Kubernetes orchestration | ECS Fargate is simpler and sufficient for single-service deployment |
| Multi-model ensemble serving | AutoML already picks best model; serving complexity not warranted |
| Agent long-term memory | Each pipeline run is independent; LangGraph checkpointer handles within-run state |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MCP-01 | Phase 1: MCP Server | Complete |
| MCP-02 | Phase 1: MCP Server | Complete |
| MCP-03 | Phase 1: MCP Server | Complete |
| MCP-04 | Phase 1: MCP Server | Complete |
| AGENT-01 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-02 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-03 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-04 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-05 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-06 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-07 | Phase 2: LangGraph Agent Pipeline | Complete |
| AGENT-08 | Phase 2: LangGraph Agent Pipeline | Complete |
| ANALYSIS-01 | Phase 3: Analysis & Explainability | Complete |
| ANALYSIS-02 | Phase 3: Analysis & Explainability | Complete |
| ANALYSIS-03 | Phase 3: Analysis & Explainability | Complete |
| ANALYSIS-04 | Phase 3: Analysis & Explainability | Complete |
| ANALYSIS-05 | Phase 3: Analysis & Explainability | Complete |
| ANALYSIS-06 | Phase 3: Analysis & Explainability | Complete |
| SLACK-01 | Phase 4: Slack Integration | Pending |
| SLACK-02 | Phase 4: Slack Integration | Pending |
| SLACK-03 | Phase 4: Slack Integration | Pending |
| SLACK-04 | Phase 4: Slack Integration | Pending |
| SLACK-05 | Phase 4: Slack Integration | Pending |
| DEPLOY-01 | Phase 5: Docker & AWS Deployment | Pending |
| DEPLOY-02 | Phase 5: Docker & AWS Deployment | Pending |
| DEPLOY-03 | Phase 5: Docker & AWS Deployment | Pending |
| DEPLOY-04 | Phase 5: Docker & AWS Deployment | Pending |
| DEPLOY-05 | Phase 5: Docker & AWS Deployment | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0

---
*Requirements defined: 2026-03-27*
*Last updated: 2026-03-27 after roadmap creation*
