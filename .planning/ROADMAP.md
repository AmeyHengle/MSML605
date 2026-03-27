# Roadmap: ML605 Agentic MLOps Pipeline

## Overview

This roadmap transforms the existing UK Carbon Intensity forecasting pipeline into an agentic MLOps system. The build follows a strict dependency chain: MCP server (data layer) enables LangGraph agents (orchestration layer), which produce analysis reports (output layer), which are surfaced via Slack (human layer), all containerized for AWS deployment (infrastructure layer). Each phase delivers a complete, independently verifiable capability.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: MCP Server** - Expose National Grid ESO API as agent-callable tools via FastMCP (completed 2026-03-27)
- [ ] **Phase 2: LangGraph Agent Pipeline** - Supervisor + worker agents orchestrating the full pipeline via shared state graph
- [ ] **Phase 3: Analysis & Explainability** - SHAP feature importance, HTML reports, LLM-generated analysis, and multi-test drift validation
- [ ] **Phase 4: Slack Integration** - Two-way Slack bot for drift alerts, pipeline triggers, model queries, and deployment commands
- [ ] **Phase 5: Docker & AWS Deployment** - Containerization, CI/CD pipeline, ECS Fargate deployment, and production secrets management

## Phase Details

### Phase 1: MCP Server
**Goal**: Agents can discover and call National Grid ESO API tools without direct API coupling
**Depends on**: Nothing (first phase)
**Requirements**: MCP-01, MCP-02, MCP-03, MCP-04
**Success Criteria** (what must be TRUE):
  1. FastMCP server starts and responds to tool discovery requests, exposing at least `fetch_intensity` and `fetch_generation_mix`
  2. A LangGraph agent can call MCP tools via `langchain-mcp-adapters` and receive structured data back
  3. MCP server runs as a separate process from the agent code (verified by process isolation in tests)
  4. Existing 29 tests continue to pass after all new dependencies are added to the project
**Plans**: 2 plans

Plans:
- [ ] 01-01-PLAN.md — Install deps (fastmcp, langchain-mcp-adapters, pytest-asyncio), configure asyncio_mode, create src/ml605_mcp package scaffold and failing test stubs
- [ ] 01-02-PLAN.md — Implement src/ml605_mcp/server.py with fetch_intensity and fetch_generation_mix tools, fix run_pipeline.py breakpoint, make all tests pass

### Phase 2: LangGraph Agent Pipeline
**Goal**: A supervisor agent orchestrates specialized workers through a complete pipeline run (fetch, test, drift-check, retrain) using a well-defined shared state
**Depends on**: Phase 1
**Requirements**: AGENT-01, AGENT-02, AGENT-03, AGENT-04, AGENT-05, AGENT-06, AGENT-07, AGENT-08
**Success Criteria** (what must be TRUE):
  1. Running the supervisor agent triggers an end-to-end pipeline: data fetch via MCP, model inference with metrics, drift detection with verdict, and conditional retraining
  2. Each worker agent (data-fetch, model-test, drift-check, report-gen, alert, retrain) executes its specific responsibility and writes results to the shared Pydantic state
  3. The supervisor routes workers in the correct order and terminates within a bounded number of steps (no infinite loops)
  4. When drift is confirmed, the retrain worker triggers AutoML and registers the new model in MLflow
  5. Pipeline state is inspectable after completion -- every field in the shared state schema reflects what happened during the run
**Plans**: TBD

Plans:
- [ ] 02-01: TBD
- [ ] 02-02: TBD
- [ ] 02-03: TBD

### Phase 3: Analysis & Explainability
**Goal**: Every pipeline run produces a comprehensive, human-readable analysis report with SHAP-based explainability and validated drift diagnosis
**Depends on**: Phase 2
**Requirements**: ANALYSIS-01, ANALYSIS-02, ANALYSIS-03, ANALYSIS-04, ANALYSIS-05, ANALYSIS-06
**Success Criteria** (what must be TRUE):
  1. SHAP TreeExplainer computes feature importance for the current model, and the summary plot plus top-N contributions are logged as MLflow artifacts
  2. Drift validation uses three independent signals (PSI threshold, KS p-value, RMSE degradation percentage) before recommending retrain -- not just a single threshold
  3. An HTML report is generated containing: performance metrics table, forecast vs. actual chart, SHAP summary plot, drift verdict with per-feature breakdown, and model comparison (current vs. previous)
  4. The HTML report includes a 2-3 paragraph LLM-generated plain-English summary of findings (drift diagnosis, top drifted features, retrain recommendation)
  5. The HTML report is saved as a file artifact, logged to MLflow, and its path is available in the pipeline state for downstream consumers (Slack)
**Plans**: TBD

Plans:
- [ ] 03-01: TBD
- [ ] 03-02: TBD
- [ ] 03-03: TBD

### Phase 4: Slack Integration
**Goal**: Humans receive actionable drift alerts in Slack and can trigger pipelines, query model status, and promote models -- all without leaving the chat interface
**Depends on**: Phase 3
**Requirements**: SLACK-01, SLACK-02, SLACK-03, SLACK-04, SLACK-05
**Success Criteria** (what must be TRUE):
  1. When drift is detected, a Block Kit message posts to Slack containing: drift verdict, top drifted features, SHAP top-3 features, and a link to the full HTML report
  2. When no drift is detected, a pipeline completion summary posts to Slack with key metrics (RMSE, forecast summary)
  3. A user can trigger a full pipeline run from Slack via slash command or @-mention, and receive the results in the same channel
  4. A user can query the current model's status (RMSE, last run timestamp, model version) from Slack and receive a structured response
  5. A user can trigger model deployment (promote retrained model to production) from Slack
**Plans**: TBD

Plans:
- [ ] 04-01: TBD
- [ ] 04-02: TBD

### Phase 5: Docker & AWS Deployment
**Goal**: The entire system runs in containers locally and deploys to AWS ECS Fargate via GitHub Actions with proper secrets management
**Depends on**: Phase 4
**Requirements**: DEPLOY-01, DEPLOY-02, DEPLOY-03, DEPLOY-04, DEPLOY-05
**Success Criteria** (what must be TRUE):
  1. `docker compose up` starts the full system locally (agent pipeline + MCP server) and a pipeline run completes successfully
  2. GitHub Actions workflow builds the Docker image, pushes to ECR, and deploys a new ECS Fargate task definition
  3. All secrets (Slack token, LLM API keys) are injected at runtime via AWS Secrets Manager in production and `.env` file locally -- never baked into images
  4. Existing 29 tests plus new agent tests all pass in the containerized CI environment
**Plans**: TBD

Plans:
- [ ] 05-01: TBD
- [ ] 05-02: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. MCP Server | 2/2 | Complete   | 2026-03-27 |
| 2. LangGraph Agent Pipeline | 0/3 | Not started | - |
| 3. Analysis & Explainability | 0/3 | Not started | - |
| 4. Slack Integration | 0/2 | Not started | - |
| 5. Docker & AWS Deployment | 0/2 | Not started | - |
