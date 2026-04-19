# AGENTS.md

## Learned User Preferences

- Always use `uv` for Python: run scripts with `uv run python <script>`, install with `uv sync`, and test with `uv run pytest`.
- Replay the task back and confirm understanding before writing code; the user often asks "did you understand the task clearly?" and dislikes unasked-for changes.
- When adding new benchmark or comparison scripts under `metrics_improvements/`, keep them standalone — do not modify or call into existing training/pipeline scripts, and do not log to MLflow or the Model Registry.
- Add progress logging and a timestamped per-run log file for any long-running script; a `logs/` directory and a shared logging helper are expected.
- Interpret imprecise wording and typos charitably, then correct framing mistakes explicitly (e.g. "AutoML vs MLflow" is really "no tuning vs AutoML tuning" since both workflows use MLflow).
- Shell is Windows PowerShell; `pwsh` is not available — use `powershell -NoProfile -Command ...`.
- Run commands from the repo root `F:\ml605-project` and write output paths relative to that root.
- Prefer ranges/presets with a fast "small" path for iteration and a "large" path for the real story (small = sanity check, large = final numbers).

## Learned Workspace Facts

- Project is an MLOps pipeline that predicts UK National Grid carbon intensity using `https://api.carbonintensity.org.uk` (`/intensity`, `/generation`, `/intensity/factors`).
- Source layout: `src/ml605_pipeline/` (core ML/data), `src/ml605_mcp/` (MCP server), `src/ml605_agent/` (LangGraph agent), `src/ml605_slack/` (Bolt + SocketMode bot); tests live in `tests/`.
- Historical dataset is `historical_data.csv` (~105k rows × ~34 cols); feature columns are listed one-per-line in `features_used.txt` and this file must exist before training or running the pipeline.
- Feature pipeline is `add_time_features` (hour, day_of_week, month, day_of_year, is_weekend) → `apply_factor_columns` (constant `factor_*` emission cols) → `one_hot_intensity_index` (low/moderate/high/very high) → `ensure_feature_columns` (pads missing with 0.0).
- Train/test split is chronological 80/20 via `ml605_pipeline.modeling.time_split`; target metric is `rmse` (lower is better); baseline model is `RandomForestRegressor(n_estimators=300, max_depth=14, random_state=42)`.
- AutoML candidate set is `ml605_pipeline.automl.CANDIDATE_MODELS` (RandomForest, ExtraTrees, HistGradientBoosting, GradientBoosting, Ridge).
- MLflow experiments used: `intensity-model-training` (baseline), `intensity-model-automl` (tuned), `daily-intensity-pipeline` (daily runs, overridable via `MLFLOW_EXPERIMENT`), `historical-data-pipeline` (fetch runs).
- Entry points at repo root: `fetch_historical_data.py`, `train_with_mlflow.py`, `train_automl.py`, `run_pipeline.py`, `detect_drift.py`; Slack bot runs via `uv run python -m ml605_slack` and accepts `/ml605 run|status|promote|retrain|report|history`.
- Pipeline env vars: `PIPELINE_WINDOW_HOURS`, `PIPELINE_INTERVAL_SECONDS`, `MLFLOW_EXPERIMENT`; Slack bot needs `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`.
- `metrics_improvements/` is a standalone benchmarking folder for time-to-target comparisons; generated markdown reports (`MLOPS_PERFORMANCE_REPORT.md`, `MANUAL_VS_AUTOML_REPORT.md`) land at repo root.
- `run_pipeline.py` has historically had a stray `breakpoint()` near the top that must be removed before any non-interactive run.
- `CLAUDE.md` at repo root is the canonical onboarding doc; architecture details live there and in `ARCHITECTURE.md` / `CODE-FLOW.md`.
