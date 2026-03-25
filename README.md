# MLflow + uv Quickstart

## Install dependencies

```bash
uv sync
```

## 1) Fetch historical data (+ factors)

```bash
uv run python fetch_historical_data.py
```

This writes:
- `historical_data.csv`
- `intensity_factors.json`

And logs both to MLflow.

## 2) Train baseline model with tracking

```bash
uv run python train_with_mlflow.py
```

## 3) Open MLflow UI

```bash
uv run mlflow ui --port 5000
```

Then open http://127.0.0.1:5000

## Logs

Each run of `fetch_historical_data.py`, `train_with_mlflow.py`, or `main.py` writes a new file under `logs/` (see `log_tracking.py`).

## End-to-end daily pipeline (12h window, run twice/day)

This runs: fetch last N hours -> build features -> train -> evaluate -> log everything to MLflow.

```bash
uv run python run_pipeline.py
```

Optional env overrides:

```bash
$env:PIPELINE_WINDOW_HOURS="12"
$env:PIPELINE_INTERVAL_SECONDS="30"
$env:MLFLOW_EXPERIMENT="daily-intensity-pipeline"
uv run python run_pipeline.py
```

To run twice per day on Windows, create two Task Scheduler triggers (e.g. 08:00 and 20:00) that execute:

```bash
uv run python run_pipeline.py
```
