# run_agent_pipeline.py
"""Terminal entry point for the MCP + LangGraph agentic pipeline.

Equivalent to `uv run python -m ml605_agent` but runnable as a plain script:

    uv run python run_agent_pipeline.py

Starts the MCP server subprocess, builds the LangGraph graph, runs one full
pipeline invocation under an MLflow parent run, and prints a summary.

Environment variables (optional):
    PIPELINE_WINDOW_HOURS       Hours of data to fetch (default: 12)
    MLFLOW_EXPERIMENT           MLflow experiment name (default: agentic-pipeline)
"""
from __future__ import annotations

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mlflow

from ml605_agent.__main__ import ensure_mcp_server
from ml605_agent.graph import build_graph
from ml605_agent.state import PipelineState
from ml605_pipeline.config import load_config_from_env


def main() -> None:
    cfg = load_config_from_env()
    experiment = cfg.mlflow_experiment or "agentic-pipeline"

    print(f"[run_agent_pipeline] window_hours={cfg.window_hours}  experiment={experiment}")

    ensure_mcp_server()
    print("[run_agent_pipeline] MCP server ready at http://localhost:8000")

    graph = build_graph()
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name="agent-pipeline-run") as parent_run:
        initial_state: PipelineState = {
            "window_hours": cfg.window_hours,
            "mlflow_run_id": parent_run.info.run_id,
            "status": "running",
            "retrain_done": False,
        }
        result = graph.invoke(initial_state, config={"recursion_limit": 25})

    status = result.get("status", "unknown")
    print(f"\n[run_agent_pipeline] Pipeline complete — status={status}")

    if result.get("error"):
        print(f"[run_agent_pipeline] Error: {result['error']}")
        sys.exit(1)

    if result.get("eval_result"):
        er = result["eval_result"]
        print(f"[run_agent_pipeline] RMSE={er.rmse:.4f}  MAE={er.mae:.4f}  R²={er.r2:.4f}")

    drift = result.get("overall_drift")
    if drift is not None:
        print(f"[run_agent_pipeline] Drift detected: {drift}")
        if drift and result.get("drift_report"):
            drifted = result["drift_report"].drifted_features
            print(f"[run_agent_pipeline] Drifted features: {drifted}")

    if result.get("new_model_version"):
        print(f"[run_agent_pipeline] New model promoted — version={result['new_model_version']}")

    if result.get("report_path"):
        print(f"[run_agent_pipeline] Report: {result['report_path']}")

    print(f"[run_agent_pipeline] MLflow run: {parent_run.info.run_id}")


if __name__ == "__main__":
    main()