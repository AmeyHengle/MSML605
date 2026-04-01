"""Entry point for the ml605 agentic pipeline.

Usage:
    uv run python -m ml605_agent

This module:
1. Starts the MCP server subprocess if not already running.
2. Builds the LangGraph pipeline graph.
3. Runs a single pipeline invocation under an MLflow parent run.
4. Prints a summary of the results.
"""
from __future__ import annotations

import atexit
import subprocess
import time
from pathlib import Path

import httpx
import mlflow

from ml605_agent.graph import build_graph
from ml605_agent.state import PipelineState
from ml605_pipeline.config import load_config_from_env

# ---------------------------------------------------------------------------
# MCP server lifecycle
# ---------------------------------------------------------------------------

_mcp_proc: subprocess.Popen | None = None


def _cleanup_mcp_server() -> None:
    """Terminate the MCP server subprocess on exit."""
    global _mcp_proc
    if _mcp_proc is not None:
        _mcp_proc.terminate()
        try:
            _mcp_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _mcp_proc.kill()
        _mcp_proc = None


def ensure_mcp_server() -> None:
    """Ensure the MCP server is running on localhost:8000.

    If already running (health check returns 200), return immediately.
    Otherwise start it as a subprocess and wait up to 15 seconds for startup.

    Raises:
        RuntimeError: If the server doesn't start within 15 seconds.
    """
    global _mcp_proc

    # Check if already running
    try:
        r = httpx.get("http://localhost:8000/health", timeout=1.0)
        if r.status_code == 200:
            return
    except Exception:
        pass

    # Start the MCP server subprocess
    project_root = Path(__file__).resolve().parent.parent.parent
    _mcp_proc = subprocess.Popen(
        ["uv", "run", "python", "src/ml605_mcp/server.py"],
        cwd=str(project_root),
    )
    atexit.register(_cleanup_mcp_server)

    # Poll for up to 15 seconds (30 attempts × 0.5s)
    for _ in range(30):
        try:
            r = httpx.get("http://localhost:8000/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)

    raise RuntimeError("MCP server failed to start within 15 seconds")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full ml605 agentic pipeline."""
    config = load_config_from_env()
    print(f"[ml605_agent] Starting pipeline (window_hours={config.window_hours})")

    ensure_mcp_server()
    print("[ml605_agent] MCP server ready")

    graph = build_graph()
    mlflow.set_experiment("agentic-pipeline")

    with mlflow.start_run(run_name="agent-pipeline-run") as parent_run:
        initial_state: PipelineState = {
            "window_hours": config.window_hours,
            "mlflow_run_id": parent_run.info.run_id,
            "status": "running",
            "retrain_done": False,
        }
        result = graph.invoke(
            initial_state,
            config={"recursion_limit": 25},
        )

    status = result.get("status", "unknown")
    print(f"[ml605_agent] Pipeline complete — status={status}")

    if result.get("error"):
        print(f"[ml605_agent] Error: {result['error']}")

    if result.get("eval_result"):
        er = result["eval_result"]
        print(f"[ml605_agent] RMSE={er.rmse:.4f}  MAE={er.mae:.4f}  R2={er.r2:.4f}")

    if result.get("overall_drift") is not None:
        print(f"[ml605_agent] Drift detected: {result['overall_drift']}")

    if result.get("new_model_version"):
        print(f"[ml605_agent] New model in Staging: version={result['new_model_version']}")


if __name__ == "__main__":
    main()
