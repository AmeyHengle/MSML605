# MSML605

## Architecture

This branch runs a FastAPI backend (`main.py`) on port `8000` with a frontend dashboard in `frontend/`.

It also includes modular pipeline utilities in `src/ml605_pipeline/` and an MCP server in `src/ml605_mcp/` running on port `8001`.

## Runtime

- FastAPI app: `uv run uvicorn main:app --host 0.0.0.0 --port 8000`
- MCP server: `uv run python src/ml605_mcp/server.py`
- Batch drift check: `uv run python run_batch_pipeline.py`

## Drift logic

Drift decisions are PSI/KS-based only.

## Tests

- Unit: `uv run pytest tests/test_pipeline.py -v --tb=short`
- API smoke (server required): `BASE_URL=http://localhost:8000 uv run pytest tests/test_api.py -v --tb=short -k "not TestSimulate"`