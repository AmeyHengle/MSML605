FROM python:3.13-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Install dependencies first (better layer caching)
# Install dependencies first (better layer caching)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev

# App code
COPY main.py pipeline.py monitoring.py ./
COPY src/ ./src/
COPY data/ ./data/

RUN mkdir -p models

# Use the uv virtualenv by default
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
