FROM python:3.12-slim AS base

# uv: fast, reproducible installs from the lockfile.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# System libs docling/pypdfium2 need at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first (cached layer) using only the manifest + lockfile.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra api

# Now the source, then the project itself.
COPY src ./src
RUN uv sync --frozen --extra api

ENV EXTRACT_BACKEND=rule \
    REDACT_PII=true \
    PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "contract_rag.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
