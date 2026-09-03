FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

EXPOSE 8000

CMD [".venv/bin/uvicorn", "corrective_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
