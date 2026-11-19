FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/app/src \
    FIA_EMBEDDER=fastembed FIA_RERANKER=fastembed \
    FIA_INDEX_DIR=/app/index FIA_MODEL_CACHE=/app/models \
    FIA_STORE_DIR=/data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data

# Bake models + retrieval index into the image: fast, deterministic start-up (no model download at boot).
RUN python -m fia.cli warmup && mkdir -p /data

EXPOSE 10000
# Render injects $PORT. One worker: SQLite audit log + Chroma are single-process stores.
CMD ["sh", "-c", "uvicorn fia.api:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
