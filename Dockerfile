# Cloud Run image for the shopping agent.
# Uses Vertex AI Gemini + Vertex embeddings; no torch / sentence-transformers.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    GOOGLE_CLOUD_PROJECT=shopping-agent-513607 \
    GOOGLE_CLOUD_LOCATION=global \
    VERTEX_CHAT_LOCATION=global \
    VERTEX_EMBED_LOCATION=us-central1 \
    VERTEX_CHAT_MODEL=gemini-3.5-flash \
    VERTEX_FALLBACK_MODEL=gemini-3.1-flash-lite \
    USE_VERTEX=1 \
    USE_LANGGRAPH=1 \
    FIRESTORE_ENABLED=1 \
    FACTS_CACHE_TTL_SEC=8

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.cloudrun.txt .
RUN pip install --no-cache-dir -r requirements.cloudrun.txt

COPY app.py ./
COPY api ./api
COPY database ./database
COPY recommenders ./recommenders
COPY workers ./workers
COPY static ./static
COPY templates ./templates
COPY recommendation.db ./
COPY cf_model.pkl cb_model.pkl cf_trainset.pkl ./
COPY products_vertex.index products_vertex_map.json products_vertex_meta.json ./

EXPOSE 8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
