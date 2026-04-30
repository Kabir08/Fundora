# ── Fundora — HuggingFace Docker Space ────────────────────────────────────
# Builds a single container that serves:
#   /ui          → Gradio UI (direct user-facing app)
#   /match       → ChatGPT Action API endpoint
#   /health      → keep-alive ping target
#   /docs        → FastAPI Swagger UI
# HuggingFace Spaces requires port 7860.
# ──────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# System deps needed to compile some sentence-transformers / chromadb wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HuggingFace Spaces runs containers as a non-root user (uid 1000).
# Pre-create writable dirs so ChromaDB and HF model cache can write to disk.
RUN mkdir -p /app/chroma_db /tmp/hf_cache \
    && chmod -R 777 /app/chroma_db /tmp/hf_cache

ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
