FROM node:22-alpine AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build


FROM python:3.11-slim

ENV CHAT_DATA_DIR=/app/data \
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen3.5:9b \
    WHISPER_MODEL=base.en \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && python -m playwright install --with-deps chromium

COPY backend ./backend
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

VOLUME ["/app/data"]
EXPOSE 8087

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8087"]
