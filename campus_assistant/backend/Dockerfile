FROM python:3.11-slim

WORKDIR /app

# System deps: ffmpeg for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# KB / model paths can be overridden via env vars or docker run -e
ENV KB_PATH=/app/data/campus_kb.json
ENV CSV_PATH=/app/data/faq_dataset.csv
ENV MODEL_SAVE_DIR=/app/models/intent_classifier
ENV FAISS_INDEX_PATH=/app/models/faiss_kb.index
ENV FAISS_ID_MAP_PATH=/app/models/faiss_id_map.json
# GROQ_API_KEY and GROK_API_KEY must be passed at runtime — never bake into image
# docker run -e GROK_API_KEY=gsk_... OR use --env-file .env

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
