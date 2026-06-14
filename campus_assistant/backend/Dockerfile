# Hugging Face Spaces requires a non-root user named "user" with home /home/user
# and the app must listen on port 7860.
FROM python:3.11-slim

# HF Spaces runs as UID 1000 — create matching user
RUN useradd -m -u 1000 user

WORKDIR /app

# System deps: ffmpeg for audio (Whisper ASR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# HF Spaces requirement: non-root user
USER user

EXPOSE 7860

# Data / model paths
ENV KB_PATH=/app/data/campus_kb.json
ENV CSV_PATH=/app/data/faq_dataset.csv
ENV MODEL_SAVE_DIR=/app/models/intent_classifier
ENV FAISS_INDEX_PATH=/app/models/faiss_kb.index
ENV FAISS_ID_MAP_PATH=/app/models/faiss_id_map.json
# GROK_API_KEY must be set as a HF Space secret (Settings → Variables and secrets)

# HF Spaces requires port 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
