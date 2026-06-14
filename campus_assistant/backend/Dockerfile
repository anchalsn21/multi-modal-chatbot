# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# System deps: ffmpeg for audio (Whisper ASR) — runs as root before USER switch above
# so we need to install system packages before the USER directive takes effect.
# Re-run as root just for apt, then drop back to user.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
USER user

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Data / model paths
ENV KB_PATH=/app/data/campus_kb.json
ENV CSV_PATH=/app/data/faq_dataset.csv
ENV MODEL_SAVE_DIR=/app/models/intent_classifier
ENV FAISS_INDEX_PATH=/app/models/faiss_kb.index
ENV FAISS_ID_MAP_PATH=/app/models/faiss_id_map.json
# GROK_API_KEY must be set as a HF Space secret (Settings → Variables and secrets)

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
