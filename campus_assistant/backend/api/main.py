"""
main.py — FastAPI application entry point.

Start the server with:
    uvicorn api.main:app --reload --port 8000

The API will be available at:
    http://localhost:8000
    http://localhost:8000/docs   ← Interactive Swagger UI (great for testing!)
    http://localhost:8000/redoc  ← Alternative ReDoc documentation
"""

import sys
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure the backend root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.routes.chat import router as chat_router
from src.inference import load_inference_model
from src.asr import load_whisper_model
from src.image_search import load_clip_model, load_faiss_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs once when the server starts.

    We load all ML models here so they are ready in memory before the first
    request arrives. Loading on the first request would cause a long delay.
    """
    print("[startup] Loading DistilBERT inference model...")
    try:
        load_inference_model()
        print("[startup] DistilBERT model loaded successfully.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Server will start but /chat will return errors until model is trained.")

    print("[startup] Loading Faster-Whisper ASR model...")
    try:
        load_whisper_model()
        print("[startup] Whisper model loaded successfully.")
    except Exception as e:
        print(f"[startup] WARNING: Whisper failed to load: {e}")
        print("[startup] Server will start but /chat/voice will return errors.")

    print("[startup] Loading CLIP image encoder...")
    try:
        load_clip_model()
        print("[startup] CLIP model loaded successfully.")
    except Exception as e:
        print(f"[startup] WARNING: CLIP failed to load: {e}")
        print("[startup] Server will start but /chat/image will return errors.")

    print("[startup] Loading FAISS campus image index...")
    try:
        load_faiss_index()
        print("[startup] FAISS index loaded successfully. Server is ready.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: {e}")
        print("[startup] Run `python src/build_image_index.py` to generate the index.")
    except Exception as e:
        print(f"[startup] WARNING: FAISS index failed to load: {e}")

    yield  # Server runs here

    print("[shutdown] Server shutting down.")


# ── Create the FastAPI app ────────────────────────────────────────────────────
app = FastAPI(
    title="Greenfield Campus Assistant API",
    description=(
        "A multimodal campus orientation assistant for Greenfield University. "
        "DistilBERT intent classification + JSON knowledge base, "
        "Whisper ASR, and CLIP image retrieval."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS middleware ───────────────────────────────────────────────────────────
# This allows the Next.js frontend (running on port 3000) to call this API.
# Without CORS, browsers block cross-origin requests for security reasons.
_extra_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        *_extra_origins,
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────────────
# The chat router adds:
#   POST /chat
#   POST /chat/voice
#   POST /chat/image
app.include_router(chat_router)


# ── Health check endpoint ─────────────────────────────────────────────────────
@app.get("/", tags=["health"])
async def root():
    """Health check — confirms the server is running."""
    return {
        "status":  "ok",
        "message": "Greenfield Campus Assistant API is running.",
        "docs":    "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    """Detailed health check for monitoring."""
    return {"status": "healthy", "version": "1.0.0"}
