"""
asr.py — Faster-Whisper ASR helper for Phase 2 voice modality.

Usage:
    Call load_whisper_model() once at startup (inside FastAPI lifespan).
    Call transcribe_audio(path) per request to get a transcript string.
"""

import logging
import time

from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once at startup, reused per request
_whisper_model: WhisperModel | None = None


def load_whisper_model(model_size: str = "small") -> None:
    """Load the Faster-Whisper model into memory. Call once at startup."""
    global _whisper_model
    logger.info("[asr] Loading Faster-Whisper model '%s' on CPU (int8)...", model_size)
    _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
    logger.info("[asr] Whisper model loaded successfully.")


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file and return the full transcript as a string.

    Args:
        audio_path: Absolute or relative path to the audio file (webm, wav, mp3, etc.)

    Returns:
        Transcribed text, stripped of leading/trailing whitespace.

    Raises:
        RuntimeError: If load_whisper_model() has not been called.
    """
    if _whisper_model is None:
        raise RuntimeError("Whisper model not loaded. Call load_whisper_model() first.")

    t0 = time.time()
    segments, _ = _whisper_model.transcribe(audio_path, beam_size=5, language="en")
    transcript = " ".join(seg.text.strip() for seg in segments).strip()
    elapsed = time.time() - t0

    logger.info("[asr] Transcribed %d chars in %.2fs", len(transcript), elapsed)
    return transcript
