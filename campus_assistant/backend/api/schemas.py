"""
schemas.py — Pydantic models for FastAPI request and response validation.

Pydantic automatically validates incoming JSON bodies and serializes
outgoing responses. If the client sends the wrong type, FastAPI returns
a 422 error with a clear explanation — no manual validation needed.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ── Request models (what the frontend sends) ──────────────────────────────────

class TextRequest(BaseModel):
    """Body for POST /chat — a plain text message with optional session tracking."""
    message: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's typed question.",
        example="Where is the main library?",
    )
    session_id: Optional[str] = Field(
        None,
        description="Client-generated UUID4 for conversation memory. Optional.",
        example="550e8400-e29b-41d4-a716-446655440000",
    )


# ── Response models (what the backend returns) ────────────────────────────────

class ChatResponse(BaseModel):
    """Response for POST /chat — intent classification result + KB reply."""
    reply:      str            = Field(..., description="Formatted markdown response from the KB.")
    intent:     str            = Field(..., description="Predicted intent label, e.g. 'ask_hours'.")
    confidence: float          = Field(..., description="Softmax confidence of the prediction (0–1).")
    entity:     Optional[str]  = Field(None, description="Resolved entity used in the KB lookup.")


class VoiceResponse(BaseModel):
    """Response for POST /chat/voice — ASR + intent pipeline."""
    reply:      str = Field(..., description="Response text to show the user.")
    transcript: str = Field(..., description="Transcribed text from audio.")


class ImageMatch(BaseModel):
    """A single CLIP retrieval candidate."""
    name:        str   = Field(..., description="Matched location name.")
    location_id: str   = Field(..., description="KB record id.")
    score:       float = Field(..., description="Cosine similarity score (0–1).")


class ImageResponse(BaseModel):
    """Response for POST /chat/image — CLIP + FAISS image retrieval."""
    reply:       str              = Field(..., description="Formatted KB response for the matched location.")
    description: str              = Field(..., description="Name of the matched campus location or entity.")
    confidence:  float            = Field(0.0, description="CLIP cosine similarity score (0–1).")
    top_matches: list[ImageMatch] = Field(default_factory=list, description="Top-3 CLIP candidates with scores.")


class MultimodalResponse(BaseModel):
    """Response for POST /chat/multimodal — fused image + text query."""
    reply:            str            = Field(..., description="Formatted KB response.")
    intent:           str            = Field(..., description="Predicted intent label.")
    confidence:       float          = Field(..., description="DistilBERT softmax confidence (0–1).")
    image_match:      Optional[str]  = Field(None, description="CLIP-identified location name, or None if not confident.")
    image_confidence: Optional[float] = Field(None, description="CLIP cosine similarity score (0–1).")
    top_matches:      list[ImageMatch] = Field(default_factory=list, description="Top-3 CLIP candidates.")


class ErrorResponse(BaseModel):
    """Returned on server errors."""
    detail: str
