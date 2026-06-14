"""
fusion.py — Late-fusion multimodal pipeline (image + text).

Combines the CLIP image retrieval pipeline (Phase 3) with the DistilBERT intent
classifier (Phase 1) to answer queries where the user provides both an image
and a text question.

Fusion strategy — "entity override":
    - Image modality determines the WHAT (which campus location).
    - Text modality determines the WHY (what the user wants to know about it).
    - If CLIP is confident (score ≥ IMAGE_MATCH_THRESHOLD), the image-identified
      location name overrides the entity extracted from the text.
    - If CLIP is not confident, the text entity is used as normal.

Example:
    User uploads a photo of the library and asks "How late is it open?"
    → CLIP identifies "Learning Resources Centre" (score 0.82, confident)
    → DistilBERT classifies intent as "ask_hours"
    → KB lookup: ask_hours("learning resources centre") → opening hours

Academic justification:
    Late fusion (process modalities independently, combine at decision layer)
    is simpler to analyse and evaluate than early fusion (joint embeddings).
    Each component remains individually evaluable, which satisfies the
    assignment's evaluation rubric. The entity-override approach is also
    fully explainable: the confidence score shows exactly when and why the
    image modality is trusted.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.image_search import search_by_image, UNKNOWN_LOCATION
from src.inference import predict_intent, load_inference_model, format_response
from src.kb_lookup import lookup, get_fallback


def answer_multimodal(
    text: str,
    pil_image: Image.Image,
    entity_candidates: Optional[list[str]] = None,
) -> dict:
    """
    Answer a query that combines a text question with a campus image.

    Pipeline:
        1. CLIP encode image → FAISS search → top location match + confidence
        2. DistilBERT classify text → intent + confidence
        3. Entity selection: image-derived name if confident, else text-extracted entity
        4. KB lookup with (intent, entity)
        5. Format and return response

    Args:
        text              : The user's typed question.
        pil_image         : Decoded PIL Image from the uploaded file.
        entity_candidates : Pre-built KB candidate list for fuzzy entity extraction
                            (passed in from the route handler to avoid reloading).

    Returns:
        Dict with keys:
            reply            — formatted KB response string (markdown)
            intent           — predicted intent label
            confidence       — DistilBERT softmax confidence (0–1)
            image_match      — name of CLIP-matched location, or None
            image_confidence — CLIP cosine similarity score (0–1)
    """
    # ── Step 1: Image retrieval ────────────────────────────────────────────────
    try:
        candidates, is_image_confident = search_by_image(pil_image, top_k=config.IMAGE_TOP_K)
        best_record, image_score = candidates[0] if candidates else (UNKNOWN_LOCATION, 0.0)
    except (RuntimeError, ValueError):
        # FAISS not loaded or empty — treat as text-only
        best_record, image_score, is_image_confident = UNKNOWN_LOCATION, 0.0, False

    # Always keep the top CLIP match name — used for generic "what is this?" queries
    # even when the score is below the normal confidence threshold.
    image_match_name: Optional[str] = best_record.get("name") if best_record != UNKNOWN_LOCATION else None
    # Whether CLIP is confident enough to drive entity selection for non-generic queries
    image_confident_for_entity = is_image_confident

    # ── Step 2: Intent classification ─────────────────────────────────────────
    model, tokenizer = load_inference_model()
    intent, text_confidence = predict_intent(text, model, tokenizer)

    # Generic image-identification phrases → treat as find_location so the
    # image modality drives the response rather than the vague text query.
    _GENERIC_IDENTIFY = {
        "what is this", "what's this", "what is this?", "what's this?",
        "identify this", "where is this", "where is this?",
        "what building is this", "what building is this?",
        "what place is this", "what place is this?",
        "show me this", "this",
    }
    if text.strip().lower() in _GENERIC_IDENTIFY:
        intent = "find_location"
        text_confidence = 1.0

    if text_confidence < config.CONFIDENCE_THRESHOLD and text.strip().lower() not in _GENERIC_IDENTIFY:
        return {
            "reply": get_fallback("low_confidence"),
            "intent": "unknown",
            "confidence": round(text_confidence, 4),
            "image_match": image_match_name,
            "image_confidence": round(image_score, 4),
        }

    # ── Step 3: Entity selection ───────────────────────────────────────────────
    # Navigation queries require the full text so route parsing works correctly.
    if intent == "find_navigation":
        entity = text
    elif image_match_name and (
        image_confident_for_entity
        or text.strip().lower() in _GENERIC_IDENTIFY
    ):
        # Image identified a location — use it as entity.
        # For generic queries ("what is this?"), always trust the top CLIP match
        # even when below the normal confidence threshold.
        entity = image_match_name.lower()
    else:
        # Image not confident — fall back to text-based entity extraction.
        if entity_candidates:
            from src.entity_extractor import extract_entity
            entity = extract_entity(text, entity_candidates)
        else:
            from src.inference import extract_entity as _extract
            entity = _extract(text)

    # ── Step 4: KB lookup ──────────────────────────────────────────────────────
    record = lookup(intent, entity)

    # ── Step 5: Generate response — Groq LLM if enabled, else template ─────────
    reply = None
    if config.USE_LLM:
        from src.llm import generate_response
        # Build a richer query string so the LLM knows an image was provided
        llm_query = text
        if image_match_name:
            llm_query = f"[Image uploaded showing: {image_match_name}] {text}"
        reply = generate_response(intent, entity, record, llm_query, history=None)

    if reply is None:
        reply = format_response(intent, record)

    return {
        "reply":            reply,
        "intent":           intent,
        "confidence":       round(text_confidence, 4),
        "image_match":      image_match_name,
        "image_confidence": round(image_score, 4),
    }
