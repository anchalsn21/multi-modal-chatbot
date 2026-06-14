"""
chat.py — FastAPI route handlers for all chat endpoints.

Endpoints:
    POST /chat            — Text query → DistilBERT intent → KB lookup → response
    POST /chat/voice      — Audio upload → Faster-Whisper ASR → intent pipeline → response
    POST /chat/image      — Image upload → CLIP + FAISS → KB response
    POST /chat/multimodal — Image + text → late-fusion pipeline → KB response  [Phase 4]

Session memory is supported on /chat and /chat/voice via optional session_id in the request body.
The /chat/multimodal endpoint accepts session_id as a form field.

Models are loaded once at startup (in main.py lifespan).
"""

import io
import json
import logging
import os
import sys
import tempfile
import time
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from api.schemas import (
    TextRequest, ChatResponse, VoiceResponse,
    ImageResponse, ImageMatch, MultimodalResponse,
)
from src.inference import answer_query, format_response, get_entity_candidates
from src.asr import transcribe_audio
from src.image_search import search_by_image, UNKNOWN_LOCATION
from src.fusion import answer_multimodal
import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponse,
    summary="Send a text message to the campus assistant",
    description="Accepts a typed question, classifies the intent with DistilBERT, "
                "looks up the campus knowledge base, and returns a formatted answer. "
                "Pass session_id to enable conversation memory.",
)
async def chat_text(request: TextRequest) -> ChatResponse:
    try:
        result = answer_query(request.message, session_id=request.session_id)
        return ChatResponse(
            reply=result["reply"],
            intent=result["intent"],
            confidence=result["confidence"],
            entity=result.get("entity"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _word_stream(result: dict) -> AsyncIterator[str]:
    """
    Yield the reply word by word as Server-Sent Events.

    Each SSE chunk carries a JSON payload so the client also receives
    intent / confidence / entity in the final event without a second request.

    Format:
        data: {"token": "Head "}\n\n        ← one word + trailing space
        ...
        data: {"done": true, "intent": "...", "confidence": 0.97, "entity": "..."}\n\n
    """
    reply: str = result["reply"]
    words = reply.split(" ")
    for i, word in enumerate(words):
        token = word if i == len(words) - 1 else word + " "
        yield f"data: {json.dumps({'token': token})}\n\n"

    # Final event carries metadata
    yield (
        f"data: {json.dumps({'done': True, 'intent': result['intent'], 'confidence': result['confidence'], 'entity': result.get('entity', '')})}\n\n"
    )


@router.post(
    "/stream",
    summary="Stream a text reply word by word (SSE)",
    description="Same pipeline as /chat but streams the reply as Server-Sent Events "
                "so the frontend can render a typing effect.",
)
async def chat_stream(request: TextRequest) -> StreamingResponse:
    try:
        result = answer_query(request.message, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        _word_stream(result),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/transcribe",
    summary="Transcribe audio only — no NLP pipeline",
    description="Accepts an audio file and returns only the transcript. "
                "The frontend uses this to populate the text input so the user "
                "can review and edit before sending.",
)
async def chat_transcribe(audio: UploadFile = File(...)) -> dict:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file received.")

    suffix = os.path.splitext(audio.filename)[-1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as e:
        logger.error("[transcribe] failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    if not transcript:
        raise HTTPException(status_code=422, detail="Audio was empty or could not be transcribed.")

    return {"transcript": transcript}


@router.post(
    "/voice",
    response_model=VoiceResponse,
    summary="Send a voice recording — Faster-Whisper ASR",
    description="Accepts an audio file (webm/wav/mp3), transcribes it with Faster-Whisper, "
                "then passes the transcript through the DistilBERT intent pipeline. "
                "Pass session_id as a form field to enable conversation memory.",
)
async def chat_voice(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> VoiceResponse:
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file received.")

    t0 = time.time()
    suffix = os.path.splitext(audio.filename)[-1] or ".webm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        transcript = transcribe_audio(tmp_path)
    except Exception as e:
        logger.error("[voice] Transcription failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        os.unlink(tmp_path)

    if not transcript:
        raise HTTPException(
            status_code=422,
            detail="Audio was empty or could not be transcribed. Please record again.",
        )

    try:
        result = answer_query(transcript, session_id=session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed = time.time() - t0
    logger.info(
        "[voice] latency=%.2fs | transcript_len=%d | intent=%s",
        elapsed, len(transcript), result.get("intent"),
    )

    return VoiceResponse(reply=result["reply"], transcript=transcript)


# All KB records now have type "location" — single entry needed
_TYPE_TO_INTENT = {
    "location": "find_location",
}


@router.post(
    "/image",
    response_model=ImageResponse,
    summary="Send a campus image — CLIP + FAISS retrieval",
    description=(
        "Accepts an image file (JPEG, PNG, WebP, etc.), encodes it with CLIP, "
        "and retrieves the best-matching campus location from the FAISS index. "
        "Returns a formatted knowledge-base response with a confidence score."
    ),
)
async def chat_image(image: UploadFile = File(...)) -> ImageResponse:
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported media type '{content_type}'. "
                "Please upload an image file (JPEG, PNG, WebP, etc.)."
            ),
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as an image. "
                   "The file may be corrupt or in an unsupported format.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {e}")

    t0 = time.time()
    try:
        candidates, is_confident = search_by_image(pil_image, top_k=config.IMAGE_TOP_K)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error("[image] search_by_image failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Image search error: {e}")

    elapsed = time.time() - t0
    best_record, best_score = candidates[0] if candidates else (UNKNOWN_LOCATION, 0.0)

    logger.info(
        "[image] latency=%.2fs | match=%s | score=%.4f",
        elapsed, best_record.get("name"), best_score,
    )

    top_matches = [
        ImageMatch(
            name=r.get("name", "Unknown"),
            location_id=r.get("id", "unknown"),
            score=round(s, 4),
        )
        for r, s in candidates
    ]

    if not is_confident:
        return ImageResponse(
            reply=(
                "I couldn't confidently match your image to a campus location. "
                "Try uploading a clearer photo, or describe what you see in the text box."
            ),
            description="No confident match found.",
            confidence=round(best_score, 4),
            top_matches=top_matches,
        )

    intent = _TYPE_TO_INTENT.get(best_record.get("type", "location"), "find_location")
    reply = format_response(intent, best_record)

    return ImageResponse(
        reply=reply,
        description=best_record.get("name", "Unknown location"),
        confidence=round(best_score, 4),
        top_matches=top_matches,
    )


@router.post(
    "/multimodal",
    response_model=MultimodalResponse,
    summary="Send a text question alongside a campus image — late-fusion pipeline",
    description=(
        "Accepts an image file and a text question together. "
        "CLIP identifies the campus location from the image; "
        "DistilBERT classifies the intent from the text. "
        "The two signals are fused: if CLIP is confident, the image-identified "
        "location overrides the text-extracted entity for the KB lookup. "
        "Pass session_id as a form field to enable conversation memory."
    ),
)
async def chat_multimodal(
    message: str = Form(..., min_length=1, max_length=500),
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
) -> MultimodalResponse:
    """
    Multimodal endpoint — image + text → late-fusion answer.

    The image identifies the WHAT (which location).
    The text identifies the WHY (what the user wants to know).
    """
    # ── Validate and decode image ─────────────────────────────────────────────
    content_type = image.content_type or ""
    if not content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{content_type}'. Please upload an image file.",
        )

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Could not decode the uploaded file as an image.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decoding error: {e}")

    # ── Run fusion pipeline ───────────────────────────────────────────────────
    t0 = time.time()
    try:
        result = answer_multimodal(
            text=message,
            pil_image=pil_image,
            entity_candidates=get_entity_candidates(),
        )
    except Exception as e:
        logger.error("[multimodal] fusion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Multimodal processing error: {e}")

    elapsed = time.time() - t0
    logger.info(
        "[multimodal] latency=%.2fs | intent=%s | image_match=%s | image_conf=%.4f",
        elapsed, result.get("intent"), result.get("image_match"), result.get("image_confidence", 0.0),
    )

    # ── Update session context ────────────────────────────────────────────────
    if session_id and result.get("intent") != "unknown":
        from src.context import update_context
        entity = result.get("image_match") or message
        update_context(
            session_id,
            entity=entity.lower() if entity else message,
            intent=result["intent"],
            reply=result["reply"],
        )

    # Build top_matches from image search for the response
    top_matches: list[ImageMatch] = []
    if result.get("image_match"):
        top_matches = [
            ImageMatch(
                name=result["image_match"],
                location_id="matched",
                score=round(result.get("image_confidence", 0.0), 4),
            )
        ]

    return MultimodalResponse(
        reply=result["reply"],
        intent=result["intent"],
        confidence=result["confidence"],
        image_match=result.get("image_match"),
        image_confidence=result.get("image_confidence"),
        top_matches=top_matches,
    )
