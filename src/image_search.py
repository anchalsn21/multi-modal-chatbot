"""
image_search.py — CLIP image encoder + FAISS similarity search.

Usage:
    # At startup (called once from main.py lifespan):
    load_clip_model()
    load_faiss_index()

    # Per request:
    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    record, score = search_by_image(pil_img)
"""

import io
import json
import logging
import os
from typing import Optional

import faiss
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once at startup) ─────────────────────────

_clip_processor = None
_clip_model = None
_faiss_index: Optional[faiss.Index] = None
_id_map: Optional[list] = None   # list[dict] — same order as FAISS index rows

# Sentinel returned when no confident match is found
UNKNOWN_LOCATION: dict = {
    "id": "unknown",
    "name": "Unknown Location",
    "type": "location",
    "description": "No confident match found.",
}


# ── Startup functions ─────────────────────────────────────────────────────────

def load_clip_model() -> None:
    """Load CLIP processor and model into module-level singletons."""
    global _clip_processor, _clip_model

    if _clip_model is not None:
        return  # already loaded

    from transformers import CLIPProcessor, CLIPModel

    model_name = "openai/clip-vit-base-patch32"
    logger.info("[CLIP] Loading %s...", model_name)

    _clip_processor = CLIPProcessor.from_pretrained(model_name)
    _clip_model = CLIPModel.from_pretrained(model_name)
    _clip_model.eval()  # disable dropout

    logger.info("[CLIP] Model loaded successfully.")


def load_faiss_index(index_path: Optional[str] = None, id_map_path: Optional[str] = None) -> None:
    """Load the pre-built FAISS index and its record id_map from disk."""
    global _faiss_index, _id_map

    if _faiss_index is not None:
        return  # already loaded

    if index_path is None or id_map_path is None:
        # Derive default paths relative to this file's location (src/)
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_path = os.path.join(base, "models", "faiss_kb.index")
        id_map_path = os.path.join(base, "models", "faiss_id_map.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run `python src/build_image_index.py` first."
        )
    if not os.path.exists(id_map_path):
        raise FileNotFoundError(
            f"FAISS id_map not found at {id_map_path}. "
            "Run `python src/build_image_index.py` first."
        )

    logger.info("[FAISS] Loading index from %s...", index_path)
    _faiss_index = faiss.read_index(index_path)

    with open(id_map_path, "r", encoding="utf-8") as f:
        _id_map = json.load(f)

    logger.info("[FAISS] Index loaded: %d records.", _faiss_index.ntotal)


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _to_unit_vec(tensor) -> np.ndarray:
    """Convert a (1, D) torch tensor to a normalised (D,) float32 numpy array."""
    vec = tensor.cpu().numpy().astype("float32")  # (1, D)
    faiss.normalize_L2(vec)                        # in-place unit norm
    return vec[0]                                  # (D,)


def encode_image(pil_image: Image.Image) -> np.ndarray:
    """
    Encode a PIL image with CLIP and return an L2-normalised float32 vector.

    Uses the full CLIPModel's vision encoder + visual projection to produce
    a 512-d embedding in the shared vision-language space.

    Returns:
        np.ndarray of shape (512,), dtype float32, unit norm.
    """
    import torch

    if _clip_processor is None or _clip_model is None:
        raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")

    inputs = _clip_processor(images=pil_image, return_tensors="pt")

    with torch.no_grad():
        # vision_model → (1, 768) pooler_output → visual_projection → (1, 512)
        vision_out = _clip_model.vision_model(**inputs)
        pooled = vision_out.pooler_output          # (1, 768)
        projected = _clip_model.visual_projection(pooled)  # (1, 512)

    return _to_unit_vec(projected)


def encode_text(text: str) -> np.ndarray:
    """
    Encode a text string with CLIP and return an L2-normalised float32 vector.

    Uses the full CLIPModel's text encoder + text projection to produce
    a 512-d embedding in the shared vision-language space.

    Used by build_image_index.py to build the KB text embeddings.

    Returns:
        np.ndarray of shape (512,), dtype float32, unit norm.
    """
    import torch

    if _clip_processor is None or _clip_model is None:
        raise RuntimeError("CLIP model not loaded. Call load_clip_model() first.")

    inputs = _clip_processor(text=[text], return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        # text_model → (1, 512) pooler_output → text_projection → (1, 512)
        text_out = _clip_model.text_model(**inputs)
        pooled = text_out.pooler_output            # (1, 512)
        projected = _clip_model.text_projection(pooled)    # (1, 512)

    return _to_unit_vec(projected)


# ── Search ────────────────────────────────────────────────────────────────────

def search_by_image(
    pil_image: Image.Image,
    top_k: int = 3,
) -> tuple[list, bool]:
    """
    Find the top-k best-matching campus KB records for an image.

    Args:
        pil_image : PIL Image (any mode; RGB conversion is handled internally).
        top_k     : Number of nearest neighbours to retrieve (default 3).

    Returns:
        (candidates, is_confident) where:
          - candidates is a list of up to top_k (record_dict, cosine_score) tuples
            sorted descending by score.
          - is_confident is True when the best score meets IMAGE_MATCH_THRESHOLD.

    Raises:
        RuntimeError : If CLIP model or FAISS index are not loaded.
        ValueError   : If the index is empty.
    """
    import config

    if _faiss_index is None or _id_map is None:
        raise RuntimeError("FAISS index not loaded. Call load_faiss_index() first.")
    if _faiss_index.ntotal == 0:
        raise ValueError("FAISS index is empty.")

    pil_image = pil_image.convert("RGB")
    query_vec = encode_image(pil_image).reshape(1, -1)   # (1, 512)

    scores, indices = _faiss_index.search(query_vec, top_k)
    candidates = [
        (_id_map[int(indices[0][i])], float(scores[0][i]))
        for i in range(top_k)
        if int(indices[0][i]) < len(_id_map)
    ]

    best_score = candidates[0][1] if candidates else 0.0
    is_confident = best_score >= config.IMAGE_MATCH_THRESHOLD
    return candidates, is_confident
