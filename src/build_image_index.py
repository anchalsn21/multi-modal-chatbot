"""
build_image_index.py — Offline script to build the FAISS index for image search.

Run once (or whenever campus_kb.json or the image dataset changes):

    python src/build_image_index.py

Strategy:
    For each KB location that has an image_dataset_path folder, encode every
    image in that folder with the CLIP image encoder, then average (mean-pool)
    the resulting embeddings into a single representative vector.  This gives
    a class centroid in CLIP embedding space that generalises well to new
    query images — much better than text-only embeddings.

    Fallback: if a location has no images in its folder (or the folder is
    missing), a text description is encoded instead so the record still
    appears in the index.

Outputs:
    models/faiss_kb.index    — FAISS IndexFlatIP over L2-normalised embeddings
    models/faiss_id_map.json — Ordered list of KB record dicts (same row order as index)
"""

import json
import logging
import os
import sys

import faiss
import numpy as np
from PIL import Image, UnidentifiedImageError

# Allow running from the repo root or from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.image_search import load_clip_model, encode_image, encode_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_PATH = os.path.join(BASE_DIR, "data", "campus_kb.json")
INDEX_PATH = os.path.join(BASE_DIR, "models", "faiss_kb.index")
ID_MAP_PATH = os.path.join(BASE_DIR, "models", "faiss_id_map.json")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Only the locations section contains visually indexable records
KB_SECTIONS = ["locations"]

SECTION_TO_TYPE = {
    "locations": "location",
}


def _build_description(record: dict, record_type: str) -> str:
    """
    Compose a rich text description for a KB record.
    Used as a fallback when no images are available for a location.
    """
    name = record.get("name", "")
    desc = record.get("description", "")
    visual_desc = record.get("visual_description", "")
    tags = record.get("tags", [])
    building = record.get("building", "")
    campus_zone = record.get("campus_zone", "")

    parts = [f"a photo of {name}", "a campus building"]

    if building:
        parts.append(f"located in {building}")
    if campus_zone:
        parts.append(f"in {campus_zone}")

    if visual_desc:
        parts.append(visual_desc)
    elif desc:
        parts.append(desc.split(".")[0].strip())

    if tags:
        parts.append("known for " + ", ".join(tags[:5]))

    return ". ".join(parts) + "."


def _load_images_from_folder(folder_path: str) -> list:
    """Load all images from a folder as PIL RGB images."""
    images = []
    if not os.path.isdir(folder_path):
        return images

    for fname in sorted(os.listdir(folder_path)):
        if os.path.splitext(fname)[1].lower() not in IMAGE_EXTENSIONS:
            continue
        fpath = os.path.join(folder_path, fname)
        try:
            img = Image.open(fpath).convert("RGB")
            images.append(img)
        except (UnidentifiedImageError, OSError) as e:
            logger.warning("  Skipping %s: %s", fname, e)

    return images


def _mean_pool_image_embeddings(images: list) -> np.ndarray:
    """
    Encode each image with CLIP and return the mean-pooled unit-norm vector.

    Mean pooling gives a class centroid — query images are closer to it than
    to any single training image, which improves generalisation.
    """
    vecs = np.stack([encode_image(img) for img in images]).astype("float32")
    avg = vecs.mean(axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg


def build_index() -> None:
    logger.info("Loading campus knowledge base from %s", KB_PATH)
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    logger.info("Loading CLIP model...")
    load_clip_model()

    records = []       # KB record dicts in index order
    embeddings = []    # One embedding vector per record

    for section in KB_SECTIONS:
        entries = kb.get(section, [])
        record_type = SECTION_TO_TYPE[section]
        logger.info("Processing %d records from '%s'", len(entries), section)

        for entry in entries:
            record = dict(entry)
            record["type"] = record_type

            img_dataset_path = record.get("image_dataset_path", "")
            folder_path = os.path.join(BASE_DIR, img_dataset_path) if img_dataset_path else ""

            images = _load_images_from_folder(folder_path) if folder_path else []

            if images:
                logger.info(
                    "  [%s] %s — encoding %d images (mean-pool)",
                    record.get("id", "?"), record.get("name", "?"), len(images)
                )
                vec = _mean_pool_image_embeddings(images)
            else:
                logger.warning(
                    "  [%s] %s — no images found at '%s', falling back to text embedding",
                    record.get("id", "?"), record.get("name", "?"), folder_path
                )
                text = _build_description(record, record_type)
                vec = encode_text(text)

            records.append(record)
            embeddings.append(vec)

    if not records:
        raise RuntimeError("No records found in campus_kb.json. Check KB_SECTIONS.")

    embeddings_np = np.stack(embeddings).astype("float32")
    # Ensure all vectors are unit-norm for cosine similarity via IndexFlatIP
    faiss.normalize_L2(embeddings_np)

    dim = embeddings_np.shape[1]   # 512 for ViT-B/32
    logger.info("Building FAISS IndexFlatIP (dim=%d, n=%d)...", dim, len(records))
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)

    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    logger.info("FAISS index saved to %s", INDEX_PATH)

    with open(ID_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    logger.info("ID map saved to %s (%d records)", ID_MAP_PATH, len(records))

    logger.info("Done. Index is ready for use.")


if __name__ == "__main__":
    build_index()
