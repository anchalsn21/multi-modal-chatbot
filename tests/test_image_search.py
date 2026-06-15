"""
tests/test_image_search.py — Unit tests for src/image_search.py

Run with:
    pytest tests/test_image_search.py -v

These tests mock the CLIP model and FAISS index so they run without GPU or
a pre-built index file — making them suitable for CI and fast local runs.
"""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pytest
from PIL import Image

# Allow imports from the backend root
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.image_search as image_search


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset module-level singletons before each test to avoid state leakage."""
    image_search._clip_processor = None
    image_search._clip_model = None
    image_search._faiss_index = None
    image_search._id_map = None
    yield
    image_search._clip_processor = None
    image_search._clip_model = None
    image_search._faiss_index = None
    image_search._id_map = None


def _make_fake_clip():
    """Return mock CLIPProcessor and CLIPModel that produce unit-norm 512-d tensors."""
    import torch

    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.zeros(1, 3, 224, 224)}

    # Produce a real numpy-backed tensor so faiss.normalize_L2 is happy
    unit_vec = torch.ones(1, 512) / (512 ** 0.5)  # unit-norm

    vision_out = MagicMock()
    vision_out.pooler_output = unit_vec

    text_out = MagicMock()
    text_out.pooler_output = unit_vec

    model = MagicMock()
    model.vision_model.return_value = vision_out
    model.text_model.return_value = text_out
    model.visual_projection.return_value = unit_vec
    model.text_projection.return_value = unit_vec
    model.eval.return_value = model

    return processor, model


def _make_fake_index(n: int = 3):
    """Return a tiny FAISS index and matching id_map list."""
    dim = 512
    vecs = np.random.randn(n, dim).astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    id_map = [
        {"id": f"loc_00{i}", "name": f"Place {i}", "type": "location"}
        for i in range(n)
    ]
    return index, id_map


# ── load_clip_model ───────────────────────────────────────────────────────────

def test_load_clip_model_sets_singletons():
    proc, model = _make_fake_clip()

    # load_clip_model imports transformers lazily inside the function,
    # so we patch at the transformers module level.
    from transformers import CLIPProcessor as RealProc, CLIPModel as RealModel
    with patch.object(RealProc, "from_pretrained", return_value=proc), \
         patch.object(RealModel, "from_pretrained", return_value=model):
        image_search.load_clip_model()

    assert image_search._clip_processor is proc
    assert image_search._clip_model is model


def test_load_clip_model_idempotent():
    """Calling load_clip_model() twice must not re-load the model."""
    proc, model = _make_fake_clip()
    image_search._clip_processor = proc
    image_search._clip_model = model

    # Should return immediately without importing transformers again
    image_search.load_clip_model()
    assert image_search._clip_processor is proc  # same object


# ── encode_image ──────────────────────────────────────────────────────────────

def test_encode_image_shape_and_norm():
    """encode_image must return a (512,) unit-norm float32 array."""
    proc, model = _make_fake_clip()
    image_search._clip_processor = proc
    image_search._clip_model = model

    pil = Image.new("RGB", (224, 224), color=(128, 64, 32))
    vec = image_search.encode_image(pil)

    assert vec.shape == (512,), f"Expected (512,) but got {vec.shape}"
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 1e-5, f"Vector is not unit-norm: {norm}"


def test_encode_image_raises_if_model_not_loaded():
    """encode_image must raise RuntimeError when CLIP is not loaded."""
    pil = Image.new("RGB", (64, 64))
    with pytest.raises(RuntimeError, match="CLIP model not loaded"):
        image_search.encode_image(pil)


# ── load_faiss_index ──────────────────────────────────────────────────────────

def test_load_faiss_index_missing_raises():
    """load_faiss_index must raise FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError):
        image_search.load_faiss_index(
            index_path="/nonexistent/faiss_kb.index",
            id_map_path="/nonexistent/faiss_id_map.json",
        )


def test_load_faiss_index_from_disk():
    """load_faiss_index must populate _faiss_index and _id_map from real files."""
    index, id_map = _make_fake_index(n=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = os.path.join(tmpdir, "faiss_kb.index")
        map_path = os.path.join(tmpdir, "faiss_id_map.json")

        faiss.write_index(index, idx_path)
        with open(map_path, "w") as f:
            json.dump(id_map, f)

        image_search.load_faiss_index(index_path=idx_path, id_map_path=map_path)

    assert image_search._faiss_index is not None
    assert image_search._faiss_index.ntotal == 4
    assert len(image_search._id_map) == 4


def test_load_faiss_index_idempotent():
    """Calling load_faiss_index() twice must not re-load."""
    index, id_map = _make_fake_index(n=2)
    image_search._faiss_index = index
    image_search._id_map = id_map

    image_search.load_faiss_index()  # should return immediately
    assert image_search._faiss_index is index  # same object


# ── search_by_image ───────────────────────────────────────────────────────────

def test_search_returns_record_and_score():
    """search_by_image must return the closest KB record with a float score."""
    proc, model = _make_fake_clip()
    image_search._clip_processor = proc
    image_search._clip_model = model

    index, id_map = _make_fake_index(n=3)
    image_search._faiss_index = index
    image_search._id_map = id_map

    pil = Image.new("RGB", (64, 64))
    record, score = image_search.search_by_image(pil)

    assert isinstance(record, dict)
    assert "name" in record
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.01  # cosine on normalised vectors is in [−1, 1]; clamp


def test_search_raises_if_index_not_loaded():
    """search_by_image must raise RuntimeError when index is not loaded."""
    pil = Image.new("RGB", (64, 64))
    with pytest.raises(RuntimeError, match="FAISS index not loaded"):
        image_search.search_by_image(pil)
