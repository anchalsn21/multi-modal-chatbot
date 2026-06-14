"""
Integration tests for POST /chat/multimodal

All ML models (CLIP, FAISS, DistilBERT) are mocked so these tests run
on any machine without GPU or pre-built index files.
"""

import io
import sys
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def _make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (64, 64), color=(100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


_FAKE_RECORD = {
    "id": "loc_001",
    "name": "Learning Resources Centre",
    "type": "location",
    "building": "Hartley Building",
    "floor": "All floors",
    "campus_zone": "North Campus",
    "map_zone": "A",
    "description": "The main campus library.",
    "hours": {"monday_friday": "8:00–22:00", "saturday": "9:00–18:00", "sunday": "10:00–18:00"},
    "nearest_entrance": "Main entrance, Hartley Building",
    "tags": ["library", "study", "books"],
}


@pytest.fixture
def client():
    with (
        patch("src.inference._model", new=object()),
        patch("src.inference._tokenizer", new=object()),
        patch("src.inference._entity_candidates", new=["learning resources centre"]),
        patch("src.inference.predict_intent", return_value=("ask_hours", 0.92)),
        patch("src.inference.load_inference_model", return_value=(object(), object())),
        patch(
            "src.image_search.search_by_image",
            return_value=([(_FAKE_RECORD, 0.87)], True),
        ),
        patch("src.context.SESSION_STORE", new={}),
    ):
        from api.main import app
        yield TestClient(app)


def test_multimodal_returns_200(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "What time does it close?"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200


def test_multimodal_response_fields(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "What time does it close?"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    body = r.json()
    assert "reply" in body
    assert "intent" in body
    assert "confidence" in body
    assert "image_match" in body
    assert "image_confidence" in body


def test_multimodal_intent_is_ask_hours(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "What time does it close?"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.json()["intent"] == "ask_hours"


def test_multimodal_image_match_is_location_name(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "What time does it close?"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert r.json()["image_match"] == "Learning Resources Centre"


def test_multimodal_reply_is_non_empty(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "What time does it close?"},
        files={"image": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
    )
    assert len(r.json()["reply"]) > 20


def test_multimodal_non_image_returns_415(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "Where is this?"},
        files={"image": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert r.status_code == 415


def test_multimodal_empty_image_returns_400(client):
    r = client.post(
        "/chat/multimodal",
        data={"message": "Where is this?"},
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert r.status_code == 400


def test_multimodal_low_confidence_image_uses_text_entity(client):
    """When CLIP confidence is low, text-based entity extraction is used."""
    with patch(
        "src.image_search.search_by_image",
        return_value=([(_FAKE_RECORD, 0.10)], False),
    ):
        r = client.post(
            "/chat/multimodal",
            data={"message": "Where is the sports centre?"},
            files={"image": ("blurry.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert r.status_code == 200
    assert r.json()["image_match"] is None
