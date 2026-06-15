"""
tests/test_image_endpoint.py — Integration tests for POST /chat/image

Run with:
    pytest tests/test_image_endpoint.py -v

These tests use FastAPI's TestClient (synchronous) and monkeypatch
search_by_image so no real CLIP/FAISS is required.
"""

import io
import os
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app, raise_server_exceptions=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    """Return raw bytes of a valid JPEG image."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 149, 237)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(34, 139, 34)).save(buf, format="PNG")
    return buf.getvalue()


_FAKE_RECORD = {
    "id": "loc_001",
    "name": "Main Library",
    "type": "location",
    "building": "Hartley Building",
    "floor": "Ground to 3rd Floor",
    "campus_zone": "North Campus",
    "map_zone": "A",
    "description": "The central academic library.",
    "hours": {"monday_friday": "08:00 - 22:00"},
    "tags": ["library", "study"],
    "nearest_entrance": "North Gate",
}


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_valid_jpeg_upload_returns_200():
    """A valid JPEG should return 200 with reply, description, and confidence."""
    with patch("api.routes.chat.search_by_image", return_value=(_FAKE_RECORD, 0.87)):
        resp = client.post(
            "/chat/image",
            files={"image": ("library.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "reply" in body
    assert "description" in body
    assert "confidence" in body
    assert body["description"] == "Main Library"
    assert body["confidence"] == pytest.approx(0.87, abs=0.001)


def test_valid_png_upload_returns_200():
    """A valid PNG should also work correctly."""
    with patch("api.routes.chat.search_by_image", return_value=(_FAKE_RECORD, 0.75)):
        resp = client.post(
            "/chat/image",
            files={"image": ("campus.png", _make_png_bytes(), "image/png")},
        )
    assert resp.status_code == 200


def test_non_image_file_returns_415():
    """Uploading a non-image file must return 415 Unsupported Media Type."""
    resp = client.post(
        "/chat/image",
        files={"image": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415
    assert "Unsupported media type" in resp.json()["detail"]


def test_corrupt_image_returns_400():
    """Random bytes declared as image/jpeg must return 400 Bad Request."""
    resp = client.post(
        "/chat/image",
        files={"image": ("broken.jpg", b"\x00\xFF\xAB\xCD" * 50, "image/jpeg")},
    )
    assert resp.status_code == 400


def test_empty_file_returns_400():
    """An empty upload must return 400 Bad Request."""
    resp = client.post(
        "/chat/image",
        files={"image": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


def test_missing_index_returns_503():
    """When search_by_image raises RuntimeError (index not loaded), return 503."""
    with patch(
        "api.routes.chat.search_by_image",
        side_effect=RuntimeError("FAISS index not loaded. Call load_faiss_index() first."),
    ):
        resp = client.post(
            "/chat/image",
            files={"image": ("img.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 503
    assert "FAISS index not loaded" in resp.json()["detail"]


def test_low_confidence_returns_200_with_no_match_message():
    """A score below IMAGE_MATCH_THRESHOLD must return 200 with a no-match reply."""
    with patch("api.routes.chat.search_by_image", return_value=(_FAKE_RECORD, 0.05)):
        resp = client.post(
            "/chat/image",
            files={"image": ("unknown.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "couldn't confidently match" in body["reply"].lower() or \
           "no confident match" in body["description"].lower()
    assert body["confidence"] == pytest.approx(0.05, abs=0.001)


def test_reply_is_non_empty_for_good_match():
    """The reply field must be a non-empty string for a confident match."""
    with patch("api.routes.chat.search_by_image", return_value=(_FAKE_RECORD, 0.90)):
        resp = client.post(
            "/chat/image",
            files={"image": ("lib.jpg", _make_jpeg_bytes(), "image/jpeg")},
        )
    assert resp.status_code == 200
    assert len(resp.json()["reply"]) > 20
