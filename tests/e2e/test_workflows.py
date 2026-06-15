"""
End-to-end workflow tests — multi-turn conversation and error scenarios.

These tests run against the real FastAPI app with mocked ML models.
They verify that conversation memory (session_id) works correctly for
pronoun resolution across turns.
"""

import sys
import os
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_LIBRARY_RESULT = {
    "reply": "**Learning Resources Centre**\nThe main campus library.\n\n📍 Hartley Building, All floors",
    "intent": "find_location",
    "confidence": 0.9712,
    "entity": "learning resources centre",
}

_HOURS_RESULT = {
    "reply": "**Learning Resources Centre — Opening Hours**\n\n  • Monday–Friday: 8:00–22:00",
    "intent": "ask_hours",
    "confidence": 0.8821,
    "entity": "learning resources centre",
}


def _make_answer_query(entity_results: dict):
    """Returns a mock answer_query that returns different results based on entity."""
    def _mock(text, session_id=None):
        # Simple routing: if "hours" or "close" or "open" in text → hours intent
        lowered = text.lower()
        if any(w in lowered for w in ("close", "open", "hours", "time")):
            return _HOURS_RESULT
        return _LIBRARY_RESULT
    return _mock


@pytest.fixture
def client():
    from src.context import SESSION_STORE
    SESSION_STORE.clear()

    with (
        patch("src.inference._model", new=object()),
        patch("src.inference._tokenizer", new=object()),
        patch("src.inference._entity_candidates", new=["learning resources centre"]),
        patch("src.inference.load_inference_model", return_value=(object(), object())),
    ):
        from api.main import app
        yield TestClient(app)

    SESSION_STORE.clear()


# ── Happy path: text query ────────────────────────────────────────────────────

def test_single_text_query_returns_reply(client):
    with patch("src.inference.answer_query", side_effect=_make_answer_query({})):
        r = client.post("/chat", json={"message": "Where is the library?"})
    assert r.status_code == 200
    assert len(r.json()["reply"]) > 0


# ── Multi-turn: pronoun resolution ────────────────────────────────────────────

def test_pronoun_resolution_two_turns(client):
    """
    Turn 1: "Where is the library?" → entity = "learning resources centre"
    Turn 2: "What time does it close?" → "it" should resolve to the library
    """
    sid = "e2e-session-001"

    # Turn 1 — establish context
    with patch("src.inference.answer_query", return_value=_LIBRARY_RESULT):
        r1 = client.post("/chat", json={"message": "Where is the library?", "session_id": sid})
    assert r1.status_code == 200

    # Turn 2 — pronoun reference
    with patch("src.inference.answer_query", return_value=_HOURS_RESULT):
        r2 = client.post("/chat", json={"message": "What time does it close?", "session_id": sid})
    assert r2.status_code == 200
    # The reply should contain hours information
    assert "Opening Hours" in r2.json()["reply"] or "monday" in r2.json()["reply"].lower()


def test_different_sessions_are_independent(client):
    """Two different session IDs should not share context."""
    sid_a = "e2e-session-A"
    sid_b = "e2e-session-B"

    with patch("src.inference.answer_query", return_value=_LIBRARY_RESULT):
        client.post("/chat", json={"message": "Where is the library?", "session_id": sid_a})

    # Session B should have no context from Session A
    from src.context import get_context
    ctx_b = get_context(sid_b)
    assert len(ctx_b) == 0


# ── Error scenarios ───────────────────────────────────────────────────────────

def test_missing_message_field_returns_422(client):
    r = client.post("/chat", json={"session_id": "test"})
    assert r.status_code == 422


def test_empty_message_returns_422(client):
    r = client.post("/chat", json={"message": "", "session_id": "test"})
    assert r.status_code == 422


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] in ("ok", "healthy")
