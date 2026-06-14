"""
Integration tests for POST /chat (text endpoint).

Tests cover:
  - Happy path responses
  - session_id handling
  - Low confidence fallback
  - Request validation errors
"""

import sys
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

_FAKE_RESULT = {
    "reply": "**Learning Resources Centre**\nThe main campus library.",
    "intent": "find_location",
    "confidence": 0.9712,
    "entity": "learning resources centre",
}

_LOW_CONF_RESULT = {
    "reply": "I'm not sure I understood that. Could you rephrase?",
    "intent": "unknown",
    "confidence": 0.21,
    "entity": "",
}


@pytest.fixture
def client():
    with (
        patch("src.inference.answer_query", return_value=_FAKE_RESULT),
        patch("src.context.SESSION_STORE", new={}),
    ):
        from api.main import app
        yield TestClient(app)


def test_chat_returns_200(client):
    r = client.post("/chat", json={"message": "Where is the library?"})
    assert r.status_code == 200


def test_chat_response_has_reply(client):
    r = client.post("/chat", json={"message": "Where is the library?"})
    assert "reply" in r.json()


def test_chat_response_has_intent(client):
    r = client.post("/chat", json={"message": "Where is the library?"})
    assert "intent" in r.json()


def test_chat_response_has_confidence(client):
    r = client.post("/chat", json={"message": "Where is the library?"})
    body = r.json()
    assert "confidence" in body
    assert 0.0 <= body["confidence"] <= 1.0


def test_chat_with_session_id(client):
    r = client.post(
        "/chat",
        json={"message": "Where is the library?", "session_id": "test-session-001"},
    )
    assert r.status_code == 200


def test_chat_empty_message_returns_422(client):
    r = client.post("/chat", json={"message": ""})
    assert r.status_code == 422


def test_chat_missing_message_returns_422(client):
    r = client.post("/chat", json={})
    assert r.status_code == 422


def test_chat_message_too_long_returns_422(client):
    r = client.post("/chat", json={"message": "a" * 501})
    assert r.status_code == 422


def test_chat_low_confidence_returns_200():
    with (
        patch("src.inference.answer_query", return_value=_LOW_CONF_RESULT),
        patch("src.context.SESSION_STORE", new={}),
    ):
        from api.main import app
        c = TestClient(app)
        r = c.post("/chat", json={"message": "purple elephant at midnight"})
    assert r.status_code == 200
    assert r.json()["intent"] == "unknown"
