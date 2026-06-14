"""
Unit tests for src/context.py

Tests cover:
  - Session creation and retrieval
  - Turn updates and rolling window
  - Pronoun resolution (it, there, that)
  - TTL pruning
  - Session isolation
"""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.context import (
    SESSION_STORE,
    get_context,
    update_context,
    resolve_entity,
    clear_session,
    session_count,
    MAX_TURNS,
    SESSION_TTL,
    Turn,
)


@pytest.fixture(autouse=True)
def clear_store():
    """Ensure a clean SESSION_STORE before each test."""
    SESSION_STORE.clear()
    yield
    SESSION_STORE.clear()


# ── Basic CRUD ────────────────────────────────────────────────────────────────

def test_get_context_new_session_returns_empty():
    ctx = get_context("new-session-1")
    assert len(ctx) == 0


def test_update_context_creates_session():
    update_context("s1", entity="library", intent="find_location", reply="The library is...")
    assert "s1" in SESSION_STORE


def test_get_context_returns_turns():
    update_context("s2", entity="gym", intent="ask_hours", reply="The gym is open...")
    ctx = get_context("s2")
    assert len(ctx) == 1
    turn = ctx[-1]
    assert turn.entity == "gym"
    assert turn.intent == "ask_hours"


def test_multiple_turns_accumulate():
    for i in range(3):
        update_context("s3", entity=f"place_{i}", intent="find_location", reply=f"reply {i}")
    ctx = get_context("s3")
    assert len(ctx) == 3


def test_rolling_window_enforced():
    for i in range(MAX_TURNS + 3):
        update_context("s4", entity=f"place_{i}", intent="find_location", reply=f"reply {i}")
    ctx = get_context("s4")
    assert len(ctx) == MAX_TURNS
    # Oldest turns should be discarded
    assert ctx[-1].entity == f"place_{MAX_TURNS + 2}"


def test_clear_session():
    update_context("s5", entity="library", intent="find_location", reply="...")
    clear_session("s5")
    assert "s5" not in SESSION_STORE


def test_session_count():
    update_context("s6", entity="a", intent="find_location", reply="...")
    update_context("s7", entity="b", intent="ask_hours", reply="...")
    assert session_count() >= 2


# ── Pronoun resolution ────────────────────────────────────────────────────────

def test_resolve_it_returns_last_entity():
    update_context("pronoun1", entity="learning resources centre", intent="find_location", reply="...")
    ctx = get_context("pronoun1")
    resolved = resolve_entity("what time does it close?", ctx)
    assert resolved == "learning resources centre"


def test_resolve_there_returns_last_entity():
    update_context("pronoun2", entity="sports centre", intent="find_location", reply="...")
    ctx = get_context("pronoun2")
    resolved = resolve_entity("how do I get there?", ctx)
    assert resolved == "sports centre"


def test_resolve_that_building():
    update_context("pronoun3", entity="health centre", intent="find_location", reply="...")
    ctx = get_context("pronoun3")
    resolved = resolve_entity("is that building open on weekends?", ctx)
    assert resolved == "health centre"


def test_no_pronoun_returns_none():
    update_context("pronoun4", entity="library", intent="find_location", reply="...")
    ctx = get_context("pronoun4")
    resolved = resolve_entity("where is the sports centre?", ctx)
    assert resolved is None


def test_resolve_empty_context_returns_none():
    ctx = get_context("empty-session-xyz")
    resolved = resolve_entity("what time does it close?", ctx)
    assert resolved is None


# ── Session isolation ─────────────────────────────────────────────────────────

def test_sessions_are_isolated():
    update_context("iso1", entity="library", intent="find_location", reply="...")
    update_context("iso2", entity="gym", intent="ask_hours", reply="...")

    ctx1 = get_context("iso1")
    ctx2 = get_context("iso2")

    assert ctx1[-1].entity == "library"
    assert ctx2[-1].entity == "gym"


# ── TTL pruning (fast test with monkeypatched time) ───────────────────────────

def test_stale_session_pruned(monkeypatch):
    update_context("stale1", entity="cafe", intent="find_location", reply="...")
    # Manually backdate the last_access
    SESSION_STORE["stale1"].last_access = time.time() - SESSION_TTL - 1

    update_context("fresh1", entity="library", intent="ask_hours", reply="...")
    # get_context triggers pruning
    _ = get_context("fresh1")

    assert "stale1" not in SESSION_STORE
    assert "fresh1" in SESSION_STORE
