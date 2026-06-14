"""
context.py — Lightweight session-based conversation memory.

Maintains a rolling window of recent turns per session so the assistant can
resolve pronominal references like "it", "there", "that place" back to the
entity mentioned in the previous turn.

Design decisions:
  - Pure Python: no LLM, no external NLP library, no database.
  - collections.deque(maxlen=MAX_TURNS): O(1) append/pop, bounded memory.
  - TTL-based pruning on each access — avoids background threads entirely.
  - session_id is a UUID4 generated client-side; the server trusts it without
    user authentication (appropriate for a demo/academic project).

Academic justification:
    Pronominal coreference without a coreference model. The closed-domain
    constraint (≤ 20 KB records, fixed intent set) means a pronoun in the
    second turn almost always refers to the entity from the first turn.
    A rolling deque of the last MAX_TURNS interactions provides enough context
    for all realistic conversational patterns while using negligible memory.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_TURNS = 5        # number of turns to keep per session
SESSION_TTL = 3600   # seconds before an idle session is pruned (1 hour)

# Pronoun patterns that trigger entity resolution from context
_PRONOUNS = {
    "it", "there", "that", "this",
    "the place", "that building", "this building",
    "that location", "this location",
}

# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """A single conversational exchange."""
    entity: str    # resolved entity used in the KB lookup, e.g. "learning resources centre"
    intent: str    # predicted intent, e.g. "find_location"
    reply:  str    # formatted assistant response (used for debugging/logging)


@dataclass
class Session:
    """Per-session state including the turn history and last-access timestamp."""
    turns:       deque  = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    last_access: float  = field(default_factory=time.time)


# ── In-memory store ───────────────────────────────────────────────────────────

# Maps session_id (str UUID4) → Session
SESSION_STORE: dict[str, Session] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def get_context(session_id: str) -> deque:
    """
    Retrieve the turn history for a session.

    Prunes stale sessions on each call to keep memory clean without
    requiring a background worker.

    Args:
        session_id : Client-provided UUID4 string.

    Returns:
        deque of Turn objects (may be empty for a new session).
    """
    _prune_stale()
    session = SESSION_STORE.get(session_id)
    if session is None:
        return deque(maxlen=MAX_TURNS)
    session.last_access = time.time()
    return session.turns


def update_context(session_id: str, entity: str, intent: str, reply: str) -> None:
    """
    Append a completed turn to the session history.

    Creates the session entry if it does not exist yet.

    Args:
        session_id : Client-provided UUID4 string.
        entity     : The entity that was resolved and used in the KB lookup.
        intent     : The predicted intent for this turn.
        reply      : The assistant's formatted response.
    """
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = Session()

    session = SESSION_STORE[session_id]
    session.turns.append(Turn(entity=entity, intent=intent, reply=reply))
    session.last_access = time.time()


def resolve_entity(query: str, context: deque) -> Optional[str]:
    """
    Check whether the query contains a pronoun and, if so, return the most
    recent entity from the session context.

    This handles patterns like:
        Turn 1: "Where is the library?" → entity = "learning resources centre"
        Turn 2: "What time does it close?" → "it" → "learning resources centre"

    Args:
        query   : The raw user input for the current turn.
        context : Turn history deque from get_context().

    Returns:
        The most recent entity string if a pronoun is detected, else None.
    """
    if not context:
        return None

    lowered = query.lower().strip()
    for char in "?!.,;:":
        lowered = lowered.replace(char, "")

    # Check for pronoun presence — must be a whole-word or short-phrase match
    words = set(lowered.split())
    if words & _PRONOUNS:
        return context[-1].entity

    # Multi-word pronoun phrases (e.g. "the place", "that building")
    for phrase in _PRONOUNS:
        if " " in phrase and phrase in lowered:
            return context[-1].entity

    return None


def clear_session(session_id: str) -> None:
    """Remove a session from the store (e.g. on logout or explicit reset)."""
    SESSION_STORE.pop(session_id, None)


def session_count() -> int:
    """Return the number of active sessions (for monitoring/debugging)."""
    return len(SESSION_STORE)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prune_stale() -> None:
    """Delete sessions that have not been accessed within SESSION_TTL seconds."""
    now = time.time()
    stale = [sid for sid, s in SESSION_STORE.items() if now - s.last_access > SESSION_TTL]
    for sid in stale:
        del SESSION_STORE[sid]
