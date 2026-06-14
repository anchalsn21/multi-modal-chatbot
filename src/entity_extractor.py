"""
entity_extractor.py — Improved entity extraction using RapidFuzz + synonym expansion.

Replaces the simple stopword-removal approach from inference.py with a two-stage
matching strategy:

  1. Synonym expansion — normalises known aliases (e.g. "lrc" → "learning resources centre")
  2. RapidFuzz partial_ratio — fuzzy-matches the cleaned query against all KB candidate
     names/tags, tolerating typos and partial mentions.

Falls back to the original stopword removal if no fuzzy match scores ≥ FUZZY_THRESHOLD.

Academic justification:
    RapidFuzz (Levenshtein-based) is O(n·m) per candidate pair and runs in microseconds
    on a 20-record vocabulary. Its score (0–100) is a directly interpretable similarity
    measure — more explainable than a neural embedding for the purposes of this project.
    It avoids the 40 MB+ overhead of spaCy NER while outperforming simple stopword
    removal on misspellings and partial names.

Usage:
    from src.entity_extractor import build_candidates, extract_entity
    CANDIDATES = build_candidates(kb)           # once at startup
    entity = extract_entity("libraary", CANDIDATES)   # per request
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from rapidfuzz import fuzz, process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Fuzzy match threshold ─────────────────────────────────────────────────────

# Minimum RapidFuzz partial_ratio score (0–100) to accept a match.
# 60 handles typos and partial names while rejecting unrelated terms.
FUZZY_THRESHOLD = 60

# ── Synonym / alias dictionary ─────────────────────────────────────────────────
# Maps common abbreviations and informal names to canonical KB names.
# Keys are lowercase; values should match what appears in KB record names or tags.

SYNONYMS: dict[str, str] = {
    # Library
    "lib": "main library",
    "lrc": "main library",
    "hartley": "main library",
    # Gymnasium
    "gym": "gymnasium",
    "fitness centre": "gymnasium",
    "fitness center": "gymnasium",
    "weights room": "gymnasium",
    "cardio": "gymnasium",
    "workout": "gymnasium",
    # Sports Centre
    "sports hall": "sports centre",
    "sports complex": "sports centre",
    "pool": "sports centre",
    "swimming pool": "sports centre",
    "squash": "sports centre",
    # Food
    "cafe": "cafeteria",
    "canteen": "cafeteria",
    "dining hall": "cafeteria",
    "food hall": "cafeteria",
    "cavendish": "cafeteria",
    # Departments
    "cs": "computer science department",
    "comp sci": "computer science department",
    "computing": "computer science department",
    "zepler": "computer science department",
    # Admin
    "admin": "administration office",
    "enrolment office": "administration office",
    "registration": "administration office",
    # Laboratory
    "lab": "science laboratory",
    "science lab": "science laboratory",
    "chem lab": "science laboratory",
    "bio lab": "science laboratory",
    # Lecture hall
    "lecture hall": "main lecture hall",
    "lecture theatre": "main lecture hall",
    # Classroom
    "teaching room": "classroom / teaching room",
    "tutorial room": "classroom / teaching room",
    # Meeting room — keep as "meeting room" so KB tags match correctly
    "seminar room": "meeting room",
    "conference room": "meeting room",
    # Navigation
    "front desk": "reception",
    "main reception": "reception",
    "information desk": "reception",
    # Main building
    "chancellor building": "main building",
    "campus centre": "main building",
    # Auditorium
    "event hall": "auditorium",
    "main auditorium": "auditorium",
}

# ── Stopword fallback (preserved from original inference.py) ──────────────────

_STOPWORDS = {
    "where", "is", "the", "a", "an", "how", "do", "i", "get", "to",
    "can", "you", "tell", "me", "what", "time", "does", "close", "open",
    "are", "there", "any", "find", "show", "look", "looking", "for",
    "on", "at", "in", "of", "my", "when", "who", "which", "near",
    "about", "this", "that", "would", "need", "want", "could", "please",
    "best", "place", "campus", "today", "week", "late", "night", "early",
    "morning", "evening", "available", "happening", "going",
}


def _stopword_fallback(text: str) -> str:
    """Original stopword-removal approach kept as the last-resort fallback."""
    cleaned = text.lower()
    for char in "?!.,;:":
        cleaned = cleaned.replace(char, "")
    words = [w for w in cleaned.split() if w not in _STOPWORDS]
    result = " ".join(words).strip()
    return result if result else text.lower()


# ── Candidate builder ─────────────────────────────────────────────────────────

def build_candidates(kb: dict) -> list[str]:
    """
    Build a flat list of all matchable entity strings from the knowledge base.

    Includes location names and their tags so that fuzzy matching can resolve
    informal terms like "printing" or "books" to the correct location.

    Args:
        kb : Loaded campus_kb.json dict.

    Returns:
        List of lowercase candidate strings (names + individual tags).
    """
    candidates: list[str] = []

    for section in ("locations", "departments", "study_areas", "events"):
        for record in kb.get(section, []):
            name = record.get("name", "").lower().strip()
            if name:
                candidates.append(name)
            for tag in record.get("tags", []):
                tag_lower = tag.lower().strip()
                if tag_lower and tag_lower not in candidates:
                    candidates.append(tag_lower)

    return candidates


# ── Main extraction function ──────────────────────────────────────────────────

def extract_entity(query: str, candidates: list[str], threshold: int = FUZZY_THRESHOLD) -> str:
    """
    Extract the most relevant campus entity from a user query.

    Strategy (in order):
      1. Apply synonym expansion to normalise known aliases.
      2. Run RapidFuzz partial_ratio against all KB candidates.
      3. If best score ≥ threshold, return the matched candidate name.
      4. Otherwise fall back to stopword removal.

    Args:
        query      : Raw user input string.
        candidates : List of known KB entity strings (from build_candidates).
        threshold  : Minimum fuzzy score to accept (default FUZZY_THRESHOLD=60).

    Returns:
        Best-matching entity string (lowercase).
    """
    if not query:
        return ""

    lowered = query.lower().strip()

    # Remove punctuation for cleaner matching
    for char in "?!.,;:":
        lowered = lowered.replace(char, "")

    # Stage 1 — synonym expansion
    for alias, canonical in SYNONYMS.items():
        if alias in lowered:
            lowered = lowered.replace(alias, canonical)

    # Stage 2a — exact candidate match after synonym expansion
    if lowered in candidates:
        return lowered

    # Stage 2b — try stopword-cleaned version of the expanded query for exact match
    cleaned = _stopword_fallback(lowered)
    if cleaned in candidates:
        return cleaned

    # Stage 2c — RapidFuzz fuzzy match using WRatio (handles length differences better)
    if candidates:
        result = process.extractOne(
            cleaned if cleaned else lowered,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )
        if result:
            matched_string, score, _ = result
            return matched_string

    # Stage 3 — fallback to stopword removal
    return _stopword_fallback(query)


def extract_entity_with_score(
    query: str,
    candidates: list[str],
    threshold: int = FUZZY_THRESHOLD,
) -> tuple[str, Optional[int]]:
    """
    Like extract_entity but also returns the fuzzy score for explainability.

    Returns:
        (entity_string, score_or_None)
        score is None when the stopword fallback was used.
    """
    if not query:
        return "", None

    lowered = query.lower().strip()
    for char in "?!.,;:":
        lowered = lowered.replace(char, "")

    for alias, canonical in SYNONYMS.items():
        if alias in lowered:
            lowered = lowered.replace(alias, canonical)

    if lowered in candidates:
        return lowered, 100

    cleaned = _stopword_fallback(lowered)
    if cleaned in candidates:
        return cleaned, 100

    if candidates:
        result = process.extractOne(
            cleaned if cleaned else lowered,
            candidates,
            scorer=fuzz.WRatio,
            score_cutoff=threshold,
        )
        if result:
            matched_string, score, _ = result
            return matched_string, int(score)

    return _stopword_fallback(query), None
