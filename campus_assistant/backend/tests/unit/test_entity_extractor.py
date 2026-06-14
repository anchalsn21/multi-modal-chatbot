"""
Unit tests for src/entity_extractor.py

Tests cover:
  - Synonym expansion
  - Fuzzy matching with typos
  - Fuzzy matching with partial names
  - Stopword fallback when no match
  - Navigation text preserved
  - Empty/edge inputs
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.entity_extractor import (
    build_candidates,
    extract_entity,
    extract_entity_with_score,
    SYNONYMS,
    FUZZY_THRESHOLD,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_KB = {
    "locations": [
        {
            "name": "Learning Resources Centre",
            "tags": ["library", "study", "books", "printing", "quiet"],
        },
        {
            "name": "Sports Centre",
            "tags": ["gym", "fitness", "swimming", "sports"],
        },
        {
            "name": "Student Restaurant",
            "tags": ["food", "cafe", "lunch", "canteen"],
        },
        {
            "name": "Health Centre",
            "tags": ["medical", "doctors", "nurse", "clinic"],
        },
    ],
    "departments": [],
    "study_areas": [],
}

CANDIDATES = build_candidates(SAMPLE_KB)


# ── build_candidates ──────────────────────────────────────────────────────────

def test_build_candidates_includes_names():
    assert "learning resources centre" in CANDIDATES
    assert "sports centre" in CANDIDATES


def test_build_candidates_includes_tags():
    assert "library" in CANDIDATES
    assert "gym" in CANDIDATES
    assert "clinic" in CANDIDATES


def test_build_candidates_no_duplicates():
    # Tags and names should not produce duplicates for well-formed KB
    seen = set()
    for c in CANDIDATES:
        assert c not in seen, f"Duplicate candidate: {c}"
        seen.add(c)


# ── Synonym expansion ─────────────────────────────────────────────────────────

def test_synonym_lrc():
    result = extract_entity("where is the lrc", CANDIDATES)
    assert "learning resources centre" in result or "learning resources" in result


def test_synonym_library():
    result = extract_entity("library hours", CANDIDATES)
    # Either the canonical name or a tag should match
    assert result in CANDIDATES


def test_synonym_gym():
    result = extract_entity("gym opening times", CANDIDATES)
    assert "sports centre" in result or "gym" in result


def test_synonym_canteen():
    result = extract_entity("where is the canteen", CANDIDATES)
    assert "restaurant" in result or "student restaurant" in result


# ── Fuzzy matching ────────────────────────────────────────────────────────────

def test_fuzzy_typo_library():
    result = extract_entity("libraary", CANDIDATES)
    assert result in CANDIDATES


def test_fuzzy_partial_name():
    result = extract_entity("resources centre", CANDIDATES)
    assert "learning resources centre" in result


def test_fuzzy_sports_partial():
    result = extract_entity("sports", CANDIDATES)
    assert "sports" in result


def test_fuzzy_returns_string():
    result = extract_entity("heelth centre", CANDIDATES)
    assert isinstance(result, str)
    assert len(result) > 0


# ── Score variant ─────────────────────────────────────────────────────────────

def test_extract_with_score_returns_tuple():
    entity, score = extract_entity_with_score("library", CANDIDATES)
    assert isinstance(entity, str)
    assert score is None or isinstance(score, int)


def test_extract_with_score_confident_match():
    entity, score = extract_entity_with_score("learning resources centre", CANDIDATES)
    assert score is not None
    assert score >= FUZZY_THRESHOLD


def test_extract_with_score_fallback_gives_none_score():
    entity, score = extract_entity_with_score("purple elephant", [])
    assert score is None


# ── Fallback behaviour ────────────────────────────────────────────────────────

def test_fallback_on_empty_candidates():
    result = extract_entity("where is the library", [])
    # Stopword fallback should remove common question words
    assert isinstance(result, str)
    assert "where" not in result.split()


def test_fallback_no_match_returns_string():
    result = extract_entity("xyzzy foobarbaz", CANDIDATES)
    assert isinstance(result, str)


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_query():
    result = extract_entity("", CANDIDATES)
    assert result == ""


def test_punctuation_stripped():
    result1 = extract_entity("library?", CANDIDATES)
    result2 = extract_entity("library", CANDIDATES)
    assert result1 == result2
