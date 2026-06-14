"""
kb_lookup.py — Search the campus knowledge base for a matching record.

Given a predicted intent and an extracted entity string, this module
looks up the most relevant record in campus_kb.json.

Matching strategy:
    - Lowercase both the query entity and each record's name + tags
    - Check if any word from the entity appears in the record's name or tags
    - For navigation queries, also match map zone labels (A–P)
    - Return the FIRST match found
    - Return a fallback message if no match is found
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_kb

_kb_cache: dict = None


def load_kb_once() -> dict:
    global _kb_cache
    if _kb_cache is None:
        _kb_cache = load_kb()
    return _kb_cache


def _matches(entity: str, record: dict) -> bool:
    entity_lower = entity.lower()
    record_name = record.get("name", "").lower()
    record_tags = " ".join(record.get("tags", [])).lower()
    searchable = record_name + " " + record_tags

    # Prefer full-phrase match first
    if entity_lower in searchable:
        return True

    # Word-by-word fallback: ALL content words must appear (not just any one)
    # Filter out very short words (≤2 chars) that cause false positives (e.g. "ai", "cs")
    entity_words = [w for w in entity_lower.split() if len(w) > 2]
    if not entity_words:
        return entity_lower in searchable
    return all(word in searchable for word in entity_words)


def _extract_zone_label(text: str) -> str | None:
    """Extract a single uppercase map zone letter (A–P) from text."""
    match = re.search(r'\b([A-Pa-p])\b', text)
    if match:
        return match.group(1).upper()
    return None


def _all_records(kb: dict) -> list:
    """Return all location, department, and study_area records together."""
    return (
        kb.get("locations", [])
        + kb.get("departments", [])
        + kb.get("study_areas", [])
    )


def find_location(entity: str) -> dict | None:
    kb = load_kb_once()
    # Try map zone label first (e.g. user says "zone A" or "I'm at A")
    zone = _extract_zone_label(entity)
    if zone:
        for record in _all_records(kb):
            if record.get("map_zone") == zone:
                return record
    for record in kb.get("locations", []):
        if _matches(entity, record):
            return record
    return None


def find_hours(entity: str) -> dict | None:
    kb = load_kb_once()
    zone = _extract_zone_label(entity)
    if zone:
        for record in _all_records(kb):
            if record.get("map_zone") == zone:
                return record
    for record in kb.get("locations", []) + kb.get("study_areas", []):
        if _matches(entity, record):
            return record
    return None


def find_event(entity: str) -> dict | None:
    kb = load_kb_once()
    for record in kb.get("events", []):
        if _matches(entity, record):
            return record
    events = kb.get("events", [])
    return events[0] if events else None


def find_department(entity: str) -> dict | None:
    kb = load_kb_once()
    # Departments are stored in `locations` with no separate `departments` array.
    # Search locations whose name/tags match, preferring type=="location" records
    # that are clearly department-like (CS, Business, etc.).
    zone = _extract_zone_label(entity)
    dept_records = [
        r for r in kb.get("locations", [])
        if "department" in r.get("name", "").lower()
        or any("department" in t.lower() for t in r.get("tags", []))
    ]
    if zone:
        for record in dept_records:
            if record.get("map_zone") == zone:
                return record
    for record in dept_records:
        if _matches(entity, record):
            return record
    # Broader fallback: any location record that matches the entity
    for record in kb.get("locations", []):
        if _matches(entity, record):
            return record
    return None


def find_study_area(entity: str) -> dict | None:
    kb = load_kb_once()
    # Study areas are stored in `locations` with study-related tags.
    study_records = [
        r for r in kb.get("locations", [])
        if any(t in r.get("tags", []) for t in ["study area", "study suite", "24 hour", "quiet", "silent"])
    ]
    zone = _extract_zone_label(entity)
    if zone:
        for record in study_records:
            if record.get("map_zone") == zone:
                return record
    for record in study_records:
        if _matches(entity, record):
            return record
    # Fallback: return first study record if any exist
    return study_records[0] if study_records else None


# Zone letters that are also common English words — require them to appear as
# standalone tokens preceded by "zone" or surrounded by digits/punctuation/start.
# Simple letters like A, B, C ... are safe; I, O are ambiguous.
_AMBIGUOUS_ZONE_LETTERS = {"I", "O", "A"}  # A is fine alone; I and O are English words

_ZONE_LETTER_RE = re.compile(r'\b([A-P])\b')


def _extract_zone_letters(text: str) -> list[str]:
    """
    Extract map zone letters (A–P) from text, avoiding false positives on
    common English words like 'I' and 'O'.
    """
    candidates = _ZONE_LETTER_RE.findall(text.upper())
    result = []
    for c in candidates:
        if c not in ("I", "O"):
            result.append(c)
        else:
            # Only accept I or O if preceded by "zone " or "at " or "->" or digit
            pattern = rf'(?:zone\s+|at\s+|from\s+|->)\s*{c}\b'
            if re.search(pattern, text, re.IGNORECASE):
                result.append(c)
    return result


def _name_to_zone(text: str, kb: dict) -> str | None:
    """
    Resolve a place name mention to a map zone label.
    Searches locations, departments, and study_areas by name/tags.
    Returns the map_zone string (e.g. "A") or None.
    """
    for record in _all_records(kb):
        if _matches(text, record):
            return record.get("map_zone")
    return None


def _split_journey(entity: str) -> tuple[str, str] | None:
    """
    Try to split a journey query into (from_part, to_part).

    Handles patterns like:
      "library to sports centre"
      "I am at the library and want to go to the gym"
      "from medical centre to admin office"
      "I'm at A want to go to K"
      "A to K"
    Returns (from_text, to_text) or None if only one place found.
    """
    # More-specific patterns must come before the generic r'\bto\b' so they
    # match before the short "to" splits mid-phrase (e.g. "want TO go to").
    separators = [
        r'\band want to go to\b',
        r'\bwant to go to\b',
        r'\band go to\b',
        r'\bgoing to\b',
        r'\bhead(?:ing)? to\b',
        r'\bdirections? to\b',
        r'\bto\b',
    ]

    text = entity.lower().strip()

    # Try "X to Y" style split on each separator
    for sep in separators:
        parts = re.split(sep, text, maxsplit=1)
        if len(parts) == 2:
            from_part = parts[0].strip()
            to_part   = parts[1].strip()
            # Strip leading filler/navigation words from from_part
            from_part = re.sub(
                r"^(?:i(?:'m| am) (?:at|in|currently at)|from|starting at|at|directions?\s+from|navigate\s+from|how\s+do\s+i\s+get\s+from)\s+",
                "", from_part
            ).strip()
            # Strip leading article "the" from both parts
            from_part = re.sub(r"^the\s+", "", from_part).strip()
            to_part   = re.sub(r"^the\s+", "", to_part).strip()
            if from_part and to_part:
                return from_part, to_part

    return None


def find_navigation(entity: str) -> dict | None:
    """
    Handle navigation / directions queries.

    Accepts:
      - Zone letters:   "A to K", "from G to M"
      - Place names:    "library to sports centre", "medical centre to admin"
      - Mixed:          "I'm at the library, want to go to K"
      - Single target:  "directions to the gym", "where is zone A"
      - Gate queries:   "from North Gate to the library"
    """
    kb = load_kb_once()
    nav   = kb.get("navigation", {})
    zones = nav.get("zones", {})
    routes  = nav.get("routes", [])
    cross   = nav.get("cross_zone_directions", [])

    entity_up  = entity.upper()
    entity_low = entity.lower()

    # ── Step 1: collect explicit unambiguous zone letters ─────────────────
    zone_letters = _extract_zone_letters(entity)

    # ── Step 2: try to split into a journey (from → to) ───────────────────
    journey = _split_journey(entity)

    from_z = to_z = None

    if len(zone_letters) >= 2:
        # Two explicit letters — direct
        from_z, to_z = zone_letters[0], zone_letters[1]

    elif journey:
        from_text, to_text = journey
        # Discard from_part if it's only filler words (no real place info)
        filler_only = re.fullmatch(
            r'[\s\w]*(how|where|can you|tell me|get|do|find|show|go|navigate|directions?|please|help)[\s\w]*',
            from_text, re.IGNORECASE
        )
        if filler_only:
            from_text = ""

        from_letters = _extract_zone_letters(from_text) if from_text else []
        to_letters   = _extract_zone_letters(to_text)
        from_z = from_letters[0] if from_letters else (_name_to_zone(from_text, kb) if from_text else None)
        to_z   = to_letters[0]   if to_letters   else _name_to_zone(to_text, kb)

    # ── Step 3: if we have two zones, return directions ────────────────────
    if from_z and to_z:
        for route in cross:
            if route.get("from_zone") == from_z and route.get("to_zone") == to_z:
                return {"type": "cross_zone", "from_zone": from_z, "to_zone": to_z,
                        "route": route, "zones": zones}
        # Synthesise generic directions
        return {
            "type": "cross_zone_generic",
            "from_zone": from_z,
            "to_zone":   to_z,
            "from_info": zones.get(from_z, {}),
            "to_info":   zones.get(to_z, {}),
        }

    # ── Step 4: single unambiguous zone letter ────────────────────────────
    if len(zone_letters) == 1 and not journey:
        z = zone_letters[0]
        if zones.get(z):
            return {"type": "single_zone", "zone": z, "zone_info": zones[z]}

    # ── Step 5: gate route ─────────────────────────────────────────────────
    for route in routes:
        gate = route.get("from", "").lower()
        if gate in entity_low or any(w in entity_low for w in gate.split()):
            return {"type": "gate_route", "route": route, "zones": zones}

    # ── Step 6: single place name → tell user its zone ────────────────────
    for record in _all_records(kb):
        if _matches(entity, record):
            zone_label = record.get("map_zone")
            return {
                "type":      "place_to_zone",
                "record":    record,
                "zone":      zone_label,
                "zone_info": zones.get(zone_label, {}),
            }

    return None


def get_fallback(intent: str) -> str:
    kb = load_kb_once()
    fallbacks = kb.get("fallbacks", {})
    mapping = {
        "find_location":   "unknown_location",
        "find_department": "unknown_department",
        "find_navigation": "no_route",
        "unknown":         "unknown_intent",
        "low_confidence":  "low_confidence",
    }
    key = mapping.get(intent, "unknown_intent")
    return fallbacks.get(key, fallbacks.get("unknown_intent", "I couldn't find that. Please try rephrasing."))


def lookup(intent: str, entity: str) -> dict | None:
    intent_to_function = {
        "find_location":   find_location,
        "ask_hours":       find_hours,
        "find_event":      find_event,
        "find_department": find_department,
        "find_study_area": find_study_area,
        "find_navigation": find_navigation,
    }
    func = intent_to_function.get(intent)
    if func is None:
        return None
    return func(entity)


if __name__ == "__main__":
    print("=== Testing kb_lookup.py ===\n")
    tests = [
        ("find_location",   "library"),
        ("find_location",   "zone A"),
        ("ask_hours",       "gym"),
        ("find_event",      "freshers fair"),
        ("find_department", "computer science"),
        ("find_study_area", "24 hour"),
        ("find_navigation", "A to K"),
        ("find_navigation", "from G to M"),
        ("find_navigation", "North Gate"),
        ("find_navigation", "sports centre"),
    ]
    for intent, entity in tests:
        result = lookup(intent, entity)
        if result is None:
            name = "NOT FOUND"
        elif isinstance(result, dict):
            name = result.get("name") or result.get("type") or str(result)[:60]
        else:
            name = str(result)[:60]
        print(f"  intent={intent:<20} entity={entity:<25} -> {name}")
