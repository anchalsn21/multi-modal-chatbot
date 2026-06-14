"""
inference.py — End-to-end pipeline: typed question → formatted response.

This is the single entry point called by the FastAPI server.
It wires together the DistilBERT intent classifier and the KB lookup.

Pipeline:
    1. Predict intent (and confidence score) using the fine-tuned DistilBERT model
    2. Extract the key entity from the user's text (simple stopword removal)
    3. Look up the matching campus record in campus_kb.json
    4. Format a natural language response string

Usage:
    from src.inference import answer_query
    response = answer_query("What time does the library close?")
"""

import os
import sys
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.kb_lookup import lookup, get_fallback

# ── Cached model (loaded once when the server starts) ────────────────────────
_model     = None
_tokenizer = None

# KB entity candidates for fuzzy matching — populated by load_inference_model()
_entity_candidates: list[str] = []


def load_inference_model():
    """
    Load the saved DistilBERT model and tokenizer from disk.

    The model is cached in module-level variables so it is loaded only once
    when the FastAPI server starts, not on every request.

    Raises:
        FileNotFoundError: If train.py has not been run yet.
    """
    global _model, _tokenizer, _entity_candidates

    if _model is not None:
        return _model, _tokenizer   # already loaded — return cached

    if not os.path.exists(config.MODEL_SAVE_DIR):
        raise FileNotFoundError(
            f"No trained model found at {config.MODEL_SAVE_DIR}. "
            "Please run `python src/train.py` first."
        )

    print("[inference] Loading model...")
    _tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_DIR)
    _model     = AutoModelForSequenceClassification.from_pretrained(config.MODEL_SAVE_DIR)
    _model.eval()   # disable dropout for inference
    print("[inference] Model ready.")

    # Build fuzzy entity candidates from KB (once at startup)
    try:
        import json
        from src.entity_extractor import build_candidates
        with open(config.KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        _entity_candidates = build_candidates(kb)
        print(f"[inference] Entity candidates built: {len(_entity_candidates)} entries.")
    except Exception as e:
        print(f"[inference] Warning: could not build entity candidates: {e}")
        _entity_candidates = []

    return _model, _tokenizer


def predict_intent(text: str, model, tokenizer) -> tuple[str, float]:
    """
    Predict the intent of a user query using the fine-tuned DistilBERT model.

    Args:
        text      : The raw user input string.
        model     : Loaded DistilBERT classification model.
        tokenizer : Matching tokenizer.

    Returns:
        intent_label : The predicted intent string (e.g. 'ask_hours').
        confidence   : Softmax probability of the top prediction (0.0 – 1.0).
    """
    # Tokenize the input — same settings used during training
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=config.MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = model(**inputs)       # forward pass
        logits  = outputs.logits        # shape: (1, num_labels)

    # Convert raw logits to probabilities with softmax
    probs = F.softmax(logits, dim=1).squeeze()  # shape: (num_labels,)

    # The predicted class is the one with the highest probability
    top_idx    = torch.argmax(probs).item()
    confidence = probs[top_idx].item()
    intent     = config.ID2LABEL[top_idx]

    return intent, confidence


def extract_entity(text: str) -> str:
    """
    Extract the key noun phrase from the user's question.

    Approach: remove common question words and verbs, return what's left.
    This is intentionally simple — no NLP library needed for 20 KB records.

    Examples:
        "Where is the main library?"        → "main library"
        "What time does the gym close?"     → "gym"
        "Are there any events this week?"   → "events"

    Args:
        text : Raw user input.

    Returns:
        A lower-case string with stopwords removed.
    """
    stopwords = {
        "where", "is", "the", "a", "an", "how", "do", "i", "get", "to",
        "can", "you", "tell", "me", "what", "time", "does", "close", "open",
        "are", "there", "any", "find", "show", "look", "looking", "for",
        "on", "at", "in", "of", "my", "when", "who", "which", "near",
        "about", "this", "that", "would", "need", "want", "could", "please",
        "best", "place", "campus", "today", "week", "late", "night", "early",
        "morning", "evening", "available", "happening", "going",
    }

    # Lowercase and remove punctuation
    cleaned = text.lower()
    for char in "?!.,;:":
        cleaned = cleaned.replace(char, "")

    # Keep only words not in the stopword list
    words = [w for w in cleaned.split() if w not in stopwords]

    entity = " ".join(words).strip()
    return entity if entity else text.lower()   # fallback: return full text if nothing remains


def format_response(intent: str, record: dict | None) -> str:
    """
    Turn a KB record into a human-readable response string.

    If the record has a `response_variants` entry for the given intent,
    one is chosen at random — producing natural, varied replies instead of
    a fixed template. The structured template is used as a fallback when no
    variant exists for that intent.

    Args:
        intent : The predicted intent label string.
        record : The matching KB record dict, or None if no match was found.

    Returns:
        A formatted string ready to display to the user.
    """
    if record is None:
        return get_fallback(intent)

    # Use pre-authored natural language variant if available for this intent
    variants = record.get("response_variants", {}).get(intent, [])
    if variants:
        return random.choice(variants)

    # ── find_navigation ─────────────────────────────────────────────────────
    if intent == "find_navigation":
        nav_type = record.get("type")

        if nav_type == "cross_zone":
            route = record["route"]
            from_z = record["from_zone"]
            to_z = record["to_zone"]
            zones = record.get("zones", {})
            from_name = zones.get(from_z, {}).get("name", from_z)
            to_name = zones.get(to_z, {}).get("name", to_z)
            steps = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(route["steps"]))
            return (
                f"**Directions: Zone {from_z} → Zone {to_z}**\n"
                f"From **{from_name}** to **{to_name}**\n\n"
                f"{steps}\n\n"
                f"🗺️ Look for zones **{from_z}** and **{to_z}** on the campus map."
            )

        if nav_type == "cross_zone_generic":
            from_z = record["from_zone"]
            to_z = record["to_zone"]
            from_info = record.get("from_info", {})
            to_info = record.get("to_info", {})
            from_name = from_info.get("name", f"Zone {from_z}")
            to_name = to_info.get("name", f"Zone {to_z}")
            from_campus = from_info.get("campus_zone", "")
            to_campus = to_info.get("campus_zone", "")
            return (
                f"**Directions: Zone {from_z} → Zone {to_z}**\n"
                f"From **{from_name}** ({from_campus}) to **{to_name}** ({to_campus}).\n\n"
                f"I don't have a step-by-step route for this specific journey, but here are some tips:\n"
                f"  1. Locate Zone **{from_z}** on the campus map.\n"
                f"  2. Head towards **{to_campus}** — follow the campus zone signs.\n"
                f"  3. Look for Zone **{to_z}** ({to_name}) when you arrive.\n\n"
                f"🗺️ Refer to the campus map for zone positions."
            )

        if nav_type == "single_zone":
            z = record["zone"]
            info = record["zone_info"]
            name = info.get("name", z)
            building = info.get("building", "")
            campus = info.get("campus_zone", "")
            return (
                f"**Zone {z} — {name}**\n"
                f"Building: {building}\n"
                f"Campus Area: {campus}\n\n"
                f"🗺️ Find Zone **{z}** on the campus map to locate this area."
            )

        if nav_type == "gate_route":
            route = record["route"]
            return (
                f"**Directions from {route['from']}**\n\n"
                f"{route['description']}\n\n"
                f"🗺️ Look for Zone **{route['to_zone']}** on the campus map."
            )

        if nav_type == "place_to_zone":
            r = record["record"]
            zone = record.get("zone", "?")
            zone_info = record.get("zone_info", {})
            return (
                f"**{r['name']}** is at Map Zone **{zone}**\n"
                f"Building: {r.get('building', 'N/A')}, {r.get('campus_zone', '')}\n\n"
                f"🗺️ Find Zone **{zone}** on the campus map to navigate there.\n"
                f"Nearest entrance: {r.get('nearest_entrance', 'See campus signage')}"
            )

        return get_fallback("find_navigation")

    # ── find_location ───────────────────────────────────────────────────────
    if intent == "find_location":
        building = record.get("building", "")
        floor    = record.get("floor", "")
        zone     = record.get("campus_zone", "")
        map_zone = record.get("map_zone", "")
        entrance = record.get("nearest_entrance", "")
        desc     = record.get("description", "")
        hours    = record.get("hours", {})
        notes    = record.get("notes", "")

        if isinstance(hours, dict):
            hours_summary = " | ".join(f"{k.replace('_',' ').title()}: {v}" for k, v in list(hours.items())[:2])
        else:
            hours_summary = str(hours)

        map_hint    = f"\n🔤 **Map Zone:** **{map_zone}** — find this label on the campus map." if map_zone else ""
        hours_hint  = f"\n🕘 **Hours (summary):** {hours_summary}" if hours_summary else ""
        notes_hint  = f"\n📌 {notes}" if notes else ""

        return (
            f"**{record['name']}**\n"
            f"{desc}\n\n"
            f"📍 **Location:** {building}, {floor}\n"
            f"🗺️ **Campus Zone:** {zone}\n"
            f"🚪 **Nearest Entrance:** {entrance}"
            f"{hours_hint}"
            f"{notes_hint}"
            f"{map_hint}"
        )

    # ── ask_hours ───────────────────────────────────────────────────────────
    elif intent == "ask_hours":
        hours = record.get("hours", {})
        map_zone = record.get("map_zone", "")

        if isinstance(hours, dict):
            hours_lines = "\n".join(
                f"  • {day.replace('_', ' ').title()}: {time}"
                for day, time in hours.items()
            )
        else:
            hours_lines = f"  • {hours}"

        # Meal service times (cafeteria)
        meal_service = record.get("meal_service", {})
        meal_block = ""
        if meal_service:
            meal_lines = "\n".join(
                f"  • {slot.replace('_', ' ').title()}: {detail}"
                for slot, detail in meal_service.items()
            )
            meal_block = f"\n\n🍽️ **Meal Service Times**\n{meal_lines}"

        # Detailed service windows (gym, library, medical, etc.)
        service_hours = record.get("service_hours", {})
        service_block = ""
        if service_hours:
            service_lines = "\n".join(
                f"  • {slot.replace('_', ' ').title()}: {detail}"
                for slot, detail in service_hours.items()
            )
            service_block = f"\n\n🕐 **Detailed Service Hours**\n{service_lines}"

        notes = record.get("notes", "")
        notes_block = f"\n\n📌 **Note:** {notes}" if notes else ""

        map_hint = f"\n🔤 **Map Zone:** **{map_zone}**" if map_zone else ""

        return (
            f"**{record['name']} — Opening Hours**\n\n"
            f"{hours_lines}"
            f"{meal_block}"
            f"{service_block}"
            f"{notes_block}"
            f"{map_hint}"
        )

    # ── find_event ──────────────────────────────────────────────────────────
    elif intent == "find_event":
        date     = record.get("date", "TBC")
        time     = record.get("time", "TBC")
        location = record.get("location_name", "TBC")
        desc     = record.get("description", "")

        return (
            f"**{record['title']}**\n"
            f"{desc}\n\n"
            f"📅 **Date:** {date}\n"
            f"⏰ **Time:** {time}\n"
            f"📍 **Location:** {location}"
        )

    # ── find_department ─────────────────────────────────────────────────────
    elif intent == "find_department":
        building = record.get("building", "")
        floor    = record.get("floor", "")
        email    = record.get("contact_email", "N/A")
        phone    = record.get("contact_phone", "N/A")
        hours    = record.get("office_hours", "N/A")
        map_zone = record.get("map_zone", "")
        desc     = record.get("description", "")

        map_hint = f"\n🔤 **Map Zone:** **{map_zone}** — find this label on the campus map." if map_zone else ""

        return (
            f"**{record['name']}**\n"
            f"{desc}\n\n"
            f"📍 **Building:** {building}, {floor}\n"
            f"🕘 **Office Hours:** {hours}\n"
            f"📧 **Email:** {email}\n"
            f"📞 **Phone:** {phone}"
            f"{map_hint}"
        )

    # ── find_study_area ─────────────────────────────────────────────────────
    elif intent == "find_study_area":
        building  = record.get("building", "")
        capacity  = record.get("capacity", "N/A")
        features  = record.get("features", [])
        hours     = record.get("hours", "N/A")
        booking   = "Yes" if record.get("booking_required") else "No"
        map_zone  = record.get("map_zone", "")
        desc      = record.get("description", "")

        features_str = ", ".join(features) if features else "N/A"
        map_hint = f"\n🔤 **Map Zone:** **{map_zone}** — find this label on the campus map." if map_zone else ""

        return (
            f"**{record['name']}**\n"
            f"{desc}\n\n"
            f"📍 **Location:** {building}\n"
            f"🕘 **Hours:** {hours}\n"
            f"👥 **Capacity:** {capacity} seats\n"
            f"✅ **Features:** {features_str}\n"
            f"📋 **Booking Required:** {booking}"
            f"{map_hint}"
        )

    # ── fallback ────────────────────────────────────────────────────────────
    else:
        map_zone = record.get("map_zone", "")
        map_hint = f" (Map Zone **{map_zone}**)" if map_zone else ""
        return (
            f"Here is what I found about **{record.get('name', 'that')}**{map_hint}: "
            f"{record.get('description', 'No details available.')}"
        )


def get_entity_candidates() -> list[str]:
    """Return the cached KB entity candidates (empty list if not yet loaded)."""
    return _entity_candidates


def answer_query(text: str, session_id: str | None = None) -> dict:
    """
    Top-level function called by the FastAPI route handler.

    Runs the full pipeline:
        resolve_entity (from context) → predict_intent → extract_entity → kb_lookup → format_response

    Args:
        text       : The user's typed question.
        session_id : Optional UUID4 from the frontend for conversation memory.

    Returns:
        A dict with keys:
            - 'reply'      : Formatted response string (markdown)
            - 'intent'     : Predicted intent label
            - 'confidence' : Softmax confidence score (0–1)
            - 'entity'     : Resolved entity used in the KB lookup
    """
    from src.context import get_context, resolve_entity, update_context
    from src.entity_extractor import extract_entity as fuzzy_extract

    model, tokenizer = load_inference_model()

    # Step 1 — classify intent
    intent, confidence = predict_intent(text, model, tokenizer)

    # Step 2 — low confidence fallback
    if confidence < config.CONFIDENCE_THRESHOLD:
        return {
            "reply": get_fallback("low_confidence"),
            "intent": "unknown",
            "confidence": round(confidence, 4),
            "entity": "",
        }

    # Step 3 — extract entity from query
    # For navigation, pass the full original text so "A to K" / "library to sports centre"
    # patterns are preserved — entity extraction strips the "to" separator and zone letters.
    if intent == "find_navigation":
        entity = text
    else:
        # Check conversation context for pronoun resolution first
        context = get_context(session_id) if session_id else []
        resolved = resolve_entity(text, context) if session_id else None

        if resolved:
            entity = resolved
        elif _entity_candidates:
            entity = fuzzy_extract(text, _entity_candidates)
        else:
            entity = extract_entity(text)

    # Step 4 — look up matching KB record
    record = lookup(intent, entity)

    # Step 5 — use Groq LLM if enabled, else fall back to template response
    reply = None
    if config.USE_LLM:
        # If DistilBERT confidence is low, ask Groq to validate/refine the intent first
        if confidence < config.LLM_INTENT_THRESHOLD and intent != "find_navigation":
            from src.llm import classify_intent
            context_for_llm = get_context(session_id) if session_id else []
            refined_intent = classify_intent(text, intent, confidence, context_for_llm)
            if refined_intent != intent:
                intent = refined_intent
                record = lookup(intent, entity)

        from src.llm import generate_response
        context_for_llm = get_context(session_id) if session_id else []
        reply = generate_response(intent, entity, record, text, context_for_llm)

    if reply is None:
        reply = format_response(intent, record)

    # Step 6 — update session context
    if session_id:
        update_context(session_id, entity=entity, intent=intent, reply=reply)

    return {
        "reply":      reply,
        "intent":     intent,
        "confidence": round(confidence, 4),
        "entity":     entity,
    }


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "Where is the main library?",
        "What time does the gym close?",
        "Tell me about the freshers fair",
        "Where is the computer science department?",
        "Is there a 24-hour study room?",
    ]

    print("=== Testing inference.py ===\n")
    for query in test_queries:
        result = answer_query(query)
        print(f"Q: {query}")
        print(f"Intent: {result['intent']} (confidence: {result['confidence']:.2f})")
        print(f"A: {result['reply']}\n{'─' * 60}\n")
