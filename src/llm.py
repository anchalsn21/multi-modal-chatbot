"""
llm.py — Groq LLM integration for intent refinement and response generation.

Two-stage pipeline:
  Stage 1 (DistilBERT): Fast first-pass intent classification — preserves the
    fine-tuned model for assignment evaluation and keeps latency low.
  Stage 2 (Groq LLaMA): Uses the classified intent + KB record to generate a
    natural, contextual response grounded in real campus data (RAG-style).
    Also validates/refines the intent when DistilBERT confidence is low.

The Groq client is initialised lazily on first call so the server still boots
if GROQ_API_KEY is missing (falls back to template responses).
"""

from __future__ import annotations

import os
import sys
import json
from typing import Optional
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Lazy-initialised Groq client — None until first call
_groq_client = None


def _get_client():
    """Return (and lazily init) the Groq client. Returns None if key is absent."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    api_key = config.GROQ_API_KEY
    if not api_key:
        print("[llm] GROQ_API_KEY not set — LLM generation disabled, using templates.")
        return None

    try:
        from groq import Groq
        _groq_client = Groq(api_key=api_key)
        print(f"[llm] Groq client ready (model: {config.GROQ_MODEL})")
        return _groq_client
    except ImportError:
        print("[llm] groq package not installed — run: pip install groq>=0.4.0")
        return None
    except Exception as e:
        print(f"[llm] Failed to initialise Groq client: {e}")
        return None


# ── Prompt builders ────────────────────────────────────────────────────────────

def _build_system_prompt(record: dict | None, intent: str) -> str:
    """
    Build the system prompt that grounds the LLM in KB data.

    The LLM is explicitly told to answer ONLY from the provided record so it
    cannot hallucinate locations, hours, or contacts that don't exist.
    """
    if record is None:
        return (
            "You are a helpful campus orientation assistant for Greenfield University. "
            "You help students find locations, opening hours, "
            "events, departments, and study areas on campus.\n\n"
            "No matching campus record was found for this query. Politely tell the user "
            "you don't have that information and suggest they visit the main reception "
            "or check the official Greenfield University website."
        )

    # Serialise the KB record as context — limit to relevant fields
    record_text = json.dumps(record, indent=2, ensure_ascii=False)

    intent_instructions = {
        "find_location": "Tell the user WHERE the place is: building, floor, campus zone, nearest entrance, and map zone. Add a brief description.",
        "ask_hours":     "Tell the user the OPENING HOURS. List every time slot clearly. Include meal service or service hours if present. Add any notes.",
        "find_event":    "Tell the user about the EVENT: title, date, time, location, and description. Be enthusiastic.",
        "find_department": "Tell the user about the DEPARTMENT: building, floor, office hours, email, and phone. Be helpful.",
        "find_study_area": "Tell the user about the STUDY AREA: location, hours, capacity, features, and whether booking is required.",
        "find_navigation": "Give the user DIRECTIONS based on the route steps provided. Be clear and step-by-step.",
    }.get(intent, "Answer the user's question using the campus record provided.")

    return (
        "You are a helpful campus orientation assistant for Greenfield University. "
        "You help students find locations, opening hours, "
        "events, departments, and study areas on campus.\n\n"
        f"TASK: {intent_instructions}\n\n"
        "RULES:\n"
        "- Answer ONLY using information from the Campus Record below.\n"
        "- Do NOT invent details, addresses, phone numbers, or hours not in the record.\n"
        "- Format your answer in clear markdown with bold headings and bullet points.\n"
        "- Keep the response concise (under 200 words) and friendly.\n"
        "- End with one helpful tip or suggestion if relevant.\n\n"
        f"CAMPUS RECORD:\n```json\n{record_text}\n```"
    )


def _build_history_messages(history: deque | list) -> list[dict]:
    """Convert the session Turn deque to Groq message format (last 3 turns only)."""
    messages = []
    turns = list(history)[-3:]  # keep last 3 turns to stay within token budget
    for turn in turns:
        if hasattr(turn, "entity"):
            messages.append({"role": "user",      "content": f"[Previous query about: {turn.entity}]"})
            messages.append({"role": "assistant",  "content": turn.reply[:300]})  # truncate long replies
    return messages


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_response(
    intent: str,
    entity: str,
    record: dict | None,
    query: str,
    history: Optional[deque | list] = None,
) -> str:
    """
    Generate a natural language response using Groq LLaMA.

    Falls back to None if the Groq client is unavailable — the caller should
    then use format_response() (template path) as fallback.

    Args:
        intent  : DistilBERT-classified intent label.
        entity  : Resolved entity used in the KB lookup.
        record  : Matching KB record dict (or None if no match found).
        query   : The original user query text.
        history : Session Turn deque for conversational context.

    Returns:
        Generated response string, or None if Groq is unavailable.
    """
    client = _get_client()
    if client is None:
        return None

    system_prompt = _build_system_prompt(record, intent)
    history_msgs  = _build_history_messages(history or [])

    messages = [
        {"role": "system", "content": system_prompt},
        *history_msgs,
        {"role": "user", "content": query},
    ]

    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.4,   # low temp for factual grounding
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[llm] Groq generation error: {e}")
        return None


def classify_intent(query: str, distilbert_intent: str, confidence: float, history: Optional[deque | list] = None) -> str:
    """
    Use Groq to validate or refine a low-confidence DistilBERT intent.

    Only called when DistilBERT confidence is below a secondary threshold
    (config.LLM_INTENT_THRESHOLD). For high-confidence predictions, the
    DistilBERT result is used directly.

    Args:
        query             : The raw user query.
        distilbert_intent : Intent label from DistilBERT.
        confidence        : DistilBERT softmax confidence score.
        history           : Session Turn deque for context.

    Returns:
        A validated intent string (one of INTENT_LABELS), or distilbert_intent
        if Groq is unavailable or returns an unexpected value.
    """
    client = _get_client()
    if client is None:
        return distilbert_intent

    valid_intents = ", ".join(config.INTENT_LABELS)
    history_ctx = ""
    if history:
        last = list(history)[-1] if history else None
        if last and hasattr(last, "entity"):
            history_ctx = f"\nPrevious query was about: '{last.entity}' (intent: {last.intent})"

    system_prompt = (
        "You are an intent classifier for a campus orientation chatbot. "
        f"Classify the user query into EXACTLY ONE of these intents:\n{valid_intents}\n\n"
        "Definitions:\n"
        "- find_location: user wants to know WHERE something is on campus\n"
        "- ask_hours: user wants to know WHEN something is open/closed\n"
        "- find_event: user wants to know about campus EVENTS or activities\n"
        "- find_department: user wants to find an ACADEMIC DEPARTMENT or office\n"
        "- find_study_area: user wants to find a place to STUDY or work\n"
        "- find_navigation: user wants DIRECTIONS or a route between places\n\n"
        f"The primary classifier predicted '{distilbert_intent}' with {confidence:.0%} confidence.{history_ctx}\n\n"
        "Reply with ONLY the intent label — no explanation, no punctuation."
    )

    try:
        completion = client.chat.completions.create(
            model=config.GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw = completion.choices[0].message.content.strip().lower()
        # Only accept known intent labels
        if raw in config.INTENT_LABELS:
            return raw
        # Try to match partial (e.g. model adds punctuation)
        for label in config.INTENT_LABELS:
            if label in raw:
                return label
        return distilbert_intent
    except Exception as e:
        print(f"[llm] Intent classification error: {e}")
        return distilbert_intent
