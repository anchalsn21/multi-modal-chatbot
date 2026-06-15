"""
generate_voice_queries.py — Generate synthetic voice queries for the campus assistant.

Uses pyttsx3 (offline TTS, no API key required) to produce WAV files organised by
intent under data/audio/<intent>/<index>.wav.

Intent categories match the FAQ dataset:
  find_location, ask_hours, find_study_area, find_events,
  ask_accessibility, general_greeting

Run:
    pip install pyttsx3
    python generate_voice_queries.py

Output:
    data/audio/
        find_location/    query_001.wav … query_010.wav
        ask_hours/        query_001.wav … query_010.wav
        find_study_area/  query_001.wav … query_006.wav
        find_events/      query_001.wav … query_006.wav
        ask_accessibility/query_001.wav … query_004.wav
        general_greeting/ query_001.wav … query_004.wav
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ── Voice queries by intent ───────────────────────────────────────────────────

QUERIES: dict[str, list[str]] = {
    "find_location": [
        "Where is the main library?",
        "How do I get to the student union?",
        "Where is the chemistry department?",
        "Can you show me where the cafeteria is?",
        "Where is the sports centre on campus?",
        "I need to find the administration office.",
        "Where is the careers service?",
        "How do I get to the lecture theatre?",
        "Where can I find the IT helpdesk?",
        "Where is the medical centre located?",
    ],
    "ask_hours": [
        "What time does the library open?",
        "When does the gym close?",
        "What are the opening hours for the cafeteria?",
        "Is the student union open on Sundays?",
        "What time does the campus close?",
        "When is the careers office open?",
        "Are the labs open on weekends?",
        "What are the reception hours?",
        "Is the library open late tonight?",
        "What time does the sports centre open in the morning?",
    ],
    "find_study_area": [
        "Where can I find a quiet study area?",
        "Show me events at the student union today.",
        "Find a quiet study area near the cafeteria.",
        "Where are the group study rooms?",
        "Is there a silent reading room on campus?",
        "Where can I study late at night?",
    ],
    "find_events": [
        "What events are happening on campus today?",
        "Are there any workshops this week?",
        "Show me events at the student union today.",
        "What is happening at the library this week?",
        "Are there any career fairs coming up?",
        "What student events are scheduled for tomorrow?",
    ],
    "ask_accessibility": [
        "Is the library wheelchair accessible?",
        "Where is the nearest accessible entrance?",
        "Are there lifts in the main building?",
        "Where can I find disabled parking on campus?",
    ],
    "general_greeting": [
        "Hello, I am new here. Can you help me?",
        "Hi, I just arrived at campus. Where do I start?",
        "Good morning. I need some help finding my way around.",
        "Hey, can you tell me about the campus facilities?",
    ],
}


def _generate_with_pyttsx3(audio_root: Path) -> None:
    """Generate WAV files using pyttsx3 (offline, no API key)."""
    try:
        import pyttsx3
    except ImportError:
        print("pyttsx3 not found. Install with:  pip install pyttsx3")
        sys.exit(1)

    engine = pyttsx3.init()
    engine.setProperty("rate", 160)   # words per minute — slightly slower for clarity
    engine.setProperty("volume", 0.9)

    # Pick a clearer voice if multiple are available
    voices = engine.getProperty("voices")
    if len(voices) > 1:
        engine.setProperty("voice", voices[1].id)

    total = sum(len(v) for v in QUERIES.values())
    done = 0

    for intent, sentences in QUERIES.items():
        intent_dir = audio_root / intent
        intent_dir.mkdir(parents=True, exist_ok=True)

        for idx, text in enumerate(sentences, start=1):
            out_path = intent_dir / f"query_{idx:03d}.wav"
            if out_path.exists():
                print(f"  [skip] {out_path.name} already exists")
                done += 1
                continue

            engine.save_to_file(text, str(out_path))
            engine.runAndWait()
            time.sleep(0.05)   # small pause to avoid engine conflicts

            done += 1
            print(f"  [{done:3d}/{total}] {intent}/query_{idx:03d}.wav  \"{text}\"")

    print(f"\nDone. {done} audio files written to {audio_root}/")


def _summarise(audio_root: Path) -> None:
    print("\n── Summary ─────────────────────────────────────────────")
    total_files = 0
    for intent_dir in sorted(audio_root.iterdir()):
        if intent_dir.is_dir():
            files = list(intent_dir.glob("*.wav"))
            total_files += len(files)
            sizes = [f.stat().st_size for f in files]
            avg_kb = sum(sizes) / len(sizes) / 1024 if sizes else 0
            print(f"  {intent_dir.name:<22}  {len(files):3d} files  avg {avg_kb:.0f} KB")
    print(f"  {'TOTAL':<22}  {total_files:3d} files")
    print("─────────────────────────────────────────────────────────")


if __name__ == "__main__":
    audio_root = Path(__file__).parent / "data" / "audio"
    print(f"Generating synthetic voice queries -> {audio_root}\n")
    _generate_with_pyttsx3(audio_root)
    if audio_root.exists():
        _summarise(audio_root)
