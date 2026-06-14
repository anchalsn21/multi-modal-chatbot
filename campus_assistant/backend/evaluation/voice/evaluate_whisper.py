"""
evaluate_whisper.py — Voice modality evaluation for Phase 2.

Measures:
  - Word Error Rate (WER) between Whisper transcript and expected transcript
  - Intent accuracy after running the transcript through the DistilBERT pipeline

Usage (run from the backend/ directory):
    python evaluation/voice/evaluate_whisper.py

Prerequisites:
  - Audio files placed in evaluation/voice/audio/ (e.g. q01.webm, q02.webm, ...)
  - faster-whisper and jiwer installed: pip install faster-whisper jiwer

Output:
  - Prints summary to console
  - Writes evaluation/voice/evaluation_results.json
"""

import csv
import json
import logging
import os
import sys

# Allow imports from the backend root (src.asr, src.inference)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from jiwer import wer
from src.asr import load_whisper_model, transcribe_audio
from src.inference import load_inference_model, answer_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "sample_queries.csv")
AUDIO_DIR = os.path.join(SCRIPT_DIR, "audio")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "evaluation_results.json")


def main():
    logger.info("Loading models...")
    load_whisper_model()
    load_inference_model()

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        audio_path = os.path.join(AUDIO_DIR, row["audio_file"])
        if not os.path.exists(audio_path):
            logger.warning("Audio file not found, skipping: %s", audio_path)
            continue

        logger.info("Processing %s...", row["audio_file"])
        transcript = transcribe_audio(audio_path)
        word_error = wer(row["expected_transcript"].lower(), transcript.lower())
        result = answer_query(transcript)

        results.append({
            "audio_file": row["audio_file"],
            "expected_transcript": row["expected_transcript"],
            "actual_transcript": transcript,
            "wer": round(word_error, 4),
            "expected_intent": row["expected_intent"],
            "predicted_intent": result["intent"],
            "intent_correct": result["intent"] == row["expected_intent"],
        })

    if not results:
        logger.error("No results — make sure audio files exist in %s", AUDIO_DIR)
        return

    intent_accuracy = sum(r["intent_correct"] for r in results) / len(results)
    avg_wer = sum(r["wer"] for r in results) / len(results)

    output = {
        "num_samples": len(results),
        "intent_accuracy": round(intent_accuracy, 4),
        "avg_wer": round(avg_wer, 4),
        "per_query": results,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Evaluation Results ===")
    print(f"Samples evaluated : {len(results)}")
    print(f"Intent Accuracy   : {intent_accuracy:.1%}")
    print(f"Average WER       : {avg_wer:.4f}")
    print(f"Results saved to  : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
