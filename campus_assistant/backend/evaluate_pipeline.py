"""
evaluate_pipeline.py — Academic evaluation script for Phase 4.

Measures:
  1. Intent classification accuracy and per-class F1 (test split of faq_dataset.csv)
  2. Entity extraction match rate (ground-truth entity column vs RapidFuzz output)
  3. Average inference latency (ms per query)
  4. CLIP threshold sweep (precision/recall at IMAGE_MATCH_THRESHOLD ∈ {0.20,0.25,0.30,0.35})
     if a labelled image eval CSV exists at data/image_eval.csv

Run:
    cd backend
    python evaluate_pipeline.py

Output:
    - Intent classification report (stdout)
    - Entity extraction match rate (stdout)
    - Latency table (stdout)
    - Optional: CLIP threshold sweep table (stdout)
"""

import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_test_split(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError(f"No rows with split=='test' found in {csv_path}")
    return test


def _print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Intent classification
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_intent(df: pd.DataFrame) -> None:
    _print_section("1. Intent Classification")

    from src.inference import load_inference_model, predict_intent

    print("Loading model...")
    model, tokenizer = load_inference_model()

    y_true, y_pred, latencies = [], [], []

    for _, row in df.iterrows():
        t0 = time.perf_counter()
        pred_intent, _ = predict_intent(row["text"], model, tokenizer)
        latencies.append((time.perf_counter() - t0) * 1000)
        y_true.append(row["intent"])
        y_pred.append(pred_intent)

    print(classification_report(y_true, y_pred, zero_division=0))
    print(f"Confusion matrix:\n{confusion_matrix(y_true, y_pred, labels=config.INTENT_LABELS)}")
    print(f"\nLatency (intent classification only):")
    print(f"  Mean:   {sum(latencies)/len(latencies):.1f} ms")
    print(f"  Median: {sorted(latencies)[len(latencies)//2]:.1f} ms")
    print(f"  P95:    {sorted(latencies)[int(len(latencies)*0.95)]:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Entity extraction match rate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_entity(df: pd.DataFrame) -> None:
    _print_section("2. Entity Extraction")

    from src.entity_extractor import build_candidates, extract_entity
    from src.inference import extract_entity as stopword_extract

    with open(config.KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)
    candidates = build_candidates(kb)

    # Only rows that have a non-empty ground truth entity
    entity_df = df[df["entity"].notna() & (df["entity"].str.strip() != "")].copy()
    if entity_df.empty:
        print("No entity ground-truth found in test split. Skipping.")
        return

    fuzzy_hits, stopword_hits, total = 0, 0, 0

    for _, row in entity_df.iterrows():
        gt = row["entity"].lower().strip()
        text = row["text"]

        fuzzy_pred = extract_entity(text, candidates).lower()
        stop_pred = stopword_extract(text).lower()

        # Flexible match: ground truth contained in prediction or vice versa
        fuzzy_match = gt in fuzzy_pred or fuzzy_pred in gt
        stop_match = gt in stop_pred or stop_pred in gt

        fuzzy_hits += int(fuzzy_match)
        stopword_hits += int(stop_match)
        total += 1

    print(f"Test examples with entity labels: {total}")
    print(f"  RapidFuzz match rate:    {fuzzy_hits/total*100:.1f}% ({fuzzy_hits}/{total})")
    print(f"  Stopword match rate:     {stopword_hits/total*100:.1f}% ({stopword_hits}/{total})")
    print(f"  Improvement:             +{(fuzzy_hits-stopword_hits)/total*100:.1f}pp")


# ─────────────────────────────────────────────────────────────────────────────
# 3. End-to-end pipeline latency
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_latency(df: pd.DataFrame) -> None:
    _print_section("3. End-to-End Pipeline Latency")

    from src.inference import answer_query

    latencies = []
    sample = df.sample(min(50, len(df)), random_state=42)

    for _, row in sample.iterrows():
        t0 = time.perf_counter()
        answer_query(row["text"])
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    print(f"Sample size: {len(latencies)}")
    print(f"  Mean:   {sum(latencies)/len(latencies):.1f} ms")
    print(f"  Median: {latencies[len(latencies)//2]:.1f} ms")
    print(f"  P95:    {latencies[int(len(latencies)*0.95)]:.1f} ms")
    print(f"  Max:    {latencies[-1]:.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-end KB retrieval accuracy
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_e2e_kb(df: pd.DataFrame) -> None:
    """
    End-to-end KB retrieval accuracy: measures whether the full pipeline
    (intent + entity extraction + KB lookup) returns any KB record for each
    query in the test split.

    A result is considered a 'hit' if answer_query() returns a non-empty
    response that is not the default "I don't understand" fallback.
    """
    _print_section("4. End-to-End KB Retrieval Accuracy")

    from src.inference import answer_query

    FALLBACK_PHRASES = [
        "i didn't understand",
        "i don't understand",
        "could you rephrase",
        "sorry, i couldn't",
        "no information",
    ]

    hits, total = 0, 0
    per_intent: dict[str, list[int]] = {}

    for _, row in df.iterrows():
        response = answer_query(row["text"])
        is_hit = response and not any(p in response.lower() for p in FALLBACK_PHRASES)
        intent = row["intent"]
        per_intent.setdefault(intent, []).append(int(is_hit))
        hits += int(is_hit)
        total += 1

    print(f"End-to-end KB retrieval accuracy: {hits}/{total}  ({hits/total*100:.1f}%)")
    print(f"\nPer-intent KB hit rate:")
    for intent, flags in sorted(per_intent.items()):
        h = sum(flags)
        n = len(flags)
        print(f"  {intent:<28} {h:3d}/{n}  ({h/n*100:.0f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLIP threshold sweep (optional)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_clip_thresholds() -> None:
    IMAGE_EVAL_PATH = os.path.join(os.path.dirname(__file__), "data", "image_eval.csv")
    if not os.path.exists(IMAGE_EVAL_PATH):
        print("\n[CLIP threshold sweep] data/image_eval.csv not found — skipping.")
        print("  To enable: create data/image_eval.csv with columns: image_path, expected_location_id")
        return

    _print_section("4. CLIP Threshold Sweep")

    from src.image_search import load_clip_model, load_faiss_index, search_by_image
    from PIL import Image

    load_clip_model()
    load_faiss_index()

    eval_df = pd.read_csv(IMAGE_EVAL_PATH)
    thresholds = [0.20, 0.25, 0.30, 0.35]

    print(f"{'Threshold':>12} | {'Top-1 Acc':>10} | {'Coverage':>10} | {'Precision':>10}")
    print("-" * 52)

    for thresh in thresholds:
        hits, covered, total = 0, 0, len(eval_df)

        for _, row in eval_df.iterrows():
            img_path = row["image_path"]
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB")
            candidates, _ = search_by_image(img, top_k=1)
            if not candidates:
                continue
            record, score = candidates[0]
            if score >= thresh:
                covered += 1
                if record.get("id") == row["expected_location_id"]:
                    hits += 1

        coverage = covered / total if total > 0 else 0
        precision = hits / covered if covered > 0 else 0
        top1 = hits / total if total > 0 else 0

        print(f"{thresh:>12.2f} | {top1*100:>9.1f}% | {coverage*100:>9.1f}% | {precision*100:>9.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Campus Assistant — Phase 4 Evaluation")
    print(f"Dataset: {config.CSV_PATH}")
    print(f"Model:   {config.MODEL_SAVE_DIR}")

    df = _load_test_split(config.CSV_PATH)
    print(f"Test examples: {len(df)}")

    evaluate_intent(df)
    evaluate_entity(df)
    evaluate_latency(df)
    evaluate_e2e_kb(df)
    evaluate_clip_thresholds()

    print("\nEvaluation complete.")
