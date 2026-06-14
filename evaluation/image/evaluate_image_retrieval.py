"""
evaluate_image_retrieval.py — Image retrieval evaluation for Phase 3.

Measures:
  - Top-1 accuracy   : predicted best match == expected location
  - Top-3 accuracy   : expected location appears in top-3 candidates
  - Per-location precision, recall, F1 (at IMAGE_MATCH_THRESHOLD)
  - Macro-averaged precision and recall

Usage (run from the backend/ directory):
    python evaluation/image/evaluate_image_retrieval.py

Prerequisites:
  - Populate data/images/<location_slug>/ with real campus photos
  - Rebuild the FAISS index: python src/build_image_index.py
  - Populate image_eval_manifest.csv with ground-truth image labels

Output:
  - Prints summary table to console
  - Writes evaluation/image/image_eval_results.json
"""

import csv
import json
import logging
import os
import sys
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.image_search import load_clip_model, load_faiss_index, search_by_image
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MANIFEST    = os.path.join(SCRIPT_DIR, "image_eval_manifest.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "image_eval_results.json")


def main() -> None:
    logger.info("Loading CLIP model and FAISS index...")
    load_clip_model()
    load_faiss_index()
    logger.info("Models loaded. Starting evaluation...")

    if not os.path.exists(MANIFEST):
        raise FileNotFoundError(
            f"Ground-truth manifest not found: {MANIFEST}\n"
            "Populate image_eval_manifest.csv with rows: image_path, expected_location_id, expected_location_name"
        )

    with open(MANIFEST, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        logger.warning("Manifest is empty — no images to evaluate.")
        return

    results = []
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for row in rows:
        img_path = os.path.join(BASE_DIR, row["image_path"])
        if not os.path.exists(img_path):
            logger.warning("Image not found, skipping: %s", img_path)
            continue

        pil_img = Image.open(img_path).convert("RGB")
        candidates, is_confident = search_by_image(pil_img, top_k=config.IMAGE_TOP_K)

        expected_id   = row["expected_location_id"]
        top1_id       = candidates[0][0].get("id") if candidates else "unknown"
        top3_ids      = [c[0].get("id") for c in candidates]
        top1_score    = candidates[0][1] if candidates else 0.0

        top1_correct = top1_id == expected_id
        top3_correct = expected_id in top3_ids

        if is_confident:
            if top1_correct:
                tp[expected_id] += 1
            else:
                fp[top1_id] += 1
                fn[expected_id] += 1
        else:
            # System abstained — counts as a missed detection for the expected location
            fn[expected_id] += 1

        results.append({
            "image_path":      row["image_path"],
            "expected_id":     expected_id,
            "expected_name":   row["expected_location_name"],
            "top1_id":         top1_id,
            "top1_name":       candidates[0][0].get("name", "unknown") if candidates else "unknown",
            "top1_score":      round(top1_score, 4),
            "top1_correct":    top1_correct,
            "top3_correct":    top3_correct,
            "is_confident":    is_confident,
            "top3_candidates": [(c[0].get("id"), round(c[1], 4)) for c in candidates],
        })

    n = len(results)
    if n == 0:
        logger.warning("No valid images found. Check image paths in the manifest.")
        return

    top1_acc = sum(r["top1_correct"] for r in results) / n
    top3_acc = sum(r["top3_correct"] for r in results) / n

    # Per-location precision and recall
    location_ids = set(tp) | set(fn)
    per_location = {}
    for loc_id in sorted(location_ids):
        denom_p = tp[loc_id] + fp[loc_id]
        denom_r = tp[loc_id] + fn[loc_id]
        p  = tp[loc_id] / denom_p if denom_p else 0.0
        r  = tp[loc_id] / denom_r if denom_r else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        per_location[loc_id] = {
            "precision": round(p,  4),
            "recall":    round(r,  4),
            "f1":        round(f1, 4),
            "tp": tp[loc_id],
            "fp": fp[loc_id],
            "fn": fn[loc_id],
        }

    macro_p = sum(v["precision"] for v in per_location.values()) / len(per_location) if per_location else 0.0
    macro_r = sum(v["recall"]    for v in per_location.values()) / len(per_location) if per_location else 0.0

    output = {
        "num_samples":     n,
        "top1_accuracy":   round(top1_acc, 4),
        "top3_accuracy":   round(top3_acc, 4),
        "macro_precision": round(macro_p,  4),
        "macro_recall":    round(macro_r,  4),
        "threshold_used":  config.IMAGE_MATCH_THRESHOLD,
        "top_k":           config.IMAGE_TOP_K,
        "per_location":    per_location,
        "per_image":       results,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\n=== Image Retrieval Evaluation ===")
    print(f"Samples         : {n}")
    print(f"Threshold used  : {config.IMAGE_MATCH_THRESHOLD}")
    print(f"Top-1 Accuracy  : {top1_acc:.1%}")
    print(f"Top-3 Accuracy  : {top3_acc:.1%}")
    print(f"Macro Precision : {macro_p:.4f}")
    print(f"Macro Recall    : {macro_r:.4f}")
    print(f"\nPer-location results:")
    for loc_id, v in per_location.items():
        print(f"  {loc_id}: P={v['precision']:.3f}  R={v['recall']:.3f}  F1={v['f1']:.3f}  (TP={v['tp']} FP={v['fp']} FN={v['fn']})")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
