"""
data_loader.py — Utility functions for loading the knowledge base and FAQ dataset.

These functions are imported by train.py, evaluate.py, inference.py, and kb_lookup.py.
Keeping data loading in one place means if the file paths change, you only update config.py.
"""

import json
import os
import sys
import pandas as pd

# Add the backend root to sys.path so we can import config from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_kb() -> dict:
    """
    Load the campus knowledge base from the JSON file.

    Returns:
        A Python dictionary with keys: 'locations', 'departments', 'events', 'study_areas'.
        Each key maps to a list of record dictionaries.

    Raises:
        FileNotFoundError: If campus_kb.json does not exist at the path in config.py.
    """
    if not os.path.exists(config.KB_PATH):
        raise FileNotFoundError(
            f"Knowledge base not found at: {config.KB_PATH}\n"
            "Make sure data/campus_kb.json exists in the backend/data/ folder."
        )

    with open(config.KB_PATH, "r", encoding="utf-8") as f:
        kb = json.load(f)

    print(f"[data_loader] Loaded KB with sections: {list(kb.keys())}")
    return kb


def load_faq_csv(split: str = None) -> pd.DataFrame:
    """
    Load the FAQ training dataset from CSV.

    Args:
        split: Optional string — 'train', 'val', or 'test'.
               If provided, only rows with that split value are returned.
               If None, the full dataset is returned.

    Returns:
        A pandas DataFrame with columns: text, intent, entity, split.

    Raises:
        FileNotFoundError: If faq_dataset.csv does not exist.
        ValueError: If the CSV is missing required columns.
    """
    if not os.path.exists(config.CSV_PATH):
        raise FileNotFoundError(
            f"FAQ dataset not found at: {config.CSV_PATH}\n"
            "Make sure data/faq_dataset.csv exists in the backend/data/ folder."
        )

    df = pd.read_csv(config.CSV_PATH)

    # Validate that the expected columns are present
    required_columns = {"text", "intent", "entity", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Drop rows whose intent label is not in the known 6-class label set
    df = df[df["intent"].isin(config.INTENT_LABELS)].reset_index(drop=True)

    # If a specific split is requested, filter the DataFrame
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)

    print(f"[data_loader] Loaded FAQ CSV — split='{split}', rows={len(df)}")
    return df


def get_label_counts(df: pd.DataFrame) -> dict:
    """
    Count how many examples exist per intent label.

    Useful for checking class balance before training — ideally all intents
    should have a similar number of examples.

    Args:
        df: A DataFrame returned by load_faq_csv().

    Returns:
        A dict like {'find_location': 30, 'ask_hours': 30, ...}
    """
    counts = df["intent"].value_counts().to_dict()
    return counts


# ── Quick sanity check when run directly ────────────────────────────────────
if __name__ == "__main__":
    print("=== Testing data_loader.py ===")

    kb = load_kb()
    print(f"KB sections: {list(kb.keys())}")
    print(f"  Locations  : {len(kb['locations'])}")
    print(f"  Departments: {len(kb['departments'])}")
    print(f"  Events     : {len(kb['events'])}")
    print(f"  Study areas: {len(kb['study_areas'])}")

    full_df = load_faq_csv()
    print(f"\nFull dataset shape: {full_df.shape}")
    print(f"Label counts:\n{get_label_counts(full_df)}")

    train_df = load_faq_csv(split="train")
    val_df   = load_faq_csv(split="val")
    test_df  = load_faq_csv(split="test")
    print(f"\nSplits — train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
