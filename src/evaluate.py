"""
evaluate.py — Evaluate the trained DistilBERT model on the test split.

Run after training:
    python src/evaluate.py

What this script does:
    1. Loads the saved model from models/intent_classifier/
    2. Runs inference on the test split of faq_dataset.csv
    3. Prints a full classification report:
         - Per-class Precision, Recall, F1-Score
         - Overall Accuracy
         - Macro and Weighted averages

You can include the printed report directly in your academic assignment report.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_faq_csv
from src.dataset import IntentDataset


def load_saved_model():
    """
    Load the fine-tuned model and tokenizer from disk.

    The model must have been saved by train.py first.
    If the model directory does not exist, a clear error message is shown.

    Returns:
        model     : The loaded DistilBERT classification model in eval mode.
        tokenizer : The matching tokenizer.
    """
    if not os.path.exists(config.MODEL_SAVE_DIR):
        raise FileNotFoundError(
            f"No saved model found at: {config.MODEL_SAVE_DIR}\n"
            "Please run `python src/train.py` first to train and save the model."
        )

    print(f"[evaluate] Loading model from: {config.MODEL_SAVE_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_SAVE_DIR)
    model.eval()    # disable dropout for evaluation

    return model, tokenizer


def predict_batch(model, loader: DataLoader, device) -> tuple:
    """
    Run inference over all batches and collect true labels and predictions.

    Args:
        model  : Loaded DistilBERT model in eval mode.
        loader : DataLoader wrapping the test IntentDataset.
        device : torch.device.

    Returns:
        all_preds  : List of predicted class indices.
        all_labels : List of true class indices.
    """
    all_preds  = []
    all_labels = []

    with torch.no_grad():   # no gradient tracking needed during evaluation
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"]

            # Forward pass — we only need logits, not loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # logits shape: (batch_size, num_labels)
            # argmax gives the index of the highest-scoring class
            preds = torch.argmax(outputs.logits, dim=1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def print_metrics(y_true: list, y_pred: list):
    """
    Print a full sklearn classification report to the console.

    The report includes:
        - Per-intent: Precision, Recall, F1-Score, Support
        - Overall Accuracy
        - Macro average (treats all classes equally)
        - Weighted average (weights by class size)

    Args:
        y_true : List of integer true labels.
        y_pred : List of integer predicted labels.
    """
    accuracy = accuracy_score(y_true, y_pred)

    print("\n" + "=" * 60)
    print("INTENT CLASSIFICATION EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
    print(classification_report(
        y_true,
        y_pred,
        target_names=config.INTENT_LABELS,   # show label names instead of numbers
        digits=4,                             # 4 decimal places for precision
    ))
    print("=" * 60)


def main():
    """
    Full evaluation pipeline: load model → load test data → predict → report.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Using device: {device}")

    # Load the saved model
    model, tokenizer = load_saved_model()
    model.to(device)

    # Load the test split
    test_df = load_faq_csv(split="test")
    print(f"[evaluate] Test examples: {len(test_df)}")

    # Build dataset and loader
    test_dataset = IntentDataset(test_df, tokenizer)
    test_loader  = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # Run predictions
    y_pred, y_true = predict_batch(model, test_loader, device)

    # Print the classification report
    print_metrics(y_true, y_pred)


if __name__ == "__main__":
    main()
