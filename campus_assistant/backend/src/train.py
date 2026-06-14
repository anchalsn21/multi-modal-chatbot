"""
train.py — Fine-tune DistilBERT for campus intent classification.

Run this script once to train the model:
    python src/train.py

What this script does:
    1. Loads the training and validation splits from faq_dataset.csv
    2. Tokenizes all text with the DistilBERT tokenizer
    3. Fine-tunes DistilBertForSequenceClassification for 5 intent classes
    4. Prints the training loss after each epoch
    5. Saves the trained model and tokenizer to models/intent_classifier/

Expected runtime: ~2–4 minutes on a modern CPU, ~30 seconds on a GPU.
"""

import os
import sys
import json
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # headless backend — no display required
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DistilBertForSequenceClassification

# Make sure we can import from the backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_loader import load_faq_csv
from src.dataset import IntentDataset


def set_seed(seed: int = config.SEED):
    """
    Set random seeds for Python, NumPy, and PyTorch to make results reproducible.
    Without this, training may produce different results on every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer():
    """
    Load the DistilBERT tokenizer and sequence classification model from HuggingFace.

    DistilBertForSequenceClassification adds a linear classification head on top of
    the DistilBERT encoder. We configure it for 5 output classes (one per intent).

    Returns:
        model     : The DistilBERT classification model (not yet trained).
        tokenizer : The matching DistilBERT tokenizer.
    """
    print(f"[train] Loading tokenizer and model: {config.BASE_MODEL}")

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)

    model = DistilBertForSequenceClassification.from_pretrained(
        config.BASE_MODEL,
        num_labels=len(config.INTENT_LABELS),   # 5 classes
        id2label=config.ID2LABEL,               # maps index → label string
        label2id=config.LABEL2ID,               # maps label string → index
    )

    return model, tokenizer


def train_one_epoch(model, loader: DataLoader, optimizer, device) -> float:
    """
    Run one full pass over the training data (one epoch).

    For each batch:
        1. Move tensors to the device (CPU or GPU)
        2. Forward pass → compute loss
        3. Backward pass → compute gradients
        4. Optimizer step → update weights
        5. Zero gradients for the next batch

    Args:
        model    : The DistilBERT model in training mode.
        loader   : DataLoader wrapping the training IntentDataset.
        optimizer: AdamW optimizer.
        device   : torch.device ('cpu' or 'cuda').

    Returns:
        Average loss across all batches in this epoch (float).
    """
    model.train()   # enables dropout (important during training)
    total_loss = 0.0

    for batch_idx, batch in enumerate(loader):
        # Move each tensor in the batch to the correct device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Zero out gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass — HuggingFace models return an object with .loss and .logits
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss  # cross-entropy loss computed internally

        # Backward pass — compute gradients for all parameters
        loss.backward()

        # Gradient clipping prevents exploding gradients (good practice)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update the model weights
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)
    return average_loss


def evaluate_accuracy(model, loader: DataLoader, device) -> float:
    """
    Compute classification accuracy on a dataset split.

    Returns:
        Fraction of examples predicted correctly (0–1 float).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total if total > 0 else 0.0


def evaluate_loss(model, loader: DataLoader, device) -> float:
    """
    Compute the average loss on the validation set without updating weights.

    Args:
        model  : The DistilBERT model.
        loader : DataLoader wrapping the validation IntentDataset.
        device : torch.device.

    Returns:
        Average validation loss (float).
    """
    model.eval()    # disables dropout
    total_loss = 0.0

    with torch.no_grad():   # no gradient computation → faster, less memory
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

    return total_loss / len(loader)


def save_training_curves(history: dict, out_dir: str):
    """
    Save loss and accuracy curves as PNG images and the raw numbers as JSON.

    Args:
        history : Dict with keys train_loss, val_loss, train_acc, val_acc
                  (each a list of floats, one per epoch).
        out_dir : Directory to write plots and JSON into.
    """
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    # ── Loss curve ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax.plot(epochs, history["val_loss"],   marker="s", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("DistilBERT Intent Classifier — Training & Validation Loss")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    loss_path = os.path.join(out_dir, "loss_curve.png")
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Loss curve saved: {loss_path}")

    # ── Accuracy curve ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [v * 100 for v in history["train_acc"]], marker="o", label="Train Accuracy")
    ax.plot(epochs, [v * 100 for v in history["val_acc"]],   marker="s", label="Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("DistilBERT Intent Classifier — Training & Validation Accuracy")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    acc_path = os.path.join(out_dir, "accuracy_curve.png")
    fig.savefig(acc_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train] Accuracy curve saved: {acc_path}")

    # ── Raw history JSON (for notebooks / reports) ────────────────────────────
    json_path = os.path.join(out_dir, "training_history.json")
    with open(json_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[train] Training history saved: {json_path}")


def save_model(model, tokenizer):
    """
    Save the fine-tuned model and tokenizer to disk.

    The saved files include:
        - config.json          (model architecture)
        - model.safetensors    (trained weights)
        - tokenizer_config.json
        - vocab.txt

    These can be reloaded with AutoModelForSequenceClassification.from_pretrained()
    and AutoTokenizer.from_pretrained() at inference time.
    """
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    model.save_pretrained(config.MODEL_SAVE_DIR)
    tokenizer.save_pretrained(config.MODEL_SAVE_DIR)
    print(f"[train] Model saved to: {config.MODEL_SAVE_DIR}")


def main():
    """
    Orchestrate the full training pipeline:
        1. Set seeds → deterministic results
        2. Load data → build datasets and loaders
        3. Build model → move to device
        4. Train loop → print loss each epoch
        5. Save model to disk
    """
    set_seed()

    # Detect GPU; fall back to CPU automatically
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    # ── Load data ──────────────────────────────────────────────────────────
    train_df = load_faq_csv(split="train")
    val_df   = load_faq_csv(split="val")

    # ── Build model and tokenizer ──────────────────────────────────────────
    model, tokenizer = build_model_and_tokenizer()
    model.to(device)

    # ── Create datasets and data loaders ──────────────────────────────────
    train_dataset = IntentDataset(train_df, tokenizer)
    val_dataset   = IntentDataset(val_df,   tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config.BATCH_SIZE, shuffle=False)

    # ── Set up optimizer ───────────────────────────────────────────────────
    # AdamW is the standard optimizer for transformer fine-tuning.
    # weight_decay adds L2 regularisation to prevent overfitting.
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)

    # ── Training loop ──────────────────────────────────────────────────────
    print(f"\n[train] Starting training for {config.EPOCHS} epochs...\n")
    print(f"  Train examples : {len(train_dataset)}")
    print(f"  Val examples   : {len(val_dataset)}")
    print(f"  Batch size     : {config.BATCH_SIZE}")
    print(f"  Learning rate  : {config.LEARNING_RATE}\n")

    # Learning rate scheduler — linear warmup / cosine decay (good practice)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # Collect metrics per epoch for curve plotting
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience = 3   # early stopping: stop if val_loss doesn't improve for 3 epochs
    no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss   = evaluate_loss(model, val_loader, device)
        train_acc  = evaluate_accuracy(model, train_loader, device)
        val_acc    = evaluate_accuracy(model, val_loader, device)

        scheduler.step()

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["train_acc"].append(round(train_acc, 6))
        history["val_acc"].append(round(val_acc, 6))

        print(f"  Epoch {epoch:02d}/{config.EPOCHS} | "
              f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc*100:.1f}%  Val Acc: {val_acc*100:.1f}%")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n[train] Early stopping triggered at epoch {epoch} "
                      f"(no val_loss improvement for {patience} epochs).")
                break

    print("\n[train] Training complete!")

    # ── Save model, curves, history ────────────────────────────────────────
    save_model(model, tokenizer)
    curves_dir = os.path.join(config.MODEL_SAVE_DIR, "training_curves")
    save_training_curves(history, curves_dir)


if __name__ == "__main__":
    main()
