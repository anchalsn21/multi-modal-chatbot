"""
efficientnet_classifier.py — EfficientNet-B0 alternative vision model (Section 4, Model Design).

This module provides the third model-design choice described in the assignment:
  EfficientNet-B0 with a frozen backbone used as a feature extractor, adapting
  only the final classification head to predict the campus building/service category.

Why EfficientNet-B0 (not B4 or higher):
  - B0 is the smallest variant: 5.3M parameters vs 19M for B4.
  - Freezing all layers except the final classifier avoids overfitting on a small
    campus dataset (~1152 images across 13 categories = ~89 images/class).
  - Treated as feature extraction rather than full fine-tuning.

Relationship to the primary CLIP+FAISS pipeline:
  - CLIP+FAISS is the primary (first choice) vision model — zero labelled training
    data required, retrieval-based, most robust.
  - EfficientNet-B0 is the third choice — requires labelled images but produces a
    classification probability distribution over fixed campus categories.
  - Both can be used in parallel; their outputs compared in evaluation.

Usage:
    from src.efficientnet_classifier import (
        build_efficientnet, train_efficientnet, predict_category
    )

    model, class_names = build_efficientnet(num_classes=13)
    # training:
    train_efficientnet(model, train_loader, val_loader, class_names)
    # inference:
    label, confidence = predict_category(model, pil_image, class_names)
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EFFICIENTNET_MODEL = "efficientnet_b0"   # torchvision model name
FEATURE_DIM        = 1280                # EfficientNet-B0 penultimate feature size
DROPOUT_P          = 0.3
BATCH_SIZE         = 32
EPOCHS             = 20
LR                 = 1e-3                # head-only LR; backbone is frozen
WEIGHT_DECAY       = 1e-4
SEED               = 42

SAVE_DIR = Path(config.BASE_DIR) / "models" / "efficientnet"


# ── Model construction ────────────────────────────────────────────────────────

def build_efficientnet(
    num_classes: int,
    dropout: float = DROPOUT_P,
    pretrained: bool = True,
) -> nn.Module:
    """
    Load EfficientNet-B0 with frozen backbone and a new classification head.

    Architecture after modification:
        EfficientNet-B0 backbone (frozen)  →  AdaptiveAvgPool2d  →  (B, 1280)
        Dropout(0.3)
        Linear(1280 → num_classes)

    Freezing strategy:
        All parameters in features (convolutional backbone) are frozen.
        Only the new classifier head (Linear layer) is trained.
        This is "feature extraction" mode: the backbone acts as a fixed
        feature extractor, only the head learns campus-specific patterns.

    Args:
        num_classes : Number of campus categories (e.g. 13 image folders).
        dropout     : Dropout probability before classifier head.
        pretrained  : Load ImageNet-pretrained weights (default True).

    Returns:
        nn.Module — modified EfficientNet-B0 ready for training.
    """
    try:
        import torchvision.models as models
    except ImportError:
        raise ImportError(
            "torchvision is required for EfficientNet. "
            "Install with:  pip install torchvision"
        )

    weights = "IMAGENET1K_V1" if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # ── Freeze backbone ──────────────────────────────────────────────────────
    for param in model.features.parameters():
        param.requires_grad = False

    # ── Replace classifier head ──────────────────────────────────────────────
    # Original head: Sequential(Dropout(0.2), Linear(1280, 1000))
    # New head:      Sequential(Dropout(0.3), Linear(1280, num_classes))
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(FEATURE_DIM, num_classes),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logger.info(
        "[EfficientNet] Built: %d trainable / %d total params (%.1f%% frozen)",
        trainable, total, 100 * (1 - trainable / total),
    )
    return model


# ── Training ──────────────────────────────────────────────────────────────────

def train_efficientnet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    epochs: int = EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    save_dir: Path = SAVE_DIR,
    patience: int = 5,
) -> dict:
    """
    Train the EfficientNet-B0 classifier head on campus images.

    Only the classifier head parameters are passed to the optimiser —
    the frozen backbone receives no gradient updates.

    Training techniques (as required by the assignment):
        - Early stopping (patience=5)
        - CosineAnnealingLR learning rate schedule
        - Weight decay (L2 regularisation via AdamW)
        - Loss/accuracy curve saving

    Args:
        model       : EfficientNet built by build_efficientnet().
        train_loader: DataLoader for training split.
        val_loader  : DataLoader for validation split.
        class_names : List of category strings (same order as label indices).
        epochs      : Maximum training epochs.
        lr          : Learning rate for the classifier head.
        weight_decay: L2 weight decay.
        save_dir    : Directory to save model weights and training curves.
        patience    : Early stopping patience (epochs without val_loss improvement).

    Returns:
        dict with keys: train_loss, val_loss, train_acc, val_acc (lists per epoch)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("[EfficientNet] Training on %s", device)
    model.to(device)

    save_dir.mkdir(parents=True, exist_ok=True)

    # Only optimise the unfrozen classifier head
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer   = AdamW(head_params, lr=lr, weight_decay=weight_decay)
    scheduler   = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion   = nn.CrossEntropyLoss()

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
    }
    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            optimizer.step()
            train_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total   += images.size(0)

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                val_loss    += loss.item() * images.size(0)
                preds        = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total   += images.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss   = val_loss   / max(val_total,   1)
        train_acc      = train_correct / max(train_total, 1)
        val_acc        = val_correct   / max(val_total,   1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        logger.info(
            "[EfficientNet] Epoch %2d/%d  "
            "train_loss=%.4f  val_loss=%.4f  "
            "train_acc=%.3f  val_acc=%.3f",
            epoch, epochs, avg_train_loss, avg_val_loss, train_acc, val_acc,
        )

        # ── Early stopping ─────────────────────────────────────────────
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            no_improve    = 0
            torch.save(model.state_dict(), save_dir / "efficientnet_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info("[EfficientNet] Early stopping at epoch %d", epoch)
                break

    # ── Save outputs ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), save_dir / "efficientnet_final.pt")

    with open(save_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    _save_efficientnet_curves(history, save_dir)
    logger.info("[EfficientNet] Training complete. Saved to %s", save_dir)
    return history


def _save_efficientnet_curves(history: dict, save_dir: Path) -> None:
    """Save loss and accuracy training curves as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#4f63d8")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#e76f51")
    axes[0].set_title("EfficientNet-B0 — Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], label="Train", color="#4f63d8")
    axes[1].plot(epochs, history["val_acc"],   label="Val",   color="#e76f51")
    axes[1].set_title("EfficientNet-B0 — Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("EfficientNet-B0 Campus Classifier — Training Curves",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_dir / "efficientnet_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ── Inference ─────────────────────────────────────────────────────────────────

def load_efficientnet(
    num_classes: int,
    checkpoint_path: Optional[Path] = None,
) -> nn.Module:
    """
    Load a saved EfficientNet-B0 classifier from disk.

    Args:
        num_classes     : Must match the value used during training.
        checkpoint_path : Path to .pt file. Defaults to models/efficientnet/efficientnet_best.pt.

    Returns:
        Loaded model in eval() mode.
    """
    if checkpoint_path is None:
        checkpoint_path = SAVE_DIR / "efficientnet_best.pt"
    model = build_efficientnet(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
    model.eval()
    return model


def predict_category(
    model: nn.Module,
    image: Image.Image,
    class_names: list[str],
) -> tuple[str, float]:
    """
    Predict the campus category for a single PIL image.

    Args:
        model       : Loaded EfficientNet-B0 (output of load_efficientnet).
        image       : PIL Image (any size/mode — preprocessed internally).
        class_names : List of category strings (output of build_dataloaders or loaded from JSON).

    Returns:
        (predicted_category, confidence) — e.g. ("library", 0.87)
    """
    from src.image_preprocessing import get_val_transform

    transform = get_val_transform()
    tensor    = transform(image.convert("RGB")).unsqueeze(0)   # (1, 3, 224, 224)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                            # (1, n_classes)
        probs  = torch.softmax(logits, dim=1)[0]         # (n_classes,)
        top_idx = int(probs.argmax().item())
        confidence = float(probs[top_idx].item())

    label = class_names[top_idx] if top_idx < len(class_names) else "unknown"
    return label, confidence


def build_dataloaders_efficientnet(
    images_root: str,
    train_fraction: float = 0.8,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Build train/val DataLoaders and return class names for EfficientNet training.

    Reuses CampusImageDataset from image_preprocessing to ensure consistent
    transforms and label mapping.

    Args:
        images_root    : Path to data/images/ directory.
        train_fraction : Fraction of data used for training (rest for validation).
        batch_size     : Batch size for DataLoaders.

    Returns:
        (train_loader, val_loader, class_names)
    """
    from src.image_preprocessing import build_dataloaders

    train_loader, val_loader = build_dataloaders(
        images_root     = images_root,
        batch_size      = batch_size,
        train_fraction  = train_fraction,
    )
    # Reconstruct class names from directory names (sorted, same order as label ids)
    class_names = sorted(
        d.name for d in Path(images_root).iterdir() if d.is_dir()
    )
    return train_loader, val_loader, class_names


# ── Demo / CLI ────────────────────────────────────────────────────────────────

def demo() -> None:
    """
    Print model architecture summary and compare with CLIP+FAISS.

    Does NOT train the model (avoids downloading ImageNet weights in demo mode).
    """
    print("=" * 65)
    print("Model Design — Vision Model Comparison")
    print("=" * 65)

    print("\n--- Choice 1 (Primary): CLIP + FAISS  ---")
    print("  Model    : openai/clip-vit-base-patch32 (HuggingFace)")
    print("  Params   : ~151M (frozen, not trained)")
    print("  Strategy : Zero-shot retrieval — encode KB text descriptions,")
    print("             find nearest image embedding via cosine similarity.")
    print("  Pros     : No labelled training data needed. Fast on CPU.")
    print("             Robust to small datasets. Generalises to unseen angles.")
    print("  Cons     : No campus-specific fine-tuning by default.")

    print("\n--- Choice 2 (Optional): CLIP fine-tuned with contrastive loss ---")
    print("  Model    : openai/clip-vit-base-patch32 (partial fine-tune)")
    print("  Strategy : Pull campus-specific image embeddings closer to")
    print("             corresponding text descriptions via InfoNCE loss.")
    print("  Requires : >= 5-10 photos per location (met: ~89/class here).")
    print("  Pros     : Adapts CLIP to campus-specific visual patterns.")
    print("  Cons     : Needs paired (image, text) training data. Slower.")

    print("\n--- Choice 3 (Alternative): EfficientNet-B0 ---")
    try:
        import torchvision.models as models
        model = build_efficientnet(num_classes=13, pretrained=False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        print(f"  Model    : EfficientNet-B0 (torchvision)")
        print(f"  Params   : {total:,} total  |  {trainable:,} trainable  "
              f"({100*trainable/total:.1f}% — head only)")
        print(f"  Frozen   : {total - trainable:,} backbone params (frozen)")
        print( "  Strategy : Feature extraction — backbone frozen, only")
        print( "             Linear(1280 -> n_classes) head trained.")
        print( "  Pros     : Lightweight, fast, well-suited to small datasets.")
        print( "  Cons     : Closed-vocabulary (fixed category set).")
        print( "             Needs labelled image dataset per category.")
        print()
        print("  Architecture:")
        print(f"    EfficientNet-B0 features (frozen)  -> AdaptiveAvgPool -> (B, 1280)")
        print(f"    Dropout({DROPOUT_P}) -> Linear(1280, 13) -> logits -> softmax")
    except ImportError:
        print("  [torchvision not installed — run: pip install torchvision]")

    print()
    print("Trade-off summary:")
    print("  CLIP+FAISS  : best generalisability, zero training data needed.")
    print("  CLIP FT     : best accuracy with sufficient paired data.")
    print("  EfficientNet: best closed-vocabulary accuracy, needs labelled data.")


if __name__ == "__main__":
    demo()
