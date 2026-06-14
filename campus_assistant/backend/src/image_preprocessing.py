"""
image_preprocessing.py — Image preprocessing and augmentation pipeline.

Implements the full image processing pipeline required by the assignment:
  - Resize and normalise images to CLIP-compatible format
  - Training augmentations: random crop, rotation, colour jitter
  - Tensor conversion and batch DataLoader creation
  - Demo function that prints sample outputs

Usage:
    python src/image_preprocessing.py          # runs demo with dummy images
    from src.image_preprocessing import get_train_transform, get_val_transform
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── CLIP-standard normalisation constants ────────────────────────────────────
# openai/clip-vit-base-patch32 was trained with these exact mean/std values.
# Using different values will degrade retrieval quality.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# Target spatial resolution for CLIP ViT-B/32
CLIP_IMAGE_SIZE = 224


# ── Transform pipelines ───────────────────────────────────────────────────────

def get_train_transform() -> transforms.Compose:
    """
    Training-time augmentation pipeline.

    Augmentations applied:
      1. RandomResizedCrop  — randomly crops a sub-region (80–100% of area) and
                              resizes to 224×224. Simulates different distances
                              and angles from which a campus building might be
                              photographed.
      2. RandomHorizontalFlip — mirrors 50% of images left-right.
      3. RandomRotation      — rotates ±15°. Accounts for slightly tilted phone
                               photos.
      4. ColorJitter         — randomly adjusts brightness (±30%), contrast
                               (±30%), saturation (±20%), and hue (±5%).
                               Handles day/lighting variation across photos.
      5. ToTensor            — converts PIL Image (H×W×C, uint8) to FloatTensor
                               (C×H×W, 0–1).
      6. Normalize           — subtracts CLIP mean, divides by CLIP std per
                               channel so values are in the range expected by the
                               frozen CLIP encoder.

    Returns:
        A torchvision.transforms.Compose pipeline.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(
            size=CLIP_IMAGE_SIZE,
            scale=(0.80, 1.0),          # crop between 80% and 100% of image area
            ratio=(0.75, 1.33),         # allow slight aspect ratio variation
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_val_transform() -> transforms.Compose:
    """
    Validation/inference-time transform pipeline (no random augmentation).

    Steps:
      1. Resize shortest side to 224 (BICUBIC interpolation, matching CLIP).
      2. CenterCrop to 224×224.
      3. ToTensor + Normalize with CLIP constants.

    This is identical to the transform applied inside the CLIP processor,
    so we can use it for evaluation without double-processing.
    """
    return transforms.Compose([
        transforms.Resize(CLIP_IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(CLIP_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Reverse the CLIP normalisation so a tensor can be displayed as an image.

    Args:
        tensor: Normalised image tensor of shape (C, H, W).

    Returns:
        Float tensor in [0, 1] range, shape (C, H, W).
    """
    mean = torch.tensor(CLIP_MEAN, dtype=tensor.dtype).view(3, 1, 1)
    std  = torch.tensor(CLIP_STD,  dtype=tensor.dtype).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)


# ── Campus Image Dataset ──────────────────────────────────────────────────────

class CampusImageDataset(Dataset):
    """
    PyTorch Dataset that loads campus location images from disk.

    Expected directory structure:
        data/images/
            main_library/       ← one subdirectory per location
                img_001.jpg
                img_002.jpg
            cafeteria/
                img_001.jpg
            ...

    The class label for each image is derived from its parent directory name,
    which is matched against the campus knowledge base to assign an integer ID.

    Args:
        images_root  : Path to the top-level images directory.
        kb_path      : Path to campus_kb.json (used to build label→id mapping).
        transform    : Optional transform to apply to each image.
        split        : 'train', 'val', or 'test' — currently used only for
                       logging; split is done at the DataLoader level.
    """

    def __init__(
        self,
        images_root: str,
        kb_path: str,
        transform: Optional[transforms.Compose] = None,
        split: str = "train",
    ):
        self.images_root = Path(images_root)
        self.transform   = transform or get_val_transform()
        self.split       = split

        # Build label → integer index from KB location names
        with open(kb_path) as f:
            kb = json.load(f)
        location_names = [r["name"].lower().replace(" ", "_")
                          for r in kb.get("locations", [])]
        self.label2id = {name: idx for idx, name in enumerate(location_names)}
        self.id2label = {idx: name for name, idx in self.label2id.items()}

        # Discover all image files
        self.samples: list[tuple[Path, int]] = []
        supported_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if self.images_root.exists():
            for class_dir in sorted(self.images_root.iterdir()):
                if not class_dir.is_dir():
                    continue
                label_key = class_dir.name.lower()
                label_id  = self.label2id.get(label_key, -1)
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in supported_exts:
                        self.samples.append((img_path, label_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image)
        return {
            "image":  tensor,
            "label":  torch.tensor(label_id, dtype=torch.long),
            "path":   str(img_path),
        }


def build_dataloaders(
    images_root: str,
    kb_path: str,
    batch_size: int = 16,
    train_fraction: float = 0.8,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from a campus images directory.

    Applies training augmentations to the train split and clean val transforms
    to the validation split. The split is performed by index (no shuffling
    before split, so the same images are always in train/val across runs).

    Args:
        images_root    : Root directory containing per-location subdirectories.
        kb_path        : Path to campus_kb.json.
        batch_size     : Batch size for both loaders.
        train_fraction : Fraction of samples used for training (default 0.8).
        num_workers    : DataLoader worker processes (0 = main process only).

    Returns:
        (train_loader, val_loader) DataLoader pair.
    """
    # Discover samples once — reuse the list for both splits via a lightweight wrapper
    full_dataset = CampusImageDataset(images_root, kb_path, transform=get_val_transform())

    n_total = len(full_dataset)
    n_train = int(n_total * train_fraction)

    if n_total == 0:
        raise ValueError(
            f"No images found in {images_root}. "
            "Add campus photos under data/images/<location_name>/ "
            "to use the image training pipeline."
        )

    train_indices = list(range(n_train))
    val_indices   = list(range(n_train, n_total))

    # Share the pre-discovered sample list; only the transform differs.
    # _SplitView wraps the same samples with a different transform so we
    # scan the filesystem only once.
    class _SplitView(Dataset):
        def __init__(self, samples, transform):
            self.samples   = samples
            self.transform = transform
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            img_path, label_id = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            return {
                "image": self.transform(image),
                "label": torch.tensor(label_id, dtype=torch.long),
                "path":  str(img_path),
            }

    train_samples = [full_dataset.samples[i] for i in train_indices]
    val_samples   = [full_dataset.samples[i] for i in val_indices]

    from torch.utils.data import Subset
    train_subset = _SplitView(train_samples, get_train_transform())
    val_subset   = _SplitView(val_samples,   get_val_transform())

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


# ── Demo ──────────────────────────────────────────────────────────────────────

def _make_dummy_image(width: int = 300, height: int = 400) -> Image.Image:
    """Generate a synthetic RGB image for demo purposes."""
    rng   = np.random.default_rng(seed=42)
    arr   = (rng.random((height, width, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(arr)
    draw  = ImageDraw.Draw(image)
    draw.rectangle([20, 20, 200, 100], fill=(80, 120, 200))
    draw.ellipse([150, 200, 260, 300], fill=(220, 80, 60))
    return image


def demo():
    """
    Demonstrate the preprocessing pipeline with a synthetic image.

    Output shows:
      - Input image size
      - Output tensor shape after each transform
      - Min/max/mean of the normalised tensor
      - Verification that denormalize reverses the operation
    """
    print("=" * 60)
    print("Image Preprocessing Pipeline — Demo")
    print("=" * 60)

    dummy = _make_dummy_image()
    print(f"\nInput image  : size={dummy.size}, mode={dummy.mode}")

    val_tf   = get_val_transform()
    train_tf = get_train_transform()

    val_tensor   = val_tf(dummy)
    train_tensor = train_tf(dummy)

    print(f"\n[Validation transform]")
    print(f"  Output tensor shape : {val_tensor.shape}  (C x H x W)")
    print(f"  dtype               : {val_tensor.dtype}")
    print(f"  min / max / mean    : {val_tensor.min():.4f} / {val_tensor.max():.4f} / {val_tensor.mean():.4f}")

    print(f"\n[Training transform (with augmentation)]")
    print(f"  Output tensor shape : {train_tensor.shape}")
    print(f"  min / max / mean    : {train_tensor.min():.4f} / {train_tensor.max():.4f} / {train_tensor.mean():.4f}")

    # Verify denormalise roundtrip
    restored = denormalize(val_tensor)
    print(f"\n[Denormalized tensor (for display)]")
    print(f"  min / max           : {restored.min():.4f} / {restored.max():.4f}  (should be 0–1)")

    print("\n[Augmentation steps applied during training]")
    print("  1. RandomResizedCrop(224, scale=0.80–1.0)  — distance / framing variation")
    print("  2. RandomHorizontalFlip(p=0.5)             — mirror symmetry")
    print("  3. RandomRotation(±15°)                    — tilted phone photos")
    print("  4. ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)")
    print("                                             — lighting / weather variation")
    print("  5. ToTensor()                              — PIL → FloatTensor [0,1]")
    print("  6. Normalize(CLIP mean, CLIP std)          — CLIP-standard normalisation")
    print()

    # Simulate a batch
    batch = torch.stack([val_tf(dummy) for _ in range(4)])  # batch of 4
    print(f"[Batch simulation]")
    print(f"  Batch tensor shape  : {batch.shape}  (N x C x H x W)")
    print(f"  Memory (float32)    : {batch.numel() * 4 / 1024:.1f} KB")
    print()
    print("Demo complete. In a real dataset, point CampusImageDataset at")
    print("  data/images/<location_name>/  and call build_dataloaders().")


if __name__ == "__main__":
    demo()
