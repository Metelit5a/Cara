"""
Unified Training Script — Cara Skin Analysis Models

Trains EfficientNetB0 classifiers using a two-phase transfer learning approach:
  Phase 1: Freeze backbone, train classification head only (fast convergence)
  Phase 2: Unfreeze upper layers (blocks 6-8), fine-tune with 10x lower LR

Supports training from scratch or resuming from an existing checkpoint.

Usage:
    # Train from scratch
    python -m model_service.training.train --model acne
    python -m model_service.training.train --model skin_type
    python -m model_service.training.train --model skin_issues
    python -m model_service.training.train --model all

    # Resume training from existing checkpoint (Phase 2 only)
    python -m model_service.training.train --model acne --resume
    python -m model_service.training.train --model acne --resume --epochs 20

    # Customize training parameters
    python -m model_service.training.train --model acne --epochs 30 --lr 5e-5 --batch-size 64

Environment:
    Works on both CPU and GPU. Automatically detects CUDA availability.
    For GPU training, ensure PyTorch is installed with CUDA support.
"""

import argparse
import copy
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════


class ImageFolderDataset(Dataset):
    """Image classification dataset from folder structure.

    Expects data organized as:
        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg

    Each subfolder name maps to a class label via class_to_idx.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            # Corrupt image fallback — grey placeholder
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self) -> List[int]:
        """All labels in dataset order (for class weight computation)."""
        return [label for _, label in self.samples]


def collect_samples(root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    """Walk folder structure and collect (image_path, label) pairs."""
    samples = []
    for class_name, idx in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.exists():
            print(f"  WARNING: {class_dir} not found, skipping")
            continue
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in ImageFolderDataset.VALID_EXTENSIONS:
                samples.append((img_path, idx))
    return samples


class MultiLabelImageFolderDataset(Dataset):
    """Multi-label classification dataset from single-label folder structure.

    Each source folder maps to a multi-hot vector via `folder_to_vector`.
    Example (2-class model with a `negative` bucket):
        pores/       → [1, 0]   (has pores, no blackheads)
        blackheads/  → [0, 1]
        negative/    → [0, 0]

    Note the noise: an image in `blackheads/` might in reality also have
    pores, but we can't know without per-image annotations. Treating each
    single-label folder as positive-only for its class is standard when
    working with folder-labelled data.
    """

    def __init__(
        self,
        samples: List[Tuple[Path, "np.ndarray"]],
        transform=None,
    ):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label_vec = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, torch.as_tensor(label_vec, dtype=torch.float32)

    @property
    def targets(self) -> "np.ndarray":
        """Stacked multi-hot label matrix (N, num_classes)."""
        return np.stack([lbl for _, lbl in self.samples])


def collect_multi_label_samples(
    root: Path,
    class_names: Dict[int, str],
    negative_class: Optional[str],
) -> List[Tuple[Path, np.ndarray]]:
    """Walk a folder structure and produce multi-hot labels.

    - A folder matching one of `class_names` values → one-hot at that index.
    - A folder matching `negative_class` → all-zero vector (no conditions).
    """
    num_classes = len(class_names)
    name_to_idx = {name: idx for idx, name in class_names.items()}
    samples: List[Tuple[Path, np.ndarray]] = []

    # Positive folders
    for name, idx in name_to_idx.items():
        d = root / name
        if not d.exists():
            print(f"  WARNING: {d} not found, skipping")
            continue
        vec = np.zeros(num_classes, dtype=np.float32)
        vec[idx] = 1.0
        for img_path in d.iterdir():
            if img_path.suffix.lower() in ImageFolderDataset.VALID_EXTENSIONS:
                samples.append((img_path, vec.copy()))

    # Negative folder (all-zero labels)
    if negative_class:
        d = root / negative_class
        if not d.exists():
            print(f"  WARNING: negative folder {d} not found — model may be biased toward positives")
        else:
            vec = np.zeros(num_classes, dtype=np.float32)
            for img_path in d.iterdir():
                if img_path.suffix.lower() in ImageFolderDataset.VALID_EXTENSIONS:
                    samples.append((img_path, vec.copy()))

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    """Training augmentation pipeline.

    Includes geometric and photometric augmentations to improve generalization.
    These make training harder so the model learns robust features.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_eval_transform():
    """Validation/test transform — deterministic, no augmentation."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# Model Architecture
# ═══════════════════════════════════════════════════════════════════════════════


def build_model(num_classes: int, pretrained: bool = True, freeze: bool = True) -> nn.Module:
    """Build EfficientNetB0 with custom classification head.

    Architecture:
        EfficientNetB0 backbone (ImageNet pretrained)
        -> AdaptiveAvgPool -> Dropout(0.3) -> Linear(1280, num_classes)

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights for backbone.
        freeze: If True, freeze all backbone parameters (Phase 1 mode).
    """
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


def unfreeze_upper(model: nn.Module):
    """Unfreeze blocks 6, 7, 8 of EfficientNet backbone for fine-tuning.

    These are the deepest convolutional blocks and contain the most
    task-specific features. Unfreezing them allows the model to adapt
    high-level representations while keeping low-level features stable.
    """
    for name, param in model.features.named_parameters():
        block_idx = name.split(".")[0]
        if block_idx in ("6", "7", "8"):
            param.requires_grad = True


# ═══════════════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Single training epoch with progress bar."""
    model.train()
    running_loss = 0.0
    running_correct = 0
    n = 0
    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        n += inputs.size(0)
        pbar.set_postfix(loss=f"{running_loss/n:.4f}", acc=f"{running_correct/n:.3f}")
    return running_loss / n, running_correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataset (no gradients)."""
    model.eval()
    running_loss = 0.0
    running_correct = 0
    n = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_correct += (preds == labels).sum().item()
        n += inputs.size(0)
    return running_loss / n, running_correct / n


# ── Multi-label training ─────────────────────────────────────────────────────
#
# For multi-label we replace argmax with per-class sigmoid + 0.5 threshold, and
# use macro-F1 across classes as the "accuracy" metric fed to the scheduler and
# early-stop logic. F1 is picked (not subset-accuracy) because with rare
# positive classes a model that predicts all-zero would score high on subset
# accuracy while being useless.


def _multi_label_f1(logits: torch.Tensor, labels: torch.Tensor,
                    threshold: float = 0.5) -> torch.Tensor:
    """Macro-F1 across output classes. Returns a scalar tensor."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    # Per-class TP/FP/FN
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1.mean()


def train_one_epoch_multi_label(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_f1_sum = 0.0
    n_batches = 0
    n_samples = 0
    pbar = tqdm(loader, desc="  train", leave=False, dynamic_ncols=True)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_f1_sum += _multi_label_f1(outputs.detach(), labels).item()
        n_samples += inputs.size(0)
        n_batches += 1
        pbar.set_postfix(
            loss=f"{running_loss/n_samples:.4f}",
            f1=f"{running_f1_sum/n_batches:.3f}",
        )
    return running_loss / n_samples, running_f1_sum / max(n_batches, 1)


@torch.no_grad()
def evaluate_multi_label(model, loader, criterion, device):
    """Evaluate multi-label model. Returns (loss, macro_f1)."""
    model.eval()
    all_logits = []
    all_labels = []
    running_loss = 0.0
    n_samples = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        n_samples += inputs.size(0)
        all_logits.append(outputs.cpu())
        all_labels.append(labels.cpu())
    logits = torch.cat(all_logits)
    labels_all = torch.cat(all_labels)
    f1 = _multi_label_f1(logits, labels_all).item()
    return running_loss / n_samples, f1


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, device, save_path, phase_name="", patience=5,
                multi_label=False):
    """Train for N epochs with early stopping, save best by validation metric.

    For single-label uses accuracy; for multi-label uses macro-F1.

    Args:
        patience: Stop if validation metric doesn't improve for this many epochs.
        multi_label: Route through the sigmoid+BCE training helpers.
    """
    best_val_metric = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    metric_name = "f1" if multi_label else "acc"
    train_step = train_one_epoch_multi_label if multi_label else train_one_epoch
    eval_step = evaluate_multi_label if multi_label else evaluate

    for epoch in range(num_epochs):
        print(f"\n  Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_metric = train_step(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = eval_step(model, val_loader, criterion, device)

        if scheduler:
            scheduler.step(val_loss)

        print(f"    train: loss={train_loss:.4f} {metric_name}={train_metric:.3f}")
        print(f"    val:   loss={val_loss:.4f} {metric_name}={val_metric:.3f}")

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(best_wts, save_path)
            patience_counter = 0
            print(f"    ✓ New best val {metric_name}: {val_metric:.3f} → saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stopping after {patience} epochs without improvement")
                break

    print(f"\n  {phase_name} best val {metric_name}: {best_val_metric:.3f}")
    model.load_state_dict(best_wts)
    return model


@torch.no_grad()
def final_test(model, test_loader, device, class_names):
    """Final evaluation on held-out test set with per-class metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    total_acc = (all_preds == all_labels).mean()

    print(f"\n{'='*50}")
    print(f"FINAL TEST ACCURACY: {total_acc:.3f} ({total_acc*100:.1f}%)")
    print(f"{'='*50}")
    print(f"\nPer-class results:")
    for idx, name in class_names.items():
        mask = all_labels == idx
        if mask.sum() == 0:
            continue
        class_acc = (all_preds[mask] == idx).mean()
        class_count = mask.sum()
        print(f"  {name:15s}: {class_acc:.3f} ({class_acc*100:.1f}%) [{class_count} samples]")

    return total_acc


@torch.no_grad()
def final_test_multi_label(model, test_loader, device, class_names, threshold: float = 0.5):
    """Final multi-label evaluation: per-class precision, recall, F1."""
    model.eval()
    all_logits = []
    all_labels = []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        all_logits.append(outputs.cpu())
        all_labels.append(labels)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    preds = (torch.sigmoid(logits) >= threshold).float()

    print(f"\n{'='*50}")
    print(f"FINAL MULTI-LABEL TEST (threshold={threshold})")
    print(f"{'='*50}")
    for idx, name in class_names.items():
        p = preds[:, idx]
        t = labels[:, idx]
        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()
        tn = ((1 - p) * (1 - t)).sum().item()
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        n_pos = int(t.sum().item())
        print(
            f"  {name:15s}: precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  "
            f"(pos={n_pos}, tp={int(tp)}, fp={int(fp)}, fn={int(fn)}, tn={int(tn)})"
        )
    macro_f1 = _multi_label_f1(logits, labels, threshold).item()
    print(f"\n  Macro F1: {macro_f1:.3f}")
    return macro_f1


# ═══════════════════════════════════════════════════════════════════════════════
# Model Configurations
# ═══════════════════════════════════════════════════════════════════════════════


MODEL_CONFIGS = {
    "acne": {
        "data_dir": "data/acne_severity",
        "class_names": {0: "clear", 1: "mild", 2: "moderate", 3: "severe"},
        "num_classes": 4,
        "checkpoint_name": "acne_model_best.pth",
        "has_splits": False,
        "epochs_frozen": 15,
        "epochs_finetune": 12,
        "batch_size": 16,
        "lr_frozen": 1e-3,
        "lr_finetune": 1e-4,
        "multi_label": False,
    },
    "skin_type": {
        "data_dir": "data/skin_type",
        "class_names": {0: "combination", 1: "dry", 2: "normal", 3: "oily"},
        "num_classes": 4,
        "checkpoint_name": "skin_type_model_best.pth",
        "has_splits": True,
        "epochs_frozen": 12,
        "epochs_finetune": 10,
        "batch_size": 32,
        "lr_frozen": 1e-3,
        "lr_finetune": 1e-4,
        "multi_label": False,
    },
    "skin_issues": {
        # DEPRECATED: single-label 5-class model. Kept so old checkpoints still
        # train, but the app now uses `skin_conditions` (multi-label) instead.
        # See data/skin_issues/README.md for why.
        "data_dir": "data/skin_issues",
        "class_names": {0: "blackheads", 1: "dark_spots", 2: "healthy", 3: "pores", 4: "wrinkles"},
        "num_classes": 5,
        "checkpoint_name": "skin_issues_model_best.pth",
        "has_splits": False,
        "epochs_frozen": 10,
        "epochs_finetune": 8,
        "batch_size": 32,
        "lr_frozen": 1e-3,
        "lr_finetune": 1e-4,
        "multi_label": False,
    },
    "skin_conditions": {
        # Multi-label: independently detects pores and blackheads. Each output
        # is a sigmoid; both, one, or neither can fire. See
        # data/skin_conditions/README.md for the dataset design.
        "data_dir": "data/skin_conditions",
        "class_names": {0: "blackheads", 1: "pores"},  # alphabetical, matches folder walk
        "negative_class": "negative",  # folder whose labels are all-zero
        "num_classes": 2,
        "checkpoint_name": "skin_conditions_model_best.pth",
        "has_splits": False,
        "epochs_frozen": 8,
        "epochs_finetune": 6,
        "batch_size": 32,
        "lr_frozen": 1e-3,
        "lr_finetune": 1e-4,
        "multi_label": True,
    },
}


def get_model_config(model_name: str) -> dict:
    """Get configuration for the specified model."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}. Choose from: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_name].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Training Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def train_model(model_name: str, resume: bool = False, epochs: Optional[int] = None,
                lr: Optional[float] = None, batch_size: Optional[int] = None):
    """Full training pipeline for a single model.

    Args:
        model_name: One of 'acne', 'skin_type', 'skin_issues'.
        resume: If True, load existing checkpoint and run Phase 2 only.
        epochs: Override number of fine-tune epochs.
        lr: Override fine-tuning learning rate.
        batch_size: Override batch size.
    """
    config = get_model_config(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Apply CLI overrides
    if batch_size:
        config["batch_size"] = batch_size
    if epochs:
        config["epochs_finetune"] = epochs
    if lr:
        config["lr_finetune"] = lr

    print(f"\n{'='*60}")
    print(f"Training: {model_name.upper()}")
    print(f"Device: {device}")
    print(f"Mode: {'RESUME (Phase 2 only)' if resume else 'Full (Phase 1 + Phase 2)'}")
    print(f"{'='*60}")

    data_dir = Path(config["data_dir"])
    class_names = config["class_names"]
    multi_label = config.get("multi_label", False)

    # Build class_to_idx from sorted class names (alphabetical for consistency).
    # For multi-label we also honour a `negative_class` folder for all-zero labels.
    sorted_classes = sorted(class_names.values())
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    print(f"\nClasses: {idx_to_class}")
    if multi_label:
        print(f"Mode: MULTI-LABEL (BCE + sigmoid, threshold 0.5, macro-F1)")

    # ── Data Loading ─────────────────────────────────────────────
    if multi_label:
        all_samples = collect_multi_label_samples(
            data_dir, idx_to_class, config.get("negative_class")
        )
        # Stratify by argmax of label vector (or -1 for all-zero negatives)
        # so each split contains representatives of every folder.
        strat = [int(np.argmax(lbl)) if lbl.sum() > 0 else -1 for _, lbl in all_samples]
        train_samples, temp_samples, _, temp_strat = train_test_split(
            all_samples, strat, test_size=0.30, random_state=42, stratify=strat
        )
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.50, random_state=42, stratify=temp_strat
        )
    elif config["has_splits"]:
        train_samples = collect_samples(data_dir / "train", class_to_idx)
        val_samples = collect_samples(data_dir / "valid", class_to_idx)
        test_samples = collect_samples(data_dir / "test", class_to_idx)
    else:
        all_samples = collect_samples(data_dir, class_to_idx)
        labels = [s[1] for s in all_samples]
        train_samples, temp_samples = train_test_split(
            all_samples, test_size=0.30, random_state=42, stratify=labels
        )
        temp_labels = [s[1] for s in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples, test_size=0.50, random_state=42, stratify=temp_labels
        )

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val:   {len(val_samples)}")
    print(f"  Test:  {len(test_samples)}")

    if multi_label:
        # Positives per output column across the training split.
        pos_counts = np.zeros(len(idx_to_class), dtype=np.int64)
        for _, lbl in train_samples:
            pos_counts += lbl.astype(np.int64)
        neg_counts = len(train_samples) - pos_counts
        print(f"\nTrain distribution (multi-label):")
        for idx, name in idx_to_class.items():
            print(f"  {name:15s}: pos={pos_counts[idx]}  neg={neg_counts[idx]}")
    else:
        train_dist = Counter(s[1] for s in train_samples)
        print(f"\nTrain distribution:")
        for idx in sorted(train_dist.keys()):
            print(f"  {idx_to_class[idx]:15s}: {train_dist[idx]}")

    # Create datasets and loaders
    if multi_label:
        train_ds = MultiLabelImageFolderDataset(train_samples, transform=get_train_transform())
        val_ds = MultiLabelImageFolderDataset(val_samples, transform=get_eval_transform())
        test_ds = MultiLabelImageFolderDataset(test_samples, transform=get_eval_transform())
    else:
        train_ds = ImageFolderDataset(train_samples, transform=get_train_transform())
        val_ds = ImageFolderDataset(val_samples, transform=get_eval_transform())
        test_ds = ImageFolderDataset(test_samples, transform=get_eval_transform())

    bs = config["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True)

    # Loss + class balancing
    if multi_label:
        # BCE with per-class pos_weight = negatives / positives (see
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).
        # Clamp to sane range so a rare class doesn't get an absurdly large weight.
        pos_weight_vals = np.clip(neg_counts / np.maximum(pos_counts, 1), 0.1, 10.0)
        pos_weight = torch.tensor(pos_weight_vals, dtype=torch.float).to(device)
        print(f"\nBCE pos_weight per class: "
              f"{dict(zip([idx_to_class[i] for i in range(len(idx_to_class))], pos_weight_vals.round(3)))}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        train_labels = np.array(train_ds.targets)
        unique_classes = np.unique(train_labels)
        weights = compute_class_weight("balanced", classes=unique_classes, y=train_labels)
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"\nClass weights: {dict(zip([idx_to_class[i] for i in unique_classes], weights.round(3)))}")
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Checkpoint path
    ckpt_dir = Path("model_service/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = str(ckpt_dir / config["checkpoint_name"])

    # ── Resume or Train from Scratch ─────────────────────────────
    if resume:
        checkpoint_path = Path(best_path)
        if not checkpoint_path.exists():
            print(f"\n  ERROR: No checkpoint found at {best_path}")
            print(f"  Cannot resume — train from scratch first.")
            sys.exit(1)

        print(f"\n{'─'*40}")
        print(f"RESUMING from: {best_path}")
        print(f"Fine-tuning ({config['epochs_finetune']} epochs, lr={config['lr_finetune']})")
        print(f"{'─'*40}")

        model = build_model(config["num_classes"], pretrained=False, freeze=False)
        state_dict = torch.load(best_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)

        # Freeze backbone, then selectively unfreeze upper layers
        for param in model.features.parameters():
            param.requires_grad = False
        unfreeze_upper(model)

    else:
        # Phase 1: Frozen backbone
        print(f"\n{'─'*40}")
        print(f"PHASE 1: Frozen backbone ({config['epochs_frozen']} epochs)")
        print(f"{'─'*40}")

        model = build_model(config["num_classes"], pretrained=True, freeze=True)
        model = model.to(device)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        optimizer = optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=config["lr_frozen"],
            weight_decay=1e-4,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        frozen_path = str(ckpt_dir / f"{model_name}_frozen.pth")
        model = train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            config["epochs_frozen"], device, frozen_path, "Phase 1",
                            multi_label=multi_label)

        # Transition to Phase 2
        print(f"\n{'─'*40}")
        print(f"PHASE 2: Fine-tune upper layers ({config['epochs_finetune']} epochs)")
        print(f"{'─'*40}")
        unfreeze_upper(model)

    # ── Phase 2: Fine-tune ───────────────────────────────────────
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr_finetune"],
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    model = train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                        config["epochs_finetune"], device, best_path, "Phase 2",
                        multi_label=multi_label)

    # ── Final Test ───────────────────────────────────────────────
    if multi_label:
        final_test_multi_label(model, test_loader, device, idx_to_class)
    else:
        final_test(model, test_loader, device, idx_to_class)
    print(f"\n✓ Model saved to: {best_path}")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Train Cara skin analysis models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train from scratch:      python -m model_service.training.train --model acne
  Train all models:        python -m model_service.training.train --model all
  Resume from checkpoint:  python -m model_service.training.train --model acne --resume
  Resume with more epochs: python -m model_service.training.train --model acne --resume --epochs 30
  Custom learning rate:    python -m model_service.training.train --model acne --resume --lr 5e-5
  GPU with large batches:  python -m model_service.training.train --model acne --resume --batch-size 64
        """,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["acne", "skin_type", "skin_issues", "skin_conditions", "all"],
        help="Which model to train",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from existing checkpoint (skips Phase 1)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of fine-tuning epochs",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override fine-tuning learning rate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size (increase for GPU, e.g. 64 or 128)",
    )
    args = parser.parse_args()

    start = time.time()

    if args.model == "all":
        # Trains the three active models. The legacy `skin_issues` model is
        # excluded — it's superseded by `skin_conditions` (multi-label).
        for name in ["acne", "skin_type", "skin_conditions"]:
            train_model(name, resume=args.resume, epochs=args.epochs,
                        lr=args.lr, batch_size=args.batch_size)
    else:
        train_model(args.model, resume=args.resume, epochs=args.epochs,
                    lr=args.lr, batch_size=args.batch_size)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"Total training time: {elapsed/60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
