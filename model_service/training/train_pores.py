"""
Training Script for Pore Severity Model

Converts COCO detection annotations into classification labels by
binning pore count per image, then fine-tunes EfficientNetB0.

Two-phase transfer learning (same as acne model):
1. Frozen backbone — train classification head only
2. Partial fine-tuning — unfreeze upper layers with smaller LR

Usage:
    python -m model_service.training.train_pores [--data-dir "skin pore.coco/train"]
"""

import os
import json
import copy
import shutil
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import numpy as np

from model_service.pores_model.model import build_pore_model, PORE_COUNT_THRESHOLDS, PORE_CLASSES


# ── Dataset ──

def compute_labels_from_coco(annotations_path: str) -> dict:
    """Parse COCO annotations and assign severity class based on pore count per image.

    Returns:
        dict mapping image filename -> severity class (0-3)
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Build image_id -> filename map
    id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}

    # Count annotations per image
    pore_counts = Counter(ann["image_id"] for ann in data["annotations"])

    # Assign severity class based on thresholds
    labels = {}
    for img_id, count in pore_counts.items():
        filename = id_to_filename[img_id]
        if count <= PORE_COUNT_THRESHOLDS[0]:
            labels[filename] = 0  # minimal
        elif count <= PORE_COUNT_THRESHOLDS[1]:
            labels[filename] = 1  # mild
        elif count <= PORE_COUNT_THRESHOLDS[2]:
            labels[filename] = 2  # moderate
        else:
            labels[filename] = 3  # severe

    # Images with no annotations get class 0 (minimal)
    for img in data["images"]:
        if img["file_name"] not in labels:
            labels[img["file_name"]] = 0

    return labels


class PoreDataset(Dataset):
    """PyTorch dataset for pore severity classification."""

    def __init__(self, image_dir: str, labels: dict, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        # Only include images that exist on disk
        self.samples = [
            (fn, label) for fn, label in labels.items()
            if (self.image_dir / fn).exists()
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.image_dir / filename
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    @property
    def targets(self):
        return [label for _, label in self.samples]


# ── Transforms ──

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Training loop ──

def train_model(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path):
    model = model.to(device)
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            loader = train_loader if phase == "train" else val_loader
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(loader.dataset)
            epoch_acc = running_corrects.double() / len(loader.dataset)

            print(f"  {phase.capitalize()} Loss: {epoch_loss:.4f}  Acc: {epoch_acc:.4f}")

            if phase == "val":
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
        print()

    print(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Pore Severity Model")
    parser.add_argument("--data-dir", default="data/pores_coco/train",
                        help="Path to COCO dataset directory with _annotations.coco.json")
    parser.add_argument("--epochs-frozen", type=int, default=10, help="Epochs for frozen backbone")
    parser.add_argument("--epochs-finetune", type=int, default=5, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr-frozen", type=float, default=0.001)
    parser.add_argument("--lr-finetune", type=float, default=0.0001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parse COCO annotations to get severity labels
    annotations_path = os.path.join(args.data_dir, "_annotations.coco.json")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"Annotations not found at {annotations_path}")

    print("Parsing COCO annotations...")
    labels = compute_labels_from_coco(annotations_path)
    print(f"Total images: {len(labels)}")

    # Print class distribution
    dist = Counter(labels.values())
    for cls_id in sorted(dist.keys()):
        print(f"  {PORE_CLASSES[cls_id]}: {dist[cls_id]} images")

    # Create full dataset
    full_dataset = PoreDataset(args.data_dir, labels, transform=train_transforms)
    print(f"Dataset loaded: {len(full_dataset)} valid images")

    # Split 80/20 train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(42))

    # Create val dataset with val transforms
    val_labels = {full_dataset.samples[i][0]: full_dataset.samples[i][1]
                  for i in val_dataset.indices}
    val_dataset_proper = PoreDataset(args.data_dir, val_labels, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_proper, batch_size=args.batch_size, shuffle=False)

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset_proper)}")

    # Compute class weights for imbalanced data
    train_labels_list = [full_dataset.samples[i][1] for i in train_dataset.indices]
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced",
                                         classes=np.unique(train_labels_list),
                                         y=train_labels_list)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights_tensor}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    checkpoints_dir = "model_service/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Phase 1: Frozen backbone
    print("\n" + "=" * 50)
    print("PHASE 1: Frozen Backbone Training")
    print("=" * 50)
    model = build_pore_model(pretrained=True, freeze=True)
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_frozen)
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_frozen, device, os.path.join(checkpoints_dir, "pores_model_frozen.pth"),
    )

    # Phase 2: Fine-tuning upper layers
    print("\n" + "=" * 50)
    print("PHASE 2: Partial Fine-Tuning")
    print("=" * 50)
    model.unfreeze_upper_layers()
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_finetune)
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_finetune, device, os.path.join(checkpoints_dir, "pores_model_best.pth"),
    )

    # Final evaluation on full val set
    print("\n" + "=" * 50)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 50)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, label_batch in val_loader:
            inputs, label_batch = inputs.to(device), label_batch.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += label_batch.size(0)
            correct += (predicted == label_batch).sum().item()
    print(f"Final Val Accuracy: {100 * correct / total:.2f}%")
    print(f"\nModel saved to: {checkpoints_dir}/pores_model_best.pth")


if __name__ == "__main__":
    main()
