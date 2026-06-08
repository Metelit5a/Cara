"""
Training Script for Skin Issues Type Model

Fine-tunes EfficientNetB0 on the Skin Issues v2 dataset (folder-based).
Classes: acne, blackheads, dark_spots, pores, wrinkles

Two-phase transfer learning:
1. Frozen backbone — train classification head only
2. Partial fine-tuning — unfreeze upper layers with smaller LR

Usage:
    python -m model_service.training.train_skin_issues [--data-dir "path/to/skin v2"]
"""

import os
import copy
import argparse
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np

from model_service.skin_issues_model.model import (
    build_skin_issues_model,
    SKIN_ISSUE_CLASSES,
    NUM_CLASSES,
)


# ── Transforms ──

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
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
    """Standard training loop with best-weight checkpointing."""
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
                    print(f"  ** Saved best model (acc={best_acc:.4f})")
        print()

    print(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train Skin Issues Type Model")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/Downloads/skin v2"),
                        help="Path to folder-based dataset (subfolders = classes)")
    parser.add_argument("--epochs-frozen", type=int, default=10, help="Epochs for frozen backbone")
    parser.add_argument("--epochs-finetune", type=int, default=8, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-frozen", type=float, default=0.001)
    parser.add_argument("--lr-finetune", type=float, default=0.0001)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {args.data_dir}")

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset not found at {args.data_dir}")

    # Load full dataset with training transforms
    full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transforms)

    # Map folder names to our canonical class names
    # ImageFolder sorts class dirs alphabetically: acne, blackheades, dark spots, pores, wrinkles
    print(f"\nDetected classes: {full_dataset.classes}")
    print(f"Class-to-index mapping: {full_dataset.class_to_idx}")
    print(f"Total images: {len(full_dataset)}")

    # Print class distribution
    dist = Counter(full_dataset.targets)
    for cls_idx in sorted(dist.keys()):
        print(f"  {full_dataset.classes[cls_idx]}: {dist[cls_idx]} images")

    # Split 80/20 train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create val dataset with proper val transforms
    val_dataset_proper = datasets.ImageFolder(args.data_dir, transform=val_transforms)
    val_subset = torch.utils.data.Subset(val_dataset_proper, val_dataset.indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    print(f"\nTrain: {len(train_dataset)} | Val: {len(val_subset)}")

    # Compute class weights for imbalanced data
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_labels),
        y=train_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights_tensor}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    checkpoints_dir = "model_service/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Phase 1: Frozen backbone
    print("\n" + "=" * 50)
    print("PHASE 1: Frozen Backbone Training")
    print("=" * 50)
    model = build_skin_issues_model(pretrained=True, freeze=True)
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_frozen)
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_frozen, device,
        os.path.join(checkpoints_dir, "skin_issues_model_frozen.pth"),
    )

    # Phase 2: Fine-tuning upper layers
    print("\n" + "=" * 50)
    print("PHASE 2: Partial Fine-Tuning")
    print("=" * 50)
    model.unfreeze_upper_layers()
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_finetune)
    model = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_finetune, device,
        os.path.join(checkpoints_dir, "skin_issues_model_best.pth"),
    )

    print("\nTraining complete!")
    print(f"Frozen checkpoint: {checkpoints_dir}/skin_issues_model_frozen.pth")
    print(f"Best checkpoint:   {checkpoints_dir}/skin_issues_model_best.pth")


if __name__ == "__main__":
    main()
