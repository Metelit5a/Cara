"""
Training Script for Acne Severity Model

Two-phase transfer learning:
1. Frozen backbone — train classification head only
2. Partial fine-tuning — unfreeze upper layers with smaller LR

Usage:
    python -m model_service.training.train_acne [--data-dir data/acne04_processed]
"""

import os
import copy
import shutil
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np

from model_service.acne_model.model import build_model


# ── Dataset Download & Preparation ──

def download_and_prepare_dataset(output_dir: str = "data/acne04_processed") -> str:
    """Download ACNE04 from Kaggle via kagglehub and organize into train/test folders."""
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        print(f"Dataset already prepared at {output_dir}, skipping download.")
        return output_dir

    print("Downloading ACNE04 dataset from Kaggle...")
    import kagglehub
    raw_path = kagglehub.dataset_download("manuelhettich/acne04")
    print(f"Downloaded to: {raw_path}")

    # Find image directories — ACNE04 typically has severity-level folders
    raw = Path(raw_path)
    samples = []

    # Recursively look for folders matching acne severity patterns
    # Dataset structure: acne_1024/acne0_1024, acne1_1024, acne2_1024, acne3_1024
    severity_map = {}

    for folder in sorted(raw.rglob("*")):
        if not folder.is_dir():
            continue
        name = folder.name.lower()
        # Match patterns like "acne0_1024", "acne1_1024", "level_0", "clear", etc.
        if "acne0" in name or "level_0" in name or name == "clear":
            severity_map[folder] = "clear"
        elif "acne1" in name or "level_1" in name or name == "mild":
            severity_map[folder] = "mild"
        elif "acne2" in name or "level_2" in name or name == "moderate":
            severity_map[folder] = "moderate"
        elif "acne3" in name or "level_3" in name or name == "severe":
            severity_map[folder] = "severe"

    # Remove parent folders if a child also matched (keep most specific)
    folders_to_remove = set()
    for f1 in severity_map:
        for f2 in severity_map:
            if f1 != f2 and str(f2).startswith(str(f1)):
                folders_to_remove.add(f1)
    for f in folders_to_remove:
        del severity_map[f]

    for folder, severity_name in severity_map.items():
        for img_file in folder.iterdir():
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                samples.append({"path": img_file, "severity": severity_name})

    if not samples:
        raise RuntimeError(
            f"No images found. Searched folders: {list(severity_map.keys())}. "
            f"Dataset root: {raw}"
        )

    print(f"Found {len(samples)} images across {len(severity_map)} classes")

    # Split 80/20 stratified
    from sklearn.model_selection import train_test_split
    labels = [s["severity"] for s in samples]
    train_samples, test_samples = train_test_split(
        samples, test_size=0.2, stratify=labels, random_state=42
    )

    # Copy into folder structure
    for split_name, split_samples in [("train", train_samples), ("test", test_samples)]:
        for sev in ["clear", "mild", "moderate", "severe"]:
            (output_path / split_name / sev).mkdir(parents=True, exist_ok=True)
        for sample in split_samples:
            dst = output_path / split_name / sample["severity"] / sample["path"].name
            shutil.copy2(sample["path"], dst)

    print(f"Dataset prepared: {len(train_samples)} train, {len(test_samples)} test")
    dist = {s: labels.count(s) for s in ["clear", "mild", "moderate", "severe"]}
    print(f"Distribution: {dist}")
    return output_dir


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
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

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

            if phase == "train":
                history["train_loss"].append(epoch_loss)
                history["train_acc"].append(epoch_acc.item())
            else:
                history["val_loss"].append(epoch_loss)
                history["val_acc"].append(epoch_acc.item())

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), save_path)
        print()

    print(f"Best Val Acc: {best_acc:.4f}")
    model.load_state_dict(best_wts)
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Train Acne Severity Model")
    parser.add_argument("--data-dir", default="data/acne04_processed", help="Processed dataset path")
    parser.add_argument("--epochs-frozen", type=int, default=10, help="Epochs for frozen backbone")
    parser.add_argument("--epochs-finetune", type=int, default=5, help="Epochs for fine-tuning")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-frozen", type=float, default=0.001)
    parser.add_argument("--lr-finetune", type=float, default=0.0001)
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    args = parser.parse_args()

    # Download and prepare dataset if not already done
    if not args.skip_download:
        download_and_prepare_dataset(args.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    full_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transforms)

    # 80/20 train/val split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply val transforms to validation subset
    val_dataset.dataset = copy.deepcopy(full_dataset)
    val_dataset.dataset.transform = val_transforms

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Classes: {full_dataset.classes}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    # Compute class weights for imbalanced data
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights_tensor}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    checkpoints_dir = "model_service/checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Phase 1: Frozen backbone
    print("\n" + "=" * 50)
    print("PHASE 1: Frozen Backbone Training")
    print("=" * 50)
    model = build_model(pretrained=True, freeze=True)
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_frozen)
    model, hist1 = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_frozen, device, os.path.join(checkpoints_dir, "acne_model_frozen.pth"),
    )

    # Phase 2: Fine-tuning upper layers
    print("\n" + "=" * 50)
    print("PHASE 2: Partial Fine-Tuning")
    print("=" * 50)
    model.unfreeze_upper_layers()
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_finetune)
    model, hist2 = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        args.epochs_finetune, device, os.path.join(checkpoints_dir, "acne_model_best.pth"),
    )

    # Final test evaluation
    print("\n" + "=" * 50)
    print("FINAL TEST EVALUATION")
    print("=" * 50)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
