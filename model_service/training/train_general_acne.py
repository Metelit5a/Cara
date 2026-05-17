"""Training script for the General Acne severity model.

Uses the Roboflow COCO export at `data/Acne new data.v1i.coco/` which
provides pre-split train/valid/test folders, each containing
`_annotations.coco.json` and a flat folder of jpg images.

Severity labels are derived from per-image lesion counts.

Usage:
    python -m model_service.training.train_general_acne
"""

import argparse
import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from model_service.acne_model.general_model import (
    GENERAL_ACNE_CLASSES,
    build_general_acne_model,
    lesion_count_to_class,
)
from model_service.training.datasets import (
    FaceCropImageDataset,
    build_eval_transform,
    build_train_augment,
)


def build_samples(split_dir: Path) -> List[Tuple[Path, int]]:
    """Read COCO annotations and produce (image_path, severity_class) samples."""
    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, "r") as f:
        coco = json.load(f)
    counts = Counter(a["image_id"] for a in coco["annotations"])
    samples: List[Tuple[Path, int]] = []
    for img in coco["images"]:
        img_path = split_dir / img["file_name"]
        if not img_path.exists():
            continue
        label = lesion_count_to_class(counts.get(img["id"], 0))
        samples.append((img_path, label))
    return samples


def train_phase(model, train_loader, val_loader, criterion, optimizer,
                num_epochs, device, save_path):
    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}", flush=True)
        for phase, loader in [("train", train_loader), ("val", val_loader)]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_correct = 0
            n = 0
            pbar = tqdm(loader, desc=f"  {phase}", leave=False, dynamic_ncols=True)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_correct += (preds == labels).sum().item()
                n += inputs.size(0)
                pbar.set_postfix(loss=f"{running_loss/n:.4f}",
                                 acc=f"{running_correct/n:.4f}")
            pbar.close()
            epoch_loss = running_loss / n
            epoch_acc = running_correct / n
            print(f"  {phase:5s} loss={epoch_loss:.4f}  acc={epoch_acc:.4f}", flush=True)
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(best_wts, save_path)
    print(f"\nBest val acc: {best_acc:.4f}  ->  {save_path}")
    model.load_state_dict(best_wts)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Acne new data.v1i.coco")
    parser.add_argument("--epochs-frozen", type=int, default=8)
    parser.add_argument("--epochs-finetune", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-frozen", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    train_samples = build_samples(data_dir / "train")
    val_samples = build_samples(data_dir / "valid")
    test_samples = build_samples(data_dir / "test")

    def dist(samples):
        c = Counter(l for _, l in samples)
        return {GENERAL_ACNE_CLASSES[k]: c[k] for k in sorted(c)}

    print(f"Train: {len(train_samples)}  {dist(train_samples)}")
    print(f"Val:   {len(val_samples)}  {dist(val_samples)}")
    print(f"Test:  {len(test_samples)}  {dist(test_samples)}")

    train_ds = FaceCropImageDataset(train_samples, transform=build_train_augment())
    val_ds = FaceCropImageDataset(val_samples, transform=build_eval_transform())
    test_ds = FaceCropImageDataset(test_samples, transform=build_eval_transform())

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=persistent)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True,
                             persistent_workers=persistent)

    # Class weights for imbalance
    labels = np.array(train_ds.targets)
    weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print(f"Class weights: {weights_tensor.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    ckpt_dir = Path("model_service/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== PHASE 1: frozen backbone ===")
    model = build_general_acne_model(pretrained=True, freeze=True)
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_frozen)
    model = train_phase(model, train_loader, val_loader, criterion, optimizer,
                        args.epochs_frozen, device,
                        str(ckpt_dir / "general_acne_model_frozen.pth"))

    print("\n=== PHASE 2: fine-tune upper layers ===")
    model.unfreeze_upper_layers()
    optimizer = optim.Adam(model.get_trainable_params(), lr=args.lr_finetune)
    model = train_phase(model, train_loader, val_loader, criterion, optimizer,
                        args.epochs_finetune, device,
                        str(ckpt_dir / "general_acne_model_best.pth"))

    print("\n=== TEST ===")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Test accuracy: {100 * correct / total:.2f}%")


if __name__ == "__main__":
    main()
