# Model Training Guide

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Train a single model from scratch
python -m model_service.training.train --model acne

# Resume training from checkpoint (fine-tune further)
python -m model_service.training.train --model acne --resume --epochs 30

# Train all models
python -m model_service.training.train --model all
```

## Training Modes

### From Scratch (default)
Runs the full two-phase pipeline:
1. **Phase 1** — Freeze backbone, train classifier head (fast convergence)
2. **Phase 2** — Unfreeze upper layers (blocks 6-8), fine-tune with 10x lower LR

### Resume (`--resume`)
Loads an existing checkpoint and runs **Phase 2 only** with fresh optimizer state. This is useful for:
- Continuing training with more epochs on a GPU
- Fine-tuning further with a lower learning rate
- Trying different hyperparameters without starting over

```bash
# Resume with defaults
python -m model_service.training.train --model acne --resume

# Resume with custom settings (recommended for GPU)
python -m model_service.training.train --model acne --resume --epochs 30 --lr 5e-5 --batch-size 64
```

## CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--model` | Which model: `acne`, `skin_type`, `skin_issues`, `all` | Required |
| `--resume` | Skip Phase 1, load checkpoint, run Phase 2 only | `False` |
| `--epochs` | Override fine-tuning epochs | Model-specific |
| `--lr` | Override fine-tuning learning rate | `1e-4` |
| `--batch-size` | Override batch size (increase for GPU) | Model-specific |

## Current Model Results

| Model | Test Accuracy | Per-Class Best | Per-Class Worst | Status |
|-------|:------------:|:--------------:|:---------------:|--------|
| Acne Severity | 69.6% | clear: 85% | mild: 55% | Needs more data/training |
| Skin Type | 77.8% | combination: 87% | dry: 76% | Good |
| Skin Issues | 98.3% | pores: 100% | dark_spots: 96% | Excellent |

## Recommendations for Retraining

### Priority 1: Acne Model (most benefit from GPU retraining)

The acne model has the smallest dataset (1,378 images) and lowest accuracy.
On a 3050 Ti GPU:

```bash
# Recommended: resume with more epochs and slightly lower LR
python -m model_service.training.train --model acne --resume --epochs 30 --lr 5e-5 --batch-size 64
```

**Expected improvement**: 69% → 75-80% with more training time on GPU.

### Priority 2: Skin Type (optional)

Already at 77.8%. Could benefit from longer training:

```bash
python -m model_service.training.train --model skin_type --resume --epochs 20 --lr 5e-5 --batch-size 64
```

**Expected improvement**: 78% → 80-83%.

### Priority 3: Skin Issues (not needed)

Already at 98.3%. Retraining is unlikely to help — the model is at ceiling.

## Dataset Structure

```
data/
├── acne_severity/          # ACNE04 dataset (1,378 images)
│   ├── acne0_1024/         # Clear skin
│   ├── acne1_1024/         # Mild acne
│   ├── acne2_1024/         # Moderate acne
│   └── acne3_1024/         # Severe acne
│
├── skin_type/              # Facial Skin Analysis dataset (4,093 images)
│   ├── train/
│   │   ├── combination/
│   │   ├── dry/
│   │   ├── normal/
│   │   └── oily/
│   ├── valid/
│   └── test/
│
└── skin_issues/            # Skin Issues v2 + healthy (8,193 images)
    ├── blackheads/
    ├── dark_spots/
    ├── healthy/            # From ACNE04 clear class
    ├── pores/
    └── wrinkles/
```

## Architecture

All models use the same architecture:
- **Backbone**: EfficientNetB0 (pretrained on ImageNet)
- **Head**: Dropout(0.3) → Linear(1280, num_classes)
- **Training**: Two-phase transfer learning with early stopping (patience=5)
- **Loss**: CrossEntropyLoss with class weights (handles imbalance)
- **Optimizer**: Adam with weight decay 1e-4
- **Scheduler**: ReduceLROnPlateau (patience=3, factor=0.5)

## GPU Tips

When training on a GPU (e.g., 3050 Ti with 4GB VRAM):
- Increase batch size: `--batch-size 64` or `--batch-size 128`
- Larger batches give more stable gradients and faster convergence
- EfficientNetB0 is small enough to fit batch_size=128 on 4GB VRAM
- Training that took 16 hours on CPU will take ~30-60 minutes on GPU

## Transferring Checkpoints Between Machines

Checkpoints are portable. To retrain on a different machine:

1. Copy the entire `model_service/checkpoints/` folder and `data/` folder
2. Install dependencies: `pip install -r requirements.txt`
3. Run with `--resume`:
   ```bash
   python -m model_service.training.train --model acne --resume --epochs 30
   ```
4. Copy the new `.pth` file back to the deployment machine

The checkpoint files are ~16MB each (EfficientNetB0 is lightweight).
