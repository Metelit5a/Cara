"""
Inference Orchestrator

Manages model loading and prediction for all three models:
  1. Acne Severity (clear/mild/moderate/severe)
  2. Skin Type (oily/dry/normal/combination)
  3. Skin Issues (healthy/blackheads/dark_spots/pores/wrinkles)
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from shared.config import settings
from shared.schemas import ModelPrediction

logger = logging.getLogger(__name__)

# Class mappings (alphabetical order — must match training)
ACNE_CLASSES = {0: "clear", 1: "mild", 2: "moderate", 3: "severe"}
SKIN_TYPE_CLASSES = {0: "combination", 1: "dry", 2: "normal", 3: "oily"}
SKIN_ISSUE_CLASSES = {0: "blackheads", 1: "dark_spots", 2: "healthy", 3: "pores", 4: "wrinkles"}

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _build_efficientnet(num_classes: int) -> nn.Module:
    """Build EfficientNetB0 with custom classifier head (matching training)."""
    from torchvision.models import efficientnet_b0

    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_model(num_classes: int, weights_path: str, device: torch.device) -> Optional[nn.Module]:
    """Load a model from checkpoint. Returns None if weights don't exist or are incompatible."""
    path = Path(weights_path)
    if not path.exists():
        logger.warning(f"Model weights not found: {path}")
        return None

    model = _build_efficientnet(num_classes)
    try:
        state_dict = torch.load(str(path), map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning(f"Incompatible checkpoint {path.name}: {e}. Skipping.")
        return None

    model = model.to(device)
    model.eval()
    logger.info(f"Loaded model: {path.name} ({num_classes} classes)")
    return model


class InferenceOrchestrator:
    """Runs inference across all registered models."""

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._models: Dict[str, nn.Module] = {}
        self._class_maps: Dict[str, Dict[int, str]] = {}
        self.confidence_threshold = settings.confidence_threshold

        # Preprocessing transform (matches training eval transform)
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        self._load_all_models()

    def _load_all_models(self):
        """Load all available model checkpoints."""
        model_configs = [
            ("acne", 4, settings.model_weights_path, ACNE_CLASSES),
            ("skin_type", 4, settings.skin_type_model_weights_path, SKIN_TYPE_CLASSES),
            ("skin_issues", 5, settings.skin_issues_model_weights_path, SKIN_ISSUE_CLASSES),
        ]

        for name, num_classes, path, class_map in model_configs:
            model = _load_model(num_classes, path, self._device)
            if model is not None:
                self._models[name] = model
                self._class_maps[name] = class_map

        logger.info(f"Models loaded: {list(self._models.keys())}")

    @property
    def loaded_models(self) -> list:
        return list(self._models.keys())

    def predict(self, model_name: str, image_tensor: torch.Tensor) -> Optional[ModelPrediction]:
        """Run inference on a single model."""
        if model_name not in self._models:
            return None

        model = self._models[model_name]
        class_map = self._class_maps[model_name]

        with torch.no_grad():
            image_tensor = image_tensor.to(self._device)
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, 1)

            pred_idx = predicted_class.item()
            conf = confidence.item()
            all_probs = [round(p, 4) for p in probs.squeeze().cpu().tolist()]

        return ModelPrediction(
            model_name=model_name,
            predicted_class=pred_idx,
            predicted_label=class_map[pred_idx],
            confidence=round(conf, 4),
            all_probabilities=all_probs,
        )

    def predict_all(self, image_tensor: torch.Tensor) -> Dict[str, ModelPrediction]:
        """Run inference on all loaded models.

        Args:
            image_tensor: Preprocessed tensor [1, 3, 224, 224] or [3, 224, 224]

        Returns:
            Dict mapping model name -> ModelPrediction
        """
        predictions = {}
        for model_name in self._models:
            pred = self.predict(model_name, image_tensor)
            if pred is not None:
                predictions[model_name] = pred
                logger.info(
                    f"  {model_name}: {pred.predicted_label} "
                    f"(conf={pred.confidence:.3f})"
                )
        return predictions

    @property
    def transform(self):
        """The image transform to apply before prediction."""
        return self._transform


# ── Singleton ────────────────────────────────────────────────────────────────

_orchestrator: Optional[InferenceOrchestrator] = None


def get_orchestrator() -> InferenceOrchestrator:
    """Get or create the singleton orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InferenceOrchestrator()
    return _orchestrator
