"""
Inference Orchestrator

Manages model loading and prediction for all registered models.
Supports acne severity (local PyTorch) and pores severity (local PyTorch).
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional

from shared.config import settings
from shared.schemas import ModelPrediction, AnalysisStatus
from model_service.acne_model.model import AcneSeverityModel, ACNE_CLASSES, NUM_CLASSES
from model_service.acne_model.general_model import (
    GeneralAcneModel,
    GENERAL_ACNE_CLASSES,
    NUM_CLASSES as GENERAL_ACNE_NUM_CLASSES,
)
from model_service.pores_model.model import PoreSeverityModel, PORE_CLASSES
from model_service.pores_model.model import NUM_CLASSES as PORE_NUM_CLASSES


class ModelRegistry:
    """Registry of available model services.

    Each model is loaded once and cached. New models can be registered
    without modifying existing code.
    """

    def __init__(self):
        self._models: Dict[str, torch.nn.Module] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self) -> torch.device:
        return self._device

    def register_acne_model(self, weights_path: Optional[str] = None) -> bool:
        """Load and register the acne severity model."""
        model = AcneSeverityModel(num_classes=NUM_CLASSES, pretrained=False)

        path = weights_path or settings.model_weights_path
        if Path(path).exists():
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            model.load_state_dict(state_dict)

        model = model.to(self._device)
        model.eval()
        self._models["acne"] = model
        return True

    def register_pores_model(self, weights_path: Optional[str] = None) -> bool:
        """Load and register the pores severity model."""
        model = PoreSeverityModel(num_classes=PORE_NUM_CLASSES, pretrained=False)

        path = weights_path or settings.pores_model_weights_path
        if Path(path).exists():
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            model.load_state_dict(state_dict)

        model = model.to(self._device)
        model.eval()
        self._models["pores"] = model
        return True

    def register_general_acne_model(self, weights_path: Optional[str] = None) -> bool:
        """Load and register the robust general-acne severity model."""
        model = GeneralAcneModel(num_classes=GENERAL_ACNE_NUM_CLASSES, pretrained=False)

        path = weights_path or settings.general_acne_model_weights_path
        if Path(path).exists():
            state_dict = torch.load(path, map_location=self._device, weights_only=True)
            model.load_state_dict(state_dict)

        model = model.to(self._device)
        model.eval()
        self._models["general_acne"] = model
        return True

    def is_loaded(self, model_name: str) -> bool:
        return model_name in self._models

    def get_model(self, model_name: str) -> Optional[torch.nn.Module]:
        return self._models.get(model_name)

    @property
    def loaded_models(self) -> list:
        return list(self._models.keys())


class InferenceOrchestrator:
    """Runs inference across registered models and applies confidence thresholding."""

    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.confidence_threshold = settings.confidence_threshold

    def predict_acne(self, image_tensor: torch.Tensor) -> ModelPrediction:
        """Run acne severity prediction on a preprocessed image tensor."""
        model = self.registry.get_model("acne")
        if model is None:
            raise RuntimeError("Acne model not loaded")

        device = self.registry.device
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        conf = confidence.item()

        return ModelPrediction(
            model_name="acne_severity",
            predicted_class=predicted_class,
            predicted_label=ACNE_CLASSES[predicted_class],
            confidence=round(conf, 4),
            all_probabilities=[round(p, 4) for p in probabilities[0].tolist()],
        )

    def predict_pores(self, image_tensor: torch.Tensor) -> ModelPrediction:
        """Run pore severity prediction on a preprocessed image tensor."""
        model = self.registry.get_model("pores")
        if model is None:
            raise RuntimeError("Pores model not loaded")

        device = self.registry.device
        input_tensor = image_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        conf = confidence.item()

        return ModelPrediction(
            model_name="pores_severity",
            predicted_class=predicted_class,
            predicted_label=PORE_CLASSES[predicted_class],
            confidence=round(conf, 4),
            all_probabilities=[round(p, 4) for p in probabilities[0].tolist()],
        )

    def predict_general_acne(self, image_tensor: torch.Tensor) -> ModelPrediction:
        """Run general-acne severity prediction on a preprocessed image tensor."""
        model = self.registry.get_model("general_acne")
        if model is None:
            raise RuntimeError("General acne model not loaded")

        device = self.registry.device
        input_tensor = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = predicted.item()
        return ModelPrediction(
            model_name="general_acne_severity",
            predicted_class=predicted_class,
            predicted_label=GENERAL_ACNE_CLASSES[predicted_class],
            confidence=round(confidence.item(), 4),
            all_probabilities=[round(p, 4) for p in probabilities[0].tolist()],
        )

    def predict_all(self, image_tensor: torch.Tensor) -> Dict[str, ModelPrediction]:
        """Run all registered models and return predictions keyed by model name.

        All models use the same preprocessed tensor.
        """
        results = {}

        if self.registry.is_loaded("acne"):
            results["acne"] = self.predict_acne(image_tensor)

        if self.registry.is_loaded("pores"):
            results["pores"] = self.predict_pores(image_tensor)

        if self.registry.is_loaded("general_acne"):
            results["general_acne"] = self.predict_general_acne(image_tensor)

        return results


# Module-level singletons
_registry: Optional[ModelRegistry] = None
_orchestrator: Optional[InferenceOrchestrator] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def get_orchestrator() -> InferenceOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = InferenceOrchestrator(get_registry())
    return _orchestrator
