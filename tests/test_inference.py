"""Tests for model inference."""

import torch
import pytest


class TestAcneModel:
    """Test acne severity model structure."""

    def test_model_builds(self):
        """Model should build without errors."""
        from model_service.acne_model.model import build_model

        model = build_model(pretrained=False, freeze=False)
        assert model is not None

    def test_model_forward_pass(self):
        """Model should produce 4-class output for correct input shape."""
        from model_service.acne_model.model import build_model

        model = build_model(pretrained=False, freeze=False)
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        assert output.shape == (1, 4)

    def test_model_freeze(self):
        """Frozen backbone should have no gradients on feature layers."""
        from model_service.acne_model.model import build_model

        model = build_model(pretrained=False, freeze=True)
        for param in model.backbone.features.parameters():
            assert param.requires_grad is False
        # Classifier should still be trainable
        for param in model.backbone.classifier.parameters():
            assert param.requires_grad is True

    def test_model_unfreeze_upper(self):
        """Unfreezing upper layers should enable gradients for blocks 6-8."""
        from model_service.acne_model.model import build_model

        model = build_model(pretrained=False, freeze=True)
        model.unfreeze_upper_layers()

        # Check that some upper-layer params now require grad
        upper_params = [
            p for n, p in model.backbone.features.named_parameters()
            if any(f"{i}" in n.split(".")[0] for i in [6, 7, 8])
        ]
        assert any(p.requires_grad for p in upper_params)

    def test_softmax_output_sums_to_one(self):
        """Softmax of model output should sum to ~1."""
        import torch.nn.functional as F
        from model_service.acne_model.model import build_model

        model = build_model(pretrained=False, freeze=False)
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
            probs = F.softmax(output, dim=1)

        assert abs(probs.sum().item() - 1.0) < 1e-5


class TestInferenceOrchestrator:
    """Test inference orchestrator."""

    def test_predict_returns_model_prediction(self):
        """Orchestrator should return a valid ModelPrediction."""
        from model_service.inference.orchestrator import ModelRegistry, InferenceOrchestrator

        registry = ModelRegistry()
        # Register model without real weights
        from model_service.acne_model.model import AcneSeverityModel
        model = AcneSeverityModel(pretrained=False)
        model.eval()
        registry._models["acne"] = model.to(registry.device)

        orchestrator = InferenceOrchestrator(registry)
        dummy_tensor = torch.randn(3, 224, 224)
        prediction = orchestrator.predict_acne(dummy_tensor)

        assert prediction.model_name == "acne_severity"
        assert prediction.predicted_class in [0, 1, 2, 3]
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.all_probabilities) == 4
