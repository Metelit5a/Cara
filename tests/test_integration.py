"""
Integration tests for the full analysis pipeline.

These tests verify that the pipeline works end-to-end:
  image → preprocessing → inference → BLP → report

Note: These tests run with whatever models are loaded (may be none
during CI when checkpoints don't exist). They verify the pipeline
logic, not model accuracy.
"""

import pytest
from PIL import Image
import io
import numpy as np

from shared.schemas import AnalysisStatus, ModelPrediction
from backend.blp.engine import BLPEngine
from backend.report_generation.builder import ReportBuilder
from model_service.inference.orchestrator import InferenceOrchestrator


class TestPipelineIntegration:
    """Test the pipeline components work together."""

    def test_image_to_tensor_transform(self):
        """Orchestrator transform produces correct tensor shape."""
        orchestrator = InferenceOrchestrator()
        img = Image.new("RGB", (500, 600), color=(200, 150, 130))
        tensor = orchestrator.transform(img)
        assert tensor.shape == (3, 224, 224)
        # Values should be normalized (not 0-255)
        assert tensor.min() < 0 or tensor.max() < 5.0

    def test_image_different_sizes_all_produce_same_tensor_shape(self):
        """Different input sizes all produce 224x224 tensors."""
        orchestrator = InferenceOrchestrator()
        for size in [(100, 100), (224, 224), (1024, 768), (50, 300)]:
            img = Image.new("RGB", size)
            tensor = orchestrator.transform(img)
            assert tensor.shape == (3, 224, 224), f"Failed for size {size}"

    def test_predict_all_returns_dict_or_empty(self):
        """predict_all returns a dict (empty if no models loaded)."""
        import torch
        orchestrator = InferenceOrchestrator()
        tensor = torch.randn(1, 3, 224, 224)
        result = orchestrator.predict_all(tensor)
        assert isinstance(result, dict)
        # Each value should be a ModelPrediction
        for name, pred in result.items():
            assert isinstance(pred, ModelPrediction)
            assert 0.0 <= pred.confidence <= 1.0
            assert len(pred.all_probabilities) > 0

    def test_blp_then_report_builder_integration(self):
        """BLP result feeds correctly into ReportBuilder."""
        engine = BLPEngine()
        predictions = {
            "acne": ModelPrediction(
                model_name="acne", predicted_class=2,
                predicted_label="moderate", confidence=0.65,
                all_probabilities=[0.1, 0.15, 0.65, 0.1]
            ),
            "skin_type": ModelPrediction(
                model_name="skin_type", predicted_class=3,
                predicted_label="oily", confidence=0.75,
                all_probabilities=[0.05, 0.1, 0.1, 0.75]
            ),
        }
        blp_result = engine.process(predictions)
        report = ReportBuilder.build_success_report(predictions, blp_result)

        assert report.status == AnalysisStatus.SUCCESS
        assert report.acne_severity.value == "moderate"
        assert report.skin_type.value == "oily"
        assert len(report.recommendations) > 0
        assert report.acne_confidence == 0.65
        assert report.skin_type_confidence == 0.75

    def test_low_confidence_report_generation(self):
        """Low confidence predictions produce correct report type."""
        predictions = {
            "acne": ModelPrediction(
                model_name="acne", predicted_class=3,
                predicted_label="severe", confidence=0.15,
                all_probabilities=[0.25, 0.3, 0.3, 0.15]
            ),
        }
        report = ReportBuilder.build_low_confidence_report(
            predictions, "Not confident enough."
        )
        assert report.status == AnalysisStatus.LOW_CONFIDENCE
        assert "Not confident" in report.message

    def test_error_report_generation(self):
        """Error reports have correct structure."""
        report = ReportBuilder.build_error_report("Something went wrong")
        assert report.status == AnalysisStatus.ERROR
        assert report.message == "Something went wrong"
        assert report.id is not None
        assert report.created_at is not None


class TestModelPredictionContract:
    """Test that ModelPrediction validates correctly."""

    def test_valid_prediction(self):
        pred = ModelPrediction(
            model_name="acne",
            predicted_class=0,
            predicted_label="clear",
            confidence=0.92,
            all_probabilities=[0.92, 0.05, 0.02, 0.01],
        )
        assert pred.confidence == 0.92
        assert pred.predicted_label == "clear"

    def test_probabilities_can_be_any_length(self):
        """Different models have different numbers of classes."""
        # 4 classes (acne, skin_type)
        pred4 = ModelPrediction(
            model_name="acne", predicted_class=0, predicted_label="clear",
            confidence=0.8, all_probabilities=[0.8, 0.1, 0.05, 0.05]
        )
        assert len(pred4.all_probabilities) == 4

        # 5 classes (skin_issues)
        pred5 = ModelPrediction(
            model_name="skin_issues", predicted_class=2, predicted_label="healthy",
            confidence=0.7, all_probabilities=[0.1, 0.05, 0.7, 0.1, 0.05]
        )
        assert len(pred5.all_probabilities) == 5
