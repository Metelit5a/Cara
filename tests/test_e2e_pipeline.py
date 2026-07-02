"""
End-to-End Pipeline Tests — Full image-to-report validation.

These tests verify the COMPLETE pipeline:
  Image → Preprocessing → All 3 Models → BLP Engine → Report

They use known ground truth images and verify the final report
contains correct predictions. This catches regressions anywhere
in the pipeline — model loading, preprocessing, inference,
BLP logic, or report generation.

Run these before pushing to catch silent failures in the pipeline.
"""

import pytest
from pathlib import Path

from PIL import Image

from backend.blp.engine import BLPEngine
from backend.report_generation.builder import ReportBuilder
from shared.schemas import AnalysisStatus, AcneSeverity, SkinIssue

CHECKPOINT_DIR = Path("model_service/checkpoints")
DATA_DIR = Path("data")

# Only run if all models are available
requires_all_models = pytest.mark.skipif(
    not all((
        (CHECKPOINT_DIR / "acne_model_best.pth").exists(),
        (CHECKPOINT_DIR / "skin_type_model_best.pth").exists(),
        (CHECKPOINT_DIR / "skin_issues_model_best.pth").exists(),
    )),
    reason="Not all model checkpoints available",
)


def _run_full_pipeline(image_path: Path) -> dict:
    """Run the complete pipeline on a single image, return report + predictions."""
    from model_service.inference.orchestrator import InferenceOrchestrator

    orchestrator = InferenceOrchestrator()
    engine = BLPEngine()

    # Load and preprocess
    img = Image.open(image_path).convert("RGB")
    tensor = orchestrator.transform(img)

    # Inference
    predictions = orchestrator.predict_all(tensor)

    # BLP processing
    blp_result = engine.process(predictions)

    # Report generation
    report = ReportBuilder.build_success_report(predictions, blp_result)

    return {
        "predictions": predictions,
        "blp_result": blp_result,
        "report": report,
    }


@requires_all_models
class TestEndToEndPipeline:
    """Full pipeline tests with ground truth images."""

    def test_clear_skin_produces_healthy_report(self):
        """A clear skin image should produce a report with no acne and healthy skin."""
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Clear images not available")

        # Get a clear skin image (these are ground truth healthy faces)
        images = [p for p in clear_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            pytest.skip("No images found")

        result = _run_full_pipeline(images[0])
        report = result["report"]

        # Pipeline should produce a valid report
        assert report.status == AnalysisStatus.SUCCESS
        assert report.id is not None
        assert report.created_at is not None

        # Clear skin should not have severe acne
        assert report.acne_severity != AcneSeverity.SEVERE

    def test_blackheads_image_detects_skin_issue(self):
        """A blackheads image should be detected as blackheads by skin_issues model."""
        blackheads_dir = DATA_DIR / "skin_issues" / "blackheads"
        if not blackheads_dir.exists():
            pytest.skip("Blackheads images not available")

        images = [p for p in blackheads_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            pytest.skip("No images found")

        result = _run_full_pipeline(images[0])
        report = result["report"]

        assert report.status == AnalysisStatus.SUCCESS
        # The skin issues model should detect blackheads
        assert report.skin_issue == SkinIssue.BLACKHEADS
        # And provide relevant recommendations
        assert len(report.recommendations) > 0

    def test_wrinkles_image_gets_anti_aging_recommendations(self):
        """A wrinkles image should get retinol/peptide recommendations."""
        wrinkles_dir = DATA_DIR / "skin_issues" / "wrinkles"
        if not wrinkles_dir.exists():
            pytest.skip("Wrinkles images not available")

        images = [p for p in wrinkles_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            pytest.skip("No images found")

        result = _run_full_pipeline(images[0])
        report = result["report"]

        assert report.status == AnalysisStatus.SUCCESS
        assert report.skin_issue == SkinIssue.WRINKLES
        # Should recommend retinol or peptides for wrinkles
        ingredients = [r.ingredient for r in report.recommendations]
        has_anti_aging = any(
            "Retinol" in i or "Peptide" in i for i in ingredients
        )
        assert has_anti_aging, f"Expected anti-aging ingredient, got: {ingredients}"

    def test_pipeline_handles_different_image_sizes(self):
        """Pipeline should handle various image dimensions without crashing."""
        from model_service.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        engine = BLPEngine()

        for size in [(50, 50), (224, 224), (1000, 750), (100, 400)]:
            img = Image.new("RGB", size, color=(180, 150, 130))
            tensor = orchestrator.transform(img)
            predictions = orchestrator.predict_all(tensor)
            blp_result = engine.process(predictions)
            assert blp_result is not None

    def test_pipeline_handles_grayscale_image(self):
        """Pipeline should handle grayscale images (converts to RGB)."""
        from model_service.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        img = Image.new("L", (300, 300), color=128)  # Grayscale
        img_rgb = img.convert("RGB")
        tensor = orchestrator.transform(img_rgb)
        predictions = orchestrator.predict_all(tensor)
        assert isinstance(predictions, dict)

    def test_report_has_all_required_fields(self):
        """End-to-end report should have all fields populated."""
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Test images not available")

        images = [p for p in clear_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if not images:
            pytest.skip("No images found")

        result = _run_full_pipeline(images[0])
        report = result["report"]

        # Required fields
        assert report.id is not None
        assert report.status is not None
        assert report.created_at is not None
        assert report.acne_severity is not None
        # Confidence scores
        assert report.acne_confidence is not None
        assert 0.0 <= report.acne_confidence <= 1.0

    def test_multiple_images_produce_consistent_format(self):
        """Different images should all produce valid reports with same structure."""
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Test images not available")

        images = [p for p in clear_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}][:3]
        if len(images) < 2:
            pytest.skip("Need at least 2 images")

        for img_path in images:
            result = _run_full_pipeline(img_path)
            report = result["report"]
            assert report.status == AnalysisStatus.SUCCESS
            assert isinstance(report.recommendations, list)


@requires_all_models
class TestPipelineRobustness:
    """Test pipeline robustness under edge cases."""

    def test_very_dark_image_doesnt_crash(self):
        """Near-black image should still produce a valid result."""
        from model_service.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        engine = BLPEngine()

        img = Image.new("RGB", (300, 300), color=(5, 5, 5))
        tensor = orchestrator.transform(img)
        predictions = orchestrator.predict_all(tensor)
        result = engine.process(predictions)
        assert result is not None

    def test_very_bright_image_doesnt_crash(self):
        """Near-white image should still produce a valid result."""
        from model_service.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        engine = BLPEngine()

        img = Image.new("RGB", (300, 300), color=(250, 250, 250))
        tensor = orchestrator.transform(img)
        predictions = orchestrator.predict_all(tensor)
        result = engine.process(predictions)
        assert result is not None

    def test_all_models_loaded_and_functional(self):
        """Verify all 3 models are loaded and producing predictions."""
        from model_service.inference.orchestrator import InferenceOrchestrator

        orchestrator = InferenceOrchestrator()
        assert "acne" in orchestrator.loaded_models
        assert "skin_type" in orchestrator.loaded_models
        assert "skin_issues" in orchestrator.loaded_models

        img = Image.new("RGB", (300, 300), color=(180, 150, 130))
        tensor = orchestrator.transform(img)
        predictions = orchestrator.predict_all(tensor)

        assert "acne" in predictions
        assert "skin_type" in predictions
        assert "skin_issues" in predictions
