"""
Model Algorithm Tests — Verify each model produces correct predictions.

These tests load the actual trained checkpoints and verify they predict
correctly on known ground truth images from the training data.

IMPORTANT: These tests require:
  1. Model checkpoints in model_service/checkpoints/
  2. Dataset images in data/ directories

If checkpoints are missing, tests are skipped (not failed).
Run these before pushing to main to catch algorithm regressions.
"""

import pytest
from pathlib import Path

import torch
from PIL import Image

# Skip all tests in this module if checkpoints don't exist
CHECKPOINT_DIR = Path("model_service/checkpoints")
ACNE_CHECKPOINT = CHECKPOINT_DIR / "acne_model_best.pth"
SKIN_TYPE_CHECKPOINT = CHECKPOINT_DIR / "skin_type_model_best.pth"
SKIN_ISSUES_CHECKPOINT = CHECKPOINT_DIR / "skin_issues_model_best.pth"

DATA_DIR = Path("data")

requires_acne_model = pytest.mark.skipif(
    not ACNE_CHECKPOINT.exists(), reason="Acne model checkpoint not found"
)
requires_skin_type_model = pytest.mark.skipif(
    not SKIN_TYPE_CHECKPOINT.exists(), reason="Skin type model checkpoint not found"
)
requires_skin_issues_model = pytest.mark.skipif(
    not SKIN_ISSUES_CHECKPOINT.exists(), reason="Skin issues model checkpoint not found"
)


def _get_orchestrator():
    """Get a fresh orchestrator instance."""
    from model_service.inference.orchestrator import InferenceOrchestrator
    return InferenceOrchestrator()


def _load_and_predict(orchestrator, image_path: Path) -> dict:
    """Load an image and run prediction through the orchestrator."""
    img = Image.open(image_path).convert("RGB")
    tensor = orchestrator.transform(img)
    return orchestrator.predict_all(tensor)


def _get_sample_images(folder: Path, count: int = 5) -> list:
    """Get up to `count` sample images from a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in extensions]
    return images[:count]


# ═══════════════════════════════════════════════════════════════════════════════
# Acne Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


@requires_acne_model
class TestAcneModelAlgorithm:
    """Verify the acne model predicts correctly on known images."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.orchestrator = _get_orchestrator()
        assert "acne" in self.orchestrator.loaded_models

    def test_clear_skin_predicted_correctly(self):
        """Images from the 'clear' class should predict as clear."""
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Clear acne images not found")

        images = _get_sample_images(clear_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "acne" in results and results["acne"].predicted_label == "clear":
                correct += 1

        # At least 70% of clear images should be predicted as clear
        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.7, (
            f"Clear skin accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_severe_acne_predicted_correctly(self):
        """Images from 'severe' class should predict as severe."""
        severe_dir = DATA_DIR / "acne_severity" / "severe"
        if not severe_dir.exists():
            pytest.skip("Severe acne images not found")

        images = _get_sample_images(severe_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "acne" in results and results["acne"].predicted_label == "severe":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.6, (
            f"Severe acne accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_acne_output_has_valid_labels(self):
        """Acne predictions should always be one of the 4 valid labels."""
        valid_labels = {"clear", "mild", "moderate", "severe"}
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Acne images not found")

        images = _get_sample_images(clear_dir, count=3)
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "acne" in results:
                assert results["acne"].predicted_label in valid_labels
                assert 0.0 <= results["acne"].confidence <= 1.0
                assert len(results["acne"].all_probabilities) == 4

    def test_acne_probabilities_sum_to_one(self):
        """Model output probabilities should sum to approximately 1.0."""
        clear_dir = DATA_DIR / "acne_severity" / "clear"
        if not clear_dir.exists():
            pytest.skip("Acne images not found")

        images = _get_sample_images(clear_dir, count=3)
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "acne" in results:
                prob_sum = sum(results["acne"].all_probabilities)
                assert abs(prob_sum - 1.0) < 0.01, f"Probabilities sum to {prob_sum}"


# ═══════════════════════════════════════════════════════════════════════════════
# Skin Type Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


@requires_skin_type_model
class TestSkinTypeModelAlgorithm:
    """Verify the skin type model predicts correctly on known images."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.orchestrator = _get_orchestrator()
        assert "skin_type" in self.orchestrator.loaded_models

    def test_oily_skin_predicted_correctly(self):
        """Images from 'oily' class should predict as oily."""
        oily_dir = DATA_DIR / "skin_type" / "test" / "oily"
        if not oily_dir.exists():
            pytest.skip("Oily skin test images not found")

        images = _get_sample_images(oily_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_type" in results and results["skin_type"].predicted_label == "oily":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.6, (
            f"Oily skin accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_dry_skin_predicted_correctly(self):
        """Images from 'dry' class should predict as dry."""
        dry_dir = DATA_DIR / "skin_type" / "test" / "dry"
        if not dry_dir.exists():
            pytest.skip("Dry skin test images not found")

        images = _get_sample_images(dry_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_type" in results and results["skin_type"].predicted_label == "dry":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.6, (
            f"Dry skin accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_skin_type_output_has_valid_labels(self):
        """Skin type predictions should always be one of 4 valid labels."""
        valid_labels = {"oily", "dry", "normal", "combination"}
        test_dir = DATA_DIR / "skin_type" / "test" / "oily"
        if not test_dir.exists():
            pytest.skip("Skin type images not found")

        images = _get_sample_images(test_dir, count=3)
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_type" in results:
                assert results["skin_type"].predicted_label in valid_labels
                assert len(results["skin_type"].all_probabilities) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# Skin Issues Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


@requires_skin_issues_model
class TestSkinIssuesModelAlgorithm:
    """Verify the skin issues model predicts correctly on known images."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.orchestrator = _get_orchestrator()
        assert "skin_issues" in self.orchestrator.loaded_models

    def test_blackheads_predicted_correctly(self):
        """Images from 'blackheads' class should predict as blackheads."""
        blackheads_dir = DATA_DIR / "skin_issues" / "blackheads"
        if not blackheads_dir.exists():
            pytest.skip("Blackheads images not found")

        images = _get_sample_images(blackheads_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_issues" in results and results["skin_issues"].predicted_label == "blackheads":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.85, (
            f"Blackheads accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_wrinkles_predicted_correctly(self):
        """Images from 'wrinkles' class should predict as wrinkles."""
        wrinkles_dir = DATA_DIR / "skin_issues" / "wrinkles"
        if not wrinkles_dir.exists():
            pytest.skip("Wrinkles images not found")

        images = _get_sample_images(wrinkles_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_issues" in results and results["skin_issues"].predicted_label == "wrinkles":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.85, (
            f"Wrinkles accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_pores_predicted_correctly(self):
        """Images from 'pores' class should predict as pores."""
        pores_dir = DATA_DIR / "skin_issues" / "pores"
        if not pores_dir.exists():
            pytest.skip("Pores images not found")

        images = _get_sample_images(pores_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_issues" in results and results["skin_issues"].predicted_label == "pores":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.90, (
            f"Pores accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_healthy_skin_predicted_correctly(self):
        """Images from 'healthy' class should predict as healthy."""
        healthy_dir = DATA_DIR / "skin_issues" / "healthy"
        if not healthy_dir.exists():
            pytest.skip("Healthy skin images not found")

        images = _get_sample_images(healthy_dir, count=10)
        correct = 0
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_issues" in results and results["skin_issues"].predicted_label == "healthy":
                correct += 1

        accuracy = correct / len(images) if images else 0
        assert accuracy >= 0.85, (
            f"Healthy skin accuracy too low: {accuracy:.0%} ({correct}/{len(images)})"
        )

    def test_skin_issues_output_has_valid_labels(self):
        """Skin issues predictions should always be one of 5 valid labels."""
        valid_labels = {"healthy", "blackheads", "dark_spots", "pores", "wrinkles"}
        blackheads_dir = DATA_DIR / "skin_issues" / "blackheads"
        if not blackheads_dir.exists():
            pytest.skip("Skin issues images not found")

        images = _get_sample_images(blackheads_dir, count=3)
        for img_path in images:
            results = _load_and_predict(self.orchestrator, img_path)
            if "skin_issues" in results:
                assert results["skin_issues"].predicted_label in valid_labels
                assert len(results["skin_issues"].all_probabilities) == 5
