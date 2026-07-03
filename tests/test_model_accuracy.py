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
# The legacy `skin_issues_model_best.pth` is no longer loaded by the app.

DATA_DIR = Path("data")

requires_acne_model = pytest.mark.skipif(
    not ACNE_CHECKPOINT.exists(), reason="Acne model checkpoint not found"
)
requires_skin_type_model = pytest.mark.skipif(
    not SKIN_TYPE_CHECKPOINT.exists(), reason="Skin type model checkpoint not found"
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
# Skin Conditions Model Tests (multi-label)
# ═══════════════════════════════════════════════════════════════════════════════
#
# The legacy single-label `TestSkinIssuesModelAlgorithm` was removed with the
# switch to multi-label. Those tests validated the deprecated `skin_issues`
# model which is no longer loaded by the orchestrator (see
# `data/skin_issues/README.md`). The multi-label successor is exercised by:
#   - Training-data smoke tests below (this file)
#   - Real-world regression: `tests/test_real_world_regression.py`


SKIN_CONDITIONS_CHECKPOINT = CHECKPOINT_DIR / "skin_conditions_model_best.pth"
requires_skin_conditions_model = pytest.mark.skipif(
    not SKIN_CONDITIONS_CHECKPOINT.exists(),
    reason="Skin conditions checkpoint not found (train it first)",
)


@requires_skin_conditions_model
class TestSkinConditionsModelAlgorithm:
    """Sanity checks for the multi-label skin_conditions model on training data.

    These test that positive examples trigger their expected finding, and
    that clean-skin examples produce empty findings. Uses training data so
    this is a memorisation smoke test, NOT proof of generalisation \u2014 that
    lives in `test_real_world_regression.py`.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        from shared.schemas import MultiLabelPrediction

        self.orchestrator = _get_orchestrator()
        assert "skin_conditions" in self.orchestrator.loaded_models
        self._MLP = MultiLabelPrediction

    def _labels_for(self, image_path: Path):
        results = _load_and_predict(self.orchestrator, image_path)
        pred = results.get("skin_conditions")
        assert isinstance(pred, self._MLP)
        return {f.label.value for f in pred.findings}

    def test_pores_positive_examples(self):
        pores_dir = DATA_DIR / "skin_conditions" / "pores"
        if not pores_dir.exists():
            pytest.skip("Pores training images not found")

        images = _get_sample_images(pores_dir, count=10)
        correct = sum(1 for p in images if "pores" in self._labels_for(p))
        rate = correct / len(images) if images else 0
        assert rate >= 0.70, (
            f"Pores recall on training data too low: {rate:.0%} ({correct}/{len(images)})"
        )

    def test_blackheads_positive_examples(self):
        d = DATA_DIR / "skin_conditions" / "blackheads"
        if not d.exists():
            pytest.skip("Blackheads training images not found")

        images = _get_sample_images(d, count=10)
        correct = sum(1 for p in images if "blackheads" in self._labels_for(p))
        rate = correct / len(images) if images else 0
        assert rate >= 0.70, (
            f"Blackheads recall on training data too low: {rate:.0%} ({correct}/{len(images)})"
        )

    def test_negative_examples_mostly_empty(self):
        """Clean-skin (`negative/`) images should mostly produce empty findings."""
        d = DATA_DIR / "skin_conditions" / "negative"
        if not d.exists():
            pytest.skip("Negative (clean skin) training images not found")

        images = _get_sample_images(d, count=10)
        empty = sum(1 for p in images if len(self._labels_for(p)) == 0)
        rate = empty / len(images) if images else 0
        assert rate >= 0.50, (
            f"Model over-flags negative examples: only {empty}/{len(images)} "
            f"({rate:.0%}) came back empty. Threshold may be too low."
        )
