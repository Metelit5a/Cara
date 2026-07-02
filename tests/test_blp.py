"""Tests for the BLP Engine with new 3-model architecture."""

import pytest
from backend.blp.engine import BLPEngine
from shared.schemas import ModelPrediction, AcneSeverity, SkinType, SkinIssue


@pytest.fixture
def engine():
    return BLPEngine()


def _make_prediction(model_name: str, label: str, confidence: float) -> ModelPrediction:
    """Helper to create a ModelPrediction with sensible defaults."""
    class_maps = {
        "acne": ["clear", "mild", "moderate", "severe"],
        "skin_type": ["combination", "dry", "normal", "oily"],
        "skin_issues": ["blackheads", "dark_spots", "healthy", "pores", "wrinkles"],
    }
    classes = class_maps[model_name]
    idx = classes.index(label)
    remainder = (1.0 - confidence) / (len(classes) - 1)
    probs = [confidence if i == idx else remainder for i in range(len(classes))]
    return ModelPrediction(
        model_name=model_name,
        predicted_class=idx,
        predicted_label=label,
        confidence=confidence,
        all_probabilities=probs,
    )


class TestBLPBasicProcessing:
    def test_acne_clear_no_recommendations(self, engine):
        predictions = {"acne": _make_prediction("acne", "clear", 0.8)}
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.CLEAR
        assert len(result.recommendations) == 0

    def test_acne_mild_has_recommendations(self, engine):
        predictions = {"acne": _make_prediction("acne", "mild", 0.7)}
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.MILD
        assert len(result.recommendations) > 0

    def test_acne_severe_has_strong_recommendations(self, engine):
        predictions = {"acne": _make_prediction("acne", "severe", 0.6)}
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.SEVERE
        assert len(result.recommendations) >= 3

    def test_skin_type_oily(self, engine):
        predictions = {"skin_type": _make_prediction("skin_type", "oily", 0.8)}
        result = engine.process(predictions)
        assert result.skin_type == SkinType.OILY
        assert len(result.recommendations) > 0

    def test_skin_type_dry(self, engine):
        predictions = {"skin_type": _make_prediction("skin_type", "dry", 0.7)}
        result = engine.process(predictions)
        assert result.skin_type == SkinType.DRY
        ingredients = [r.ingredient for r in result.recommendations]
        assert any("Hyaluronic" in i for i in ingredients)

    def test_skin_issue_healthy_no_recommendations(self, engine):
        predictions = {"skin_issues": _make_prediction("skin_issues", "healthy", 0.8)}
        result = engine.process(predictions)
        assert result.skin_issue == SkinIssue.HEALTHY
        assert len(result.recommendations) == 0

    def test_skin_issue_blackheads(self, engine):
        predictions = {"skin_issues": _make_prediction("skin_issues", "blackheads", 0.7)}
        result = engine.process(predictions)
        assert result.skin_issue == SkinIssue.BLACKHEADS
        assert len(result.recommendations) > 0

    def test_skin_issue_dark_spots_recommends_vitamin_c(self, engine):
        predictions = {"skin_issues": _make_prediction("skin_issues", "dark_spots", 0.6)}
        result = engine.process(predictions)
        assert result.skin_issue == SkinIssue.DARK_SPOTS
        ingredients = [r.ingredient for r in result.recommendations]
        assert any("Vitamin C" in i for i in ingredients)


class TestBLPMultiModel:
    def test_all_three_models_combined(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "mild", 0.7),
            "skin_type": _make_prediction("skin_type", "oily", 0.8),
            "skin_issues": _make_prediction("skin_issues", "blackheads", 0.6),
        }
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.MILD
        assert result.skin_type == SkinType.OILY
        assert result.skin_issue == SkinIssue.BLACKHEADS
        assert len(result.recommendations) > 3

    def test_healthy_across_all_models(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "clear", 0.9),
            "skin_type": _make_prediction("skin_type", "normal", 0.85),
            "skin_issues": _make_prediction("skin_issues", "healthy", 0.8),
        }
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.CLEAR
        assert result.skin_type == SkinType.NORMAL
        assert result.skin_issue == SkinIssue.HEALTHY

    def test_no_duplicate_ingredients(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "mild", 0.7),
            "skin_type": _make_prediction("skin_type", "oily", 0.8),
        }
        result = engine.process(predictions)
        ingredients = [r.ingredient for r in result.recommendations]
        assert len(ingredients) == len(set(ingredients))


class TestBLPConfidenceThreshold:
    def test_below_threshold_ignored(self, engine):
        predictions = {"acne": _make_prediction("acne", "severe", 0.2)}
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.CLEAR
        assert len(result.recommendations) == 0

    def test_above_threshold_used(self, engine):
        predictions = {"acne": _make_prediction("acne", "moderate", 0.35)}
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.MODERATE
        assert len(result.recommendations) > 0

    def test_mixed_confidence(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "severe", 0.15),
            "skin_type": _make_prediction("skin_type", "oily", 0.9),
        }
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.CLEAR
        assert result.skin_type == SkinType.OILY


class TestBLPAllRulesReachable:
    def test_all_acne_rules(self, engine):
        for severity in ["clear", "mild", "moderate", "severe"]:
            preds = {"acne": _make_prediction("acne", severity, 0.8)}
            result = engine.process(preds)
            assert result.acne_severity == AcneSeverity(severity)

    def test_all_skin_type_rules(self, engine):
        for stype in ["oily", "dry", "normal", "combination"]:
            preds = {"skin_type": _make_prediction("skin_type", stype, 0.8)}
            result = engine.process(preds)
            assert result.skin_type == SkinType(stype)

    def test_all_skin_issue_rules(self, engine):
        for issue in ["healthy", "blackheads", "dark_spots", "pores", "wrinkles"]:
            preds = {"skin_issues": _make_prediction("skin_issues", issue, 0.8)}
            result = engine.process(preds)
            assert result.skin_issue == SkinIssue(issue)

    def test_empty_predictions(self, engine):
        result = engine.process({})
        assert result.acne_severity == AcneSeverity.CLEAR
        assert result.explanation == "Analysis complete."
