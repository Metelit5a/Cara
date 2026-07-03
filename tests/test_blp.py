"""Tests for the BLP Engine with the multi-label skin_conditions model."""

import pytest
from backend.blp.engine import BLPEngine
from shared.schemas import (
    ModelPrediction,
    MultiLabelPrediction,
    SkinConditionFinding,
    AcneSeverity,
    SkinType,
    SkinCondition,
)


@pytest.fixture
def engine():
    return BLPEngine()


def _make_prediction(model_name: str, label: str, confidence: float) -> ModelPrediction:
    """Helper to create a single-label ModelPrediction."""
    class_maps = {
        "acne": ["clear", "mild", "moderate", "severe"],
        "skin_type": ["combination", "dry", "normal", "oily"],
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


def _make_multi_label(*findings) -> MultiLabelPrediction:
    """Build a MultiLabelPrediction from (label, confidence) tuples.

    Examples:
        _make_multi_label()                                     # empty (clean)
        _make_multi_label(("pores", 0.9))                       # one finding
        _make_multi_label(("pores", 0.8), ("blackheads", 0.7))  # both
    """
    all_scores = {"blackheads": 0.0, "pores": 0.0}
    findings_list = []
    for label, conf in findings:
        all_scores[label] = conf
        findings_list.append(
            SkinConditionFinding(label=SkinCondition(label), confidence=conf)
        )
    return MultiLabelPrediction(
        model_name="skin_conditions",
        findings=findings_list,
        all_scores=all_scores,
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


class TestBLPSkinConditionsMultiLabel:
    """New multi-label semantics — 0, 1, or 2 findings."""

    def test_no_findings_yields_no_condition_recommendations(self, engine):
        """Empty findings list = 'no notable conditions', report says nothing."""
        predictions = {"skin_conditions": _make_multi_label()}
        result = engine.process(predictions)
        assert result.skin_conditions == []
        assert len(result.recommendations) == 0

    def test_pores_only(self, engine):
        predictions = {"skin_conditions": _make_multi_label(("pores", 0.9))}
        result = engine.process(predictions)
        assert result.skin_conditions == [SkinCondition.PORES]
        assert any("Niacinamide" in r.ingredient for r in result.recommendations)

    def test_blackheads_only(self, engine):
        predictions = {"skin_conditions": _make_multi_label(("blackheads", 0.85))}
        result = engine.process(predictions)
        assert result.skin_conditions == [SkinCondition.BLACKHEADS]
        assert any("Salicylic" in r.ingredient for r in result.recommendations)

    def test_both_pores_and_blackheads(self, engine):
        """The whole point of multi-label: report BOTH when both are present."""
        predictions = {
            "skin_conditions": _make_multi_label(("pores", 0.8), ("blackheads", 0.75))
        }
        result = engine.process(predictions)
        assert set(result.skin_conditions) == {SkinCondition.PORES, SkinCondition.BLACKHEADS}
        ingredients = [r.ingredient for r in result.recommendations]
        assert len(ingredients) == len(set(ingredients))
        assert len(ingredients) > 0


class TestBLPMultiModel:
    def test_all_three_models_combined(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "mild", 0.7),
            "skin_type": _make_prediction("skin_type", "oily", 0.8),
            "skin_conditions": _make_multi_label(("blackheads", 0.75)),
        }
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.MILD
        assert result.skin_type == SkinType.OILY
        assert result.skin_conditions == [SkinCondition.BLACKHEADS]
        assert len(result.recommendations) > 3

    def test_clean_skin_across_all_models(self, engine):
        """Clear acne + normal skin + no conditions → minimal report."""
        predictions = {
            "acne": _make_prediction("acne", "clear", 0.9),
            "skin_type": _make_prediction("skin_type", "normal", 0.85),
            "skin_conditions": _make_multi_label(),
        }
        result = engine.process(predictions)
        assert result.acne_severity == AcneSeverity.CLEAR
        assert result.skin_type == SkinType.NORMAL
        assert result.skin_conditions == []

    def test_no_duplicate_ingredients(self, engine):
        predictions = {
            "acne": _make_prediction("acne", "mild", 0.7),
            "skin_type": _make_prediction("skin_type", "oily", 0.8),
            "skin_conditions": _make_multi_label(("pores", 0.9)),
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

    def test_all_skin_condition_rules(self, engine):
        """Both multi-label classes should produce non-empty recommendations."""
        for cond in ["pores", "blackheads"]:
            preds = {"skin_conditions": _make_multi_label((cond, 0.8))}
            result = engine.process(preds)
            assert SkinCondition(cond) in result.skin_conditions
            assert len(result.recommendations) > 0

    def test_empty_predictions(self, engine):
        result = engine.process({})
        assert result.acne_severity == AcneSeverity.CLEAR
        assert result.explanation == "Analysis complete."
