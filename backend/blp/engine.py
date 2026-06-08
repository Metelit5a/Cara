"""
Business Logic Processing Engine

Rule-based engine that interprets model outputs and generates
explainable skincare recommendations.

Rules are loaded from a JSON configuration file, making the system
data-driven and easily configurable without code changes.

Supports multi-model aggregation:
  acne model + pores model → combined BLP output
"""

import json
from pathlib import Path
from typing import Dict, Optional

from shared.schemas import (
    AcneSeverity,
    PoreSeverity,
    BLPResult,
    ModelPrediction,
    Recommendation,
)


class BLPEngine:
    """Rule-based business logic processing engine."""

    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            rules_path = str(Path(__file__).parent / "rules.json")
        with open(rules_path, "r") as f:
            config = json.load(f)
        self._acne_rules = config["acne_rules"]
        self._pore_rules = config["pore_rules"]
        self._thresholds = config["confidence_thresholds"]
        self._low_confidence_msg = config["low_confidence_message"]
        self._disagreement_msg = config.get(
            "disagreement_message",
            "Our models show mixed signals \u2014 consider retaking the photo in better lighting.",
        )

    def _resolve_acne_severity(self, predictions: Dict[str, ModelPrediction]) -> Optional[ModelPrediction]:
        """Resolve final acne severity from acne + general_acne models.

        Strategy: use the prediction with higher confidence. If only one
        model is available, use that one. This gives the BLP a consensus
        mechanism across the two acne classifiers.
        """
        acne_pred = predictions.get("acne")
        general_pred = predictions.get("general_acne")

        if acne_pred and general_pred:
            # Use whichever model is more confident
            return acne_pred if acne_pred.confidence >= general_pred.confidence else general_pred
        return acne_pred or general_pred

    def _detect_disagreement(self, predictions: Dict[str, ModelPrediction]) -> bool:
        """Detect a meaningful conflict between the two acne models.

        Returns True when both acne classifiers produced a prediction and
        their severities differ by two or more levels (e.g. clear vs severe).
        Adjacent disagreements (clear vs mild) are treated as normal noise.
        """
        acne_pred = predictions.get("acne")
        general_pred = predictions.get("general_acne")
        if not (acne_pred and general_pred):
            return False

        order = [s.value for s in AcneSeverity]
        try:
            acne_idx = order.index(acne_pred.predicted_label)
            general_idx = order.index(general_pred.predicted_label)
        except ValueError:
            return False
        return abs(acne_idx - general_idx) >= 2

    def process(self, predictions: Dict[str, ModelPrediction]) -> BLPResult:
        """Process multiple model predictions into a combined BLP result.

        Args:
            predictions: dict keyed by model name ("acne", "pores", "general_acne")
        """
        recommendations = []
        explanations = []
        educational_notes = []

        acne_severity = None
        pore_severity = None
        general_acne_severity = None

        # Resolve acne severity (consensus between acne + general_acne models)
        resolved_acne = self._resolve_acne_severity(predictions)

        # Process acne prediction (using resolved best prediction)
        if resolved_acne:
            acne_severity = AcneSeverity(resolved_acne.predicted_label)
            acne_rule = self._acne_rules[acne_severity.value]

            acne_recs = [
                Recommendation(
                    ingredient=r["ingredient"],
                    reason=r["reason"],
                    category=r.get("category", ""),
                )
                for r in acne_rule["recommendations"]
            ]
            recommendations.extend(acne_recs)

            explanation = acne_rule["explanation"]
            explanation += self._confidence_note(resolved_acne.confidence)
            explanations.append(explanation)
            educational_notes.append(acne_rule["educational_note"])

        # Process pores prediction
        if "pores" in predictions:
            pore_pred = predictions["pores"]
            pore_severity = PoreSeverity(pore_pred.predicted_label)
            pore_rule = self._pore_rules[pore_severity.value]

            pore_recs = [
                Recommendation(
                    ingredient=r["ingredient"],
                    reason=r["reason"],
                    category=r.get("category", ""),
                )
                for r in pore_rule["recommendations"]
            ]
            recommendations.extend(pore_recs)

            pore_explanation = pore_rule["explanation"]
            pore_explanation += self._confidence_note(pore_pred.confidence)
            explanations.append(pore_explanation)
            educational_notes.append(pore_rule["educational_note"])

        # Track general_acne separately for reporting
        if "general_acne" in predictions:
            general_pred = predictions["general_acne"]
            general_acne_severity = AcneSeverity(general_pred.predicted_label)

        # Deduplicate recommendations by ingredient
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec.ingredient not in seen:
                seen.add(rec.ingredient)
                unique_recs.append(rec)

        combined_explanation = " ".join(explanations)
        combined_educational = "\n\n".join(educational_notes)

        models_disagree = self._detect_disagreement(predictions)
        disagreement_message = self._disagreement_msg if models_disagree else None

        return BLPResult(
            acne_severity=acne_severity or AcneSeverity.CLEAR,
            pore_severity=pore_severity,
            general_acne_severity=general_acne_severity,
            recommendations=unique_recs,
            explanation=combined_explanation,
            educational_note=combined_educational,
            models_disagree=models_disagree,
            disagreement_message=disagreement_message,
        )

    def _confidence_note(self, confidence: float) -> str:
        """Generate confidence context string."""
        if confidence >= self._thresholds["high"]:
            return " This assessment was made with high confidence."
        elif confidence >= self._thresholds["medium"]:
            return " This assessment was made with moderate confidence."
        else:
            return " This assessment was made with lower confidence — results should be interpreted cautiously."

    @property
    def low_confidence_message(self) -> str:
        return self._low_confidence_msg

    @property
    def confidence_threshold(self) -> float:
        return self._thresholds["low"]


# Module-level singleton
_engine: Optional[BLPEngine] = None


def get_blp_engine() -> BLPEngine:
    global _engine
    if _engine is None:
        _engine = BLPEngine()
    return _engine
