"""
Business Logic Processing Engine

Rule-based engine that interprets model outputs and generates
explainable skincare recommendations.

Rules are loaded from a JSON configuration file, making the system
data-driven and easily configurable without code changes.

Designed to support future multi-model aggregation:
  acne model + skin type model + redness model → combined BLP output
"""

import json
from pathlib import Path
from typing import Optional

from shared.schemas import (
    AcneSeverity,
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
        self._rules = config["rules"]
        self._thresholds = config["confidence_thresholds"]
        self._low_confidence_msg = config["low_confidence_message"]

    def process(self, prediction: ModelPrediction) -> BLPResult:
        """Process a model prediction into a BLP result with recommendations.

        Currently processes only acne severity. Architecture supports
        future multi-model inputs by accepting additional predictions.
        """
        severity = AcneSeverity(prediction.predicted_label)
        rule = self._rules[severity.value]

        recommendations = [
            Recommendation(
                ingredient=r["ingredient"],
                reason=r["reason"],
                category=r.get("category", ""),
            )
            for r in rule["recommendations"]
        ]

        explanation = rule["explanation"]

        # Add confidence context to explanation
        if prediction.confidence >= self._thresholds["high"]:
            confidence_note = " This assessment was made with high confidence."
        elif prediction.confidence >= self._thresholds["medium"]:
            confidence_note = " This assessment was made with moderate confidence."
        else:
            confidence_note = " This assessment was made with lower confidence — results should be interpreted cautiously."
        explanation += confidence_note

        return BLPResult(
            severity=severity,
            recommendations=recommendations,
            explanation=explanation,
            educational_note=rule["educational_note"],
        )

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
