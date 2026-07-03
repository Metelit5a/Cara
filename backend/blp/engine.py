"""
Business Logic Processing Engine

Rule-based engine that interprets model outputs and generates
explainable skincare recommendations.

Processes outputs from 3 models:
  1. Acne Severity  → acne-specific recommendations
  2. Skin Type      → skin-type-appropriate products
  3. Skin Conditions → multi-label: pores and/or blackheads
     (may be empty — meaning no notable conditions detected)

Rules are loaded from rules.json, keeping the system data-driven
and easily configurable without code changes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from shared.schemas import (
    AcneSeverity,
    SkinType,
    SkinCondition,
    BLPResult,
    ModelPrediction,
    MultiLabelPrediction,
    Recommendation,
)

logger = logging.getLogger(__name__)


class BLPEngine:
    """Rule-based business logic processing engine."""

    def __init__(self, rules_path: Optional[str] = None):
        if rules_path is None:
            rules_path = str(Path(__file__).parent / "rules.json")
        with open(rules_path, "r") as f:
            config = json.load(f)

        self._acne_rules = config["acne_rules"]
        self._skin_type_rules = config["skin_type_rules"]
        self._skin_issue_rules = config["skin_issue_rules"]
        self._confidence_threshold = config["confidence_threshold"]
        self._low_confidence_msg = config["low_confidence_message"]

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def low_confidence_message(self) -> str:
        return self._low_confidence_msg

    def process(
        self,
        predictions: Dict[str, Union[ModelPrediction, MultiLabelPrediction]],
    ) -> BLPResult:
        """Process model predictions into a combined BLP result.

        Args:
            predictions: Dict with keys 'acne', 'skin_type', 'skin_conditions'.
                         'acne' and 'skin_type' are `ModelPrediction`s;
                         'skin_conditions' is a `MultiLabelPrediction` whose
                         `findings` may be empty (nothing to report).

        Returns:
            BLPResult with combined recommendations and explanations.
        """
        all_recommendations: List[Recommendation] = []
        explanations: List[str] = []
        educational_notes: List[str] = []

        # ── Acne Severity ──
        acne_pred = predictions.get("acne")
        acne_severity = AcneSeverity.CLEAR
        if (
            isinstance(acne_pred, ModelPrediction)
            and acne_pred.confidence >= self._confidence_threshold
        ):
            acne_severity = AcneSeverity(acne_pred.predicted_label)
            rule = self._acne_rules.get(acne_pred.predicted_label, {})
            if rule.get("explanation"):
                explanations.append(rule["explanation"])
            if rule.get("educational_note"):
                educational_notes.append(rule["educational_note"])
            for rec in rule.get("recommendations", []):
                all_recommendations.append(Recommendation(**rec))

        # ── Skin Type ──
        skin_type_pred = predictions.get("skin_type")
        skin_type = None
        if (
            isinstance(skin_type_pred, ModelPrediction)
            and skin_type_pred.confidence >= self._confidence_threshold
        ):
            skin_type = SkinType(skin_type_pred.predicted_label)
            rule = self._skin_type_rules.get(skin_type_pred.predicted_label, {})
            if rule.get("explanation"):
                explanations.append(rule["explanation"])
            if rule.get("educational_note"):
                educational_notes.append(rule["educational_note"])
            for rec in rule.get("recommendations", []):
                if not any(r.ingredient == rec["ingredient"] for r in all_recommendations):
                    all_recommendations.append(Recommendation(**rec))

        # ── Skin Conditions (multi-label) ──
        #
        # The multi-label model already applied its own threshold in the
        # orchestrator, so any finding here is above the sigmoid threshold.
        # We simply iterate and combine recommendations. If `findings` is
        # empty we say nothing about skin conditions — the user's instinct
        # ("if it doesn't find blackheads or pores, it can assume he is
        # healthy or just not talk about it") is captured here.
        skin_pred = predictions.get("skin_conditions")
        conditions_found: List[SkinCondition] = []
        if isinstance(skin_pred, MultiLabelPrediction):
            for finding in skin_pred.findings:
                conditions_found.append(finding.label)
                rule = self._skin_issue_rules.get(finding.label.value, {})
                if rule.get("explanation"):
                    explanations.append(rule["explanation"])
                if rule.get("educational_note"):
                    educational_notes.append(rule["educational_note"])
                for rec in rule.get("recommendations", []):
                    if not any(r.ingredient == rec["ingredient"] for r in all_recommendations):
                        all_recommendations.append(Recommendation(**rec))

        # ── Combine ──
        combined_explanation = " ".join(explanations) if explanations else "Analysis complete."
        combined_educational = " ".join(educational_notes) if educational_notes else ""

        return BLPResult(
            acne_severity=acne_severity,
            skin_type=skin_type,
            skin_conditions=conditions_found,
            recommendations=all_recommendations,
            explanation=combined_explanation,
            educational_note=combined_educational,
        )


# ── Singleton ────────────────────────────────────────────────────────────────

_engine: Optional[BLPEngine] = None


def get_blp_engine() -> BLPEngine:
    global _engine
    if _engine is None:
        _engine = BLPEngine()
    return _engine
