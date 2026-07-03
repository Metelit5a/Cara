"""
Report Builder

Assembles the final AnalysisReport from BLP results and model predictions.
Supports the 3-model pipeline (acne + skin_type + skin_conditions),
where `skin_conditions` is multi-label.
"""

import uuid
from datetime import datetime
from typing import Dict, Union

from shared.schemas import (
    AnalysisReport,
    AnalysisStatus,
    BLPResult,
    ModelPrediction,
    MultiLabelPrediction,
)


class ReportBuilder:
    """Builds user-facing analysis reports."""

    @staticmethod
    def build_success_report(
        predictions: Dict[str, Union[ModelPrediction, MultiLabelPrediction]],
        blp_result: BLPResult,
    ) -> AnalysisReport:
        acne_pred = predictions.get("acne")
        skin_type_pred = predictions.get("skin_type")
        skin_conditions_pred = predictions.get("skin_conditions")

        # Multi-label findings come out of the BLP already filtered by the
        # sigmoid threshold. Grab the per-finding confidences the model
        # emitted (from `MultiLabelPrediction.findings`).
        finding_conf = {
            f.label.value: f.confidence
            for f in (skin_conditions_pred.findings if isinstance(skin_conditions_pred, MultiLabelPrediction) else [])
        }
        skin_conditions = [
            # Rebuild finding objects so the report carries confidence too.
            _finding_for(label, finding_conf)
            for label in blp_result.skin_conditions
        ]

        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.SUCCESS,
            acne_severity=blp_result.acne_severity,
            skin_type=blp_result.skin_type,
            skin_conditions=skin_conditions,
            acne_confidence=acne_pred.confidence if isinstance(acne_pred, ModelPrediction) else None,
            skin_type_confidence=skin_type_pred.confidence if isinstance(skin_type_pred, ModelPrediction) else None,
            explanation=blp_result.explanation,
            recommendations=blp_result.recommendations,
            educational_note=blp_result.educational_note,
        )

    @staticmethod
    def build_low_confidence_report(
        predictions: Dict[str, Union[ModelPrediction, MultiLabelPrediction]], message: str
    ) -> AnalysisReport:
        # Use the highest confidence from any single-label model. Multi-label
        # scores are per-class and not directly comparable, so we skip them.
        best_conf = max(
            (p.confidence for p in predictions.values() if isinstance(p, ModelPrediction)),
            default=None,
        )
        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.LOW_CONFIDENCE,
            acne_confidence=best_conf,
            message=message,
        )

    @staticmethod
    def build_no_face_report(message: str) -> AnalysisReport:
        """Face detection failed — the user needs to retake the photo."""
        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.NO_FACE,
            message=message,
        )

    @staticmethod
    def build_error_report(message: str) -> AnalysisReport:
        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.ERROR,
            message=message,
        )


def _finding_for(label, conf_lookup: Dict[str, float]):
    """Build a `SkinConditionFinding` for the report from the BLP label list."""
    from shared.schemas import SkinConditionFinding, SkinCondition

    key = label.value if hasattr(label, "value") else str(label)
    return SkinConditionFinding(
        label=SkinCondition(key),
        confidence=conf_lookup.get(key, 0.0),
    )
