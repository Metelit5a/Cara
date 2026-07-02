"""
Report Builder

Assembles the final AnalysisReport from BLP results and model predictions.
Supports the 3-model pipeline (acne + skin_type + skin_issues).
"""

import uuid
from datetime import datetime
from typing import Dict

from shared.schemas import (
    AnalysisReport,
    AnalysisStatus,
    BLPResult,
    ModelPrediction,
)


class ReportBuilder:
    """Builds user-facing analysis reports."""

    @staticmethod
    def build_success_report(
        predictions: Dict[str, ModelPrediction],
        blp_result: BLPResult,
    ) -> AnalysisReport:
        acne_pred = predictions.get("acne")
        skin_type_pred = predictions.get("skin_type")
        skin_issue_pred = predictions.get("skin_issues")

        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.SUCCESS,
            acne_severity=blp_result.acne_severity,
            skin_type=blp_result.skin_type,
            skin_issue=blp_result.skin_issue,
            acne_confidence=acne_pred.confidence if acne_pred else None,
            skin_type_confidence=skin_type_pred.confidence if skin_type_pred else None,
            skin_issue_confidence=skin_issue_pred.confidence if skin_issue_pred else None,
            explanation=blp_result.explanation,
            recommendations=blp_result.recommendations,
            educational_note=blp_result.educational_note,
        )

    @staticmethod
    def build_low_confidence_report(
        predictions: Dict[str, ModelPrediction], message: str
    ) -> AnalysisReport:
        # Use the highest confidence from any model
        best_conf = max((p.confidence for p in predictions.values()), default=None)
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
