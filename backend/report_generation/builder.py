"""
Report Builder

Assembles the final AnalysisReport from BLP results and model predictions.
Supports multi-model pipeline (acne + pores).
"""

import uuid
from datetime import datetime
from typing import Dict, Optional

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
        pore_pred = predictions.get("pores")

        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.SUCCESS,
            acne_severity=blp_result.acne_severity,
            pore_severity=blp_result.pore_severity,
            confidence=acne_pred.confidence if acne_pred else None,
            pore_confidence=pore_pred.confidence if pore_pred else None,
            pore_count=pore_pred.predicted_class if pore_pred else None,
            explanation=blp_result.explanation,
            recommendations=blp_result.recommendations,
            educational_note=blp_result.educational_note,
        )

    @staticmethod
    def build_low_confidence_report(prediction: ModelPrediction, message: str) -> AnalysisReport:
        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.LOW_CONFIDENCE,
            confidence=prediction.confidence,
            message=message,
        )

    @staticmethod
    def build_no_face_report(message: str) -> AnalysisReport:
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
