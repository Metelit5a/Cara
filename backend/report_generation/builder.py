"""
Report Builder

Assembles the final AnalysisReport from BLP results and model predictions.
"""

import uuid
from datetime import datetime

from shared.schemas import (
    AnalysisReport,
    AnalysisStatus,
    BLPResult,
    ModelPrediction,
)


class ReportBuilder:
    """Builds user-facing analysis reports."""

    @staticmethod
    def build_success_report(prediction: ModelPrediction, blp_result: BLPResult) -> AnalysisReport:
        return AnalysisReport(
            id=str(uuid.uuid4()),
            created_at=datetime.utcnow(),
            status=AnalysisStatus.SUCCESS,
            severity=blp_result.severity,
            confidence=prediction.confidence,
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
