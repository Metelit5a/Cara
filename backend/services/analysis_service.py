"""
Analysis Service

Orchestrates the full analysis pipeline:
  image bytes → preprocessing → model inference → BLP → report → storage
"""

from typing import List, Optional

from shared.schemas import AnalysisReport
from model_service.preprocessing.pipeline import get_pipeline
from model_service.inference.orchestrator import get_orchestrator
from backend.blp.engine import get_blp_engine
from backend.report_generation.builder import ReportBuilder
from backend.database.repository import StorageRepository


class AnalysisService:
    """Main service that coordinates the end-to-end analysis pipeline."""

    def __init__(self, repository: StorageRepository):
        self.repository = repository

    async def analyze_image(self, image_bytes: bytes) -> AnalysisReport:
        """Run the full analysis pipeline on an uploaded image."""

        # Step 1: Preprocessing
        pipeline = get_pipeline()
        preprocess_result, tensor = pipeline.process(image_bytes)

        if not preprocess_result.success:
            if not preprocess_result.face_detected:
                report = ReportBuilder.build_no_face_report(preprocess_result.message)
            else:
                report = ReportBuilder.build_error_report(preprocess_result.message)
            await self.repository.save_report(report)
            return report

        # Step 2: Model inference
        orchestrator = get_orchestrator()
        try:
            prediction = orchestrator.predict_acne(tensor)
        except RuntimeError as e:
            report = ReportBuilder.build_error_report(str(e))
            await self.repository.save_report(report)
            return report

        # Step 3: Confidence check
        blp_engine = get_blp_engine()
        if prediction.confidence < blp_engine.confidence_threshold:
            report = ReportBuilder.build_low_confidence_report(
                prediction, blp_engine.low_confidence_message
            )
            await self.repository.save_report(report)
            return report

        # Step 4: Business Logic Processing
        blp_result = blp_engine.process(prediction)

        # Step 5: Report generation
        report = ReportBuilder.build_success_report(prediction, blp_result)

        # Step 6: Persist
        await self.repository.save_report(report)

        return report

    async def get_report(self, report_id: str) -> Optional[AnalysisReport]:
        return await self.repository.get_report(report_id)

    async def list_reports(self, limit: int = 50) -> List[AnalysisReport]:
        return await self.repository.list_reports(limit)
