"""
Analysis Service

Orchestrates the full analysis pipeline:
  image bytes → face-crop preprocessing → model inference (3 models)
  → BLP → report → storage

All three models run on the same face-cropped image. Face-cropping is
mandatory: without it the models see too much background/hair/clothing
that they were never trained on, and hallucinate confidently-wrong
labels (see tests/test_real_world_regression.py). If no face is
detected we return a NO_FACE status so the user can retake the photo.
"""

import logging

from shared.schemas import AnalysisReport, AnalysisStatus, ModelPrediction, MultiLabelPrediction
from model_service.inference.orchestrator import get_orchestrator
from model_service.preprocessing.pipeline import get_pipeline
from backend.blp.engine import get_blp_engine
from backend.report_generation.builder import ReportBuilder
from backend.database.repository import StorageRepository

logger = logging.getLogger(__name__)


class AnalysisService:
    """Main service that coordinates the end-to-end analysis pipeline."""

    def __init__(self, repository: StorageRepository):
        self.repository = repository

    async def analyze_image(self, image_bytes: bytes) -> AnalysisReport:
        """Run the full analysis pipeline on an uploaded image."""

        # Step 1: Face-crop preprocessing (produces a 224x224 face tensor)
        pipeline = get_pipeline()
        preprocess_result, image_tensor = pipeline.process(image_bytes)

        if not preprocess_result.success:
            # "No face detected" is user-actionable (retake the photo).
            # "Invalid image data" is a real error (corrupt/unsupported file).
            if preprocess_result.message.startswith("No face"):
                logger.info(f"Face detection failed: {preprocess_result.message}")
                report = ReportBuilder.build_no_face_report(preprocess_result.message)
            else:
                logger.error(f"Preprocessing failed: {preprocess_result.message}")
                report = ReportBuilder.build_error_report(preprocess_result.message)
            await self.repository.save_report(report)
            return report

        # Step 2: Run all models on the cropped face tensor
        orchestrator = get_orchestrator()
        try:
            predictions = orchestrator.predict_all(image_tensor)
        except RuntimeError as e:
            logger.error(f"Model inference failed: {e}")
            report = ReportBuilder.build_error_report(str(e))
            await self.repository.save_report(report)
            return report

        if not predictions:
            report = ReportBuilder.build_error_report("No models are loaded. Please check model checkpoints.")
            await self.repository.save_report(report)
            return report

        # Step 3: Check confidence — at least one signal must be usable.
        #
        # For single-label models: `confidence >= threshold` on the argmax.
        # For multi-label models: at least one class score >= its threshold
        #   (i.e. a non-empty `findings` list, or "we detected something").
        # If neither the single-label nor multi-label models had anything
        # confident, we return a low-confidence report so the user can retry.
        blp_engine = get_blp_engine()

        def _prediction_is_confident(pred) -> bool:
            if isinstance(pred, MultiLabelPrediction):
                return len(pred.findings) > 0
            if isinstance(pred, ModelPrediction):
                return pred.confidence >= blp_engine.confidence_threshold
            return False

        any_confident = any(_prediction_is_confident(p) for p in predictions.values())

        if not any_confident:
            report = ReportBuilder.build_low_confidence_report(
                predictions, blp_engine.low_confidence_message
            )
            await self.repository.save_report(report)
            return report

        # Step 4: Business Logic Processing
        blp_result = blp_engine.process(predictions)

        # Step 5: Build report
        report = ReportBuilder.build_success_report(predictions, blp_result)

        # Step 6: Persist
        await self.repository.save_report(report)

        return report

    async def get_report(self, report_id: str):
        """Retrieve a saved report by ID."""
        return await self.repository.get_report(report_id)

    async def list_reports(self, limit: int = 50):
        """List recent reports."""
        return await self.repository.list_reports(limit=limit)
