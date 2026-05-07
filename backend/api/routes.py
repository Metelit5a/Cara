"""
API Router: Analysis Endpoints

POST /api/v1/analyze   - Upload image, receive analysis report
GET  /api/v1/report/{id} - Retrieve a saved report
GET  /api/v1/reports   - List all reports
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional

from shared.schemas import AnalysisReport, AnalyzeResponse
from backend.services.analysis_service import AnalysisService
from backend.database.repository import create_repository

router = APIRouter(prefix="/api/v1", tags=["analysis"])

# Service instance (initialized once)
_service: Optional[AnalysisService] = None


def _get_service() -> AnalysisService:
    global _service
    if _service is None:
        _service = AnalysisService(create_repository())
    return _service


ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Upload a face image for skincare analysis."""

    # Validate content type
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Accepted: JPEG, PNG, WebP, BMP.",
        )

    # Read and validate size
    image_bytes = await file.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    service = _get_service()
    report = await service.analyze_image(image_bytes)

    return AnalyzeResponse(report=report)


@router.get("/report/{report_id}", response_model=AnalysisReport)
async def get_report(report_id: str):
    """Retrieve a previously generated report by ID."""
    service = _get_service()
    report = await service.get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@router.get("/reports", response_model=List[AnalysisReport])
async def list_reports(limit: int = 50):
    """List recent analysis reports."""
    service = _get_service()
    return await service.list_reports(limit=min(limit, 100))
