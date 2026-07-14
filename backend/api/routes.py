"""
API Router: Analysis Endpoints

POST /api/v1/analyze   - Upload image, receive analysis report
GET  /api/v1/report/{id} - Retrieve a saved report
GET  /api/v1/reports   - List all reports
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from backend.api.auth import get_current_user
from backend.database.repository import create_repository
from backend.services.analysis_service import AnalysisService
from shared.schemas import AnalysisReport, AnalyzeResponse

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
async def analyze_image(
    file: UploadFile = File(...),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
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

    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    service = _get_service()
    report = await service.analyze_image(image_bytes, user_id=current_user["id"])

    return AnalyzeResponse(report=report)


@router.get("/report/{report_id}", response_model=AnalysisReport)
async def get_report(
    report_id: str,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """Retrieve a previously generated report by ID."""
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    service = _get_service()
    report = await service.get_report(report_id, user_id=current_user["id"])
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@router.get("/reports", response_model=List[AnalysisReport])
async def list_reports(
    limit: int = 50,
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user),
):
    """List recent analysis reports."""
    if current_user is None:
        raise HTTPException(status_code=401, detail="Authentication required")

    service = _get_service()
    return await service.list_reports(limit=min(limit, 100), user_id=current_user["id"])
