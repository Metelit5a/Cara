"""
API Router: Analysis Endpoints

POST /api/v1/analyze   - Upload image, receive analysis report
POST /api/v1/detect-face - Upload image, detect faces (no model inference)
GET  /api/v1/report/{id} - Retrieve a saved report
GET  /api/v1/reports   - List all reports
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Optional
import numpy as np
import cv2

from shared.schemas import AnalysisReport, AnalyzeResponse, FaceDetectionResponse
from backend.services.analysis_service import AnalysisService
from backend.database.repository import create_repository
from model_service.preprocessing.pipeline import get_pipeline

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


@router.post("/detect-face", response_model=FaceDetectionResponse)
async def detect_face(file: UploadFile = File(...)):
    """Detect faces in an image without running model inference."""

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

    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image data.")

    # Detect faces
    pipeline = get_pipeline()
    face_count, bbox = pipeline.face_detector.detect_all(image)

    if face_count == 0:
        return FaceDetectionResponse(
            face_count=0,
            faces_detected=False,
            message="No face detected in image. Please upload a clear face photo.",
            bbox=None,
        )
    elif face_count == 1:
        return FaceDetectionResponse(
            face_count=1,
            faces_detected=True,
            message="Face detected successfully.",
            bbox={"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])} if bbox else None,
        )
    else:
        return FaceDetectionResponse(
            face_count=face_count,
            faces_detected=False,
            message=f"Multiple faces detected ({face_count}). Please upload a photo with only your face.",
            bbox={"x": int(bbox[0]), "y": int(bbox[1]), "w": int(bbox[2]), "h": int(bbox[3])} if bbox else None,
        )


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
