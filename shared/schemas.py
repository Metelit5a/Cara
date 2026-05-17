"""Shared Pydantic schemas used across backend, model service, and BLP."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# ── Enums ──

class AcneSeverity(str, Enum):
    CLEAR = "clear"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class PoreSeverity(str, Enum):
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    LOW_CONFIDENCE = "low_confidence"
    NO_FACE = "no_face_detected"
    ERROR = "error"


# ── Model inference schemas ──

class ModelPrediction(BaseModel):
    """Raw output from a single model."""
    model_config = {"protected_namespaces": ()}

    model_name: str
    predicted_class: int
    predicted_label: str
    confidence: float = Field(ge=0.0, le=1.0)
    all_probabilities: List[float]


class PreprocessingResult(BaseModel):
    """Result of the preprocessing pipeline."""
    success: bool
    face_detected: bool
    message: str = ""


# ── BLP schemas ──

class Recommendation(BaseModel):
    """A single skincare recommendation."""
    ingredient: str
    reason: str
    category: str = ""


class BLPResult(BaseModel):
    """Output from the Business Logic Processing layer."""
    acne_severity: AcneSeverity
    pore_severity: Optional[PoreSeverity] = None
    general_acne_severity: Optional[AcneSeverity] = None
    recommendations: List[Recommendation]
    explanation: str
    educational_note: str


# ── Report schemas ──

class AnalysisReport(BaseModel):
    """Complete analysis report returned to the user."""
    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: AnalysisStatus
    acne_severity: Optional[AcneSeverity] = None
    pore_severity: Optional[PoreSeverity] = None
    general_acne_severity: Optional[AcneSeverity] = None
    confidence: Optional[float] = None
    pore_confidence: Optional[float] = None
    general_acne_confidence: Optional[float] = None
    pore_count: Optional[int] = None
    explanation: Optional[str] = None
    recommendations: List[Recommendation] = []
    educational_note: Optional[str] = None
    message: Optional[str] = None


# ── API schemas ──

class FaceDetectionResponse(BaseModel):
    """API response for POST /detect-face."""
    face_count: int
    faces_detected: bool
    message: str
    bbox: Optional[dict] = None  # {x, y, w, h} for single face


class AnalyzeResponse(BaseModel):
    """API response for POST /analyze."""
    report: AnalysisReport


class HealthResponse(BaseModel):
    """API response for GET /health."""
    model_config = {"protected_namespaces": ()}

    status: str = "healthy"
    model_loaded: bool
    pores_model_loaded: bool = False
    general_acne_model_loaded: bool = False
    version: str = "0.1.0"
