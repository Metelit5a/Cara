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


class SkinType(str, Enum):
    OILY = "oily"
    DRY = "dry"
    NORMAL = "normal"
    COMBINATION = "combination"


class SkinIssue(str, Enum):
    HEALTHY = "healthy"
    BLACKHEADS = "blackheads"
    DARK_SPOTS = "dark_spots"
    PORES = "pores"
    WRINKLES = "wrinkles"


class PoreSeverity(str, Enum):
    """Legacy — kept for backward compatibility."""
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
    skin_type: Optional[SkinType] = None
    skin_issue: Optional[SkinIssue] = None
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
    skin_type: Optional[SkinType] = None
    skin_issue: Optional[SkinIssue] = None
    acne_confidence: Optional[float] = None
    skin_type_confidence: Optional[float] = None
    skin_issue_confidence: Optional[float] = None
    explanation: Optional[str] = None
    recommendations: List[Recommendation] = []
    educational_note: Optional[str] = None
    message: Optional[str] = None


# ── API schemas ──

class AnalyzeResponse(BaseModel):
    """API response for POST /analyze."""
    report: AnalysisReport


class HealthResponse(BaseModel):
    """API response for GET /health."""
    model_config = {"protected_namespaces": ()}

    status: str = "healthy"
    models_loaded: List[str] = []
    version: str = "0.1.0"
