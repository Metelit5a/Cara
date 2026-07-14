"""Shared Pydantic schemas used across backend, model service, and BLP."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
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
    """DEPRECATED — single-label output of the old skin_issues model.

    Kept so old reports in `storage/reports/` still deserialise. New
    reports use `skin_conditions: List[SkinConditionFinding]` (see below)
    from the multi-label model.
    """
    HEALTHY = "healthy"
    BLACKHEADS = "blackheads"
    DARK_SPOTS = "dark_spots"
    PORES = "pores"
    WRINKLES = "wrinkles"


class SkinCondition(str, Enum):
    """Multi-label conditions detected by the current skin_conditions model.

    Each is independent — an image can trigger any subset (including none).
    """
    PORES = "pores"
    BLACKHEADS = "blackheads"


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


class SkinConditionFinding(BaseModel):
    """A single condition the multi-label model flagged as present."""
    label: SkinCondition
    confidence: float = Field(ge=0.0, le=1.0)


class MultiLabelPrediction(BaseModel):
    """Raw output from a multi-label model (independent sigmoid per class).

    `findings` is the list of (label, probability) pairs above the
    decision threshold. `all_scores` keeps every class score (regardless
    of threshold) for debugging and per-condition confidence display.
    """
    model_config = {"protected_namespaces": ()}

    model_name: str
    findings: List[SkinConditionFinding] = []
    all_scores: Dict[str, float] = {}


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
    skin_conditions: List[SkinCondition] = []
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
    # Multi-label output: 0, 1, or 2 findings (empty = "no notable conditions").
    skin_conditions: List[SkinConditionFinding] = []
    acne_confidence: Optional[float] = None
    skin_type_confidence: Optional[float] = None
    # DEPRECATED: `skin_issue` / `skin_issue_confidence` from the old
    # single-label model. Left here so old reports in `storage/reports/`
    # still deserialise, but never populated by new runs.
    skin_issue: Optional[SkinIssue] = None
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


class UserCreate(BaseModel):
    """Schema for registering a new user."""
    username: str
    email: str
    password: str


class UserLogin(BaseModel):
    """Schema for logging in an existing user."""
    email: str
    password: str


class Token(BaseModel):
    """JWT response payload for authentication."""
    access_token: str
    token_type: str = "bearer"
