"""
Cara Backend - FastAPI Application

Main entry point for the backend server.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from shared.config import settings
from shared.schemas import HealthResponse
from backend.api.routes import router as analysis_router
from model_service.inference.orchestrator import get_registry

logger = logging.getLogger("cara")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: load models
    logger.info("Loading models...")
    registry = get_registry()

    weights_path = settings.model_weights_path
    if Path(weights_path).exists():
        registry.register_acne_model(weights_path)
        logger.info("Acne severity model loaded successfully")
    else:
        logger.warning(f"Model weights not found at {weights_path} — running without model")
        # Register model without weights for structural testing
        registry.register_acne_model(weights_path)

    # Ensure storage directories exist
    Path(settings.storage_path, "reports").mkdir(parents=True, exist_ok=True)

    logger.info("Cara backend ready")
    yield
    # Shutdown
    logger.info("Shutting down Cara backend")


app = FastAPI(
    title="Cara - AI Skincare Analysis",
    description="Explainable cosmetic skincare analysis API. NOT a medical diagnosis system.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(analysis_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    registry = get_registry()
    return HealthResponse(
        status="healthy",
        model_loaded=registry.is_loaded("acne"),
        version="0.1.0",
    )
