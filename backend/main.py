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
from backend.api.auth import router as auth_router
from model_service.inference.orchestrator import get_orchestrator

logger = logging.getLogger("cara")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    # Startup: load models (orchestrator handles loading)
    logger.info("Loading models...")
    orchestrator = get_orchestrator()

    # Ensure storage directories exist
    Path(settings.storage_path, "reports").mkdir(parents=True, exist_ok=True)

    logger.info(f"Cara backend ready — models loaded: {orchestrator.loaded_models}")
    yield
    # Shutdown
    logger.info("Shutting down Cara backend")


app = FastAPI(
    title="Cara - AI Skincare Analysis",
    description="Explainable cosmetic skincare analysis API. NOT a medical diagnosis system.",
    version="0.2.0",
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
app.include_router(auth_router)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    orchestrator = get_orchestrator()
    return HealthResponse(
        status="healthy",
        models_loaded=orchestrator.loaded_models,
        version="0.2.0",
    )
