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

    # Load acne model
    weights_path = settings.model_weights_path
    if Path(weights_path).exists():
        registry.register_acne_model(weights_path)
        logger.info("Acne severity model loaded successfully")
    else:
        logger.warning(f"Model weights not found at {weights_path} — running without model")
        registry.register_acne_model(weights_path)

    # Load pores model
    pores_path = settings.pores_model_weights_path
    if Path(pores_path).exists():
        registry.register_pores_model(pores_path)
        logger.info("Pores severity model loaded successfully")
    else:
        logger.warning(f"Pores weights not found at {pores_path} — running without pores model")
        registry.register_pores_model(pores_path)

    # Load general acne model
    general_path = settings.general_acne_model_weights_path
    if Path(general_path).exists():
        registry.register_general_acne_model(general_path)
        logger.info("General acne severity model loaded successfully")
    else:
        logger.warning(f"General acne weights not found at {general_path} — running without general acne model")
        registry.register_general_acne_model(general_path)

    # Load skin issues model
    skin_issues_path = settings.skin_issues_model_weights_path
    if Path(skin_issues_path).exists():
        registry.register_skin_issues_model(skin_issues_path)
        logger.info("Skin issues type model loaded successfully")
    else:
        logger.warning(f"Skin issues weights not found at {skin_issues_path} — running without skin issues model")

    # Ensure storage directories exist
    Path(settings.storage_path, "reports").mkdir(parents=True, exist_ok=True)

    logger.info(f"Cara backend ready — models loaded: {registry.loaded_models}")
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
        pores_model_loaded=registry.is_loaded("pores"),
        general_acne_model_loaded=registry.is_loaded("general_acne"),
        skin_issues_model_loaded=registry.is_loaded("skin_issues"),
        version="0.1.0",
    )
