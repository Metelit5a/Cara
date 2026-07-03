"""Application-wide configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    debug: bool = False

    # Model - Acne Severity
    model_weights_path: str = "model_service/checkpoints/acne_model_best.pth"
    confidence_threshold: float = 0.30

    # Model - Skin Type
    skin_type_model_weights_path: str = "model_service/checkpoints/skin_type_model_best.pth"

    # Model - Skin Issues (DEPRECATED — legacy single-label model, replaced by
    # `skin_conditions` below. Kept only so historical checkpoints still load.)
    skin_issues_model_weights_path: str = "model_service/checkpoints/skin_issues_model_best.pth"

    # Model - Skin Conditions (multi-label: pores, blackheads)
    # Threshold applied to each sigmoid output. Findings above the threshold
    # are reported; below → not mentioned (i.e. "no notable conditions").
    skin_conditions_model_weights_path: str = "model_service/checkpoints/skin_conditions_model_best.pth"
    skin_conditions_threshold: float = 0.50

    # Storage
    storage_backend: str = "json"  # "json" or "mongodb"
    storage_path: str = "storage"

    # MongoDB (prepared, not connected)
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db_name: str = "cara"

    # CORS
    cors_origins: str = '["http://localhost:3000"]'

    @property
    def cors_origins_list(self) -> List[str]:
        return json.loads(self.cors_origins)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        protected_namespaces = ("settings_",)


settings = Settings()
