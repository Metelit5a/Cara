"""Application-wide configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    # Server
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    debug: bool = False

    # Model - Acne (acne04)
    model_weights_path: str = "model_service/checkpoints/acne_model_best.pth"
    # Lowered from 0.4: the 4-class softmax often peaks at 0.3-0.45 on real-world
    # photos even when correct, especially with class-balanced training.
    confidence_threshold: float = 0.25

    # Model - Pores
    pores_model_weights_path: str = "model_service/checkpoints/pores_model_best.pth"

    # Model - General Acne (Roboflow COCO, lesion-count severity)
    general_acne_model_weights_path: str = "model_service/checkpoints/general_acne_model_best.pth"

    # Model - Skin Issues (type classification)
    skin_issues_model_weights_path: str = "model_service/checkpoints/skin_issues_model_best.pth"

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
