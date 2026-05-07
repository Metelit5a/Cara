"""
Storage Repository Interface

Defines an abstract interface for data storage. Implementations:
- JsonStorageRepository: File-based JSON storage (current POC)
- MongoStorageRepository: MongoDB storage (prepared, not connected)

Switching backends requires only changing STORAGE_BACKEND in .env.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from shared.schemas import AnalysisReport


class StorageRepository(ABC):
    """Abstract storage interface for reports."""

    @abstractmethod
    async def save_report(self, report: AnalysisReport) -> str:
        ...

    @abstractmethod
    async def get_report(self, report_id: str) -> Optional[AnalysisReport]:
        ...

    @abstractmethod
    async def list_reports(self, limit: int = 50) -> List[AnalysisReport]:
        ...


class JsonStorageRepository(StorageRepository):
    """File-based JSON storage for POC. Each report is a separate JSON file."""

    def __init__(self, storage_path: str = "storage"):
        self.reports_dir = Path(storage_path) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def save_report(self, report: AnalysisReport) -> str:
        filepath = self.reports_dir / f"{report.id}.json"
        data = report.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return report.id

    async def get_report(self, report_id: str) -> Optional[AnalysisReport]:
        filepath = self.reports_dir / f"{report_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            data = json.load(f)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return AnalysisReport(**data)

    async def list_reports(self, limit: int = 50) -> List[AnalysisReport]:
        reports = []
        files = sorted(self.reports_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        for filepath in files[:limit]:
            with open(filepath, "r") as f:
                data = json.load(f)
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            reports.append(AnalysisReport(**data))
        return reports


class MongoStorageRepository(StorageRepository):
    """MongoDB storage (prepared for future connection).

    To activate:
    1. Set STORAGE_BACKEND=mongodb in .env
    2. Set MONGODB_URI and MONGODB_DB_NAME
    3. Install motor: pip install motor
    """

    def __init__(self, uri: str, db_name: str):
        self._uri = uri
        self._db_name = db_name
        self._client = None
        self._db = None

    async def _get_collection(self):
        if self._client is None:
            from motor.motor_asyncio import AsyncIOMotorClient
            self._client = AsyncIOMotorClient(self._uri)
            self._db = self._client[self._db_name]
        return self._db["reports"]

    async def save_report(self, report: AnalysisReport) -> str:
        collection = await self._get_collection()
        data = report.model_dump()
        data["_id"] = report.id
        await collection.insert_one(data)
        return report.id

    async def get_report(self, report_id: str) -> Optional[AnalysisReport]:
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": report_id})
        if doc is None:
            return None
        doc.pop("_id", None)
        return AnalysisReport(**doc)

    async def list_reports(self, limit: int = 50) -> List[AnalysisReport]:
        collection = await self._get_collection()
        cursor = collection.find().sort("created_at", -1).limit(limit)
        reports = []
        async for doc in cursor:
            doc.pop("_id", None)
            reports.append(AnalysisReport(**doc))
        return reports


def create_repository() -> StorageRepository:
    """Factory: create the appropriate storage repository based on config."""
    from shared.config import settings

    if settings.storage_backend == "mongodb":
        return MongoStorageRepository(settings.mongodb_uri, settings.mongodb_db_name)
    return JsonStorageRepository(settings.storage_path)
