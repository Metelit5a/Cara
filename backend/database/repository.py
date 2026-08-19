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
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from typing import List, Optional, Dict, Any

from shared.schemas import AnalysisReport


class StorageRepository(ABC):
    """Abstract storage interface for reports."""

    @abstractmethod
    async def save_report(self, report: AnalysisReport, user_id: Optional[str] = None) -> str:
        ...

    @abstractmethod
    async def get_report(self, report_id: str, user_id: Optional[str] = None) -> Optional[AnalysisReport]:
        ...

    @abstractmethod
    async def list_reports(self, limit: int = 50, user_id: Optional[str] = None) -> List[AnalysisReport]:
        ...

    # User storage methods
    @abstractmethod
    async def save_user(self, user: Dict[str, Any]) -> str:
        ...

    @abstractmethod
    async def get_user(self, user_id: Optional[str] = None, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    async def list_users(self) -> List[Dict[str, Any]]:
        ...


class JsonStorageRepository(StorageRepository):
    """File-based JSON storage for POC. Each report is a separate JSON file."""

    def __init__(self, storage_path: str = "storage"):
        self.reports_dir = Path(storage_path) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.users_file = Path(storage_path) / "users.json"
        # ensure users file exists
        if not self.users_file.exists():
            self.users_file.parent.mkdir(parents=True, exist_ok=True)
            self.users_file.write_text("[]", encoding="utf-8")

    async def save_report(self, report: AnalysisReport, user_id: Optional[str] = None) -> str:
        filepath = self.reports_dir / f"{report.id}.json"
        data = report.model_dump()
        data["created_at"] = data["created_at"].isoformat()
        data["user_id"] = user_id or getattr(report, "user_id", None)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return report.id

    async def get_report(self, report_id: str, user_id: Optional[str] = None) -> Optional[AnalysisReport]:
        filepath = self.reports_dir / f"{report_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, "r") as f:
            data = json.load(f)
        if user_id is not None and data.get("user_id") != user_id:
            return None
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return AnalysisReport(**data)

    async def list_reports(self, limit: int = 50, user_id: Optional[str] = None) -> List[AnalysisReport]:
        reports = []
        files = sorted(self.reports_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        for filepath in files:
            with open(filepath, "r") as f:
                data = json.load(f)
            if user_id is not None and data.get("user_id") != user_id:
                continue
            data["created_at"] = datetime.fromisoformat(data["created_at"])
            reports.append(AnalysisReport(**data))
            if len(reports) >= limit:
                break
        return reports

    # -- User methods (JSON file) --
    async def save_user(self, user: Dict[str, Any]) -> str:
        # Load existing users, replace if id present, otherwise append
        users = []
        with open(self.users_file, "r", encoding="utf-8") as f:
            users = json.load(f)

        if "id" in user and any(u.get("id") == user.get("id") for u in users):
            users = [user if u.get("id") == user.get("id") else u for u in users]
        else:
            # assign id if missing
            if "id" not in user:
                user = dict(user)
                user["id"] = str(len(users) + 1)
            users.append(user)

        with open(self.users_file, "w", encoding="utf-8") as f:
            json.dump(users, f, indent=2)

        return user["id"]

    async def get_user(self, user_id: Optional[str] = None, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with open(self.users_file, "r", encoding="utf-8") as f:
            users = json.load(f)
        for u in users:
            if user_id is not None and str(u.get("id")) == str(user_id):
                return u
            if email is not None and u.get("email") == email:
                return u
        return None

    async def list_users(self) -> List[Dict[str, Any]]:
        with open(self.users_file, "r", encoding="utf-8") as f:
            return json.load(f)


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

    async def _get_users_collection(self):
        if self._client is None:
            from motor.motor_asyncio import AsyncIOMotorClient
            self._client = AsyncIOMotorClient(self._uri)
            self._db = self._client[self._db_name]
        return self._db["users"]

    async def save_report(self, report: AnalysisReport, user_id: Optional[str] = None) -> str:
        collection = await self._get_collection()
        data = report.model_dump()
        data["_id"] = report.id
        data["user_id"] = user_id or getattr(report, "user_id", None)
        await collection.insert_one(data)
        return report.id

    async def get_report(self, report_id: str, user_id: Optional[str] = None) -> Optional[AnalysisReport]:
        collection = await self._get_collection()
        query = {"_id": report_id}
        if user_id is not None:
            query["user_id"] = user_id
        doc = await collection.find_one(query)
        if doc is None:
            return None
        doc.pop("_id", None)
        return AnalysisReport(**doc)

    async def list_reports(self, limit: int = 50, user_id: Optional[str] = None) -> List[AnalysisReport]:
        collection = await self._get_collection()
        query = {}
        if user_id is not None:
            query["user_id"] = user_id
        cursor = collection.find(query).sort("created_at", -1).limit(limit)
        reports = []
        async for doc in cursor:
            doc.pop("_id", None)
            reports.append(AnalysisReport(**doc))
        return reports

    # -- User methods (Mongo) --
    async def save_user(self, user: Dict[str, Any]) -> str:
        users_col = await self._get_users_collection()
        user_doc = dict(user)
        if "id" in user_doc:
            user_doc["_id"] = user_doc.pop("id")
            await users_col.replace_one({"_id": user_doc["_id"]}, user_doc, upsert=True)
            return user_doc["_id"]
        else:
            result = await users_col.insert_one(user_doc)
            return str(result.inserted_id)

    async def get_user(self, user_id: Optional[str] = None, email: Optional[str] = None) -> Optional[Dict[str, Any]]:
        users_col = await self._get_users_collection()
        # Prefer querying by email if provided, otherwise by id.
        # When querying by id, try both the raw string and an ObjectId.
        if email is not None and user_id is None:
            doc = await users_col.find_one({"email": email})
        else:
            # try direct string match first
            doc = None
            if user_id is not None:
                doc = await users_col.find_one({"_id": user_id})
                if doc is None:
                    # try treating id as ObjectId
                    try:
                        from bson.objectid import ObjectId

                        oid = ObjectId(user_id)
                        doc = await users_col.find_one({"_id": oid})
                    except Exception:
                        doc = None
            # if not found by id and email provided, try email fallback
            if doc is None and email is not None:
                doc = await users_col.find_one({"email": email})

        if doc is None:
            return None
        # normalize id to string
        raw_id = doc.pop("_id", None)
        doc["id"] = str(raw_id) if raw_id is not None else None
        return doc

    async def list_users(self) -> List[Dict[str, Any]]:
        users_col = await self._get_users_collection()
        docs = []
        cursor = users_col.find({})
        async for doc in cursor:
            raw_id = doc.pop("_id", None)
            doc["id"] = str(raw_id) if raw_id is not None else None
            docs.append(doc)
        return docs


def create_repository() -> StorageRepository:
    """Factory: create the appropriate storage repository based on config."""
    from shared.config import settings

    if settings.storage_backend == "mongodb":
        return MongoStorageRepository(settings.mongodb_uri, settings.mongodb_db_name)
    return JsonStorageRepository(settings.storage_path)
