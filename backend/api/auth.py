"""Authentication endpoints and helpers for the Cara backend."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from shared.schemas import Token, UserCreate, UserLogin

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


def get_users_file() -> Path:
    users_file = Path("storage/users.json")
    users_file.parent.mkdir(parents=True, exist_ok=True)
    return users_file


def load_users() -> List[Dict[str, Any]]:
    users_file = get_users_file()
    if not users_file.exists():
        users_file.write_text("[]", encoding="utf-8")
    with users_file.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_users(users: List[Dict[str, Any]]) -> None:
    users_file = get_users_file()
    with users_file.open("w", encoding="utf-8") as handle:
        json.dump(users, handle, indent=2)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed_password: str) -> bool:
    return pwd_context.verify(password, hashed_password)


def create_access_token(subject: str, expires_delta: timedelta | None = None) -> str:
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(subject),
        "exp": datetime.now(timezone.utc) + expires_delta,
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[Dict[str, Any]]:
    if credentials is None:
        return None

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials") from exc

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    users = load_users()
    user = next(
        (u for u in users if str(u.get("id")) == str(subject) or u.get("email") == str(subject)),
        None,
    )
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/register", response_model=Dict[str, str])
async def register_user(user: UserCreate):
    users = load_users()

    if any(existing_user["email"] == user.email for existing_user in users):
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = {
        "id": str(len(users) + 1),
        "username": user.username,
        "email": user.email,
        "hashed_password": hash_password(user.password),
    }
    users.append(new_user)
    save_users(users)

    return {"message": "User registered successfully"}


@router.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    users = load_users()

    stored_user = next((u for u in users if u["email"] == user.email), None)
    if stored_user is None or not verify_password(user.password, stored_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(stored_user["id"])
    return Token(access_token=access_token, token_type="bearer")