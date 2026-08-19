"""Authentication endpoints and helpers for the Cara backend."""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext

from backend.database.repository import create_repository
from shared.schemas import Token, UserCreate, UserLogin

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

SECRET_KEY = "change-me-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


_repository = create_repository()

async def load_users() -> List[Dict[str, Any]]:
    return await _repository.list_users()


async def save_users(users: List[Dict[str, Any]]) -> None:
    # Replace entire users file/collection
    # For JSON backend, list_users + save_user for each user is used by callers.
    # Keep a simple implementation that writes each user via save_user.
    # Clear existing by writing via repository (JSON backend will overwrite file when saving users sequentially).
    # First remove all existing users (only JSON backend supports replace via file in our implementation).
    existing = await _repository.list_users()
    # naive replace: save each provided user (they should contain ids when relevant)
    for u in users:
        await _repository.save_user(u)


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


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Optional[Dict[str, Any]]:
    # Make this dependency async so it can query the storage backend.
    if credentials is None:
        return None

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials") from exc

    subject = payload.get("sub")
    if not subject:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    # subject is usually the user id; try to load by id first, then by email
    user = None
    user = await _repository.get_user(user_id=subject)

    if user is None:
        user = await _repository.get_user(email=str(subject))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")

    return user


@router.post("/register", response_model=Dict[str, str])
async def register_user(user: UserCreate):
    # Check existing by email
    existing = await _repository.get_user(email=user.email)
    if existing is not None:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = {
        "username": user.username,
        "email": user.email,
        "hashed_password": hash_password(user.password),
    }
    user_id = await _repository.save_user(new_user)

    return {"message": "User registered successfully", "id": user_id}


@router.post("/login", response_model=Token)
async def login_user(user: UserLogin):
    stored_user = await _repository.get_user(email=user.email)
    if stored_user is None or not verify_password(user.password, stored_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(stored_user["id"])
    return Token(access_token=access_token, token_type="bearer")