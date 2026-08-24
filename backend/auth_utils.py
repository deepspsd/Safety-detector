"""
auth_utils.py — JWT + password hashing

Uses bcrypt directly (not passlib) to avoid the known
passlib 1.7.4 + bcrypt 4.x incompatibility that causes
AttributeError on password hash/verify, leading to HTTP 500.

Standard bcrypt hashes ($2b$...) are interoperable: any
existing hashes created by passlib will still verify correctly.
"""

from datetime import datetime, timedelta
from typing import Optional

import bcrypt
from jose import JWTError, jwt

from config import settings


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the stored bcrypt hash."""
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Hash a plain-text password with bcrypt and return the hash string."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Optional[dict]:
    try:
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except JWTError:
        return None
