import jwt
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from config import settings

security_scheme = HTTPBearer()

# Cryptographically secure password hashing using standard library hashlib
SALT = "visionguard_secure_salt_2026"
ITERATIONS = 100000

def get_hashed_password(password: str) -> str:
    """Hashes a password using PBKDF2-HMAC-SHA256."""
    dk = hashlib.pbkdf2_hmac('sha256', password.encode(), SALT.encode(), ITERATIONS)
    return dk.hex()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies that the hashed password matches the hashed plain password."""
    return get_hashed_password(plain_password) == hashed_password

# In-Memory Users Registry (For demo/testing RBAC)
MOCK_USERS = {
    "admin_user": {
        "username": "admin_user",
        "hashed_password": get_hashed_password("admin_pass"),
        "role": "Admin"
    },
    "operator_user": {
        "username": "operator_user",
        "hashed_password": get_hashed_password("operator_pass"),
        "role": "Operator"
    },
    "investigator_user": {
        "username": "investigator_user",
        "hashed_password": get_hashed_password("investigator_pass"),
        "role": "Investigator"
    },
    "auditor_user": {
        "username": "auditor_user",
        "hashed_password": get_hashed_password("auditor_pass"),
        "role": "Auditor"
    }
}

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a secure JWT token encoding claims and expiry using UTC time.time()."""
    to_encode = data.copy()
    expire_seconds = expires_delta.total_seconds() if expires_delta else (settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    expire_timestamp = int(time.time() + expire_seconds)
    to_encode.update({"exp": expire_timestamp})
    
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str) -> Optional[dict]:
    """Decodes and validates a JWT token using current epoch time."""
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        exp = payload.get("exp")
        if exp and time.time() > exp:
            return None
        return payload
    except jwt.PyJWTError:
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> Dict[str, Any]:
    """Dependency that extracts and validates the JWT from HTTP Authorization header."""
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload

class RoleChecker:
    """Dependency injector to enforce Role-Based Access Control on endpoints."""
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles

    def __call__(self, current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role")
        if user_role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {self.allowed_roles}. Your role: '{user_role}'"
            )
        return current_user
