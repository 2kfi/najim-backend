from datetime import datetime, timedelta
from typing import Any, Optional
from fastapi import HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

from core.config import get_settings
from core.schemas import DeviceInfo


class TokenPayload(BaseModel):
    user_id: str
    device_id: str
    permissions: list[str] = []
    exp: datetime
    iat: datetime


class JWTManager:
    def __init__(self):
        self._settings = get_settings()

    @property
    def secret(self) -> str:
        s = self._settings.jwt.secret
        if not s:
            raise RuntimeError("JWT_SECRET not configured")
        return s

    @property
    def algorithm(self) -> str:
        return self._settings.jwt.algorithm

    @property
    def expiry_minutes(self) -> int:
        return self._settings.jwt.expiry_minutes

    def create_token(self, user_id: str, device_id: str, permissions: list[str] = None) -> str:
        now = datetime.utcnow()
        payload = {
            "user_id": user_id,
            "device_id": device_id,
            "permissions": permissions or [],
            "iat": now.timestamp(),
            "exp": (now + timedelta(minutes=self.expiry_minutes)).timestamp(),
        }
        return jwt.encode(payload, self.secret, algorithm=self.algorithm)

    def verify_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(token, self.secret, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")

    def decode_payload(self, token: str) -> TokenPayload:
        raw = self.verify_token(token)
        return TokenPayload(
            user_id=raw["user_id"],
            device_id=raw["device_id"],
            permissions=raw.get("permissions", []),
            exp=datetime.fromtimestamp(raw["exp"]),
            iat=datetime.fromtimestamp(raw["iat"]),
        )


_jwt_manager: Optional[JWTManager] = None


def get_jwt_manager() -> JWTManager:
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


security = HTTPBearer(auto_error=False)


async def verify_jwt(credentials: Optional[HTTPAuthorizationCredentials] = None, token: Optional[str] = Query(None)) -> dict[str, Any]:
    t = None
    if credentials:
        t = credentials.credentials
    elif token:
        t = token
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    manager = get_jwt_manager()
    return manager.verify_token(t)


async def ws_verify(token: str) -> dict[str, Any]:
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    manager = get_jwt_manager()
    return manager.verify_token(token)