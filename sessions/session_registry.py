from datetime import datetime
from typing import Optional
from core.redis_manager import RedisManager
from core.schemas import SessionData, SessionConfig
from core.config import get_settings


class SessionRegistry:
    KEY_PREFIX = "session"

    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()

    def _key(self, device_id: str) -> str:
        return f"{self.KEY_PREFIX}:{device_id}"

    async def create(self, device_id: str, user_id: str, config: Optional[SessionConfig] = None) -> SessionData:
        now = datetime.utcnow()
        session = SessionData(
            device_id=device_id,
            user_id=user_id,
            created_at=now,
            last_active=now,
            config=config or SessionConfig(),
        )
        data = {
            "device_id": device_id,
            "user_id": user_id,
            "created_at": now.isoformat(),
            "last_active": now.isoformat(),
            "config": session.config.model_dump_json(),
            "status": "active",
            "language": config.language if config else None,
        }
        await self.redis.hset_dict(self._key(device_id), data)
        await self.redis.expire(self._key(device_id), self._settings.session.ttl_seconds)
        return session

    async def get(self, device_id: str) -> Optional[SessionData]:
        raw = await self.redis.hget_all(self._key(device_id))
        if not raw:
            return None
        return SessionData(
            device_id=raw["device_id"],
            user_id=raw["user_id"],
            created_at=datetime.fromisoformat(raw["created_at"]),
            last_active=datetime.fromisoformat(raw["last_active"]),
            config=SessionConfig.model_validate_json(raw.get("config", "{}")),
            status=raw.get("status", "active"),
            language=raw.get("language"),
        )

    async def exists(self, device_id: str) -> bool:
        return await self.redis.exists(self._key(device_id))

    async def touch(self, device_id: str) -> None:
        await self.redis.hset_json(self._key(device_id), "last_active", datetime.utcnow().isoformat())
        await self.redis.expire(self._key(device_id), self._settings.session.ttl_seconds)

    async def update_config(self, device_id: str, config: SessionConfig) -> None:
        await self.redis.hset_json(self._key(device_id), "config", config.model_dump_json())
        await self.redis.hset_json(self._key(device_id), "language", config.language)
        await self.touch(device_id)

    async def set_status(self, device_id: str, status: str) -> None:
        await self.redis.hset_json(self._key(device_id), "status", status)
        await self.touch(device_id)

    async def delete(self, device_id: str) -> None:
        await self.redis.delete(self._key(device_id))

    async def list_all(self) -> list[SessionData]:
        client = self.redis.client
        keys = []
        async for key in client.scan_iter(match="session:*"):
            keys.append(key.decode() if isinstance(key, bytes) else key)
        sessions = []
        for key in keys:
            device_id = key.replace("session:", "")
            session = await self.get(device_id)
            if session:
                sessions.append(session)
        return sessions

    async def list_by_user(self, user_id: str) -> list[SessionData]:
        all_sessions = await self.list_all()
        return [s for s in all_sessions if s.user_id == user_id]