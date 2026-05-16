from core.redis_manager import RedisManager


class PermissionStore:
    KEY_PREFIX = "perms"

    def __init__(self, redis: RedisManager):
        self.redis = redis

    def _key(self, device_id: str) -> str:
        return f"{self.KEY_PREFIX}:{device_id}"

    async def set_permission(self, device_id: str, tool_name: str, allowed: bool) -> None:
        value = "allow" if allowed else "deny"
        await self.redis.client.hset(self._key(device_id), tool_name, value)

    async def get_permission(self, device_id: str, tool_name: str) -> str:
        val = await self.redis.client.hget(self._key(device_id), tool_name)
        if val is None:
            return "deny"
        return val.decode() if isinstance(val, bytes) else val

    async def has_permission(self, device_id: str, tool_name: str) -> bool:
        perm = await self.get_permission(device_id, tool_name)
        return perm == "allow"

    async def get_all(self, device_id: str) -> dict[str, str]:
        raw = await self.redis.client.hgetall(self._key(device_id))
        if not raw:
            return {}
        out = {}
        for k, v in raw.items():
            out[k.decode() if isinstance(k, bytes) else k] = v.decode() if isinstance(v, bytes) else v
        return out

    async def set_all(self, device_id: str, permissions: dict[str, str]) -> None:
        flat = {k: v for k, v in permissions.items()}
        await self.redis.client.hset(self._key(device_id), mapping=flat)

    async def clear(self, device_id: str) -> None:
        await self.redis.delete(self._key(device_id))

    async def bulk_check(self, device_id: str, tool_names: list[str]) -> dict[str, bool]:
        results = {}
        for name in tool_names:
            results[name] = await self.has_permission(device_id, name)
        return results