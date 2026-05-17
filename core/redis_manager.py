import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, Optional
import redis.asyncio as redis
from redis.asyncio.client import Redis
from redis.asyncio.connection import ConnectionPool, SSLConnection

from core.config import Settings, get_settings


class RedisManager:
    _instance: Optional["RedisManager"] = None
    _pool: Optional[ConnectionPool] = None
    _client: Optional[Redis] = None
    _lock: Optional[asyncio.Lock] = None
    _initialized: bool = False

    def __init__(self):
        self._settings = get_settings()

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._lock is None:
            cls._lock = asyncio.Lock()
        return cls._lock

    @classmethod
    async def get_instance(cls) -> "RedisManager":
        if cls._instance is None or not cls._initialized:
            async with cls._get_lock():
                if cls._instance is None:
                    cls._instance = cls()
                if not cls._initialized:
                    await cls._instance._initialize()
        return cls._instance

    async def _initialize(self):
        s = self._settings.redis

        if s.url:
            self._pool = ConnectionPool.from_url(s.url, max_connections=s.pool_size)
        else:
            pool_kwargs = {
                "host": s.host,
                "port": s.port,
                "password": s.password if s.password else None,
                "max_connections": s.pool_size,
                "socket_keepalive": s.socket_keepalive,
                "socket_connect_timeout": s.socket_connect_timeout,
                "health_check_interval": s.health_check_interval,
            }
            if s.tls:
                pool_kwargs["connection_class"] = SSLConnection
            self._pool = ConnectionPool(**pool_kwargs)

        self._client = Redis(connection_pool=self._pool)
        self._initialized = True

    @property
    def client(self) -> Redis:
        if self._client is None:
            raise RuntimeError("RedisManager not initialized")
        return self._client

    async def ping(self) -> bool:
        try:
            await self._client.ping()
            return True
        except Exception:
            return False

    async def hset_json(self, key: str, field: str, value: Any) -> None:
        await self._client.hset(key, field, json.dumps(value))

    async def hget_json(self, key: str, field: str) -> Any:
        raw = await self._client.hget(key, field)
        return json.loads(raw) if raw else None

    async def hset_dict(self, key: str, data: dict[str, Any]) -> None:
        flat = {}
        for k, v in data.items():
            if isinstance(v, dict):
                flat[k] = json.dumps(v)
            elif isinstance(v, str):
                flat[k] = v
            else:
                flat[k] = json.dumps(v)
        await self._client.hset(key, mapping=flat)

    async def hget_all(self, key: str) -> dict[str, Any]:
        raw = await self._client.hgetall(key)
        if not raw:
            return {}
        out = {}
        for k, v in raw.items():
            k_str = k.decode() if isinstance(k, bytes) else k
            if isinstance(v, bytes):
                v_str = v.decode()
            elif isinstance(v, str):
                v_str = v
            else:
                v_str = str(v)
            try:
                out[k_str] = json.loads(v_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                out[k_str] = v_str
        return out

    async def set_with_ttl(self, key: str, value: Any, ttl: int) -> None:
        encoded = json.dumps(value) if not isinstance(value, str) else value
        await self._client.set(key, encoded, ex=ttl)

    async def get(self, key: str) -> Any:
        raw = await self._client.get(key)
        if isinstance(raw, bytes):
            raw_str = raw.decode()
        elif isinstance(raw, str):
            raw_str = raw
        elif raw is None:
            return None
        else:
            raw_str = str(raw)
        try:
            return json.loads(raw_str)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return raw_str

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def exists(self, key: str) -> bool:
        return bool(await self._client.exists(key))

    async def lpush(self, key: str, *values: Any) -> int:
        encoded = [json.dumps(v) if not isinstance(v, str) else v for v in values]
        return await self._client.lpush(key, *encoded)

    async def rpush(self, key: str, *values: Any) -> int:
        encoded = [json.dumps(v) if not isinstance(v, str) else v for v in values]
        return await self._client.rpush(key, *encoded)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list[Any]:
        raw = await self._client.lrange(key, start, end)
        out = []
        for item in raw:
            item_str = item.decode() if isinstance(item, bytes) else (item if isinstance(item, str) else str(item))
            try:
                out.append(json.loads(item_str))
            except (json.JSONDecodeError, UnicodeDecodeError):
                out.append(item_str)
        return out

    async def ltrim(self, key: str, start: int, end: int) -> None:
        await self._client.ltrim(key, start, end)

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        return await self._client.zadd(key, mapping)

    async def zrange(self, key: str, start: int = 0, end: int = -1, withscores: bool = False) -> list[Any]:
        raw = await self._client.zrange(key, start, end, withscores=withscores)
        def _decode(x):
            if isinstance(x, bytes):
                try:
                    return json.loads(x.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return x.decode()
            return x
        if not withscores:
            return [_decode(x) for x in raw]
        return [(_decode(k), s) for k, s in raw]

    async def publish(self, channel: str, message: Any) -> int:
        encoded = json.dumps(message) if not isinstance(message, str) else message
        return await self._client.publish(channel, encoded)

    async def blpop(self, key: str, timeout: int = 0) -> tuple[str, Any]:
        result = await self._client.blpop(key, timeout=timeout)
        if result:
            _, raw = result
            raw_str = raw.decode() if isinstance(raw, bytes) else (raw if isinstance(raw, str) else str(raw))
            try:
                return result[0], json.loads(raw_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return result[0], raw_str
        return None, None

    async def xadd(self, stream: str, fields: dict[str, Any], maxlen: int = 1000) -> str:
        flat: dict[str, str] = {}
        for k, v in fields.items():
            if isinstance(v, str):
                flat[k] = v
            elif isinstance(v, bytes):
                flat[k] = v.decode()
            else:
                flat[k] = json.dumps(v, ensure_ascii=False)
        return await self._client.xadd(stream, flat, maxlen=maxlen)

    async def xread(self, streams: dict[str, str], count: int = 10, block: int = None) -> list[Any]:
        raw = await self._client.xread(streams, count=count, block=block)
        results = []
        for stream_name, messages in raw:
            for msg_id, data in messages:
                decoded = {}
                for k, v in data.items():
                    k_str = k.decode() if isinstance(k, bytes) else k
                    if isinstance(v, bytes):
                        v_str = v.decode()
                    elif isinstance(v, str):
                        v_str = v
                    else:
                        v_str = str(v)
                    try:
                        decoded[k_str] = json.loads(v_str)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        decoded[k_str] = v_str
                results.append({
                    "stream": stream_name.decode() if isinstance(stream_name, bytes) else stream_name,
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    "data": decoded,
                })
        return results

    async def expire(self, key: str, ttl: int) -> None:
        await self._client.expire(key, ttl)

    async def incr(self, key: str) -> int:
        return await self._client.incr(key)

    async def xgroup_create(self, stream: str, group: str, id: str = "$", mkstream: bool = True) -> None:
        try:
            await self._client.xgroup_create(stream, group, id=id, mkstream=mkstream)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def xreadgroup(
        self,
        group: str,
        consumer: str,
        streams: dict[str, str],
        count: int = 1,
        block: int = 5000,
        noack: bool = False,
    ) -> list[Any]:
        raw = await self._client.xreadgroup(
            group, consumer, streams, count=count, block=block, noack=noack
        )
        results = []
        for stream_name, messages in raw:
            for msg_id, data in messages:
                decoded = {}
                for k, v in data.items():
                    k_str = k.decode() if isinstance(k, bytes) else k
                    if isinstance(v, bytes):
                        v_str = v.decode()
                    elif isinstance(v, str):
                        v_str = v
                    else:
                        v_str = str(v)
                    try:
                        decoded[k_str] = json.loads(v_str)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        decoded[k_str] = v_str
                results.append({
                    "stream": stream_name.decode() if isinstance(stream_name, bytes) else stream_name,
                    "id": msg_id.decode() if isinstance(msg_id, bytes) else msg_id,
                    "data": decoded,
                })
        return results

    async def xack(self, stream: str, group: str, *ids: str) -> int:
        return await self._client.xack(stream, group, *ids)

    async def xlen(self, stream: str) -> int:
        return await self._client.xlen(stream)

    async def xpending(self, stream: str, group: str) -> dict[str, Any]:
        raw = await self._client.xpending(stream, group)
        if raw:
            return {
                "pending": raw[0],
                "min_id": raw[1],
                "max_id": raw[2],
                "consumers": raw[3],
            }
        return {"pending": 0, "min_id": None, "max_id": None, "consumers": []}

    async def xpending_detail(self, stream: str, group: str, msg_id: str) -> Optional[dict[str, Any]]:
        raw = await self._client.xpending_range(stream, group, min="-", max="+", count=10)
        for entry in raw:
            if entry["message_id"] == msg_id:
                return {
                    "message_id": entry["message_id"],
                    "delivery_count": entry["times_delivered"],
                }
        return None

    async def close(self):
        try:
            if self._client:
                await self._client.aclose()
        finally:
            if self._pool:
                await self._pool.disconnect()


async def get_redis() -> RedisManager:
    return await RedisManager.get_instance()