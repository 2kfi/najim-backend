import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Optional

from core.config import get_settings
from core.redis_manager import RedisManager
from core.schemas import ToolCallRecord, ToolCallResult, ToolResultStatus

logger = logging.getLogger(__name__)


class ToolBridge:
    RESPONSE_KEY_PREFIX = "tool_resp"
    CORRELATION_KEY_PREFIX = "tool_corr"
    MAX_POLL_ITERATIONS = 100

    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()

    def _resp_key(self, correlation_id: str) -> str:
        return f"{self.RESPONSE_KEY_PREFIX}:{correlation_id}"

    def _corr_key(self, correlation_id: str) -> str:
        return f"{self.CORRELATION_KEY_PREFIX}:{correlation_id}"

    async def initiate_remote_call(
        self,
        device_id: str,
        tool_name: str,
        params: dict[str, Any],
        session_id: str,
        timeout: float = None,
    ) -> ToolCallRecord:
        if timeout is None:
            timeout = self._settings.tool.remote_timeout

        correlation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        record = ToolCallRecord(
            correlation_id=correlation_id,
            tool_name=tool_name,
            params=params,
            device_id=device_id,
            session_id=session_id,
            initiated_at=now,
            status="pending",
        )

        await self.redis.set_with_ttl(
            self._corr_key(correlation_id),
            {"device_id": device_id, "tool_name": tool_name, "status": "pending"},
            60
        )

        from sessions.device_registry import DeviceRegistry
        registry = DeviceRegistry(self.redis)
        phone_node = await registry.get_node_for_device(device_id)
        if not phone_node:
            logger.warning(f"Device {device_id} not found in registry")
            phone_node = self._settings.cluster.node_id
        pub_channel = f"najim:ws_send:{phone_node}"
        ws_message = {
            "type": "tool_request",
            "correlation_id": correlation_id,
            "device_id": device_id,
            "tool_name": tool_name,
            "params": params,
            "timestamp": now.isoformat(),
        }
        await self.redis.publish(pub_channel, ws_message)
        logger.info(f"Tool call {correlation_id}: {tool_name} on {device_id}")

        return record

    async def await_remote_response(self, correlation_id: str, timeout: float) -> Any:
        resp_key = self._resp_key(correlation_id)
        try:
            result = await asyncio.wait_for(
                self._poll_response(resp_key), timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            logger.warning(f"Tool call {correlation_id} timed out after {timeout}s")
            await self.redis.delete(resp_key)
            return {"error": "Tool call timed out"}
        except Exception as e:
            logger.error(f"Tool call {correlation_id} error: {e}")
            return {"error": str(e)}

    async def _poll_response(self, resp_key: str) -> Any:
        iterations = 0
        while iterations < self.MAX_POLL_ITERATIONS:
            key, value = await self.redis.blpop(resp_key, timeout=2)
            if value is not None:
                return value
            iterations += 1
            await asyncio.sleep(0.1)
        raise TimeoutError(f"Poll response exceeded max iterations for {resp_key}")

    async def handle_response(self, correlation_id: str, result: Any, error: Optional[str] = None) -> None:
        resp_key = self._resp_key(correlation_id)
        corr_key = self._corr_key(correlation_id)
        payload = result if error is None else {"error": error, "result": result}
        await self.redis.rpush(resp_key, payload)
        await self.redis.expire(resp_key, 60)
        await self.redis.delete(corr_key)


_bridge: Optional[ToolBridge] = None
_bridge_lock: asyncio.Lock = asyncio.Lock()


async def get_tool_bridge() -> ToolBridge:
    global _bridge
    if _bridge is None:
        async with _bridge_lock:
            if _bridge is None:
                redis = await RedisManager.get_instance()
                _bridge = ToolBridge(redis)
    return _bridge