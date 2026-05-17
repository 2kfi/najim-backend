import asyncio
import json
import logging
from typing import Any

from core.config import get_settings
from core.redis_manager import RedisManager
from core.schemas import ToolCallResult, ToolResultStatus
from tools.call_client_tool import get_tool_bridge
from tools.internal_tools import run_internal_tool
from tools.registry import get_tool_registry

logger = logging.getLogger(__name__)


class UnknownToolError(Exception):
    pass


async def route_tool_call(
    device_id: str,
    session_id: str,
    tool_name: str,
    params: dict[str, Any],
) -> ToolCallResult:
    settings = get_settings()
    registry = await get_tool_registry()

    if registry.is_internal(tool_name):
        logger.info(f"Routing internal tool: {tool_name}")
        return await run_internal_tool(tool_name, params, timeout=settings.tool.internal_timeout)

    elif registry.is_remote(tool_name):
        from sessions.permissions import PermissionStore
        from core.redis_manager import RedisManager
        
        redis = await RedisManager.get_instance()
        perms = PermissionStore(redis)
        
        if not await perms.has_permission(device_id, tool_name):
            logger.warning(f"Permission denied for tool {tool_name} on device {device_id}")
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=f"Permission denied for tool: {tool_name}",
                duration_ms=0,
            )
        
        logger.info(f"Routing remote tool: {tool_name} to device {device_id}")
        bridge = await get_tool_bridge()
        record = await bridge.initiate_remote_call(
            device_id=device_id,
            tool_name=tool_name,
            params=params,
            session_id=session_id,
            timeout=settings.tool.remote_timeout,
        )
        result = await bridge.await_remote_response(record.correlation_id, timeout=settings.tool.remote_timeout)
        if "error" in result:
            return ToolCallResult(
                tool_name=tool_name,
                success=False,
                error=result["error"],
                duration_ms=0,
            )
        return ToolCallResult(tool_name=tool_name, success=True, result=result, duration_ms=0)

    else:
        raise UnknownToolError(f"Unknown tool: {tool_name}")


async def route_tool_calls_batch(
    device_id: str,
    session_id: str,
    tool_calls: list[dict[str, Any]],
) -> list[ToolCallResult]:
    tasks = []
    for tc in tool_calls:
        tool_name = tc.get("name") or tc.get("function", {}).get("name", "")
        params = tc.get("params") or tc.get("function", {}).get("arguments", {})
        if isinstance(params, str):
            params = json.loads(params)
        tasks.append(route_tool_call(device_id, session_id, tool_name, params))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    output = []
    for r in results:
        if isinstance(r, Exception):
            output.append(ToolCallResult(tool_name="unknown", success=False, error=str(r), duration_ms=0))
        else:
            output.append(r)
    return output