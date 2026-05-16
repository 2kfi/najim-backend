import asyncio
import logging
import time
from typing import Any, Optional

from core.config import get_settings
from core.redis_manager import RedisManager
from core.schemas import ToolCallResult, ToolDefinition
from tools.call_client_tool import ToolBridge, get_tool_bridge

logger = logging.getLogger(__name__)


class ToolRegistry:
    _instance: Optional["ToolRegistry"] = None
    _lock: asyncio.Lock = None

    def __init__(self):
        self._internal: dict[str, ToolDefinition] = {}
        self._remote: dict[str, ToolDefinition] = {}
        self._lock = asyncio.Lock()
        self._register_defaults()

    def _register_defaults(self):
        self._internal["get_time"] = ToolDefinition(
            name="get_time",
            description="Get the current time",
            input_schema={"type": "object", "properties": {}, "required": []},
            is_internal=True,
        )
        self._internal["get_weather"] = ToolDefinition(
            name="get_weather",
            description="Get weather for a location",
            input_schema={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
            is_internal=True,
        )
        self._internal["calculator"] = ToolDefinition(
            name="calculator",
            description="Perform a calculation",
            input_schema={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Math expression"}},
                "required": ["expression"],
            },
            is_internal=True,
        )

    async def register_remote_tool(self, name: str, definition: ToolDefinition) -> None:
        async with self._lock:
            self._remote[name] = definition

    async def register_internal_tool(self, name: str, definition: ToolDefinition) -> None:
        async with self._lock:
            self._internal[name] = definition

    def is_remote(self, tool_name: str) -> bool:
        return tool_name in self._remote

    def is_internal(self, tool_name: str) -> bool:
        return tool_name in self._internal

    def get_all(self) -> dict[str, ToolDefinition]:
        all_tools = {}
        all_tools.update(self._internal)
        all_tools.update(self._remote)
        return all_tools

    def get_remote_tools(self) -> list[dict[str, Any]]:
        return [t.model_dump() for t in self._remote.values()]

    def get_internal_tools(self) -> list[dict[str, Any]]:
        return [t.model_dump() for t in self._internal.values()]

    async def remove_remote_tool(self, name: str) -> bool:
        async with self._lock:
            if name in self._remote:
                del self._remote[name]
                return True
        return False


_registry: Optional[ToolRegistry] = None


async def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry