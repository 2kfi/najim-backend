import asyncio
import logging
from datetime import datetime
from typing import Any

from core.schemas import ToolCallResult

logger = logging.getLogger(__name__)


async def run_internal_tool(tool_name: str, params: dict[str, Any], timeout: float = 10.0) -> ToolCallResult:
    start = asyncio.get_event_loop().time()
    try:
        if tool_name == "get_time":
            result = await _get_time(params)
        elif tool_name == "get_weather":
            result = await _get_weather(params)
        elif tool_name == "calculator":
            result = await _calculator(params)
        else:
            return ToolCallResult(tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}", duration_ms=0)

        duration = int((asyncio.get_event_loop().time() - start) * 1000)
        return ToolCallResult(tool_name=tool_name, success=True, result=result, duration_ms=duration)
    except Exception as e:
        duration = int((asyncio.get_event_loop().time() - start) * 1000)
        logger.error(f"Internal tool {tool_name} failed: {e}")
        return ToolCallResult(tool_name=tool_name, success=False, error=str(e), duration_ms=duration)


async def _get_time(params: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(0.01)
    now = datetime.utcnow()
    return {
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "iso": now.isoformat(),
        "timezone": "UTC",
    }


async def _get_weather(params: dict[str, Any]) -> dict[str, Any]:
    location = params.get("location", "unknown")
    await asyncio.sleep(0.05)
    return {
        "location": location,
        "temperature": 22,
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_speed": 12,
        "units": "metric",
    }


async def _calculator(params: dict[str, Any]) -> dict[str, Any]:
    expression = params.get("expression", "0")
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError("Invalid characters in expression")
    result = eval(expression)
    return {"expression": expression, "result": result, "type": typeof(result)}


def typeof(v):
    return type(v).__name__