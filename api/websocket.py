import asyncio
import base64
import json
import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from starlette.websockets import WebSocketState

from core.config import get_settings
from core.jwt_auth import ws_verify
from core.redis_manager import RedisManager, get_redis
from core.schemas import DeviceInfo, DeviceStatus, WSMessage, WSMessageType
from sessions.device_registry import DeviceRegistry
from tools.registry import get_tool_registry
from tools.registry import ToolDefinition

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["websocket"])

_active_connections: dict[str, WebSocket] = {}
_connection_locks: dict[str, asyncio.Lock] = {}


def _get_lock(device_id: str) -> asyncio.Lock:
    if device_id not in _connection_locks:
        _connection_locks[device_id] = asyncio.Lock()
    return _connection_locks[device_id]


async def _start_ws_listener(node_id: str, redis: RedisManager) -> None:
    pubsub = redis.client.pubsub()
    channel = f"najim:ws_send:{node_id}"
    await pubsub.subscribe(channel)
    try:
        async for msg in pubsub.listen():
            if msg["type"] == "message":
                try:
                    data = json.loads(msg["data"])
                    device_id = data.get("device_id")
                    if device_id and device_id in _active_connections:
                        ws = _active_connections[device_id]
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json(data)
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"WS send error: {e}")
    finally:
        await pubsub.unsubscribe(channel)


async def _register_phone_tools(device_id: str, capabilities: list[str], tools_list: list[dict]):
    registry = await get_tool_registry()
    for tool_def in tools_list:
        name = tool_def.get("name", "")
        if name:
            await registry.register_remote_tool(name, ToolDefinition(
                name=name,
                description=tool_def.get("description", ""),
                input_schema=tool_def.get("input_schema", {"type": "object", "properties": {}, "required": []}),
            ))
            logger.info(f"Registered remote tool [{name}] from device {device_id}")


@router.websocket("/connect")
async def ws_connect(websocket: WebSocket, token: Optional[str] = Query(None)):
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    try:
        claims = await ws_verify(token)
    except Exception as e:
        await websocket.close(code=4002, reason=f"Invalid token: {e}")
        return

    device_id = claims.get("device_id")
    user_id = claims.get("user_id")
    if not device_id:
        await websocket.close(code=4003, reason="Missing device_id in token")
        return

    await websocket.accept()

    settings = get_settings()
    redis = await get_redis()
    device_registry = DeviceRegistry(redis)

    capabilities = []
    remote_tools = []
    try:
        caps_raw = await asyncio.wait_for(websocket.receive_json(), timeout=10)
        if caps_raw.get("type") == "connect":
            capabilities = caps_raw.get("capabilities", [])
            remote_tools = caps_raw.get("tools", [])
    except asyncio.TimeoutError:
        logger.warning(f"Device {device_id} did not send connect message within 10s")
    except Exception as e:
        logger.warning(f"Failed to parse connect message from {device_id}: {e}")

    device_info = DeviceInfo(
        device_id=device_id,
        user_id=user_id,
        capabilities=capabilities,
        status=DeviceStatus.ONLINE,
        node_id=settings.cluster.node_id,
    )
    await device_registry.register(device_id, device_info)

    await _register_phone_tools(device_id, capabilities, remote_tools)

    async with _get_lock(device_id):
        _active_connections[device_id] = websocket

    await websocket.send_json({
        "type": "connected",
        "device_id": device_id,
        "node_id": settings.cluster.node_id,
    })

    try:
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=settings.session.heartbeat_interval + 5,
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
                await device_registry.heartbeat(device_id)
                continue

            msg_type = data.get("type")
            if msg_type == "heartbeat":
                await device_registry.heartbeat(device_id)
                await websocket.send_json({"type": "heartbeat_ack", "timestamp": data.get("timestamp")})

            elif msg_type == "tool_response":
                correlation_id = data.get("correlation_id")
                result = data.get("result")
                error = data.get("error")
                from tools.call_client_tool import get_tool_bridge
                bridge = await get_tool_bridge()
                await bridge.handle_response(correlation_id, result, error)

            elif msg_type == "disconnect":
                break

            elif msg_type == "audio":
                await _handle_audio(websocket, device_id, data, redis)

            elif msg_type == "tools_update":
                tools_list = data.get("tools", [])
                await _register_phone_tools(device_id, [], tools_list)
                await websocket.send_json({"type": "tools_updated", "count": len(tools_list)})

            else:
                await websocket.send_json({"type": "error", "message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"Device {device_id} disconnected")
    except Exception as e:
        logger.error(f"WS error for device {device_id}: {e}")
    finally:
        async with _get_lock(device_id):
            _active_connections.pop(device_id, None)
        await device_registry.set_status(device_id, DeviceStatus.OFFLINE)
        await device_registry.unregister(device_id)


async def _handle_audio(websocket: WebSocket, device_id: str, data: dict, redis: RedisManager) -> None:
    audio_b64 = data.get("audio_data", "")
    if not audio_b64:
        await websocket.send_json({"type": "error", "message": "Missing audio_data"})
        return

    settings = get_settings()
    stream_key = settings.pipeline.stt_stream

    await redis.xadd(stream_key, {
        "device_id": device_id,
        "session_id": device_id,
        "audio_data": audio_b64,
        "language": data.get("language", ""),
        "task": data.get("task", ""),
    }, maxlen=1000)

    await websocket.send_json({
        "type": "accepted",
        "message": "Processing started",
    })


async def send_to_device(device_id: str, message: dict) -> bool:
    ws = _active_connections.get(device_id)
    if ws and ws.client_state == WebSocketState.CONNECTED:
        await ws.send_json(message)
        return True
    return False


def get_active_connection(device_id: str) -> Optional[WebSocket]:
    return _active_connections.get(device_id)


def is_device_connected(device_id: str) -> bool:
    ws = _active_connections.get(device_id)
    return ws is not None and ws.client_state == WebSocketState.CONNECTED