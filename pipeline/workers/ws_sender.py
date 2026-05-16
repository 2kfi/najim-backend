import asyncio
import logging
from starlette.websockets import WebSocketState

from core.config import get_settings
from core.redis_manager import RedisManager
from api.websocket import get_active_connection

logger = logging.getLogger(__name__)


async def response_handler(data: dict) -> None:
    device_id = data.get("device_id", "")
    ws = get_active_connection(device_id)
    if ws is None or ws.client_state != WebSocketState.CONNECTED:
        logger.debug(f"Device {device_id} not connected to this node, skipping")
        return

    audio = data.get("audio", "")
    text = data.get("text", "")
    msg = {"type": "audio_chunk", "audio_data": audio, "text": text}
    try:
        await ws.send_json(msg)
        logger.info(f"Sent response to {device_id}")
    except Exception as e:
        logger.error(f"Failed to send to {device_id}: {e}")


async def process_responses(redis: RedisManager, consumer: str):
    from pipeline.workers.base import BaseWorker

    settings = get_settings()
    worker = BaseWorker(
        redis=redis,
        stream=settings.pipeline.response_stream,
        group=settings.pipeline.consumer_group,
        consumer=consumer,
        handler=response_handler,
        poll_timeout=settings.pipeline.poll_timeout_ms,
        max_retries=1,
    )
    await worker.start()