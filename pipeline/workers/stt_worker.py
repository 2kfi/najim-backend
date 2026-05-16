import asyncio
import base64
import logging
import os
import tempfile

from core.app_state import get_app_state
from core.config import get_settings
from core.redis_manager import RedisManager

logger = logging.getLogger(__name__)


async def stt_handler(data: dict) -> dict:
    settings = get_settings()
    state = get_app_state()

    audio_b64 = data.get("audio_data", "")
    if not audio_b64:
        raise ValueError("No audio_data in job")

    audio = base64.b64decode(audio_b64)
    device_id = data.get("device_id", "")
    language = data.get("language")
    stt_task = data.get("task")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        temp_path = f.name

    try:
        segments, info = await asyncio.to_thread(
            lambda: state.whisper_model.transcribe(
                temp_path,
                beam_size=settings.stt.beam_size,
                vad_filter=settings.stt.vad_filter,
                language=language or settings.stt.language,
                task=stt_task,
            )
        )
        text = "".join(segment.text for segment in segments).strip()
        logger.info(f"STT [{device_id}]: {text}")
        return {
            "device_id": device_id,
            "session_id": data.get("session_id", device_id),
            "text": text,
            "language": info.language,
            "probability": info.language_probability,
        }
    finally:
        os.unlink(temp_path)


async def process_stt_jobs(redis: RedisManager, consumer: str):
    import asyncio
    from pipeline.workers.base import BaseWorker

    settings = get_settings()
    worker = BaseWorker(
        redis=redis,
        stream=settings.pipeline.stt_stream,
        group=settings.pipeline.consumer_group,
        consumer=consumer,
        handler=stt_handler,
        poll_timeout=settings.pipeline.poll_timeout_ms,
        max_retries=settings.pipeline.stt_max_retries,
    )
    await worker.start()