import asyncio
import json
import logging

from core.config import get_settings
from core.redis_manager import RedisManager
from pipeline.llm_runner import LLMRunner

logger = logging.getLogger(__name__)


async def llm_handler(data: dict) -> dict:
    redis = await RedisManager.get_instance()
    from sessions.conversation_store import ConversationStore
    conv_store = ConversationStore(redis)

    device_id = data.get("device_id", "")
    session_id = data.get("session_id", device_id)
    text = data.get("text", "")
    language = data.get("language", "")

    runner = LLMRunner(redis, conv_store)

    if not text:
        logger.warning(f"LLM [{device_id}]: empty transcription text, skipping")
        return {"device_id": device_id, "session_id": session_id, "text": "", "response": "", "language": language}

    response = await runner.run_query(session_id, text)
    response_preview = response[:100] if response else "(empty)"
    logger.info(f"LLM [{device_id}]: response ({len(response)} chars) -> {response_preview}")

    return {
        "device_id": device_id,
        "session_id": session_id,
        "input_text": text,
        "response": response,
        "language": language,
    }


async def process_llm_jobs(redis: RedisManager, consumer: str):
    from pipeline.workers.base import BaseWorker

    settings = get_settings()
    worker = BaseWorker(
        redis=redis,
        stream=settings.pipeline.llm_stream,
        group=settings.pipeline.consumer_group,
        consumer=consumer,
        handler=llm_handler,
        poll_timeout=settings.pipeline.poll_timeout_ms,
        max_retries=settings.pipeline.llm_max_retries,
        target_stream=settings.pipeline.tts_stream,
    )
    await worker.start()