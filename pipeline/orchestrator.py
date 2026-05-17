import asyncio
import logging
from typing import Optional

from core.config import get_settings
from core.redis_manager import RedisManager
from sessions.conversation_store import ConversationStore
from sessions.session_registry import SessionRegistry

logger = logging.getLogger(__name__)


class WorkerManager:
    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()
        self._tasks: list[asyncio.Task] = []
        self._running = False
        self._consumer_id: Optional[str] = None

    async def start_all(self):
        if self._running:
            return
        self._running = True

        node = self._settings.cluster.node_id
        prefix = self._settings.pipeline.consumer_prefix
        self._consumer_id = f"{prefix}:{node}"

        from pipeline.workers.stt_worker import process_stt_jobs
        from pipeline.workers.llm_worker import process_llm_jobs
        from pipeline.workers.tts_worker import process_tts_jobs
        from pipeline.workers.ws_sender import process_responses

        stage_configs = [
            ("STT", self._settings.pipeline.stt_workers, process_stt_jobs, "stt"),
            ("LLM", self._settings.pipeline.llm_workers, process_llm_jobs, "llm"),
            ("TTS", self._settings.pipeline.tts_workers, process_tts_jobs, "tts"),
            ("WS", self._settings.pipeline.ws_workers, process_responses, "ws"),
        ]

        for name, count, fn, suffix in stage_configs:
            for i in range(count):
                consumer = f"{self._consumer_id}:{suffix}-{i}"
                coro = fn(self.redis, consumer)
                task = asyncio.create_task(coro, name=f"worker:{name}-{i}")
                self._tasks.append(task)
                logger.info(f"Started {name} worker {i+1}/{count} [{consumer}]")

    async def stop_all(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        logger.info("All workers stopped")