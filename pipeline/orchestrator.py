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

        workers = [
            ("STT", process_stt_jobs(self.redis, f"{self._consumer_id}:stt")),
            ("LLM", process_llm_jobs(self.redis, f"{self._consumer_id}:llm")),
            ("TTS", process_tts_jobs(self.redis, f"{self._consumer_id}:tts")),
            ("WS", process_responses(self.redis, f"{self._consumer_id}:ws")),
        ]

        for name, coro in workers:
            task = asyncio.create_task(coro, name=f"worker:{name}")
            self._tasks.append(task)
            logger.info(f"Started {name} worker [{self._consumer_id}]")

    async def stop_all(self):
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
            self._tasks.clear()
        logger.info("All workers stopped")