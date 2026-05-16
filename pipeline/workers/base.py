import asyncio
import logging
from typing import Any, Callable, Optional

from core.config import get_settings
from core.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class BaseWorker:
    def __init__(
        self,
        redis: RedisManager,
        stream: str,
        group: str,
        consumer: str,
        handler: Callable,
        poll_timeout: int = 5000,
        max_retries: int = 3,
    ):
        self.redis = redis
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.handler = handler
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self._running = False

    async def start(self):
        self._running = True
        await self.redis.xgroup_create(self.stream, self.group)
        logger.info(f"Worker {self.consumer} starting on stream {self.stream}")
        while self._running:
            try:
                await self._process_one()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {self.consumer} error: {e}", exc_info=True)
                await asyncio.sleep(1)

    def stop(self):
        self._running = False

    async def _process_one(self):
        messages = await self.redis.xreadgroup(
            group=self.group,
            consumer=self.consumer,
            streams={self.stream: ">"},
            count=1,
            block=self.poll_timeout,
        )
        if not messages:
            return

        msg = messages[0]
        msg_id = msg["id"]
        data = msg["data"]

        try:
            result = await self.handler(data)
            await self.redis.xack(self.stream, self.group, msg_id)
            return result
        except Exception as e:
            logger.error(f"Handler failed on {msg_id}: {e}")
            pending_detail = await self.redis.xpending_detail(self.stream, self.group, msg_id)
            delivery_count = pending_detail.get("delivery_count", 0) if pending_detail else 0
            if delivery_count >= self.max_retries:
                logger.warning(f"Discarding {msg_id} after {delivery_count} attempts")
                await self.redis.xack(self.stream, self.group, msg_id)
            return None