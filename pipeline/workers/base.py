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
        target_stream: Optional[str] = None,
        backoff_base: float = 1.0,
    ):
        self.redis = redis
        self.stream = stream
        self.group = group
        self.consumer = consumer
        self.handler = handler
        self.poll_timeout = poll_timeout
        self.max_retries = max_retries
        self.target_stream = target_stream
        self.backoff_base = backoff_base
        self._running = False

    async def start(self):
        self._running = True
        await self.redis.xgroup_create(self.stream, self.group)
        logger.info(f"Worker {self.consumer} starting on stream {self.stream}")
        while self._running:
            try:
                result = await self._process_one()
                if result:
                    keys = list(result.keys())
                    logger.debug(f"Worker {self.consumer} got result: keys={keys}")
                    if self.target_stream:
                        await self.redis.xadd(self.target_stream, result, maxlen=1000)
                        logger.info(f"Forwarded to {self.target_stream}: {len(result)} fields")
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
            delivery_count = 0
            if pending_detail:
                delivery_count = pending_detail.get("delivery_count", 0)
            if delivery_count >= self.max_retries:
                logger.warning(f"Discarding {msg_id} after {delivery_count} attempts")
                await self.redis.xack(self.stream, self.group, msg_id)
            else:
                backoff_time = self.backoff_base * (2 ** delivery_count)
                logger.info(f"Retrying {msg_id} after {backoff_time}s (attempt {delivery_count + 1}/{self.max_retries})")
                await asyncio.sleep(backoff_time)
            return None