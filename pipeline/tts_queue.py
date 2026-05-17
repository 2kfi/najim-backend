import asyncio
import base64
import logging
import os
import tempfile
import wave

from core.app_state import get_app_state
from core.config import get_settings
from core.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class TTSQueue:
    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()
        self._queue_key = "tts_queue"

    async def enqueue(self, device_id: str, text: str, language: str = None, priority: int = 0) -> int:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        score = loop.time() + priority
        return await self.redis.zadd(self._queue_key, {f"{device_id}:{text[:50]}": score})

    async def dequeue(self, count: int = 1) -> list[tuple[str, str]]:
        items = await self.redis.zrange(self._queue_key, 0, count - 1, withscores=True)
        results = []
        for item, score in items:
            if isinstance(item, bytes):
                item = item.decode()
            parts = item.split(":", 1)
            if len(parts) == 2:
                results.append((parts[0], parts[1]))
            await self.redis.client.zrem(self._queue_key, item)
        return results

    async def synthesize(self, text: str, language: str = None) -> bytes:
        state = get_app_state()
        lang = language or self._settings.tts.default_voice
        voice = state.get_tts_voice(lang)
        if not voice:
            raise RuntimeError(f"TTS voice for language '{lang}' not available")

        syn_config = state.get_synthesis_config()
        output_file = tempfile.mktemp(suffix=".wav")
        try:
            if syn_config:
                await asyncio.to_thread(
                    lambda: voice.synthesize_wav(text, wave.open(output_file, "wb"), syn_config=syn_config)
                )
            else:
                await asyncio.to_thread(voice.execute, text, output_file)
            with open(output_file, "rb") as f:
                audio_data = f.read()
            return audio_data
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    async def synthesize_and_b64(self, text: str, language: str = None) -> str:
        audio = await self.synthesize(text, language)
        return base64.b64encode(audio).decode()