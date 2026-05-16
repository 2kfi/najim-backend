import asyncio
import base64
import logging
import os
import tempfile
from pathlib import Path

from core.app_state import get_app_state
from core.config import get_settings
from core.redis_manager import RedisManager

logger = logging.getLogger(__name__)


class STTBuffer:
    def __init__(self, redis: RedisManager):
        self.redis = redis
        self._settings = get_settings()
        self._queue_key = "stt_queue"
        self._result_prefix = "stt_result"

    async def enqueue_audio(self, device_id: str, audio_data: bytes, chunk_index: int = 0, total_chunks: int = 1) -> str:
        audio_b64 = base64.b64encode(audio_data).decode()
        job_id = f"{device_id}:{chunk_index}"
        msg_id = await self.redis.xadd(
            self._queue_key,
            {
                "device_id": device_id,
                "job_id": job_id,
                "audio_data": audio_b64,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            },
            maxlen=1000,
        )
        return msg_id

    async def get_audio_stream(self, device_id: str, timeout: int = 5000):
        stream_key = f"stt_stream:{device_id}"
        result = await self.redis.xread({self._queue_key: "0"}, count=10)
        for item in result:
            if item["data"].get("device_id") == device_id:
                audio_b64 = item["data"].get("audio_data", "")
                if audio_b64:
                    yield base64.b64decode(audio_b64)
        await asyncio.sleep(0.1)

    async def process_top(self, timeout: int = None) -> dict | None:
        if timeout is None:
            timeout = self._settings.tool.internal_timeout * 1000
        results = await self.redis.xread({self._queue_key: "0"}, count=1, block=timeout)
        if not results:
            return None
        item = results[0]
        data = item.get("data", {})
        audio_b64 = data.get("audio_data")
        if not audio_b64:
            return None
        audio_data = base64.b64decode(audio_b64)
        return {
            "device_id": data.get("device_id"),
            "job_id": data.get("job_id"),
            "audio_data": audio_data,
            "chunk_index": data.get("chunk_index", 0),
        }

    async def transcribe(self, audio_data: bytes, language: str = None, task: str = None) -> str:
        state = get_app_state()
        if not state.whisper_model:
            raise RuntimeError("Whisper model not loaded")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_data)
            temp_path = f.name

        try:
            segments, info = await asyncio.to_thread(
                lambda: state.whisper_model.transcribe(
                    temp_path,
                    beam_size=self._settings.stt.beam_size,
                    vad_filter=self._settings.stt.vad_filter,
                    language=language or self._settings.stt.language,
                    task=task,
                )
            )
            text = "".join(segment.text for segment in segments)
            return text.strip()
        finally:
            os.unlink(temp_path)

    async def transcribe_job(self, job_data: dict) -> str:
        audio = job_data.get("audio_data")
        if not audio:
            return ""
        return await self.transcribe(audio)