import logging
import time
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from core.app_state import get_app_state
from core.jwt_auth import verify_jwt
from core.redis_manager import get_redis
from core.schemas import HealthResponse
from core.config import get_settings
from sessions.device_registry import DeviceRegistry

logger = logging.getLogger(__name__)

_start_time = time.time()

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    redis = await get_redis()
    redis_ok = await redis.ping()

    state = get_app_state()
    whisper_ok = state.whisper_model is not None
    tts_voices = list(state.tts_voice_paths.keys())

    device_reg = DeviceRegistry(redis)
    connected_devices = await device_reg.count()

    return HealthResponse(
        status="healthy" if (redis_ok and whisper_ok) else "degraded",
        node_id=get_settings().cluster.node_id,
        redis=redis_ok,
        whisper_model=whisper_ok,
        tts_voices=tts_voices,
        uptime=time.time() - _start_time,
        connected_devices=connected_devices,
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics(claims: dict = Depends(verify_jwt)):
    redis = await get_redis()
    device_reg = DeviceRegistry(redis)
    connected = await device_reg.count()

    state = get_app_state()
    whisper_loaded = state.whisper_model is not None

    output = []
    output.append("# HELP najim_connected_devices Number of connected Android devices")
    output.append("# TYPE najim_connected_devices gauge")
    output.append(f"najim_connected_devices {connected}")
    output.append("# HELP najim_whisper_model_loaded Whisper model load status")
    output.append("# TYPE najim_whisper_model_loaded gauge")
    output.append(f"najim_whisper_model_loaded {1 if whisper_loaded else 0}")
    output.append("# HELP najim_uptime_seconds Node uptime in seconds")
    output.append("# TYPE najim_uptime_seconds counter")
    output.append(f"najim_uptime_seconds {time.time() - _start_time:.2f}")
    output.append("# HELP najim_tts_voices_loaded Number of TTS voices loaded")
    output.append("# TYPE najim_tts_voices_loaded gauge")
    output.append(f"najim_tts_voices_loaded {len(state.tts_voice_paths)}")

    return "\n".join(output)


@router.get("/ready")
async def readiness():
    redis = await get_redis()
    if not await redis.ping():
        return {"ready": False, "reason": "Redis unavailable"}
    state = get_app_state()
    if not state.initialized:
        return {"ready": False, "reason": "Models not loaded"}
    return {"ready": True}


@router.get("/live")
async def liveness():
    return {"alive": True}