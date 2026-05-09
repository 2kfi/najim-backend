"""
Najim Backend API - Refactored with security, performance, and observability improvements
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
import wave
import glob
import time
from contextlib import asynccontextmanager
from typing import Any, Optional
from functools import lru_cache

import yaml
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential

from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openai import AsyncOpenAI

from scripts.mcp import MCPWrapper

# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


config = load_config()


class Settings:
    API_HOST: str = config.get("api", {}).get("host", "0.0.0.0")
    API_PORT: int = config.get("api", {}).get("port", 8080)
    BASE_PATH: str = config.get("api", {}).get("base_path", "/api/v1")
    MAX_AUDIO_SIZE_MB: int = config.get("api", {}).get("max_audio_size_mb", 10)
    ALLOWED_AUDIO_TYPES: list[str] = config.get("api", {}).get("allowed_audio_types", 
        ["audio/wav", "audio/mpeg", "audio/ogg"])
    
    AUTH_API_KEYS: dict = config.get("auth", {}).get("api_keys", {})
    DEFAULT_RATE_LIMIT: int = config.get("auth", {}).get("default_rate_limit", 60)
    
    STT_CONFIG: dict = config.get("stt", {})
    MODEL_PATH: str = STT_CONFIG.get("model_path", "models/whisper-medium")
    DEVICE: str = STT_CONFIG.get("device", "cpu")
    COMPUTE_TYPE: str = STT_CONFIG.get("compute_type", "int8")
    STT_TASK: str = STT_CONFIG.get("task", "transcribe")
    BEAM_SIZE: int = STT_CONFIG.get("beam_size", 5)
    VAD_FILTER: bool = STT_CONFIG.get("vad_filter", True)
    VAD_PARAMS: dict = {
        "threshold": STT_CONFIG.get("vad_threshold", 0.5),
        "min_speech_duration_ms": STT_CONFIG.get("vad_min_speech_duration_ms", 250),
        "min_silence_duration_ms": STT_CONFIG.get("vad_min_silence_duration_ms", 200),
    }
    
    TTS_CONFIG: dict = config.get("tts", {})
    LLM_CONFIG: dict = config.get("llm", {})
    LLM_API_URL: str = LLM_CONFIG.get("api_url", "")
    LLM_API_KEY: str = LLM_CONFIG.get("api_key", "")
    LLM_MODEL: str = LLM_CONFIG.get("model", "gpt-4o-mini")
    LLM_TIMEOUT: int = LLM_CONFIG.get("timeout", 120)
    
    MCP_CONFIG: dict = config.get("mcp", {})
    MCP_SERVERS: list = MCP_CONFIG.get("servers", [])
    MCP_SETTINGS: dict = {
        "sse_read_timeout": MCP_CONFIG.get("sse_read_timeout", 300.0),
        "connect_timeout": MCP_CONFIG.get("connect_timeout", 30.0),
        "tool_timeout": MCP_CONFIG.get("tool_timeout", 60.0),
        "max_retries": MCP_CONFIG.get("max_retries", 2),
        "max_tool_loops": MCP_CONFIG.get("max_tool_loops", 5),
    }
    
    OBS_CONFIG: dict = config.get("observability", {})
    LOG_LEVEL: str = OBS_CONFIG.get("log_level", "INFO")
    LOG_FORMAT: str = OBS_CONFIG.get("log_format", "json")
    METRICS_ENABLED: bool = OBS_CONFIG.get("metrics_enabled", True)


settings = Settings()

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging() -> logging.Logger:
    logger = logging.getLogger("najim")
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    if settings.LOG_FORMAT == "json":
        try:
            from pythonjsonlogger import jsonlogger
            handler = logging.StreamHandler()
            handler.setFormatter(jsonlogger.JsonFormatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            ))
            logger.addHandler(handler)
        except ImportError:
            pass
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(handler)
    
    return logger


logger = setup_logging()

# ============================================================================
# METRICS
# ============================================================================

METRICS_AVAILABLE = False
if settings.METRICS_ENABLED:
    try:
        from prometheus_client import Counter, Histogram, Gauge, generate_latest
        
        REQUEST_COUNT = Counter(
            "http_requests_total", "Total HTTP requests",
            ["method", "endpoint", "status"]
        )
        REQUEST_DURATION = Histogram(
            "http_request_duration_seconds", "HTTP request duration",
            ["endpoint"]
        )
        PROCESS_TIME = Histogram(
            "najim_process_duration_seconds", "Audio processing duration",
            ["stage"]
        )
        STT_LANGUAGE = Counter(
            "najim_stt_language_total", "STT detected languages",
            ["language"]
        )
        ACTIVE_REQUESTS = Gauge(
            "najim_active_requests", "Active audio processing requests"
        )
        METRICS_AVAILABLE = True
    except ImportError:
        logger.warning("prometheus-client not available, metrics disabled")


def record_metric(name: str, **labels):
    pass


# ============================================================================
# APP STATE
# ============================================================================

class AppState:
    whisper_model: Optional[WhisperModel] = None
    tts_voices: dict[str, PiperVoice] = {}
    tts_voice_paths: dict[str, str] = {}
    llm_client: Optional[AsyncOpenAI] = None
    mcp_wrapper: Optional[MCPWrapper] = None
    initialized: bool = False
    _llm_cache: dict[str, str] = {}
    
    def get_llm_client(self) -> AsyncOpenAI:
        if self.llm_client is None:
            self.llm_client = AsyncOpenAI(
                base_url=settings.LLM_API_URL,
                api_key=settings.LLM_API_KEY,
                timeout=settings.LLM_TIMEOUT,
                max_retries=0,
            )
        return self.llm_client
    
    def get_tts_voice(self, lang: str) -> PiperVoice:
        if lang not in self.tts_voices:
            model_path = self.tts_voice_paths.get(lang)
            if not model_path:
                raise ValueError(f"No TTS model for language: {lang}")
            use_cuda = get_tts_use_cuda(lang)
            self.tts_voices[lang] = PiperVoice.load(model_path, use_cuda=use_cuda)
            logger.info(f"Lazy loaded TTS voice for '{lang}' from {model_path}")
        return self.tts_voices[lang]


state = AppState()

# ============================================================================
# RATE LIMITING
# ============================================================================

limiter = Limiter(key_func=get_remote_address)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class HealthResponse(BaseModel):
    status: str
    stt_model: str
    stt_device: str
    tts_voices: list[str]
    mcp_servers: list[str]
    llm_url: str


class ErrorResponse(BaseModel):
    error: str
    error_code: str
    request_id: Optional[str] = None


# ============================================================================
# HELPERS
# ============================================================================

def find_onnx_file(folder_path: str) -> str:
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {folder_path}")
    return onnx_files[0]


def get_voice_models() -> dict:
    return {
        lang: find_onnx_file(data.get("local_path", ""))
        for lang, data in settings.TTS_CONFIG.items()
    }


def get_tts_use_cuda(lang: str) -> bool:
    lang_cfg = settings.TTS_CONFIG.get(lang, {})
    use_cuda = lang_cfg.get("use_cuda")
    if use_cuda is not None:
        return use_cuda
    return config.get("settings", {}).get("use_cuda", False)


def get_synthesis_config() -> SynthesisConfig:
    synth_settings = config.get("settings", {})
    return SynthesisConfig(
        volume=synth_settings.get("volume", 0.5),
        length_scale=synth_settings.get("length_scale", 1.0),
        noise_scale=synth_settings.get("noise_scale", 1.0),
        noise_w_scale=synth_settings.get("noise_w_scale", 1.0),
        normalize_audio=synth_settings.get("normalize_audio", False),
    )


syn_config = get_synthesis_config()


# ============================================================================
# STT - with retry logic
# ============================================================================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
async def transcribe_audio_with_retry(audio_path: str, task: str = None) -> tuple:
    """Transcribe audio with retry logic"""
    if task is None:
        task = settings.STT_TASK
    segments, info = await asyncio.to_thread(
        lambda: state.whisper_model.transcribe(
            audio_path,
            beam_size=settings.BEAM_SIZE,
            vad_filter=settings.VAD_FILTER,
            vad_parameters=settings.VAD_PARAMS,
            task=task,
        )
    )
    segments_list = list(segments)
    return segments_list, info.language, info.language_probability


# ============================================================================
# MCP - with fallback
# ============================================================================

async def call_llm_with_mcp_fallback(user_message: str) -> str:
    if not state.mcp_wrapper:
        raise RuntimeError("MCP wrapper not initialized")
    
    try:
        return await state.mcp_wrapper.run_query(user_message)
    except Exception as e:
        logger.warning(f"MCP query failed, using fallback: {e}")
        client = state.get_llm_client()
        response = await client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are a concise voice assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content or ""


# ============================================================================
# TTS - lazy loading
# ============================================================================

async def synthesize_one(text: str, voice: PiperVoice, output_path: str):
    def _synth():
        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)
    await asyncio.to_thread(_synth)
    return output_path


async def synthesize_multiple(texts: list[str], language: str, output_dir: str):
    voice = state.get_tts_voice(language)
    tasks = [
        synthesize_one(text, voice, f"{output_dir}/output_{i}.wav")
        for i, text in enumerate(texts)
    ]
    return await asyncio.gather(*tasks)


# ============================================================================
# FILE HELPERS
# ============================================================================

def combine_wav_files(input_paths: list[str], output_path: str):
    if not input_paths:
        raise ValueError("No input files to combine")
    
    with wave.open(input_paths[0], "rb") as first:
        sample_rate = first.getframerate()
        sample_width = first.getsampwidth()
        channels = first.getnchannels()
        first_data = first.readframes(first.getnframes())
    
    for path in input_paths[1:]:
        with wave.open(path, "rb") as w:
            if w.getframerate() != sample_rate:
                raise ValueError("Sample rate mismatch")
            if w.getsampwidth() != sample_width:
                raise ValueError("Sample width mismatch")
            if w.getnchannels() != channels:
                raise ValueError("Channel mismatch")
            first_data += w.readframes(w.getnframes())
    
    with wave.open(output_path, "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(sample_width)
        out.setframerate(sample_rate)
        out.writeframes(first_data)


async def cleanup_files(*paths):
    for path in paths:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError as e:
            logger.debug(f"Cleanup failed for {path}: {e}")


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global state
    
    logger.info("Loading Whisper model...")
    state.whisper_model = WhisperModel(
        settings.MODEL_PATH, 
        device=settings.DEVICE, 
        compute_type=settings.COMPUTE_TYPE
    )
    logger.info("Whisper model loaded!")
    
    state.tts_voice_paths = get_voice_models()
    logger.info(f"TTS voice paths: {state.tts_voice_paths}")
    
    mcp_servers = []
    if settings.MCP_SERVERS:
        servers_list = []
        for srv in settings.MCP_SERVERS:
            if isinstance(srv, str):
                servers_list.append({"url": srv, "api_key": ""})
            else:
                servers_list.append(srv)
        
        state.mcp_wrapper = MCPWrapper(
            llama_base_url=settings.LLM_API_URL,
            llama_model=settings.LLM_MODEL,
            mcp_servers=servers_list,
            api_key=settings.LLM_API_KEY,
            timeout=settings.LLM_TIMEOUT,
            max_tool_loops=settings.MCP_SETTINGS.get("max_tool_loops", 5),
            max_retries=settings.MCP_SETTINGS.get("max_retries", 2),
            mcp_defaults=settings.MCP_SETTINGS,
        )
        try:
            await state.mcp_wrapper.initialize_servers()
            for mgr in state.mcp_wrapper.mcp_managers:
                if mgr.connected:
                    mcp_servers.append(mgr.url)
        except Exception as e:
            logger.warning(f"MCP initialization failed (will use fallback): {e}")
    
    state.initialized = True
    logger.info("Application initialized successfully")
    
    yield
    
    logger.info("Closing MCP connections...")
    if state.mcp_wrapper:
        await state.mcp_wrapper.close()
    logger.info("Shutdown complete")


# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Najim Backend API",
    version="2.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"error": "Rate limit exceeded", "error_code": "RATE_LIMITED"}
    )


# ============================================================================
# MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def api_key_auth(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", f"{settings.BASE_PATH}/health", f"{settings.BASE_PATH}/metrics"]:
        return await call_next(request)
    
    if not settings.AUTH_API_KEYS:
        return await call_next(request)
    
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {"error": "Missing or invalid Authorization header"},
            status_code=401
        )
    
    api_key = auth_header.replace("Bearer ", "").strip()
    if api_key not in settings.AUTH_API_KEYS:
        logger.warning(f"Unauthorized request with key: {api_key[:8]}...")
        return JSONResponse({"error": "Invalid API key"}, status_code=401)
    
    return await call_next(request)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ============================================================================
# ROUTES
# ============================================================================

@app.get(f"{settings.BASE_PATH}/health", response_model=HealthResponse)
async def health():
    mcp_servers = []
    if state.mcp_wrapper:
        for mgr in state.mcp_wrapper.mcp_managers:
            if mgr.connected:
                mcp_servers.append(mgr.url)
    return {
        "status": "healthy",
        "stt_model": settings.MODEL_PATH,
        "stt_device": settings.DEVICE,
        "tts_voices": list(state.tts_voice_paths.keys()),
        "mcp_servers": mcp_servers,
        "llm_url": settings.LLM_API_URL,
    }


@app.get("/health")
async def health_redirect():
    return await health()


@app.get(f"{settings.BASE_PATH}/metrics")
async def metrics():
    if not METRICS_AVAILABLE:
        return {"error": "Metrics not available"}
    from starlette.responses import Response
    return Response(generate_latest(), media_type="text/plain")


@app.post(f"{settings.BASE_PATH}/process")
@limiter.limit(lambda request: settings.DEFAULT_RATE_LIMIT)
async def process(request: Request, file: UploadFile = File(...), data: str = Form(None)):
    start_time = time.time()
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    
    temp_input = None
    temp_output = None
    temp_dir = None
    
    try:
        if METRICS_AVAILABLE:
            ACTIVE_REQUESTS.inc()
        
        content = await file.read()
        max_size = settings.MAX_AUDIO_SIZE_MB * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file exceeds maximum size of {settings.MAX_AUDIO_SIZE_MB}MB"
            )
        
        if file.content_type not in settings.ALLOWED_AUDIO_TYPES:
            logger.warning(f"Unexpected content type: {file.content_type}")
        
        override_lang = None
        stt_task = None
        if data:
            try:
                payload = json.loads(data)
                override_lang = payload.get("lang")
                stt_task = payload.get("task")
            except json.JSONDecodeError:
                pass
        
        temp_input = f"/tmp/{uuid.uuid4()}_input.wav"
        temp_output = f"/tmp/{uuid.uuid4()}_output.wav"
        temp_dir = f"/tmp/{uuid.uuid4()}"
        
        os.makedirs(temp_dir, exist_ok=True)
        
        with open(temp_input, "wb") as f:
            f.write(content)
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="stt").start_time = time.time()
        
        segments, detected_lang, lang_prob = await transcribe_audio_with_retry(temp_input, task=stt_task)
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="stt").observe(time.time() - start_time)
            STT_LANGUAGE.labels(language=detected_lang).inc()
        
        transcribed_text = " ".join(segment.text for segment in segments)
        logger.info(f"STT: lang={detected_lang}, text='{transcribed_text}'")
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="llm").start_time = time.time()
        
        llm_response = await call_llm_with_mcp_fallback(transcribed_text)
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="llm").observe(time.time() - start_time)
        
        logger.info(f"LLM response: {llm_response}")
        
        tts_lang = override_lang or detected_lang
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="tts").start_time = time.time()
        
        output_files = await synthesize_multiple([llm_response], tts_lang, temp_dir)
        combine_wav_files(list(output_files), temp_output)
        
        if METRICS_AVAILABLE:
            PROCESS_TIME.labels(stage="tts").observe(time.time() - start_time)
        
        total_time = time.time() - start_time
        logger.info(f"Process complete: lang={tts_lang}, time={total_time:.2f}s")
        
        return FileResponse(
            temp_output,
            media_type="audio/wav",
            filename="output.wav",
            background=lambda: cleanup_files(temp_input, temp_output, temp_dir)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Process error: {e}", exc_info=True)
        await cleanup_files(temp_input, temp_output, temp_dir)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if METRICS_AVAILABLE:
            ACTIVE_REQUESTS.dec()


@app.post("/process")
async def process_redirect(request: Request, file: UploadFile = File(...), data: str = Form(None)):
    return await process(request, file, data)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)