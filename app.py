import asyncio
import logging
import os
import uuid
import wave
import glob
import yaml
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openai import AsyncOpenAI
from scripts.mcp import MCPWrapper


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    
    with open(config_path) as f:
        return yaml.safe_load(f)


config = load_config()


def get_stt_config() -> dict:
    return config.get("stt", {})


def get_tts_config() -> dict:
    return config.get("tts", {})


def get_synthesis_settings() -> dict:
    return config.get("settings", {})


def get_api_config() -> dict:
    return config.get("api", {})


def get_auth_config() -> dict:
    return config.get("auth", {})


def get_llm_config() -> dict:
    return config.get("llm", {})


def get_mcp_config() -> dict:
    return config.get("mcp", {})


def get_mcp_servers() -> list[dict]:
    """Get MCP servers list with url and api_key."""
    mcp = get_mcp_config()
    servers = mcp.get("servers", [])
    if not servers:
        return []
    if isinstance(servers[0], str):
        return [{"url": s, "api_key": ""} for s in servers]
    return servers


def get_mcp_settings() -> dict:
    """Get MCP timing and other settings."""
    mcp = get_mcp_config()
    return {
        "sse_read_timeout": mcp.get("sse_read_timeout", 300.0),
        "connect_timeout": mcp.get("connect_timeout", 30.0),
        "tool_timeout": mcp.get("tool_timeout", 60.0),
        "max_retries": mcp.get("max_retries", 2),
        "max_tool_loops": mcp.get("max_tool_loops", 5),
    }


def find_onnx_file(folder_path: str) -> str:
    onnx_files = glob.glob(os.path.join(folder_path, "*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx file found in {folder_path}")
    return onnx_files[0]


def get_voice_models() -> dict:
    tts = get_tts_config()
    return {
        lang: find_onnx_file(data.get("local_path", ""))
        for lang, data in tts.items()
    }


def get_tts_use_cuda(lang: str) -> bool:
    tts = get_tts_config()
    lang_cfg = tts.get(lang, {})
    use_cuda = lang_cfg.get("use_cuda")
    if use_cuda is not None:
        return use_cuda
    return settings.get("use_cuda", False)


stt_cfg = get_stt_config()
tts_cfg = get_tts_config()
settings = get_synthesis_settings()
api_cfg = get_api_config()
auth_cfg = get_auth_config()
llm_cfg = get_llm_config()
mcp_cfg = get_mcp_config()

AUTH_API_KEYS = auth_cfg.get("api_keys", {})

LLM_API_URL = llm_cfg.get("api_url", "")
LLM_API_KEY = llm_cfg.get("api_key", "")
LLM_MODEL = llm_cfg.get("model", "gpt-4o-mini")
LLM_TIMEOUT = llm_cfg.get("timeout", 120)

MCP_SERVERS = get_mcp_servers()
MCP_SETTINGS = get_mcp_settings()

VOICE_MODELS = get_voice_models()

MODEL_PATH = stt_cfg.get("model_path", "models/whisper-medium")
DEVICE = stt_cfg.get("device", "cpu")
COMPUTE_TYPE = stt_cfg.get("compute_type", "int8")
BEAM_SIZE = stt_cfg.get("beam_size", 5)
VAD_FILTER = stt_cfg.get("vad_filter", True)
VAD_PARAMS = {
    "threshold": stt_cfg.get("vad_threshold", 0.5),
    "min_speech_duration_ms": stt_cfg.get("vad_min_speech_duration_ms", 250),
    "min_silence_duration_ms": stt_cfg.get("vad_min_silence_duration_ms", 200),
}

syn_config = SynthesisConfig(
    volume=settings.get("volume", 0.5),
    length_scale=settings.get("length_scale", 1.0),
    noise_scale=settings.get("noise_scale", 1.0),
    noise_w_scale=settings.get("noise_w_scale", 1.0),
    normalize_audio=settings.get("normalize_audio", False),
)

TTS_VOICES = {}
mcp_wrapper: Optional[MCPWrapper] = None

logger.info(f"Loading Whisper model: {MODEL_PATH}")
whisper_model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
logger.info("Whisper model loaded!")


def preload_voices():
    for lang, model_path in VOICE_MODELS.items():
        use_cuda = get_tts_use_cuda(lang)
        TTS_VOICES[lang] = PiperVoice.load(model_path, use_cuda=use_cuda)
        logger.info(f"TTS voice for '{lang}' loaded from {model_path} (use_cuda={use_cuda})")


async def transcribe_audio(audio_path: str):
    segments, info = await asyncio.to_thread(
        lambda: whisper_model.transcribe(
            audio_path,
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            vad_parameters=VAD_PARAMS,
        )
    )
    segments_list = list(segments)
    return segments_list, info.language, info.language_probability


def mcp_tools_to_openai_tools(mcp_tools: list) -> list:
    tools = []
    for tool in mcp_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        })
    return tools


async def call_llm_with_mcp(user_message: str) -> str:
    if not mcp_wrapper:
        raise RuntimeError("MCP wrapper not initialized")
    return await mcp_wrapper.run_query(user_message)


async def _synthesize_one(text: str, voice: PiperVoice, output_path: str):
    def _synth():
        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file, syn_config=syn_config)

    await asyncio.to_thread(_synth)
    return output_path


async def synthesize_multiple(texts: list[str], language: str, output_dir: str):
    voice = TTS_VOICES.get(language)
    if not voice:
        raise ValueError(f"No voice model for language: {language}")

    tasks = [
        _synthesize_one(text, voice, f"{output_dir}/output_{i}.wav")
        for i, text in enumerate(texts)
    ]
    return await asyncio.gather(*tasks)


async def process_audio(audio_path: str, texts_to_synthesize: list[str], output_dir: str):
    segments, detected_lang, lang_prob = await transcribe_audio(audio_path)

    logger.info(f"Detected language: {detected_lang} (probability: {lang_prob:.2f})")

    full_text = " ".join(segment.text for segment in segments)
    logger.info(f"Transcribed: {full_text}")

    audio_files = await synthesize_multiple(texts_to_synthesize, detected_lang, output_dir)
    return audio_files


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
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global mcp_wrapper

    preload_voices()
    logger.info("All TTS voices preloaded on startup")

    mcp_wrapper = MCPWrapper(
        llama_base_url=LLM_API_URL,
        llama_model=LLM_MODEL,
        mcp_servers=MCP_SERVERS,
        api_key=LLM_API_KEY,
        timeout=LLM_TIMEOUT,
        max_tool_loops=MCP_SETTINGS.get("max_tool_loops", 5),
        max_retries=MCP_SETTINGS.get("max_retries", 2),
        mcp_defaults=MCP_SETTINGS,
    )
    await mcp_wrapper.initialize_servers()

    yield

    logger.info("Closing MCP connections...")
    if mcp_wrapper:
        await mcp_wrapper.close()
    logger.info("Shutting down")


app = FastAPI(title="Najim Backend API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def api_key_auth(request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    if not AUTH_API_KEYS:
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(
            {"error": "Missing or invalid Authorization header"},
            status_code=401
        )

    api_key = auth_header.replace("Bearer ", "").strip()
    if api_key not in AUTH_API_KEYS:
        logger.warning(f"Unauthorized request with key: {api_key[:8]}...")
        return JSONResponse({"error": "Invalid API key"}, status_code=401)

    client_info = AUTH_API_KEYS[api_key]
    logger.debug(f"Authorized request from: {client_info.get('name', 'unknown')}")

    return await call_next(request)


@app.get("/health")
async def health():
    """Health check endpoint"""
    mcp_servers = []
    if mcp_wrapper:
        for mgr in mcp_wrapper.mcp_managers:
            mcp_servers.append(mgr.url)
    return {
        "status": "healthy",
        "stt_model": MODEL_PATH,
        "stt_device": DEVICE,
        "tts_voices": list(TTS_VOICES.keys()),
        "mcp_servers": mcp_servers,
        "llm_url": LLM_API_URL,
    }


@app.post("/process")
async def process(
    file: UploadFile = File(...),
    data: str = Form(None)
):
    """Upload audio → STT → LLM+MCP → TTS → audio"""
    temp_input = None
    temp_output = None
    temp_dir = None

    try:
        override_lang = None
        if data:
            try:
                payload = json.loads(data)
                override_lang = payload.get("lang")
            except json.JSONDecodeError:
                pass

        content = await file.read()

        temp_input = f"/tmp/{uuid.uuid4()}_input.wav"
        temp_output = f"/tmp/{uuid.uuid4()}_output.wav"
        temp_dir = f"/tmp/{uuid.uuid4()}"

        os.makedirs(temp_dir, exist_ok=True)

        with open(temp_input, "wb") as f:
            f.write(content)

        segments, detected_lang, lang_prob = await transcribe_audio(temp_input)
        transcribed_text = " ".join(segment.text for segment in segments)

        logger.info(f"STT: lang={detected_lang}, text='{transcribed_text}'")

        llm_response = await call_llm_with_mcp(transcribed_text)

        logger.info(f"LLM response: {llm_response}")

        tts_lang = override_lang or detected_lang

        output_files = await synthesize_multiple([llm_response], tts_lang, temp_dir)
        combine_wav_files(list(output_files), temp_output)

        logger.info(f"Process complete: lang={tts_lang}")

        return FileResponse(
            temp_output,
            media_type="audio/wav",
            filename="output.wav",
            background=lambda: cleanup_files(temp_input, temp_output, temp_dir)
        )
    except Exception as e:
        logger.error(f"Process error: {e}")
        await cleanup_files(temp_input, temp_output, temp_dir)
        raise


if __name__ == "__main__":
    import uvicorn
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8080)
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)