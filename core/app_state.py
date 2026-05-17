import asyncio
import logging
import glob
import os
from pathlib import Path
from typing import Optional
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig
from openai import AsyncOpenAI

from core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class AppState:
    whisper_model: Optional[WhisperModel] = None
    tts_voices: dict[str, PiperVoice] = {}
    tts_voice_paths: dict[str, str] = {}
    llm_client: Optional[AsyncOpenAI] = None
    syn_config: Optional[SynthesisConfig] = None
    initialized: bool = False
    _load_lock: Optional[asyncio.Lock] = None

    @classmethod
    def _get_lock(cls) -> asyncio.Lock:
        if cls._load_lock is None:
            cls._load_lock = asyncio.Lock()
        return cls._load_lock

    @classmethod
    async def initialize(cls) -> None:
        if cls.initialized:
            return
        async with cls._get_lock():
            if cls.initialized:
                return
            settings = get_settings()

            cls.syn_config = SynthesisConfig(
                volume=settings.tts.synthesis.volume,
                length_scale=settings.tts.synthesis.length_scale,
                noise_scale=settings.tts.synthesis.noise_scale,
                noise_w_scale=settings.tts.synthesis.noise_w_scale,
            )

            if settings.stt.model_dir:
                await cls._load_whisper(settings)
            await cls._load_tts(settings)
            await cls._load_llm(settings)
            cls.initialized = True

    @classmethod
    async def _load_whisper(cls, settings: Settings) -> None:
        model_path = Path(settings.stt.model_dir) / f"whisper-{settings.stt.model_name}"
        if model_path.exists():
            logger.info(f"Loading Whisper model from {model_path}")
            cls.whisper_model = WhisperModel(
                str(model_path),
                device=settings.stt.device,
                compute_type=settings.stt.compute_type,
            )
            logger.info("Whisper model loaded")
        else:
            logger.warning(f"Whisper model not found at {model_path}")

    @classmethod
    async def _load_tts(cls, settings: Settings) -> None:
        model_dir = Path(settings.tts.model_dir)
        for lang, voice_cfg in settings.tts.voices.items():
            local_path = voice_cfg.get("local_path", "")
            if not local_path:
                continue
            voice_dir = model_dir / local_path
            if not voice_dir.exists():
                logger.warning(f"TTS voice dir not found: {voice_dir}")
                continue
            onnx_files = list(voice_dir.glob("*.onnx"))
            if onnx_files:
                cls.tts_voice_paths[lang] = str(onnx_files[0])
                logger.info(f"TTS voice registered: {lang} -> {onnx_files[0]}")
            else:
                logger.warning(f"No .onnx file found in {voice_dir}")

    @classmethod
    async def _load_llm(cls, settings: Settings) -> None:
        if settings.llm.api_key:
            cls.llm_client = AsyncOpenAI(
                base_url=settings.llm.api_base_url,
                api_key=settings.llm.api_key,
                timeout=settings.llm.timeout,
                max_retries=settings.llm.max_retries,
            )
            logger.info("LLM client initialized")

    @classmethod
    def get_tts_voice(cls, language: str) -> Optional[PiperVoice]:
        if language in cls.tts_voices:
            return cls.tts_voices[language]
        path = cls.tts_voice_paths.get(language)
        if not path:
            path = cls.tts_voice_paths.get(get_settings().tts.default_voice)
        if path:
            if path not in cls.tts_voices:
                try:
                    cls.tts_voices[language] = PiperVoice.load(path)
                except Exception as e:
                    logger.error(f"Failed to load TTS voice {path}: {e}")
                    return None
            return cls.tts_voices.get(language)
        return None

    @classmethod
    def get_llm_client(cls) -> Optional[AsyncOpenAI]:
        return cls.llm_client

    @classmethod
    def get_synthesis_config(cls) -> Optional[SynthesisConfig]:
        return cls.syn_config

    @classmethod
    async def shutdown(cls) -> None:
        cls.tts_voices.clear()
        cls.whisper_model = None
        cls.llm_client = None
        cls.syn_config = None
        cls.initialized = False
        logger.info("AppState shut down")


_state: Optional[AppState] = None


def get_app_state() -> AppState:
    global _state
    if _state is None:
        _state = AppState()
    return _state