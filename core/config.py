import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_")

    url: str = ""
    host: str = "localhost"
    port: int = 6379
    password: str = ""
    tls: bool = True
    pool_size: int = 20
    socket_keepalive: bool = True
    socket_connect_timeout: int = 5
    health_check_interval: int = 30


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STT_")

    model_name: str = "medium"
    model_dir: str = "./models"
    hf_repo: str = "Systran/faster-whisper-medium"
    device: str = "auto"
    compute_type: str = "int8"
    beam_size: int = 5
    vad_filter: bool = True
    language: Optional[str] = None


class TTSVoiceConfig(BaseSettings):
    local_path: str = ""
    hf_repo: str = ""
    voice: str = ""
    use_cuda: bool = False


class SynthesisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_SYNTH_")

    volume: float = 0.75
    length_scale: float = 1.0
    noise_scale: float = 0.75
    noise_w_scale: float = 0.5
    normalize_audio: bool = True
    nchannels: int = 1
    sampwidth: int = 2
    framerate: int = 22050


class TTSSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TTS_")

    model_dir: str = "./models"
    voices: dict[str, dict[str, str]] = {
        "en": {"local_path": "TTS-CORI-EN", "voice": "en.en_GB.cori.high"},
        "ar": {"local_path": "TTS-KAREEM-ARABIC", "voice": "ar.ar_JO.kareem.medium"},
    }
    default_voice: str = "en"
    max_length: int = 500
    synthesis: SynthesisSettings = SynthesisSettings()


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")

    api_base_url: str = "https://api.groq.com/openai/v1"
    api_key: str = ""
    model: str = "llama-3.3-70b-versatile"
    timeout: float = 60.0
    max_retries: int = 2


class MCPSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_")

    servers: list[str] = []
    sse_read_timeout: float = 300.0
    tool_timeout: float = 30.0
    max_tool_loops: int = 5
    max_retries: int = 2


class JWTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JWT_")

    secret: str = ""
    algorithm: str = "HS256"
    expiry_minutes: int = 1440


class SessionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SESSION_")

    ttl_seconds: int = 86400
    max_history: int = 100
    heartbeat_interval: int = 30


class ToolSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOOL_")

    remote_timeout: float = 30.0
    internal_timeout: float = 10.0
    max_retries: int = 2


class ClusterSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLUSTER_")

    node_id: str = os.uname().nodename
    node_role: str = "worker"
    pubsub_channel: str = "najim:events"


class PipelineSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_")

    stt_stream: str = "stt_jobs"
    llm_stream: str = "llm_jobs"
    tts_stream: str = "tts_jobs"
    response_stream: str = "responses"
    consumer_group: str = "najim_workers"
    consumer_prefix: str = "worker"
    stt_max_retries: int = 3
    llm_max_retries: int = 2
    tts_max_retries: int = 3
    poll_timeout_ms: int = 5000


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_")

    api_keys: dict[str, dict[str, str]] = {}
    jwt_only: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="config.yaml", env_file_encoding="utf-8", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8080
    debug: bool = False
    cors_origins: list[str] = ["*"]
    rate_limit: str = "60/minute"
    max_audio_size_mb: int = 10
    allowed_audio_types: list[str] = ["audio/wav", "audio/mpeg", "audio/ogg"]

    redis: RedisSettings = RedisSettings()
    jwt: JWTSettings = JWTSettings()
    llm: LLMSettings = LLMSettings()
    mcp: MCPSettings = MCPSettings()
    stt: STTSettings = STTSettings()
    tts: TTSSettings = TTSSettings()
    session: SessionSettings = SessionSettings()
    tool: ToolSettings = ToolSettings()
    cluster: ClusterSettings = ClusterSettings()
    auth: AuthSettings = AuthSettings()
    pipeline: PipelineSettings = PipelineSettings()


@lru_cache
def get_settings() -> Settings:
    return Settings()