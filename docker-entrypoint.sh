#!/bin/sh
# =============================================================================
# Najim Backend — Docker Entrypoint
# Reads environment variables, generates config.yaml, starts uvicorn.
# =============================================================================
set -e

if [ -n "$REDIS_URL" ]; then
    echo "[najim] Using REDIS_URL: $REDIS_URL"
elif [ -n "$REDIS_HOST" ]; then
    echo "[najim] Using REDIS_HOST: $REDIS_HOST"
fi

if [ -z "$JWT_SECRET" ]; then
    echo "[najim] WARNING: JWT_SECRET not set! Generating ephemeral secret."
    JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
fi

cat > /app/config.yaml <<EOF
api:
  host: "0.0.0.0"
  port: ${API_PORT:-8080}
  debug: ${DEBUG:-false}
  cors_origins: ["*"]
  rate_limit: "${RATE_LIMIT:-60/minute}"
  max_audio_size_mb: ${MAX_AUDIO_SIZE_MB:-10}

redis:
  url: "${REDIS_URL:-}"
  host: "${REDIS_HOST:-localhost}"
  port: ${REDIS_PORT:-6379}
  password: "${REDIS_PASSWORD:-}"
  tls: ${REDIS_TLS:-false}
  pool_size: ${REDIS_POOL_SIZE:-20}

jwt:
  secret: "${JWT_SECRET}"
  algorithm: "HS256"
  expiry_minutes: ${JWT_EXPIRY_MINUTES:-1440}

cluster:
  node_id: "${CLUSTER_NODE_ID:-$(hostname)}"
  node_role: "worker"
  pubsub_channel: "${PUBSUB_CHANNEL:-najim:events}"

auth:
  jwt_only: ${AUTH_JWT_ONLY:-true}
  api_keys: {}

session:
  ttl_seconds: ${SESSION_TTL:-86400}
  max_history: ${SESSION_MAX_HISTORY:-100}
  heartbeat_interval: ${SESSION_HEARTBEAT:-30}

tool:
  remote_timeout: ${TOOL_REMOTE_TIMEOUT:-30.0}
  internal_timeout: ${TOOL_INTERNAL_TIMEOUT:-10.0}
  max_retries: ${TOOL_MAX_RETRIES:-2}

stt:
  model_name: "${STT_MODEL_NAME:-medium}"
  model_dir: "${STT_MODEL_DIR:-./models}"
  hf_repo: "${STT_HF_REPO:-Systran/faster-whisper-medium}"
  device: "${STT_DEVICE:-auto}"
  compute_type: "${STT_COMPUTE_TYPE:-int8}"
  beam_size: ${STT_BEAM_SIZE:-5}
  vad_filter: ${STT_VAD_FILTER:-true}
  language: ${STT_LANGUAGE:-null}

tts:
  model_dir: "${TTS_MODEL_DIR:-./models}"
  voices:
    en:
      local_path: "${TTS_EN_PATH:-TTS-CORI-EN}"
      voice: "${TTS_EN_VOICE:-en.en_GB.cori.high}"
    ar:
      local_path: "${TTS_AR_PATH:-TTS-KAREEM-ARABIC}"
      voice: "${TTS_AR_VOICE:-ar.ar_JO.kareem.medium}"
  default_voice: "${TTS_DEFAULT_VOICE:-en}"
  max_length: ${TTS_MAX_LENGTH:-500}
  synthesis:
    volume: ${TTS_VOLUME:-0.75}
    length_scale: ${TTS_LENGTH_SCALE:-1.0}
    noise_scale: ${TTS_NOISE_SCALE:-0.75}
    noise_w_scale: ${TTS_NOISE_W_SCALE:-0.5}
    normalize_audio: true
    nchannels: 1
    sampwidth: 2
    framerate: ${TTS_FRAMERATE:-22050}

llm:
  api_base_url: "${LLM_API_BASE_URL:-https://api.groq.com/openai/v1}"
  api_key: "${LLM_API_KEY:-}"
  model: "${LLM_MODEL:-llama-3.3-70b-versatile}"
  timeout: ${LLM_TIMEOUT:-60.0}
  max_retries: ${LLM_MAX_RETRIES:-2}

mcp:
  servers: []
  sse_read_timeout: 300.0
  tool_timeout: 30.0
  max_retries: 2
  max_tool_loops: 5

pipeline:
  stt_stream: "stt_jobs"
  llm_stream: "llm_jobs"
  tts_stream: "tts_jobs"
  response_stream: "responses"
  consumer_group: "${PIPELINE_CONSUMER_GROUP:-najim_workers}"
  consumer_prefix: "${CLUSTER_NODE_ID:-node}"
  stt_max_retries: ${PIPELINE_STT_RETRIES:-3}
  llm_max_retries: ${PIPELINE_LLM_RETRIES:-2}
  tts_max_retries: ${PIPELINE_TTS_RETRIES:-3}
  poll_timeout_ms: ${PIPELINE_POLL_TIMEOUT:-5000}

models:
  storage_path: "${MODEL_DIR:-./models}"
  download_on_startup: ${MODEL_DOWNLOAD:-true}
EOF

echo "[najim] config.yaml generated"

if [ "$MODEL_DOWNLOAD" = "true" ] && [ -f "scripts/downloader.py" ]; then
    echo "[najim] Downloading models..."
    python3 scripts/downloader.py || echo "[najim] Model download skipped"
fi

# Generate JWT for admin testing if DEBUG is on
if [ "${DEBUG:-false}" = "true" ]; then
    echo "[najim] DEBUG mode — generating test admin token..."
    python3 -c "
from core.jwt_auth import get_jwt_manager
mgr = get_jwt_manager()
token = mgr.create_token('admin', 'admin-device', permissions=['admin'])
print(f'[najim] Admin JWT: {token}')
"
fi

echo "[najim] Starting uvicorn on 0.0.0.0:${API_PORT:-8080}"
exec python3 -m uvicorn app:app \
    --host 0.0.0.0 \
    --port ${API_PORT:-8080} \
    --log-level ${LOG_LEVEL:-info} \
    --workers 1 \
    --loop uvloop \
    --http h11
