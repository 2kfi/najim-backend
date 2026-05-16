# Najim Backend

Multi-tenant, distributed voice assistant backend for Intel Atom clusters.  
Android App → STT → LLM → Tool Calls → TTS. All state in Redis.

## Architecture

```
                    ┌─────────────────┐
                    │  Load Balancer   │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
    │  Node 1   │      │  Node 2   │      │  Node 3   │
    │  (FastAPI)│      │  (FastAPI)│      │  (FastAPI)│
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │     Redis       │
                    │  Sessions,      │
                    │  Conversations, │
                    │  Tool Bridge,   │
                    │  Pub/Sub,       │
                    │  Checkpoints    │
                    └─────────────────┘
```

**Checkpoint Pipeline** — Every stage writes to a Redis stream before the next picks up:

```
WS Receive → [stt_jobs] → STT → [llm_jobs] → LLM → [tts_jobs] → TTS → [responses] → WS Send
                                            ↗              ↖
                                     Tool Router      Phone Tools
```

If a node crashes mid-stage, another node picks up the pending job from the stream.  
No state in local memory. Shared-nothing architecture.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/2kfi/najim-backend
cd najim-backend

# 2. Set env vars
cp .env.example .env
# Edit .env — set JWT_SECRET and LLM_API_KEY

# 3. Place models in ./models/ (or set MODEL_DOWNLOAD=true)
#    whisper-medium, TTS-CORI-EN, TTS-KAREEM-ARABIC

# 4. Start
docker compose up -d

# 5. Check
curl http://localhost:8080/health
```

## 3-Node Cluster

```bash
docker compose -f docker-compose.cluster.yml up -d
# Node 1: http://localhost:8081
# Node 2: http://localhost:8082
# Node 3: http://localhost:8083
```

All nodes share the same Redis. Each has a unique `CLUSTER_NODE_ID`.  
A load balancer (nginx/haproxy) in front distributes WebSocket connections.

## Standalone (No Docker Redis)

```bash
REDIS_URL="redis://:pass@my-redis:6379" docker compose \
  -f docker-compose.standalone.yml up -d
```

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `JWT_SECRET` | Signing key for auth tokens (use `openssl rand -hex 32`) |
| `LLM_API_KEY` | API key for LLM provider (Groq, OpenAI, etc.) |

### Connection

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | — | `redis://:pass@host:6379/0` overrides host/port |
| `REDIS_HOST` | `localhost` | Redis host (ignored if REDIS_URL set) |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_PASSWORD` | `najim_pass` | Redis password |
| `REDIS_TLS` | `false` | Enable TLS for Redis |

### Cluster

| Variable | Default | Description |
|----------|---------|-------------|
| `CLUSTER_NODE_ID` | hostname | Unique node name |
| `PUBSUB_CHANNEL` | `najim:events` | Redis pub/sub channel |

### STT (Whisper)

| Variable | Default | Description |
|----------|---------|-------------|
| `STT_DEVICE` | `auto` | `cpu`, `cuda`, or `auto` |
| `STT_MODEL_NAME` | `medium` | Whisper model size |
| `STT_COMPUTE_TYPE` | `int8` | `int8`, `fp16`, `fp32` |
| `STT_BEAM_SIZE` | `5` | Beam search width |

### LLM

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_BASE_URL` | `https://api.groq.com/openai/v1` | API endpoint |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Model name |
| `LLM_TIMEOUT` | `60.0` | Request timeout seconds |

### TTS (Piper)

| Variable | Default | Description |
|----------|---------|-------------|
| `TTS_VOLUME` | `0.75` | Output volume |
| `TTS_LENGTH_SCALE` | `1.0` | Speed (lower = faster) |
| `TTS_NOISE_SCALE` | `0.75` | Voice variation |
| `TTS_NOISE_W_SCALE` | `0.5` | Spectral variation |
| `TTS_FRAMERATE` | `22050` | Output sample rate |
| `TTS_EN_PATH` | `TTS-CORI-EN` | English voice path |
| `TTS_AR_PATH` | `TTS-KAREEM-ARABIC` | Arabic voice path |

### Pipeline

| Variable | Default | Description |
|----------|---------|-------------|
| `PIPELINE_STT_RETRIES` | `3` | Max STT retries |
| `PIPELINE_LLM_RETRIES` | `2` | Max LLM retries |
| `PIPELINE_TTS_RETRIES` | `3` | Max TTS retries |
| `PIPELINE_POLL_TIMEOUT` | `5000` | Worker poll (ms) |

### Auth

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTH_JWT_ONLY` | `true` | `false` = also accept API keys |
| `RATE_LIMIT` | `60/minute` | Per-device rate limit |
| `DEBUG` | `false` | Prints admin JWT on startup |

## API Reference

### WebSocket

```
ws://host:8080/api/v1/connect?token=<JWT>
```

**Phone → Server messages:**
| Type | Payload | Description |
|------|---------|-------------|
| `connect` | `{capabilities, tools}` | Handshake, register device capabilities |
| `audio` | `{audio_data (base64)}` | Send audio for processing |
| `tool_response` | `{correlation_id, result}` | Respond to a remote tool call |
| `tools_update` | `{tools}` | Update registered tool list |
| `heartbeat` | `{}` | Keepalive (every 30s) |

**Server → Phone messages:**
| Type | Payload | Description |
|------|---------|-------------|
| `connected` | `{device_id, node_id}` | Connection accepted |
| `accepted` | `{}` | Audio received, processing started |
| `processing` | `{text}` | "I'm thinking" response |
| `response` | `{audio_data, text}` | Final TTS audio output |
| `tool_request` | `{correlation_id, tool, params}` | Execute a tool on device |
| `error` | `{message}` | Error description |

### REST

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Node health + connected devices |
| GET | `/ready` | No | Readiness (Redis + models loaded) |
| GET | `/live` | No | Liveness |
| GET | `/metrics` | No | Prometheus metrics |
| POST | `/api/v1/sessions` | JWT | Create session |
| GET | `/api/v1/sessions` | JWT | List all sessions |
| GET | `/api/v1/sessions/{id}` | JWT | Get session |
| DELETE | `/api/v1/sessions/{id}` | JWT | Delete session |
| PATCH | `/api/v1/sessions/{id}/config` | JWT | Update session config |
| GET | `/api/v1/conversations/{id}` | JWT | Get conversation history |
| GET | `/api/v1/devices` | JWT | List connected devices |
| GET | `/api/v1/devices/{id}` | JWT | Get device info |
| GET | `/api/v1/permissions/{id}` | JWT | List device permissions |
| PUT | `/api/v1/permissions/{id}/{tool}` | JWT | Set tool permission |

## Generating a JWT

```bash
# Install python-jose
pip install python-jose[cryptography]

# Generate token
python3 -c "
from jose import jwt
import time
token = jwt.encode({
    'user_id': 'admin',
    'device_id': 'test-phone',
    'permissions': ['admin'],
    'iat': time.time(),
    'exp': time.time() + 86400
}, 'YOUR_JWT_SECRET', algorithm='HS256')
print(token)
"
```

Or set `DEBUG=true` — the entrypoint prints an admin JWT on startup.

## Testing without a Phone

```bash
# Install wscat
npm install -g wscat

# Connect (use a real JWT)
wscat -c "ws://localhost:8080/api/v1/connect?token=<JWT>"

# Once connected:
{"type": "connect", "capabilities": ["test"]}

# Send audio (base64 encoded WAV):
{"type": "audio", "audio_data": "<base64>"}

# You'll get back:
# {"type": "thinking", "text": "..."}
# Then later:
# {"type": "response", "audio_data": "...", "text": "..."}
```

## Model Files

Place these in `./models/`:

```
models/
├── whisper-medium/
│   ├── model.bin
│   └── config.json
├── TTS-CORI-EN/
│   └── en.en_GB.cori.high.onnx
└── TTS-KAREEM-ARABIC/
    └── ar.ar_JO.kareem.medium.onnx
```

Download with:
```bash
python3 scripts/downloader.py
```

Or set `MODEL_DOWNLOAD=true` in Docker.

## Architecture Details

For a deep dive into how everything works, see [understand.md](understand.md) — covers JWT, Redis data structures, WebSocket lifecycle, tool bridge, correlation IDs, load balancing, crash recovery, and the checkpoint pipeline.

## License

Open source.
