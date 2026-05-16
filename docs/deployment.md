# Deployment

## Prerequisites

- Docker and Docker Compose
- At least 4GB RAM per node (for Whisper + Piper models)
- Redis 7+ (provided by Docker)

## Quick Start (Single Node)

```bash
docker compose up -d
```

This starts the app + Redis. App available at `http://localhost:8080`.

## Cluster (3 Nodes)

```bash
docker compose -f docker-compose.cluster.yml up -d
```

Starts:
- `redis` — Redis 7 on `:6379`
- `node-1` — app on `:8081`
- `node-2` — app on `:8082`
- `node-3` — app on `:8083`

Put a load balancer (nginx/haproxy) in front of `:8081-:8083`.

## Standalone (External Redis)

```bash
docker compose -f docker-compose.standalone.yml up -d
```

Set `REDIS_URL` environment variable to your Redis instance.

## Environment Variables

All config can be set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8080` | HTTP port |
| `API_DEBUG` | `false` | Enable debug mode |
| `API_CORS_ORIGINS` | `*` | CORS origins (comma-separated) |
| `API_RATE_LIMIT` | `60/minute` | Global rate limit |
| `REDIS_URL` | `redis://:password@redis:6379/0` | Redis connection string |
| `REDIS_TLS` | `false` | Enable TLS for Redis |
| `REDIS_POOL_SIZE` | `20` | Connection pool size |
| `JWT_SECRET` | `change-me-in-production` | JWT signing key |
| `JWT_ALGORITHM` | `HS256` | JWT algorithm |
| `JWT_EXPIRY_MINUTES` | `1440` | Token lifetime |
| `CLUSTER_NODE_ID` | hostname | Unique node identifier |
| `CLUSTER_NODE_ROLE` | `worker` | Node role |
| `AUTH_JWT_ONLY` | `true` | Disable API key fallback |
| `SESSION_TTL` | `86400` | Session TTL in seconds |
| `SESSION_MAX_HISTORY` | `100` | Max conversation messages |
| `SESSION_HEARTBEAT_INTERVAL` | `30` | Heartbeat interval in seconds |
| `TOOL_REMOTE_TIMEOUT` | `30.0` | Remote tool timeout |
| `TOOL_INTERNAL_TIMEOUT` | `10.0` | Internal tool timeout |
| `TOOL_MAX_RETRIES` | `2` | Max tool retries |
| `STT_MODEL_NAME` | `medium` | Whisper model size |
| `STT_MODEL_DIR` | `./models` | Model storage path |
| `STT_DEVICE` | `auto` | Compute device (`cpu`/`cuda`/`auto`) |
| `STT_COMPUTE_TYPE` | `int8` | Compute precision |
| `TTS_MODEL_DIR` | `./models` | TTS model path |
| `TTS_DEFAULT_VOICE` | `en` | Default TTS voice |
| `TTS_VOLUME` | `0.75` | TTS output volume |
| `TTS_LENGTH_SCALE` | `1.0` | Speech speed |
| `TTS_NOISE_SCALE` | `0.75` | Voice variance |
| `TTS_NOISE_W_SCALE` | `0.5` | Voice variance (width) |
| `LLM_API_BASE_URL` | `https://api.groq.com/openai/v1` | LLM API endpoint |
| `LLM_API_KEY` | `gsk_...` | LLM API key |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `LLM_TIMEOUT` | `60.0` | LLM request timeout |

## Model Files

Models are loaded from `STT_MODEL_DIR` / `TTS_MODEL_DIR` (default: `./models/`).

### Whisper

- Directory: `{stt.model_dir}/whisper-{stt.model_name}/`
- Downloaded from HuggingFace on first run
- Default: `./models/whisper-medium/`

### Piper TTS

- Directory: `{tts.model_dir}/{voice.local_path}/`
- Downloaded from HuggingFace on first run
- Default English: `./models/TTS-CORI-EN/`
- Default Arabic: `./models/TTS-KAREEM-ARABIC/`

## Testing

### Health Check

```bash
curl http://localhost:8080/health
# → {"status":"healthy","redis":"ok","models":"ok","uptime_seconds":123}
```

### WebSocket

```bash
# Get an admin JWT from the server (set DEBUG=true in docker-compose.yml):
docker compose run -e DEBUG=true -e JWT_SECRET=$JWT_SECRET najim
# → [najim] Admin JWT: eyJhbGciOi...

# Connect
wscat -c "ws://localhost:8080/api/v1/connect?token=<ADMIN_JWT>"
```

### Send Audio

```python
import base64, json, asyncio, websockets

async def test():
    async with websockets.connect(f"ws://localhost:8080/api/v1/connect?token={TOKEN}") as ws:
        await ws.send(json.dumps({"type": "connect", "capabilities": ["gps"]}))
        
        with open("test.wav", "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        
        await ws.send(json.dumps({
            "type": "audio", "data": b64,
            "mime_type": "audio/wav",
            "chunk_index": 0, "total_chunks": 1
        }))
        
        async for msg in ws:
            print(msg)

asyncio.run(test())
```

## Verified Config

Reference `config.yaml` contains every section with defaults. See the `config/` directory or `understand.md` Section 22 Q16 for the complete annotated config.
