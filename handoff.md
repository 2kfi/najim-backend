# Najim Backend — AI Session Handoff

## Project Overview

Multi-tenant distributed voice assistant backend. Audio flows through a Redis-stream-backed pipeline:

```
WS Receive → [stt_jobs] → STT (faster-whisper) → [llm_jobs] → LLM (Groq API + Tool Loop) → [tts_jobs] → TTS (Piper) → [responses] → WS Send
```

Runs on Python 3.11, FastAPI, Redis Streams. Designed for Intel Atom clusters with shared-nothing Redis-backed state.

**Location:** `/run/media/2kfi/DATA/Work-files/Projects/najim-backend`
**Entry point:** `app.py` — `python app.py` (or `uvicorn app:app`)
**Venue:** `.venv/`
**Redis:** Docker `localhost:6379`, password `69397516`
**Config:** `config.yaml` (YAML, not `.env` — `.env` is only for Docker Compose)

---

## What Was Fixed This Session

### 1. Config loading: YAML parsing broken
**File:** `core/config.py`
**Issue:** `model_config = SettingsConfigDict(env_file="config.yaml")` used the dotenv parser, not YAML. All YAML values were silently dropped — `redis.url`, `jwt.secret`, etc. loaded as defaults (empty strings).
**Fix:** Replaced with custom `YamlConfigSettingsSource` via `settings_customise_sources()`. Also changed nested model defaults from `= ClassName()` (evaluated once at class-definition time, preventing pydantic-settings from applying env vars) to `Field(default_factory=ClassName)`.

### 2. Config type annotations: YAML bools/ints rejected
**File:** `core/config.py`
**Issue:** `dict[str, dict[str, str]]` for `TTSSettings.voices` and `AuthSettings.api_keys` rejected non-string YAML values (bools, ints).
**Fix:** Changed to `dict[str, dict[str, Any]]`.

### 3. Pipeline data forwarding broken
**File:** `pipeline/workers/base.py`
**Issue:** `BaseWorker.start()` called `self.handler(data)` but discarded the return value. Results never flowed between pipeline stages.
**Fix:** Captured `result = await self._process_one()`, then `if result and self.target_stream: await self.redis.xadd(...)`.

### 4. Pipeline worker target_streams not set
**Files:** `pipeline/workers/stt_worker.py`, `pipeline/workers/llm_worker.py`, `pipeline/workers/tts_worker.py`
**Fix:** Each stage now sets `target_stream=settings.pipeline.llm_stream` (stt), `.tts_stream` (llm), `.response_stream` (tts).

### 5. Pipeline `__init__.py` wrong export
**File:** `pipeline/__init__.py`
**Fix:** Exported `WorkerManager` instead of nonexistent `VoicePipeline`.

### 6. Nested env var overrides don't work
**File:** `core/config.py`
**Issue:** pydantic-settings' `EnvSettingsSource` only processes env vars based on the top-level model's config. `PIPELINE_STT_WORKERS=4` was never found because the `PIPELINE_` prefix is on `PipelineSettings`, not `Settings`.
**Fix:** Added `_NestedEnvSource` — a custom settings source that reads `PIPELINE_*`, `REDIS_*`, `JWT_*`, etc. env vars and maps them to nested dict keys (`pipeline.stt_workers`, etc.). Placed **before** `YamlConfigSettingsSource` in the priority chain so env vars override YAML.

### 7. Redis stream data decode crash
**File:** `core/redis_manager.py`
**Issue:** All `.decode()` calls assumed values were always `bytes`, but redis-py can return already-decoded strings depending on connection config. When `json.loads()` returned a `dict` and the code tried `.decode()` on it: `'dict' object has no attribute 'decode'`. Also `xadd` didn't handle non-str non-bytes values properly.
**Fix:** Every decode path (`xadd`, `xreadgroup`, `xread`, `hget_all`, `get`, `lrange`, `blpop`, `zrange`) now uses a 3-way branch: `isinstance(v, bytes) → v.decode()`, `isinstance(v, str) → v`, else `str(v)`. All `json.loads()` calls are guarded with `try/except (JSONDecodeError, UnicodeDecodeError)`.

### 8. Conversation store .decode() on dict
**File:** `sessions/conversation_store.py`
**Issue:** `get_history()` called `.decode()` on items from `lrange()`, but `lrange` returns **parsed Python objects** (via `json.loads` inside the method). When an existing conversation message was read, it arrived as a `dict`, and `.decode()` crashed.
**Fix:** Added `isinstance(item, dict)` check → `Message.model_validate(item)`, `isinstance(item, str)` → `.model_validate_json(item)`, else (bytes) → `.decode()` first.

### 9. WebSocket keepalive ping timeout
**File:** `app.py`
**Issue:** uvicorn's default `ws_ping_interval=20` / `ws_ping_timeout=20` closed the connection during long pipeline processing (STT ~15s + LLM ~10s + TTS ~5s = ~30s). The application already has its own heartbeat mechanism (`heartbeat_interval=30`), making protocol-level ping/pong redundant.
**Fix:** Added `ws_ping_interval=None` to `uvicorn.run()`.

---

## Known Remaining Issues

1. **LLM query response handling after client disconnect** — If the WebSocket client disconnects before LLM/TTS finishes, the pipeline still completes but the ws_sender skips sending (device not connected). Currently harmless, but wasted work. Could add a cancellation mechanism.
2. **TTS queue not used** — `TTSQueue` has `enqueue`/`dequeue` methods but `tts_handler` calls `synthesize_and_b64` directly, bypassing the queue. The queue requires concurrent dequeue + zrange with proper locking.
3. **Client test script has no progress feedback** — The test client (`/home/2kfi/test.py`) just prints received JSON. No way to know if the server is still processing.
4. **Multiple worker threads for STT/TTS** — Multiple STT workers share the same Whisper model (thread-safe). Multiple TTS workers share the same Piper voices (lazy-loaded, should be thread-safe). LLM/WS workers are async-safe (no shared state).
5. **Concurrent history writes** — Multiple LLM workers could write conversation history for the same device_id concurrently. The conversation store uses `rpush` without any locking, which could interleave messages.

---

## Key Files to Read for Client Issues

| Issue Area | Files |
|---|---|
| WebSocket connection/auth | `api/websocket.py` (lines 67-209), `core/jwt_auth.py` |
| Audio receive & pipeline submit | `api/websocket.py` lines 150-192 (`_handle_audio`) |
| Pipeline worker infrastructure | `pipeline/workers/base.py`, `pipeline/orchestrator.py` |
| STT stage | `pipeline/workers/stt_worker.py`, `core/app_state.py` (Whisper loading) |
| LLM stage | `pipeline/workers/llm_worker.py`, `pipeline/llm_runner.py` |
| TTS stage | `pipeline/workers/tts_worker.py`, `pipeline/tts_queue.py`, `core/app_state.py` (voice loading) |
| Response sending | `pipeline/workers/ws_sender.py`, `api/websocket.py` `_start_ws_listener` |
| Conversation history | `sessions/conversation_store.py` |
| Redis operations | `core/redis_manager.py` |
| Config/env/overrides | `core/config.py`, `config.yaml`, `.env.example` |
| Session/device management | `sessions/session_registry.py`, `sessions/device_registry.py` |

---

## How to Test

**Terminal 1** (server, in project dir):
```bash
cd /run/media/2kfi/DATA/Work-files/Projects/najim-backend
PIPELINE_STT_WORKERS=4 PIPELINE_TTS_WORKERS=4 PIPELINE_LLM_WORKERS=4 PIPELINE_WS_WORKERS=4 python app.py
```

**Terminal 2** (client):
```bash
python /home/2kfi/test.py /home/2kfi/output.wav
```

The test client JWT is generated by the script using `core.jwt_auth.JWTManager` with the config's JWT secret. `os.chdir(APP_DIR)` is needed for relative `config.yaml` path resolution.

**Server log levels:** Controlled by `api.debug` in `config.yaml`. Sets `logging.basicConfig(level=DEBUG)` and `uvicorn.run(log_level="debug")`.

---

## Critical Config Details

- `PIPELINE_STT_WORKERS` / `PIPELINE_LLM_WORKERS` / `PIPELINE_TTS_WORKERS` / `PIPELINE_WS_WORKERS` — env vars that override `config.yaml`. Works via `_NestedEnvSource`.
- Stream names: `stt_jobs`, `llm_jobs`, `tts_jobs`, `responses` (from `PipelineSettings`)
- Consumer group: `najim_workers`
- Redis URL from YAML: `redis://:69397516@localhost:6379/0`
- First run loads Whisper medium model (~45s CPU). Subsequent runs load from cached `models/whisper-medium/`.
- WebSocket audio field: `"audio_data"` (base64), task: `"transcribe"` or `"translate"`

---

## Redis Stream Data Format

**stt_jobs** (written by `_handle_audio`):
```python
{
    "device_id": str,
    "session_id": str,
    "audio_data": str,  # base64 WAV
    "language": str,
    "task": str,
}
```

**llm_jobs** (written by `stt_handler`):
```python
{
    "device_id": str,
    "session_id": str,
    "text": str,            # transcription
    "language": str,        # detected language
    "probability": float,   # language confidence
}
```

**tts_jobs** (written by `llm_handler`):
```python
{
    "device_id": str,
    "session_id": str,
    "input_text": str,      # user's query
    "response": str,        # LLM response text
    "language": str,
}
```

**responses** (written by `tts_handler`):
```python
{
    "device_id": str,
    "session_id": str,
    "audio": str,           # base64 WAV (Piper synthesized)
    "text": str,            # response text
}
```

---

## Data Flow (with 4 workers per stage)

```
                     ┌─────────────┐
                     │  Client WS  │
                     └──────┬──────┘
                            │ audio, task="transcribe"
                            ▼
                     ┌─────────────┐
                     │ _handle_audio│
                     │  (websocket) │
                     └──────┬──────┘
                            │ xadd to stt_jobs
                            ▼
                ┌──────────────────────┐
                │  STT workers x4       │
                │  (faster-whisper)     │
                └──────────┬───────────┘
                           │ xadd to llm_jobs
                           ▼
                ┌──────────────────────┐
                │  LLM workers x4       │
                │  (Groq API + tools)   │
                └──────────┬───────────┘
                           │ xadd to tts_jobs
                           ▼
                ┌──────────────────────┐
                │  TTS workers x4       │
                │  (Piper synthesis)    │
                └──────────┬───────────┘
                           │ xadd to responses
                           ▼
                ┌──────────────────────┐
                │  WS workers x4        │
                │  (send via pubsub)    │
                └──────────┬───────────┘
                           │ send_json
                           ▼
                     ┌─────────────┐
                     │  Client WS  │
                     └─────────────┘
```

The data forwarding inside `BaseWorker.start()`:
```python
result = await self._process_one()    # reads stream, calls handler
if result and self.target_stream:
    await self.redis.xadd(self.target_stream, result, maxlen=1000)
```

---

## Pipeline Worker Configuration

Each stage spawns N workers as asyncio tasks. All share the same consumer group (`najim_workers`) with unique consumer names:
`worker:{node_id}:{stage}-{index}` (e.g., `worker:node-1:stt-0`).

Redis Stream consumer groups ensure each message is delivered to exactly **one** consumer within the group, enabling horizontal scaling without duplicates.

---

## Environment Setup

- Python: 3.11+ (`requirements.txt` lists all deps)
- Redis: Docker `redis:7-alpine` on `localhost:6379` with `--requirepass 69397516`
- Models (auto-downloaded on first run or via `scripts/downloader.py`):
  - STT: `models/whisper-medium/` (faster-whisper, ~1.5GB)
  - TTS voices: `models/TTS-CORI-EN/`, `models/TTS-KAREEM-ARABIC/` (Piper, ~50MB each)
- LLM API key required (Groq): set `LLM_API_KEY` env var or in `config.yaml`
