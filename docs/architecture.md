# Architecture

## Overview

Najim is a **voice assistant backend** designed for a cluster of 3 Intel Atom computers. An Android app connects via WebSocket, sends audio, and the backend runs a pipeline: **STT → LLM → Tool Calls → TTS**, streaming audio back. All state lives in Redis so any node can handle any request.

## Design Tenets

| Tenet | Why |
|-------|-----|
| **Shared-nothing** | No session data in local RAM. Everything in Redis. A crash or rebalance never loses session state. |
| **Stateless nodes** | Any request → any node. The load balancer round-robins freely. |
| **Checkpoints between stages** | STT → LLM → TTS each write to a Redis stream before the next stage starts. If a node crashes mid-stage, another node claims the pending job from the stream. |
| **Async non-blocking I/O** | All I/O (Redis, HTTP, disk) uses `asyncio`. CPU-heavy work (Whisper, Piper) runs in threads. |
| **Phone as a tool server** | The Android app isn't just an audio I/O device — it executes remote tools (GPS, file index, HTTP server) on demand. |

## System Diagram

```
                    ┌─────────────────┐
                    │  Load Balancer   │
                    │  (nginx/haproxy) │
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
    │  Node 1   │      │  Node 2   │      │  Node 3   │
    │  (FastAPI)│      │  (FastAPI)│      │  (FastAPI)│
    │  WS conns │      │  WS conns │      │  WS conns │
    │  Workers  │      │  Workers  │      │  Workers  │
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │     Redis       │
                    │  Sessions       │
                    │  Conversations  │
                    │  Device Registry│
                    │  Tool Bridge    │
                    │  Pub/Sub        │
                    │  Pipeline Jobs  │
                    └─────────────────┘
```

## Components

### `core/` — Foundation

- **config.py**: Pydantic Settings with 15 nested model classes. Reads from `config.yaml`. Every section (Redis, JWT, STT, TTS with synthesis, LLM, pipeline, auth, cluster) is typed and validated at boot.
- **redis_manager.py**: Async Redis client wrapping `redis.asyncio`. Methods: `hset_dict`, `xadd`, `xreadgroup`, `xack`, `xpending`, `xgroup_create`, `blpop`, `publish`, `subscribe`. Connection pool with 20 connections. TLS support.
- **schemas.py**: All Pydantic models — `SessionData`, `ChatMessage`, `ToolCall`, `ToolDefinition`, `WSMessage`, `DeviceInfo`, `SessionConfig`.
- **jwt_auth.py**: `create_token(device_id, user_id, permissions)`, `verify_token(token)`, FastAPI dependency `get_current_device_id`. Optional API key fallback.
- **app_state.py**: Singleton holding loaded Whisper model, Piper TTS instances, and configured LLM HTTP client.

### `api/` — Entry Points

- **websocket.py**: `/api/v1/connect` — WebSocket endpoint. JWT auth on connect. Message loop: heartbeat, audio → pipeline, tool_response, dynamic phone tool registration. Pushes audio as a `stt_jobs` stream entry.
- **sessions.py**: REST CRUD — `GET/POST /sessions`, `GET/PUT /sessions/{id}`, `GET/DELETE /conversations/{id}`, `GET/PUT /devices`, `GET/PUT /permissions`.
- **health.py**: `/health` (summary), `/ready` (all dependencies OK), `/live` (process alive), `/metrics` (prometheus).

### `sessions/` — Redis State Management

- **session_registry.py**: Hash `session:{device_id}` with TTL (24h). Touch on activity. Config per device (language, voice).
- **conversation_store.py**: List `conv:{device_id}`. Append-only, trim to `max_history` (100). Roles: system, user, assistant, tool.
- **device_registry.py**: Hash `devices` (all devices) + key `device_ws:{id}` → node_id (with TTL). Heartbeat every 30s. Tracks which node a device is connected to.
- **permissions.py**: Hash `perms:{device_id}`. Per-device allow/deny for tools. Default deny.

### `tools/` — Tool System

- **registry.py**: `ToolRegistry` — maps tool names to `ToolDefinition`. Two dicts: `_internal` and `_remote`. Register/check/unregister.
- **internal_tools.py**: `get_time` (UTC), `get_weather` (mock), `calculator` (safe eval). Extensible.
- **call_client_tool.py**: The cross-node remote tool bridge. `initiate_remote_call` generates a correlation UUID, stores in Redis `tool_corr:{id}`, publishes to `najim:ws_send:{node_id}`. `await_remote_response` uses Redis `BLPOP` on `tool_resp:{id}` to wait for the phone's response (no local futures). `handle_response` pushes the response to the BLPOP-able key.
- **router.py**: `ToolRouter.route_tool_call(name, device_id)` — checks if internal → runs locally, checks if remote → checks permission → sends via bridge. Raises `UnknownToolError` if not found.

### `pipeline/` — STT → LLM → TTS

- **workers/base.py**: `BaseWorker` with `xreadgroup` loop, max retries (3), exponential backoff (1s → 3s), ack-on-success.
- **workers/stt_worker.py**: Reads `stt_jobs` stream → writes temp WAV → Whisper transcribes in thread → writes `llm_jobs` stream entry with transcript + metadata (language, confidence, duration).
- **workers/llm_worker.py**: Reads `llm_jobs` → loads conversation history from Redis → sends to Groq API → tool loop (up to 5 iterations, parallel tool calls) → appends to history → writes `tts_jobs` with response text.
- **workers/tts_worker.py**: Reads `tts_jobs` → Piper synthesizes in thread (with synthesis settings: volume, length_scale, etc.) → base64-encodes WAV → writes `responses` stream entry.
- **workers/ws_sender.py**: Reads `responses` stream → checks if device is connected on this node via `get_active_connection(device_id)` → sends audio chunk via WebSocket. If device moved to another node, silently drops (future: pub/sub routing back to new node).
- **orchestrator.py**: `WorkerManager` — creates all 4 workers, runs them as asyncio tasks, handles graceful shutdown.

## Data Flow

```
WS Handler → [stt_jobs] → STT → [llm_jobs] → LLM → [tts_jobs] → TTS → [responses] → WS Sender
                                   ↗              ↖
                            Internal Tools    Phone Tools (via Redis BLPOP bridge)
```

1. **WS Handler** receives audio, `XADD`s to `stt_jobs` stream, responds with `{"accepted": true}`.
2. **STT Worker** (any node) `XREADGROUP`s from `stt_jobs`, transcribes, `XADD`s transcript to `llm_jobs`.
3. **LLM Worker** (any node) reads `llm_jobs`, runs LLM with tool loop (internal tools run locally, remote tools go via Redis bridge), `XADD`s response text to `tts_jobs`.
4. **TTS Worker** (any node) reads `tts_jobs`, synthesizes audio, `XADD`s base64 audio to `responses`.
5. **WS Sender** (the node where the device is connected) reads `responses`, sends audio via WebSocket.

## Pipeline vs Monolithic

| Aspect | Monolithic (old) | Checkpoint Pipeline (current) |
|--------|------------------|-------------------------------|
| Processing node | Same node that received audio | Any node per stage |
| Crash during STT | Audio lost, phone re-sends | STT job in stream, retried by another node |
| Crash during LLM | Future lost, phone re-sends | LLM job in stream, consumer group retries |
| Crash during TTS | WAV temp file lost | TTS job in stream, retried |
| Scaling | N/A — tied to WS node | STT on node-1, LLM on node-2, TTS on node-3 |
