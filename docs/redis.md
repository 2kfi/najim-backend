# Redis — The Shared Brain

All 3 cluster nodes connect to the same Redis instance. It stores all runtime state.

## Connection

```yaml
redis:
  url: "redis://:password@host:6379/0"
  # or rediss:// for TLS
  tls: false
  pool_size: 20
```

## Data Structures

### Hash — session metadata & device registry & permissions

```
session:{device_id}          → fields: device_id, user_id, created_at, last_active, config, status, language
devices                      → fields: {device_id → JSON info}, each value has node_id, capabilities, status
perms:{device_id}            → fields: {tool_name → "allow"|"deny"}
```

### List — conversation history

```
conv:{device_id}             → elements: JSON {"role": "user"|"assistant"|"tool"|"system", "content": "...", ...}
```

### String — device-to-node mapping, tool correlation

```
device_ws:{device_id}        → value: "node-1" (TTL: 35s, refreshed on heartbeat)
tool_corr:{correlation_id}   → value: JSON with device_id, tool, params, status (TTL: 35s)
tool_resp:{correlation_id}   → value: JSON with tool result (used for BLPOP bridge)
```

### Stream — pipeline checkpoints

```
stt_jobs                     → fields: device_id, audio_data, mime_type, chunk_index, total_chunks
llm_jobs                     → fields: device_id, transcript, language, confidence, duration
tts_jobs                     → fields: device_id, text, language, voice
responses                    → fields: device_id, audio_data, mime_type, text
```

Each stream uses a **consumer group** (`najim-workers`) so jobs are distributed across nodes and retried on failure.

### Pub/Sub Channels — cross-node messaging

```
najim:events                 → Cluster-wide announcements (node up/down)
najim:ws_send:{node_id}      → Direct message to a specific node (tool requests, WS forwarding)
```

## Key Access Patterns

| Operation | Redis Command | Wrapper |
|-----------|--------------|---------|
| Create/update session | `HSET session:{id} field value` | `redis.hset_dict(key, data)` |
| Read session | `HGETALL session:{id}` | `redis.hgetall(key)` |
| Extend TTL | `EXPIRE session:{id} 86400` | `await redis.expire(key, ttl)` |
| Append conversation | `RPUSH conv:{id} json_msg` | `redis.rpush(key, json)` |
| Read recent history | `LRANGE conv:{id} -50 -1` | `redis.lrange(key, -50, -1)` |
| Trim history | `LTRIM conv:{id} 0 100` | `redis.ltrim(key, 0, 100)` |
| Register device | `HSET devices {id} json_info` | `redis.hset_dict("devices", {id: info})` |
| Set WS mapping | `SET device_ws:{id} node-1 EX 35` | `await redis.setex(key, 35, node_id)` |
| Check permission | `HGET perms:{id} tool_name` | `redis.hget(key, tool_name)` |
| Push to pipeline | `XADD stt_jobs * field val ...` | `redis.xadd("stt_jobs", data)` |
| Read from pipeline | `XREADGROUP group consumer streams >` | `redis.xreadgroup(...)` |
| Acknowledge job | `XACK stt_jobs group id` | `redis.xack(stream, group, id)` |
| Check pending | `XPENDING stt_jobs group` | `redis.xpending(stream, group)` |
| Publish message | `PUBLISH najim:ws_send:node-1 msg` | `redis.publish(channel, msg)` |
| Wait for tool response | `BLPOP tool_resp:{id} 30` | `redis.blpop(key, timeout=30)` |

## Why Redis (Not PostgreSQL)?

| | PostgreSQL | Redis |
|---|---|---|
| Read speed | ~1-10ms | ~0.1ms |
| Write speed | ~5-20ms | ~0.1ms |
| Data model | Tables, rows | Key-value, hashes, lists, streams, sets |
| Pub/Sub | No (polling) | Yes (instant push) |
| TTL | Manual cleanup | Auto-expire keys |
| Session data | Overkill | Perfect fit |

## Why Not Python Memory?

If node-1 stores session data in RAM, node-2 can't access it. And if node-1 crashes, all sessions are gone. Redis is a separate process — if node-1 crashes, node-2 reads from Redis and continues.

## Memory Estimates

| Item | Size | Notes |
|------|------|-------|
| Session metadata | ~500B | Per device, stored once |
| Audio in stt_jobs | ~200KB | 5s of audio base64 (in stream transiently) |
| Transcript message | ~200B | Per turn |
| LLM response | ~500B | Per turn |
| Tool call record | ~300B | TTL 35s, auto-deleted |

**Per turn**: ~201KB (mostly base64 audio in stream), or ~1KB for text-only.
**Per session (50 turns)**: ~10.5MB with streaming.
