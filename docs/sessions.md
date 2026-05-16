# Sessions, Conversations & Devices

## Session Registry

One session per device. Stored as a Redis hash with TTL (24h).

```
Key:  session:{device_id}
Type: Hash
TTL:  86400s (refreshed on activity)
```

Fields:

| Field | Type | Example |
|-------|------|---------|
| `device_id` | string | `"phone-android-123"` |
| `user_id` | string | `"user-abc"` |
| `created_at` | ISO datetime | `"2025-01-01T00:00:00Z"` |
| `last_active` | ISO datetime | `"2025-01-02T12:00:00Z"` |
| `language` | string | `"en"` |
| `config` | JSON string | `{"tts_voice": "en"}` |
| `status` | string | `"active"` |

If no activity for 24 hours, the session auto-deletes. Every message resets the TTL.

## Conversation Store

One conversation list per device. Append-only, trimmed to `max_history` (default 100).

```
Key:  conv:{device_id}
Type: List
```

Message roles:

| Role | Description |
|------|-------------|
| `system` | Instructions to the LLM ("You are a helpful assistant") |
| `user` | The human's transcribed speech |
| `assistant` | The LLM's response |
| `tool` | Result of a tool call |

The LLM receives the last N messages for context. Example:

```json
[
  {"role": "user", "content": "What's the weather in Cairo?", "timestamp": "..."},
  {"role": "tool", "content": "{\"temp\": 32}", "tool_call_id": "call_1"},
  {"role": "assistant", "content": "The weather in Cairo is 32°C", "timestamp": "..."}
]
```

## Device Registry

Tracks which devices are online and which node they're connected to.

### Hash (all devices)

```
Key:  devices
Type: Hash
```

```json
{
  "phone-android-123": {
    "device_id": "phone-android-123",
    "user_id": "user-abc",
    "capabilities": ["gps", "file_index"],
    "status": "online",
    "node_id": "node-1",
    "last_heartbeat": "2025-01-02T12:00:00Z"
  }
}
```

### Key per device (fast WS routing)

```
Key:  device_ws:{device_id}
Type: String (value = node_id)
TTL:  35s (refreshed on each heartbeat)
```

**Why two places?** The hash is good for listing all devices. The individual key with TTL is fast for routing and auto-expires if the node crashes.

### Heartbeats

Phone sends `{"type": "heartbeat"}` every 30 seconds. Server:
- Updates `last_heartbeat` in the `devices` hash
- Refreshes TTL on `device_ws:{id}` (extend by 35s)
- If server sees no message for 35 seconds, it sends a probe. No response → close connection, mark offline.

## REST Endpoints

```bash
# List sessions
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/sessions

# Get session for a device
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/sessions/device-123

# Update session config
curl -X PATCH -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"language": "ar"}' \
  http://localhost:8000/api/v1/sessions/device-123/config

# Delete session (also clears conversation history)
curl -X DELETE -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/sessions/device-123

# Get conversation history
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/conversations/device-123

# List devices
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/devices

# Get device info
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/devices/device-123

# Get permissions
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/permissions/device-123

# Set permission
curl -X PUT "http://localhost:8000/api/v1/permissions/device-123/get_gps?allowed=true" \
  -H "Authorization: Bearer <token>"
```
