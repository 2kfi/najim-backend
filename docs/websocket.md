# WebSocket Protocol

## Endpoint

```
ws://<host>:<port>/api/v1/connect?token=<JWT>
```

Requires a valid JWT as a query parameter. Invalid tokens get a 401 close frame.

## Connection Lifecycle

```
1. Phone connects with JWT token
2. Server validates JWT
3. Server accepts WebSocket
4. Phone sends capabilities: {"type": "connect", "capabilities": ["gps", "file_index"]}
5. Server registers phone in device registry with dynamic tools
6. Message loop (until disconnect or timeout):
   - Phone → heartbeat (every 30s)
   - Phone → audio (base64 WAV chunks)
   - Server → tool_request (when LLM needs phone-side tool)
   - Phone → tool_response (result of tool execution)
7. On disconnect: cleanup device registry
```

## Message Types

### Phone → Server

#### `connect`
Sent once immediately after WS handshake. Registers phone capabilities.

```json
{"type": "connect", "capabilities": ["gps", "file_index"]}
```

#### `heartbeat`
Sent every 30 seconds to keep the connection alive. Server refreshes TTL on `device_ws:{id}`.

```json
{"type": "heartbeat"}
```

#### `audio`
Pushes audio data into the pipeline. Server responds with `{"type": "accepted"}` and later sends results.

```json
{
  "type": "audio",
  "data": "<base64-encoded WAV audio>",
  "mime_type": "audio/wav",
  "chunk_index": 0,
  "total_chunks": 1
}
```

#### `tool_response`
Response to a `tool_request` from the server.

```json
{
  "type": "tool_response",
  "correlation_id": "uuid-from-server",
  "result": {"temperature": 32, "humidity": 60},
  "error": null
}
```

### Server → Phone

#### `accepted`
Acknowledges that audio was received and pushed to the pipeline.

```json
{"type": "accepted", "message": "Audio queued for processing"}
```

#### `interim_transcript`
Partial transcription sent back during STT processing.

```json
{"type": "interim_transcript", "text": "What's the weather in..."}
```

#### `thinking`
Sent when processing takes noticeable time. Phone plays a "processing" chime.

```json
{"type": "thinking", "text": "Give me a moment..."}
```

#### `tool_request`
The LLM needs the phone to execute a tool. Phone should execute and respond.

```json
{
  "type": "tool_request",
  "correlation_id": "uuid",
  "tool_name": "get_gps",
  "params": {"accuracy": 10}
}
```

#### `audio_chunk`
The final TTS audio output, streamed back to the phone.

```json
{
  "type": "audio_chunk",
  "data": "<base64-encoded WAV audio>",
  "mime_type": "audio/wav",
  "text": "The weather in Cairo is 32°C",
  "chunk_index": 0,
  "total_chunks": 1
}
```

## Testing with wscat

```bash
npm install -g wscat

# Get an admin JWT (set DEBUG=true in docker-compose.yml):
docker compose run -e DEBUG=true -e JWT_SECRET=$JWT_SECRET najim
# → [najim] Admin JWT: eyJhbGciOi...

# Connect
wscat -c "ws://localhost:8000/api/v1/connect?token=<ADMIN_JWT>"

# Send messages interactively:
{"type": "connect", "capabilities": ["gps", "test"]}
{"type": "heartbeat"}
# (audio messages via script)
```

## Android Client Requirements

The Android app must:

1. Obtain a JWT (via REST `/api/v1/sessions` POST or pre-configured token)
2. Connect via WebSocket URL with `?token=<JWT>`
3. Send `{"type": "connect", "capabilities": [...]}` on open
4. Send `{"type": "heartbeat"}` every 30 seconds
5. Send audio as base64 WAV chunks
6. Listen for `tool_request` messages, execute the tool, send `tool_response`
7. Play `audio_chunk` data through speakers
8. Handle reconnection on disconnect with exponential backoff
