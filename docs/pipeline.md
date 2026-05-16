# Pipeline — STT → LLM → TTS

The pipeline processes voice in 4 stages, each decoupled by a Redis stream checkpoint.

## Diagram

```
WS Handler → [stt_jobs] → STT → [llm_jobs] → LLM → [tts_jobs] → TTS → [responses] → WS Sender
                               ↗              ↖
                        Internal Tools    Phone Tools (Redis bridge)
```

## Stages

### 1. STT (Speech-to-Text)

| | |
|---|---|
| **Stream** | `stt_jobs` |
| **Worker** | `pipeline.workers.stt_worker.SttWorker` |
| **Model** | faster-whisper-medium |
| **Input** | base64 WAV audio (16kHz) |
| **Output** | transcript + language + confidence |
| **Time** | ~100-500ms |
| **Runs** | Local (CPU thread) |

The worker writes a temp WAV file, runs Whisper in a thread, extracts the text and detected language, then writes to `llm_jobs`.

### 2. LLM (Language Model)

| | |
|---|---|
| **Stream** | `llm_jobs` |
| **Worker** | `pipeline.workers.llm_worker.LlmWorker` |
| **Model** | llama-3.3-70b-versatile (Groq API) |
| **Input** | transcript + conversation history from Redis |
| **Output** | response text |
| **Time** | ~500ms-3s |
| **Runs** | Remote API (needs internet) |

**The Tool Loop**: The LLM may call 0-5 tools per turn. Each tool call is routed:
- **Internal tool** → runs on the cluster node (e.g., `get_weather`)
- **Remote tool** → goes through the Redis bridge to the phone (e.g., `get_gps`)

Up to 5 iterations (configurable via `mcp.max_tool_loops`). In each iteration, the LLM can call multiple tools in parallel.

### 3. TTS (Text-to-Speech)

| | |
|---|---|
| **Stream** | `tts_jobs` |
| **Worker** | `pipeline.workers.tts_worker.TtsWorker` |
| **Model** | Piper |
| **Input** | text + language |
| **Output** | base64 WAV audio |
| **Time** | ~200-500ms |
| **Runs** | Local (CPU thread) |

Pipeline runs Piper in a thread with synthesis settings (volume, length_scale, noise_scale, noise_w_scale). Output is base64-encoded WAV.

### 4. WS Sender

| | |
|---|---|
| **Stream** | `responses` |
| **Worker** | `pipeline.workers.ws_sender.WsSender` |
| **Input** | device_id + audio_data |
| **Output** | WebSocket `audio_chunk` message |

Checks `get_active_connection(device_id)` — if the device is connected to this node, sends the audio via WebSocket. If not connected to this node (e.g., device reconnected to another node after crash), the message is dropped.

## Workers

All workers extend `BaseWorker`:

```python
class BaseWorker:
    async def run(self):
        while True:
            messages = await self.redis.xreadgroup(
                self.stream, self.group, self.consumer, block=5000
            )
            for stream_name, entries in messages:
                for msg_id, data in entries:
                    try:
                        await self.process(data)
                        await self.redis.xack(self.stream, self.group, msg_id)
                    except Exception:
                        # retry up to 3 times with backoff
                        ...
```

- **Consumer group**: `najim-workers` (same across all nodes)
- **Consumer name**: `{node_id}-{stage}` (e.g., `node-1-stt`)
- **Retry**: 3 attempts, exponential backoff (1s → 3s → 9s)
- **Ack**: Only on success. Unacked messages are picked up by another node after timeout.

## Crash Recovery

| Crash during | Before checkpoint fix | After fix |
|---|---|---|
| Receiving audio | Audio lost, phone re-sends | Audio in `stt_jobs`. Any node picks up. |
| STT | Temp WAV on node-1's disk, lost | Job in `stt_jobs`. Another node resumes. |
| STT→LLM | Transcript in node-1 RAM, lost | Transcript in `llm_jobs`. Any node runs LLM. |
| LLM awaiting Groq | asyncio.Future lost | Consumer group retries the LLM job. |
| Tool awaiting phone | Future lost, phone's response goes nowhere | Tool response goes to `tool_resp:{id}` in Redis. Any node reads it. |
| TTS | Temp WAV on node-1's disk, lost | Response text in `tts_jobs`. Any node synthesizes. |
| Sending to phone | Partial audio, then WS drops | TTS audio in `responses` stream. Phone fetches on reconnect. |

## Configuration

```yaml
pipeline:
  stt_stream: "stt_jobs"
  llm_stream: "llm_jobs"
  tts_stream: "tts_jobs"
  response_stream: "responses"
  consumer_group: "najim_workers"
  stt_max_retries: 3
  llm_max_retries: 2
  tts_max_retries: 3
  poll_timeout_ms: 5000
```

## Scaling

Because stages are decoupled by streams, you can have:

- 3 nodes all running all stages (current setup)
- Dedicated STT nodes, dedicated LLM nodes, dedicated TTS nodes
- More STT workers if Whisper is the bottleneck
- The WS sender must run on the node the phone is connected to (currently checked via local dict)
