# Frequently Asked Questions

### Q1: What if a node crashes mid-pipeline?

Each pipeline stage writes its output to a Redis stream *before* the next stage starts. The `BaseWorker` uses consumer groups with retry logic (3 attempts, exponential backoff). Unacked messages are picked up by another node after the visibility timeout.

| Crash during | What happens |
|---|---|
| Receiving audio via WS | Next chunk is pushed to `stt_jobs` stream. Phone re-sends if WS drops. |
| STT transcribing | Job in `stt_jobs` stream. Another node claims it. |
| STT→LLM handoff | Transcript in `llm_jobs` stream. Any node runs LLM. |
| LLM awaiting Groq | Consumer group retries the LLM job on another node. |
| Tool awaiting phone | Phone's `tool_response` goes into `tool_resp:{id}` in Redis. Any node reads it via BLPOP. |
| TTS synthesizing | Response text in `tts_jobs` stream. Another node synthesizes. |
| Sending to phone | TTS audio in `responses` stream. Phone fetches on reconnect (future). |

### Q2: How much Redis memory per request?

| Item | Size |
|------|------|
| Session metadata | ~500B |
| Audio base64 in stream | ~200KB (5s audio) |
| Transcript/LLM message | ~200-500B |
| Tool call record (TTL 35s) | ~300B |

**Per turn**: ~201KB (with audio in stream). **Per session (50 turns)**: ~10.5MB.

### Q3: How many tools per turn?

Up to 5 iterations (configurable). Each iteration can call multiple tools in parallel (typically 5-10). Theoretical max: ~50 tool calls per user turn.

### Q4: Redis TLS setup?

For LAN: skip TLS, use a password. For internet: generate CA + server certs with openssl, configure Redis with `tls-port`, and use `rediss://` URL.

### Q5: Can I run a single node without a load balancer?

Yes. The architecture scales from 1 to N nodes. With 1 node, connect directly.

### Q6: Can I run without Redis?

No. Redis is not optional. It's the backbone for session persistence, conversation history, tool correlation, pub/sub, and pipeline checkpoints.

### Q7: How does the node get its ID?

From the OS hostname by default (`socket.gethostname()`). Override with `CLUSTER_NODE_ID` env var.

### Q8: What if my nodes are weak and processing takes 2+ minutes?

Use the "thinking" pattern: immediately send `{"type": "thinking"}` back to the phone so the user hears a "processing" chime. Process asynchronously via the pipeline workers.

### Q9: How to test without an Android phone?

Use `wscat` for WebSocket or `curl` for REST endpoints. See [deployment.md](deployment.md) for examples.

### Q10: How does the server know what tools a phone has?

The phone announces capabilities on connect: `{"type": "connect", "capabilities": ["gps", "file_index"]}`. These are mapped to predefined remote tool definitions in the ToolRegistry.

### Q11: Does the old API key system still work?

Yes, as a fallback. Set `auth.jwt_only: false` and add keys to `auth.api_keys`. Default is JWT only.

### Q12: What config values changed from the old version?

See the full migration table in `understand.md` Section 22 Q12. Key changes: STT/TTS paths restructured, synthesis settings moved under `tts.synthesis`, new sections for Redis/JWT/cluster/session/tool.

### Q13: How do I generate an admin JWT?

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{"device_id": "admin", "user_id": "admin"}'
```

Or in debug mode, the server prints an admin JWT on startup.

### Q14: What happens if Redis goes down?

The nodes lose all session state, conversation history, and pipeline coordination. WebSocket connections stay open but can't process new audio. Once Redis recovers, nodes reconnect and resume. All data in Redis is in-memory — configure Redis persistence (RDB/AOF) for restart survival.

### Q15: How do I add a new voice/language?

1. Download the Piper voice model files (.onnx + .json) to `{tts.model_dir}/{voice.local_path}/`
2. Add a voice entry to `tts.voices` in config.yaml
3. The phone can request it via session config language field

### Q16: How do I monitor the cluster?

- `/health` — overall status (Redis + models)
- `/ready` — readiness probe (all dependencies OK)
- `/live` — liveness probe (process alive)
- `/metrics` — Prometheus metrics (request count, latency, active connections)
