# Glossary

| Term                | Meaning                                                                                      |
| ------------------- | -------------------------------------------------------------------------------------------- |
| **STT**             | Speech-to-Text. Converts audio to text using Whisper.                                        |
| **TTS**             | Text-to-Speech. Converts text to audio using Piper.                                          |
| **LLM**             | Large Language Model. The "brain" using Groq/llama.                                          |
| **JWT**             | JSON Web Token. Signed authentication token with device_id + permissions.                    |
| **WebSocket**       | Persistent full-duplex TCP connection between client and server.                             |
| **Redis**           | In-memory database used as shared state, pub/sub, and pipeline streams.                      |
| **Pub/Sub**         | Publish/Subscribe. Redis channels for cross-node messaging.                                  |
| **Correlation ID**  | UUID that ties a tool request to its response across nodes.                                  |
| **BLPOP**           | Redis blocking list pop. Used to wait for tool responses across nodes.                       |
| **Stream**          | Redis data type for append-only logs with consumer groups. Used for pipeline checkpoints.    |
| **Consumer Group**  | Redis stream feature that distributes messages across consumers and handles retries.         |
| **XADD**            | Redis command to append to a stream.                                                         |
| **XREADGROUP**      | Redis command to read messages as a consumer group member.                                   |
| **XACK**            | Redis command to acknowledge a stream message as processed.                                  |
| **Load Balancer**   | Distributes incoming connections across 3 cluster nodes (nginx/haproxy).                     |
| **Session**         | Per-device state in Redis (language, config, status). TTL 24h.                               |
| **Conversation**    | Message history between user and LLM. Stored as a Redis list.                                |
| **Device Registry** | Tracks which devices are online and which node they're connected to.                         |
| **Internal Tool**   | Tool that runs on the cluster node (e.g., get_weather, calculator).                          |
| **Remote Tool**     | Tool that runs on the Android phone (e.g., get_gps, index_files).                            |
| **Tool Bridge**     | Cross-node mechanism for sending tool requests to phones via Redis pub/sub + BLPOP.          |
| **TTL**             | Time To Live. Auto-delete Redis key after N seconds.                                         |
| **Base64**          | Binary-to-text encoding used for audio data in JSON.                                         |
| **FastAPI**         | Python async web framework the server is built on.                                           |
| **ASGI**            | Async Server Gateway Interface (uvicorn serves it).                                          |
| **Node**            | One computer in the cluster (we have 3 Intel Atom nodes).                                    |
| **Pipeline**        | Sequence: STT → LLM → TTS, each stage writing to a Redis stream.                             |
| **Checkpoint**      | Redis stream entry that persists a stage's output so another node can recover after a crash. |
| **Rate Limiting**   | Max requests per time window per device (60 req/min).                                        |
| **HMAC-SHA256**     | Algorithm used to sign JWTs.                                                                 |
| **asyncio**         | Python's async/await concurrency framework.                                                  |
| **slowapi**         | Python library for rate limiting in FastAPI.                                                 |
| **Tenacity**        | Python library for retry logic with backoff.                                                 |
| **Whisper**         | OpenAI's open-source speech-to-text model (faster-whisper).                                  |
| **Piper**           | Fast neural text-to-speech system, runs locally on CPU.                                      |
| **Groq**            | Cloud LLM API providing llama-3.3-70b.                                                       |
| **MCP**             | Model Context Protocol. Tool definitions for LLM function calling.                           |
| **VAD**             | Voice Activity Detection. Filters silence from audio before STT.                             |
| **WAV**             | Audio file format. 16kHz mono for STT, 22050Hz mono for TTS.                                 |
