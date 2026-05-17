# Najim Backend — Complete Architecture Guide

> **Who is this for?** You know how to code. You understand APIs. But terms like "load balancer", "JWT", "Redis pub/sub", "correlation ID", "WebSocket" are fuzzy.  
> By the end of this, you'll understand every bit that enters and leaves this cluster.

**Organized docs also available in [`docs/`](docs/):**
- [Overview](docs/overview.md) — One-sentence + diagram
- [Architecture](docs/architecture.md) — Full design + components
- [Authentication](docs/authentication.md) — JWT + API keys
- [Redis](docs/redis.md) — All data structures + keys
- [WebSocket](docs/websocket.md) — Protocol + message types
- [Pipeline](docs/pipeline.md) — STT→LLM→TTS stages
- [Tools](docs/tools.md) — Internal + remote + bridge
- [Sessions](docs/sessions.md) — Sessions, conversations, devices
- [Cluster](docs/cluster.md) — Multi-node + crash recovery
- [Deployment](docs/deployment.md) — Docker + env vars
- [FAQ](docs/faq.md) — Frequently asked questions
- [Glossary](docs/glossary.md) — Terms

---

# Table of Contents

1. [The One-Sentence Summary](#1-the-one-sentence-summary)
2. [How Voice Flows Through The System](#2-how-voice-flows-through-the-system)
3. [What Is a Load Balancer?](#3-what-is-a-load-balancer)
4. [What Is WebSocket? Why Not HTTP?](#4-what-is-websocket-why-not-http)
5. [What Is JWT? (Authentication)](#5-what-is-jwt-authentication)
6. [What Is Redis? The Shared Brain](#6-what-is-redis-the-shared-brain)
7. [Redis Data Structures We Use](#7-redis-data-structures-we-use)
8. [The Session Registry](#8-the-session-registry)
9. [The Device Registry](#9-the-device-registry)
10. [The Conversation Store](#10-the-conversation-store)
11. [The Tool Registry](#11-the-tool-registry)
12. [Internal vs Remote Tools — The Router](#12-internal-vs-remote-tools--the-router)
13. [The Remote Tool Bridge (Correlation IDs)](#13-the-remote-tool-bridge-correlation-ids)
14. [Redis Pub/Sub — How Nodes Talk To Each Other](#14-redis-pubsub--how-nodes-talk-to-each-other)
15. [The Permission Store](#15-the-permission-store)
16. [The Voice Pipeline (STT → LLM → TTS)](#16-the-voice-pipeline-stt--llm--tts)
17. [What Happens On Startup](#17-what-happens-on-startup)
18. [What Happens On Shutdown](#18-what-happens-on-shutdown)
19. [The Full Data Flow — A Complete Trace](#19-the-full-data-flow--a-complete-trace)
20. [Security Model](#20-security-model)
21. [Glossary](#21-glossary)
22. [Frequently Asked Questions](#22-frequently-asked-questions)

---

## 1. The One-Sentence Summary

> Najim is a **voice assistant backend** running on 3 Intel Atom computers. An Android app connects via WebSocket, sends audio, the backend runs STT → LLM → Tool Calls → TTS, and streams audio back — all using **Redis** as the shared brain so any node can handle any request.

---

## 2. How Voice Flows Through The System

```
  Android App               Any Node (WS Handler)               Redis Streams               Worker (any node)
     │                              │                               │                             │
     │  (1) WS connect + JWT       │                               │                             │
     │ ──────────────────────────► │                               │                             │
     │                              │  (2) Verify JWT               │                             │
     │                              │ ─────────────────────────────► devices                      │
     │                              │                               │                             │
     │  (3) Audio data (base64)     │                               │                             │
     │ ──────────────────────────► │                               │                             │
     │                              │                               │                             │
     │                              │  (4) XADD stt_jobs           │                             │
     │                              │ ─────────────────────────────► stt_jobs                    │
     │                              │                               │                             │
     │  (5) {"accepted"}            │                               │                             │
     │ ◄────────────────────────── │                               │                             │
     │                              │                               │                             │
     │                              │                               │  (6) XREADGROUP stt_jobs    │
     │                              │                               │ ◄───────────────────────── │
     │                              │                               │                             │
     │                              │                               │  (7) Whisper transcribe     │
     │                              │                               │         │                   │
     │                              │                               │  (8) XADD llm_jobs         │
     │                              │                               │ ──────────────────────────►│
     │                              │                               │                             │
     │                              │                               │  (9) XREADGROUP llm_jobs   │
     │                              │                               │ ◄───────────────────────── │
     │                              │                               │                             │
     │                              │                               │  (10) LLM → maybe tool     │
     │                              │                               │       │                    │
     │                              │                               │       ▼                    │
     │                              │                               │  Internal → run on node    │
     │  (11) Tool request           │                               │  Remote  → pub/sub to phone│
     │ ◄────────────────────────── │                               │                             │
     │  (12) Tool response         │                               │                             │
     │ ──────────────────────────► │                               │                             │
     │                              │                               │                             │
     │                              │  (13) XADD tts_jobs          │                             │
     │                              │ ─────────────────────────────► tts_jobs                    │
     │                              │                               │                             │
     │                              │                               │  (14) XREADGROUP tts_jobs  │
     │                              │                               │ ◄───────────────────────── │
     │                              │                               │                             │
     │                              │                               │  (15) Piper synthesize     │
     │                              │                               │         │                   │
     │                              │                               │  (16) XADD responses       │
     │                              │                               │ ──────────────────────────►│
     │                              │                               │                             │
     │                              │  (17) XREADGROUP responses   │                             │
     │                              │ ◄────────────────────────── │                             │
     │                              │  (if device on this node)    │                             │
     │                              │                               │                             │
     │  (18) TTS audio (base64)     │                               │                             │
     │ ◄────────────────────────── │                               │                             │
```

**Key insight**: Steps 6-16 can run on **any node**. The WS handler only receives audio and sends back results. The STT, LLM, and TTS workers are decoupled via Redis streams. If a node crashes mid-step, the job stays in the stream and another node picks it up. This is the **checkpoint pipeline** — every stage saves its output to a checkpoint (Redis stream) before the next stage begins.

---

## 3. What Is a Load Balancer?

### The Problem

You have 3 Atom nodes. Your Android app only knows ONE URL (like `ws://najim.example.com/connect`). It doesn't know there are 3 computers behind it.

### The Solution

A **load balancer** sits in front of your 3 nodes. It receives all incoming connections and distributes them across the nodes.

```
                      ┌─────────────────┐
                      │  Load Balancer   │
                      │  (nginx/haproxy) │
                      └────────┬────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
    ┌─────▼─────┐       ┌─────▼─────┐       ┌─────▼─────┐
    │  node-1   │       │  node-2   │       │  node-3   │
    └───────────┘       └───────────┘       └───────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │
                         ┌─────▼─────┐
                         │   Redis   │  ← shared by all 3
                         └───────────┘
```

### How Does the Load Balancer Decide Where to Send a Connection?

There are several strategies:

| Strategy | How It Works | Used Here? |
|----------|-------------|------------|
| **Round-robin** | Send to node-1, then node-2, then node-3, then repeat | Yes (for initial WS connect) |
| **Least connections** | Send to whichever node has fewest active WebSockets | Better if some users are more active |
| **Source IP hash** | Always send the same phone to the same node | Can cause uneven load |
| **Sticky sessions** | Once connected to node-1, stay on node-1 | Not needed — we use Redis |

**Important**: Once a phone connects to node-1, if node-1 crashes, the phone reconnects and gets node-2 or node-3. Because all session data is in Redis, node-2 can resume the conversation seamlessly. This is the "shared nothing" architecture.

### What Happens When a Node Goes Down?

```
1. Phone A is connected to node-1 (stored in Redis as device_ws:phoneA = "node-1")
2. Node-1 crashes
3. Phone A's WebSocket disconnects
4. Phone A reconnects → load balancer sends to node-2
5. node-2 reads from Redis: oh, phoneA's session is here, conversation history is here
6. node-2 continues the conversation. Phone A never knew a node died.
```

---

## 4. What Is WebSocket? Why Not HTTP?

### HTTP (Old Way)

```
Phone ─── HTTP POST /process ──► Server
      ◄── HTTP Response ─────── 
```

- Phone sends audio, waits for response, gets audio back. Done.
- **Problem**: Server can't send anything to phone unless phone asks first.
- **Problem**: Every request requires headers, authentication, setup overhead.
- **Problem**: You can't stream partial results (like interim transcript while LLM is still thinking).

### WebSocket (Our Way)

```
Phone ─── WebSocket Connect ──► Server
      ◄── WebSocket Accept ──── (persistent connection)

Now both sides can send messages freely:

Phone: "Here's audio chunk 1"
Phone: "Here's audio chunk 2"
Phone: "Here's audio chunk 3"

Server: "Here's the transcript so far"
Server: "I need to call a tool on your phone"
Phone:  "Here's the tool result"
Server: "Here's the final audio response"
```

- **Full-duplex**: Both sides can send at any time, without asking permission.
- **Persistent**: One TCP connection, stays open until one side closes it. No per-request overhead.
- **Low latency**: No HTTP headers per message. Just raw data.
- **Server push**: Server can send tool requests to phone without phone polling.

### What Happens During the WebSocket Connection?

See the `api/websocket.py` file. This is the connection lifecycle:

```
1. Phone connects: ws://host/api/v1/connect?token=<JWT>
2. Server validates JWT
3. Server accepts WebSocket
4. Phone sends capabilities: {"type": "connect", "capabilities": ["gps", "file_index"]}
5. Server registers phone in Redis (device_registry)
6. Server starts message loop:

   while connected:
       wait for message from phone
       
       if message type == "heartbeat":
           update last_seen in Redis
           send back ack
       
       if message type == "audio":
           run pipeline: STT → LLM → maybe tool → TTS
           send back transcript + audio chunks
       
       if message type == "tool_response":
           resolve the pending tool call (unblock the LLM)
       
       if message type == "disconnect":
           break
       
       if no message for 35 seconds:
           send heartbeat to check if phone is still alive

7. On disconnect:
   remove from Redis
   mark device as offline
```

---

## 5. What Is JWT? (Authentication)

### Problem

How does the server know which phone is talking to it? Anyone could connect and pretend to be your phone.

### JWT = JSON Web Token

Think of JWT as a **digital ID card** that the server gives to the phone once, and the phone shows it on every request.

### Three Parts of a JWT

```
                        Header         Payload                   Signature
                        ┌──────┐   ┌──────────────────────┐   ┌──────────┐
JWT looks like:  xxxxx.  yyyyy.   zzzzz
                   │        │          │
                   │        │          │
              algorithm    user_id,    HMAC-SHA256 of
              HS256        device_id,  header + payload
                           permissions signed with secret
                           exp, iat
```

### How Our JWT Works

```
1. Phone registers → gets JWT:
   {
     "user_id": "user-abc",
     "device_id": "phone-android-123",
     "permissions": ["index_files", "get_gps"],
     "iat": 1700000000,          // issued at (unix timestamp)
     "exp": 1700086400           // expires in 24 hours
   }
   → Signed with SECRET_KEY using HMAC-SHA256
   → Phone stores this token

2. Phone connects via WebSocket:
   ws://server/connect?token=<JWT>

3. Server validates:
   - Decode the token
   - Verify signature (using SECRET_KEY — if tampered, signature won't match)
   - Check expiration (if expired, reject)
   - Extract device_id, user_id, permissions
   - If valid → accept connection
```

### Why JWT and not just a password?

| | Password/API Key | JWT |
|---|---|---|
| State | Server must store it in a database | Stateless — server just verifies signature |
| Expiry | Manual revocation | Built-in via `exp` field |
| Permissions | One key = all access | Token can limit specific tools |
| Multiple devices | Need separate keys per device | `device_id` in the token |

### Where is the JWT validated?

- **WebSocket**: On connection (query parameter `token`)
- **REST API**: Every endpoint via `Authorization: Bearer <token>` header

---

## 6. What Is Redis? The Shared Brain

### Problem

You have 3 nodes. If node-1 stores session data in its own memory (RAM), what happens when node-2 needs it? It can't — the data is on node-1's RAM.

You need a **shared memory** that all 3 nodes can read/write at the same time.

### Redis = Remote Dictionary Server

Redis is an **in-memory database** that lives on its own server (or in the cloud). All 3 nodes connect to the same Redis. It's extremely fast because it keeps everything in RAM.

```
  Node-1                  Node-2                  Node-3
    │                       │                       │
    └──────────────────────┬┴──────────────────────┘
                           │
                    ┌──────▼──────┐
                    │    Redis    │
                    │   (RAM)     │
                    │             │
                    │ session:ph1 │  ← hash
                    │ conv:ph1    │  ← list
                    │ devices     │  ← hash
                    │ tool_corr:  │  ← string
                    │ ...         │
                    └─────────────┘
```

### Why Redis and not PostgreSQL/MySQL?

| | PostgreSQL (Disk DB) | Redis (In-memory DB) |
|---|---|---|
| Read speed | ~1-10ms | ~0.1ms |
| Write speed | ~5-20ms | ~0.1ms |
| Data model | Tables, rows | Key-value, hashes, lists, streams, sets |
| Pub/Sub | No (polling required) | Yes (instant push notifications) |
| TTL/Expiry | Manual cleanup | Auto-expire keys |
| Session data | Overkill | Perfect fit |

### Why not just store it in Python memory?

Because then node-2 can't access it. And if node-1 crashes, all sessions are gone. Redis is a separate process — if node-1 crashes, node-2 reads from Redis and continues.

---

## 7. Redis Data Structures We Use

Redis isn't just a key-value store. It has several data types, each useful for different things:

### Hash (Like a Python dict)

```
Key: "session:phone-android-123"
Fields:
  ├── device_id: "phone-android-123"
  ├── user_id: "user-abc"
  ├── created_at: "2025-01-01T00:00:00"
  ├── last_active: "2025-01-02T12:00:00"
  ├── config: {"language": "en", "tts_voice": "en"}
  └── status: "active"

Python: redis.hset("session:phone-123", "status", "active")
        redis.hget("session:phone-123", "status") → "active"
```

**Use**: Session metadata, device registration, permissions.

### List (Like a Python list)

```
Key: "conv:phone-android-123"
Values (ordered):
  [0]: {"role": "user", "content": "what's the weather?"}
  [1]: {"role": "tool", "content": "..."}
  [2]: {"role": "assistant", "content": "The weather is..."}
  [3]: {"role": "user", "content": "and in Cairo?"}

Python: redis.rpush("conv:phone-123", message_json)
        redis.lrange("conv:phone-123", -50, -1)  # last 50 messages
```

**Use**: Conversation history. New messages go to the right (rpush). Read the rightmost messages for recent history.

### Sorted Set (Like a Python dict with scores)

```
Key: "tts_queue"
Members with scores:
  ┌───────────────────────────────────────┐
  │ phone-123:What's the weather? → 17345 │
  │ phone-456:And in Cairo?      → 17346 │
  └───────────────────────────────────────┘
               ↑ score = timestamp

Python: redis.zadd("tts_queue", {"phone-123:text...": timestamp})
        redis.zrange("tts_queue", 0, 1)  # earliest job first
```

**Use**: Job queues where you want to process items in order.

### Stream (Like an append-only log)

```
Key: "stt_queue"
Entries:
  ┌──────┬──────────────────────────────────────────┐
  │ ID-1 │ device_id: phone-123                     │
  │      │ audio_data: <base64 audio chunk>         │
  │      │ chunk_index: 0, total_chunks: 3          │
  ├──────┼──────────────────────────────────────────┤
  │ ID-2 │ device_id: phone-123                     │
  │      │ audio_data: <base64 audio chunk>         │
  │      │ chunk_index: 1, total_chunks: 3          │
  └──────┴──────────────────────────────────────────┘
  
Python: redis.xadd("stt_queue", {"device_id": "...", "audio_data": "..."})
        redis.xread({"stt_queue": "0"}, block=5000)  # wait for new items
```

**Use**: Processing pipelines where you want to track what's been processed (like Kafka but simpler).

### Pub/Sub (Like a radio broadcast)

```
Channels:
  ┌─────────────────────────────┐
  │ najim:events               │  ← Cluster-wide announcements
  ├─────────────────────────────┤
  │ najim:ws_send:node-1       │  ← Only node-1 listens here
  │ najim:ws_send:node-2       │  ← Only node-2 listens here
  │ najim:ws_send:node-3       │  ← Only node-3 listens here
  └─────────────────────────────┘

Python (Publisher): redis.publish("najim:ws_send:node-1", message)
Python (Subscriber): pubsub.subscribe("najim:ws_send:node-1")
```

**Use**: Cross-node communication. Explained in section 14.

---

## 8. The Session Registry

### What It Does

The `SessionRegistry` manages one session per device. A session is created when the phone first connects, and it stores the language preference, creation time, etc.

### Redis Key

```
session:{device_id}  →  Hash
```

Example: `session:phone-android-123`

```python
{
  "device_id": "phone-android-123",
  "user_id": "user-abc",
  "created_at": "2025-01-01T00:00:00Z",
  "last_active": "2025-01-02T12:00:00Z",
  "config": "{\"language\": \"en\", \"tts_voice\": \"en\", ...}",
  "status": "active",
  "language": "en"
}
```

### Key Behaviors

- **TTL (Time To Live)**: 86400 seconds (24 hours). If the phone doesn't send any data for 24 hours, the session auto-deletes. Every time the phone sends a message, the TTL resets (via `touch()`).
- **Lazy loading**: If no session exists, the code creates a new one.
- **Config updates**: The phone can update language/voice mid-session via the PATCH endpoint.

### Why a Session per Device?

Because one user could have multiple Android devices (phone + tablet). Each device has its own conversation history, its own tools, its own language preference.

---

## 9. The Device Registry

### What It Does

Tracks which Android devices are currently connected and **which node** they are connected to.

### Redis Key

```
devices  →  Hash  (field = device_id, value = device info dict)
```

```python
{
  "phone-android-123": {
    "device_id": "phone-android-123",
    "user_id": "user-abc",
    "capabilities": "gps,file_index,http_server",
    "status": "online",
    "connected_at": "...",
    "last_heartbeat": "...",
    "node_id": "node-1"        // ← KEY FIELD
  },
  "phone-android-456": {
    ...
    "node_id": "node-2"
  }
}
```

Additionally:

```
device_ws:phone-android-123  →  String  "node-1"
```

### Why Two Places?

`devices` is a hash of all devices (good for listing). `device_ws:{id}` is a single key with TTL (fast lookup for routing). If the node crashes, the TTL expires and the device is assumed disconnected.

### Heartbeats

Every 30 seconds, the phone sends a heartbeat. The server updates `last_heartbeat` in Redis. If the server hasn't heard from the phone for ~35 seconds, it sends its own heartbeat probe. If no response, the connection is closed and the device marked offline.

---

## 10. The Conversation Store

### What It Does

Stores the history of the conversation between the user and the LLM, plus any tool call results.

### Redis Key

```
conv:{device_id}  →  List (each element is a JSON message)
```

```python
[
  {"role": "user", "content": "What's the weather in Cairo?", "timestamp": "..."},
  {"role": "tool", "content": "{'location': 'Cairo', 'temp': 32}", "tool_call_id": "call_1"},
  {"role": "assistant", "content": "The weather in Cairo is 32°C...", "timestamp": "..."}
]
```

### Why Does the LLM Need History?

When you say "What's the weather in Cairo?" and then "And in Alexandria?", the LLM needs to know you're still talking about weather. The conversation history provides that context. Without it, every turn would be a standalone question with no memory.

### What Message Roles Exist?

| Role | Description |
|------|-------------|
| `system` | Instructions to the LLM ("You are a helpful assistant") |
| `user` | The human speaking |
| `assistant` | The LLM's response |
| `tool` | Result of a tool call (weather data, file index, etc.) |

### Message Count Limit

We store a maximum of `max_history` messages (default: 100). Oldest messages are trimmed. This keeps the LLM context window within limits.

---

## 11. The Tool Registry

### What It Does

The `ToolRegistry` is a central list of all tools the system knows about. It has two categories: **internal** and **remote**.

### Internal Tools (Built-in)

These run directly on the cluster node. No network needed.

| Tool Name | Description | Speed |
|-----------|-------------|-------|
| `get_time` | Returns current UTC time | <1ms |
| `get_weather` | Returns mock weather data | ~50ms |
| `calculator` | Safe math expression evaluation | <1ms |

These are for testing. You can add more cluster-side tools easily.

### Remote Tools (On the Phone)

These are tools that live on the Android app. The cluster doesn't run them — it asks the phone to run them.

| Tool Name | Description |
|-----------|-------------|
| `index_files` | Index files on the phone's storage |
| `get_gps` | Get the phone's current GPS location |
| `start_http_server` | Start an HTTP server on the phone |

### How Do Tools Get Registered?

```python
# Manual registration in code:
registry.register_remote_tool("get_gps", ToolDefinition(
    name="get_gps",
    description="Get the phone's current GPS location",
    input_schema={
        "type": "object",
        "properties": {
            "accuracy": {"type": "number", "description": "desired accuracy in meters"}
        },
        "required": []
    }
))
```

The phone also announces its capabilities during the WebSocket handshake:

```json
{"type": "connect", "capabilities": ["gps", "file_index", "http_server"]}
```

---

## 12. Internal vs Remote Tools — The Router

The `ToolRouter` decides where to execute a tool call. The LLM says "I need to call `get_weather`" and the router figures out where `get_weather` lives.

```
                  ┌──────────────────┐
                  │  route_tool_call │
                  │  (tool_name)     │
                  └────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐       ┌───────────────────┐
    │  Internal Tool   │       │   Remote Tool     │
    │                 │       │                   │
    │  is_internal()  │       │  is_remote()      │
    │  = True         │       │  = True           │
    │                 │       │                   │
    │  Run locally    │       │  Send to phone    │
    │  <10ms          │       │  Wait for phone   │
    │                 │       │  30s timeout      │
    └─────────────────┘       └───────────────────┘
```

### What Happens If the Tool Doesn't Exist?

```python
raise UnknownToolError(f"Unknown tool: {tool_name}")
```

The LLM gets back an error message and tries something else.

---

## 13. The Remote Tool Bridge (Correlation IDs)

This is the most important piece of the architecture. Let me explain **why** it exists and **how** it works.

### The Problem

```
Node-1: "I need to call index_files on phone-android-123"
Node-1: "I'll wait for the result"

BUT: phone-android-123 might be connected to Node-2 (different node)

How does Node-1 know when the phone has responded?
```

### The Solution: Correlation IDs

A **correlation ID** is a unique identifier (UUID) that ties a request to its response.

### Step-by-Step

```
                    Node-1                              Redis                      Phone (on Node-2)
                      │                                  │                            │
   LLM says:         │                                  │                            │
   "call index_files"│                                  │                            │
                      │                                  │                            │
   1. Generate ID:   │                                  │                            │
      corr_abc       │                                  │                            │
                      │                                  │                            │
   2. Store pending  │                                  │                            │
      ──────────────►│  SET tool_corr:corr_abc          │                            │
                      │      {device_id, tool, etc}      │                            │
                      │      EX 35 (auto-delete in 35s)  │                            │
                      │                                  │                            │
   3. Publish to     │                                  │                            │
      phone's node   │  PUBLISH najim:ws_send:node-2    │                            │
      ──────────────►│      {type: "tool_request",      │                            │
                      │       correlation_id: "corr_abc",│                            │
                      │       tool: "index_files",       │  ◄── Node-2 receives      │
                      │       params: {...}}             │       pub/sub message     │
                      │                                  │       and sends via WS   │
                      │                                  │            │              │
                      │                                  │            │              │
                      │                                  │            ▼              │
   4. Create future: │                                  │  Phone executes tool      │
      await future   │                                  │                            │
      (blocked)      │                                  │  Phone sends response:     │
                      │                                  │  {type: "tool_response",   │
                      │                                  │   correlation_id: "corr_abc"│
                      │                                  │   result: {...}}          │
                      │                                  │            │              │
                      │                                  │  ◄─────────┘              │
                      │                                  │                            │
   5. Handle response│                                  │                            │
      ◄──────────────│  Node-2 receives WS message      │                            │
                      │  from phone                     │                            │
                      │                                  │                            │
   6. Resolve future:│                                  │                            │
      future.set_    │                                  │                            │
      result(result) │                                  │                            │
                      │                                  │                            │
   7. Unblock!       │                                  │                            │
      await returns  │                                  │                            │
      with result    │                                  │                            │
                      │                                  │                            │
   8. LLM continues  │                                  │                            │
```

### Why This Works Across Nodes

Node-1 doesn't know which node the phone is on. But it doesn't matter:
- It publishes to the channel for **all nodes** (or the specific node via `najim:ws_send:node-2`)
- Whichever node has the phone's WebSocket will receive the request
- That node sends it to the phone
- When the phone responds, that node calls `bridge.handle_response(correlation_id, result)`
- `handle_response` resolves the original future on **Node-1** (because futures are local to the Python process that created them)

Wait — futures are in Python memory on Node-1. How does Node-2 resolve it?

**It doesn't.** Here's the corrected flow:

```
Actually, the phone is connected to Node-2's WebSocket.
Node-1 wants to call a tool on the phone.

PRACTICAL REALITY:
- Phone is connected to Node-2 via WebSocket
- Node-1 asks Node-2 to deliver the tool request
- Phone responds to Node-2 via WebSocket
- Node-2 stores the response in Redis (tool_resp:{corr_id})
- Node-1 polls/watches Redis for the response

This is why tool_corr has a TTL of 35s — Node-1 keeps checking
if the response has arrived in Redis.

If Node-1 doesn't get a response within 30s → timeout error.
```

---

## 14. Redis Pub/Sub — How Nodes Talk To Each Other

### What Is Pub/Sub?

**Pub/Sub** stands for **Publish/Subscribe**. It's a messaging pattern where:

- **Publishers** send messages to a **channel** (don't know who receives them)
- **Subscribers** listen on a channel (don't know who sent the messages)
- Redis handles the routing

Think of it like a **radio station**:
- The radio host (publisher) talks on frequency 104.5 FM (channel)
- Anyone with a radio tuned to 104.5 FM (subscriber) hears the broadcast
- The host doesn't know who's listening
- Listeners don't know who's broadcasting

### Our Channels

| Channel | Who Publishes | Who Subscribes | Purpose |
|---------|--------------|----------------|---------|
| `najim:events` | Any node | All nodes | Cluster-wide announcements |
| `najim:ws_send:{node_id}` | Any node | Only that specific node | Send message to device on that node |

### Example: Node-1 needs to send a tool request to a phone on Node-2

```python
# Node-1 code (in ToolBridge.initiate_remote_call):
# 1. Find which node the phone is on
phone_node = await device_registry.get_node_for_device("phone-android-123")
# Returns: "node-2"

# 2. Publish to node-2's channel
channel = f"najim:ws_send:{phone_node}"  # "najim:ws_send:node-2"
await redis.publish(channel, {
    "type": "tool_request",
    "correlation_id": "corr_abc",
    "device_id": "phone-android-123",
    "tool_name": "index_files",
    "params": {"path": "/sdcard/Documents"},
})
```

```python
# Node-2 code (in _start_ws_listener, runs on startup):
# Subscribes to "najim:ws_send:node-2"
# When message arrives:
device_id = message["device_id"]
ws = _active_connections[device_id]
await ws.send_json(message)
```

### Why Not Just Direct HTTP Between Nodes?

Because the nodes might not be able to reach each other directly (firewalls, different networks). Redis acts as the central hub — every node talks to Redis, and Redis routes messages.

---

## 15. The Permission Store

### What It Does

Controls which tools a device is allowed to call. Not every device should be able to run `index_files` (maybe you only want the admin tablet to have that permission).

### Redis Key

```
perms:{device_id}  →  Hash
```

```python
"perms:phone-android-123": {
    "get_gps": "allow",
    "index_files": "deny",
    "start_http_server": "deny"
}
```

### How It's Checked

Before the router sends a remote tool call to the phone, it checks:

```python
if not await permission_store.has_permission(device_id, "index_files"):
    return ToolCallResult(
        success=False,
        error="Permission denied: device is not allowed to call index_files"
    )
```

### Default Behavior

If no permission is set for a tool → **deny**. Explicit allow is required.

### How to Set Permissions

```http
PUT /api/v1/permissions/{device_id}/{tool_name}?allowed=true
Authorization: Bearer <admin_token>
```

---

## 16. The Voice Pipeline (STT → LLM → TTS)

### What Is a "Pipeline"?

A pipeline is a sequence of processing steps. Audio goes in one end, audio comes out the other end, with text transformations in the middle.

### The Four Stages (Actually!)

```
Audio In  ──►  STT  ──►  LLM  ──►  TTS  ──►  WS Sender  ──►  Audio Out
                │            │            │           │
                ▼            ▼            ▼           ▼
           Speech to     Language     Text to     WebSocket
           Text          Model        Speech      Delivery
           (Whisper)     (Groq API)   (Piper)     (to phone)
           Local         Remote API   Local       Local
```

Each stage is **decoupled** by a Redis Stream. This is the **checkpoint pipeline** — every stage writes its output to a stream before the next stage begins.

### Redis Streams: The Checkpoints

```
┌─────────────┐    XADD        ┌─────────────┐    XADD        ┌─────────────┐
│ WS Handler  │ ─────────────► │  stt_jobs   │ ─────────────► │  llm_jobs   │
│             │                │   Stream    │                │   Stream    │
└─────────────┘                └──────┬──────┘                └──────┬──────┘
                                      │                              │
                                      ▼                              ▼
                               ┌─────────────┐               ┌─────────────┐
                               │ STT Worker  │               │ LLM Worker  │
                               │ XREADGROUP  │               │ XREADGROUP  │
                               └─────────────┘               └─────────────┘
                                                                     │
                                                                     │ XADD
                                                                     ▼
┌─────────────┐    send_json      ┌─────────────┐    XADD        ┌─────────────┐
│    Phone    │ ◄─────────────── │ WS Sender   │ ◄───────────── │  tts_jobs   │
│             │                   │ Worker      │                │   Stream    │
└─────────────┘                   └─────────────┘                └──────┬──────┘
                                        ▲                              │
                                        │ XREADGROUP                   ▼
                                        │                       ┌─────────────┐
                                        │                       │ TTS Worker  │
                                        │                       │ XREADGROUP  │
                                        │                       └─────────────┘
```

**What this means:**

| Stream | Who Writes | Who Reads | What's Inside |
|--------|-----------|-----------|---------------|
| `stt_jobs` | WS Handler | STT Worker | audio_data (base64), device_id, language |
| `llm_jobs` | STT Worker | LLM Worker | transcript, device_id, language |
| `tts_jobs` | LLM Worker | TTS Worker | response text, device_id, language |
| `responses` | TTS Worker | WS Sender | audio_data (base64), device_id, text |

### Why Streams Instead of Function Calls?

**Old way (function calls):**
```
Phone sends audio → Node-1 calls stt() → Node-1 calls llm() → Node-1 calls tts()
                                    │
                                    └── If Node-1 crashes here, EVERYTHING IS LOST
```

**New way (streams):**
```
Phone sends audio → XADD stt_jobs → STT Worker picks it up
                                         │
                                         └── If this worker crashes, job is still in stream
                                             Another node's STT worker picks it up
```

Each stream entry is a **checkpoint**. If any worker crashes, the job stays in the stream. Another node picks it up.

### Stage 1: STT (Speech-to-Text)

```
Location: pipeline/workers/stt_worker.py
Stream:   stt_jobs

Input (from stream):
  - audio_data: base64-encoded WAV (16kHz mono)
  - device_id: which phone sent it
  - language: optional language hint

Process:
  1. Base64 decode → raw audio bytes
  2. Write to temp WAV file
  3. Run Whisper in a thread (CPU-bound, blocking)
  4. Get: text + detected language + confidence
  5. XADD to llm_jobs stream
  6. XACK to acknowledge this job is done
  7. Delete temp file

Output (to llm_jobs):
  - text: transcribed text
  - device_id, session_id
  - language, probability

Time: ~100-500ms depending on audio length
Model: faster-whisper-medium (loaded once on startup)
```

### Stage 2: LLM (Large Language Model)

```
Location: pipeline/workers/llm_worker.py
Stream:   llm_jobs

Input (from stream):
  - text: what the user said
  - device_id: which phone
  - language: detected language

Process:
  1. Load conversation history from Redis (conv:{device_id})
  2. Build messages: [system prompt, history..., user text]
  3. Send to Groq API (llama-3.3-70b-versatile)
  4. If LLM calls tools → execute them (up to 5 iterations)
  5. Get final response text
  6. Save to conversation history
  7. XADD to tts_jobs stream
  8. XACK to acknowledge

Output (to tts_jobs):
  - response: LLM's reply text
  - device_id, session_id
  - input_text, language

Time: ~500ms-3s (depends on Groq API + tool calls)
Runs: needs internet (calls Groq API)
```

**The Tool Loop** — This is key. The LLM doesn't just answer. It can say "I need to call a tool first, then I'll answer." The code allows up to 5 tool calls per request. Example:

```
User: "What's the weather in Cairo and Alexandria?"

LLM: "I need to call get_weather twice"
    → call get_weather("Cairo") → result: {temp: 32}
    → call get_weather("Alexandria") → result: {temp: 28}
LLM: "The weather in Cairo is 32°C and in Alexandria it's 28°C."
```

### Stage 3: TTS (Text-to-Speech)

```
Location: pipeline/workers/tts_worker.py
Stream:   tts_jobs

Input (from stream):
  - response: text to speak
  - device_id: which phone
  - language: which voice to use

Process:
  1. Get Piper voice for this language
  2. Run synthesis in a thread (CPU-bound)
  3. Get WAV audio bytes
  4. Base64 encode
  5. XADD to responses stream
  6. XACK to acknowledge

Output (to responses):
  - audio: base64-encoded WAV (22050Hz mono)
  - text: what was spoken
  - device_id

Time: ~200-500ms
Model: Piper (loaded lazily on first use)
Voices: English (Cori), Arabic (Kareem)
```

### Stage 4: WS Sender (WebSocket Delivery)

```
Location: pipeline/workers/ws_sender.py
Stream:   responses

Input (from stream):
  - audio: base64-encoded audio
  - text: the spoken text
  - device_id: which phone to send to

Process:
  1. Check if device is connected to THIS node
  2. If yes → send via WebSocket
  3. If no → silently drop (phone reconnected elsewhere or disconnected)
  4. XACK to acknowledge

Output:
  - WebSocket message to phone: {"type": "audio_chunk", "audio_data": ..., "text": ...}

Time: <10ms (just sending data)
```

**Why check if device is on this node?**

The phone is connected to ONE node (e.g., node-1). But TTS Worker could be on node-2. The `responses` stream is read by all nodes. Only the node with the active WebSocket should send.

### Consumer Groups: How Multiple Nodes Share Work

Each stream has a **consumer group** called `najim_workers`. This is a Redis feature that distributes messages across consumers.

```
Stream: stt_jobs
Consumer Group: najim_workers
Consumers: worker:node-1:stt-0, worker:node-2:stt-0, worker:node-3:stt-0

When a job arrives:
  - Redis gives it to ONE consumer (round-robin or idlest)
  - That consumer processes it
  - Other consumers don't see it (unless it's not acknowledged)

If consumer crashes:
  - Job stays "pending" in the stream
  - After timeout, another consumer can claim it
```

### Retry Logic with Exponential Backoff

When a handler fails, we don't immediately retry. We use **exponential backoff**:

```
Attempt 1 fails → wait 1 second → retry
Attempt 2 fails → wait 2 seconds → retry
Attempt 3 fails → wait 4 seconds → retry
Attempt 4 fails → DISCARD (give up, acknowledge to remove from stream)
```

This prevents hammering a failing service. The backoff doubles each time: 1s → 2s → 4s → 8s.

```python
# In pipeline/workers/base.py
backoff_time = backoff_base * (2 ** delivery_count)  # 1, 2, 4, 8, ...
await asyncio.sleep(backoff_time)
```

### Why Is the Pipeline Async?

Remember: multiple phones can be connected at the same time. If Phone A is waiting for a slow tool call (like indexing 10,000 files), Phone B's weather query should not be delayed.

```python
# This is how it works — each phone's processing runs independently:
async def process_audio(device_id, audio_data):
    text = await stt.transcribe(audio_data)        # non-blocking
    response = await llm.run_query(device_id, text) # non-blocking  
    audio = await tts.synthesize(response)          # non-blocking
    return audio

# Multiple calls run concurrently:
await asyncio.gather(
    process_audio("phone-A", audio_a),
    process_audio("phone-B", audio_b),
    process_audio("phone-C", audio_c),
)
```

Python's `asyncio` handles the concurrency. While Phone A's STT is running in a thread, Phone B's LLM call can proceed. They don't block each other.

---

## 17. What Happens On Startup

When you start the server (`uvicorn app:app`):

```
1. Load configuration from config.yaml
   - Redis host/port/password
   - JWT secret
   - Model paths
   - etc.

2. Connect to Redis
   - Create connection pool (20 connections)
   - Ping to verify

3. Load ML Models
   - Whisper model (STT) — loads from disk, takes 5-10 seconds
   - Register TTS voice paths (English, Arabic) — models load lazily

4. Initialize LLM client
   - Not a connection, just a configured HTTP client

5. Start Pub/Sub listener
   - Subscribe to najim:ws_send:{this_node_id}
   - This allows other nodes to route messages through this node

6. Start serving
   - WebSocket endpoint ready
   - REST endpoints ready
   - Health check ready
```

---

## 18. What Happens On Shutdown

```
1. Close all active WebSocket connections
   - Send disconnect message to each device
   - Mark devices as offline in Redis

2. Stop pipeline workers
   - Cancel asyncio tasks
   - Workers finish current job, then exit

3. Close Pub/Sub subscriptions

4. Close Redis connection pool

5. Unload ML models (free memory)
```

---

## 19. Deep Dive: Redis Streams and Consumer Groups

### What Is a Redis Stream?

A **stream** is an append-only log. You can only ADD to the right, never modify in the middle, never delete individual entries (except trimming).

```
Stream: stt_jobs

Entry 1: {id: "1700000000001-0", data: {device_id: "phone-1", audio_data: "..."}} 
Entry 2: {id: "1700000000002-0", data: {device_id: "phone-2", audio_data: "..."}}
Entry 3: {id: "1700000000003-0", data: {device_id: "phone-1", audio_data: "..."}}
         ▲
         │
         └── ID is auto-generated: timestamp-sequence
```

### Basic Stream Commands

```bash
# Add entry
XADD stt_jobs * device_id phone-1 audio_data "base64..."
# Returns: "1700000000001-0" (the generated ID)

# Read all entries
XRANGE stt_jobs - +

# Read new entries (blocking)
XREAD BLOCK 5000 STREAMS stt_jobs $
# Waits up to 5 seconds for new entries

# Trim old entries (keep last 1000)
XTRIM stt_jobs MAXLEN 1000
```

### What Is a Consumer Group?

A **consumer group** is like a team of workers sharing a queue. Each message is delivered to only ONE consumer in the group.

```
Stream: stt_jobs
Group: najim_workers
Consumers: [worker-node-1-stt, worker-node-2-stt, worker-node-3-stt]

Entry 1 → goes to worker-node-1-stt
Entry 2 → goes to worker-node-2-stt
Entry 3 → goes to worker-node-3-stt
Entry 4 → goes to worker-node-1-stt (round-robin)
```

### Consumer Group Commands

```bash
# Create group (once)
XGROUP CREATE stt_jobs najim_workers $ MKSTREAM
# $ = start from now (new messages only)
# 0 = start from beginning (all messages)
# MKSTREAM = create stream if it doesn't exist

# Read as a consumer
XREADGROUP GROUP najim_workers consumer-node-1-stt COUNT 1 BLOCK 5000 STREAMS stt_jobs >
# The ">" means "give me new messages I haven't seen"

# Acknowledge (mark as processed)
XACK stt_jobs najim_workers 1700000000001-0

# Check pending messages (not yet acknowledged)
XPENDING stt_jobs najim_workers
# Returns: total pending, min ID, max ID, consumers with pending count

# Detailed pending info
XPENDING stt_jobs najim_workers - + 10
# Returns: message ID, consumer, idle time (ms), delivery count
```

### The Lifecycle of a Stream Entry

```
1. XADD stt_jobs → Entry created, ID = "1700000000001-0"

2. XREADGROUP by worker-node-1-stt → Entry is now "pending"
   Redis tracks: this entry belongs to worker-node-1-stt
   Delivery count = 1

3a. SUCCESS: Worker processes, XACK → Entry is "processed" (still in stream, but marked done)

3b. FAILURE: Worker crashes without XACK → Entry stays "pending"
   After visibility timeout, another worker can claim it
   Delivery count increments

4. After max_retries deliveries → We XACK to discard it (give up)
```

### Claiming Pending Messages

If a consumer crashed with pending messages:

```bash
# See what's pending
XPENDING stt_jobs najim_workers - + 10

# Claim pending messages for yourself
XCLAIM stt_jobs najim_workers worker-node-2-stt 5000 1700000000001-0
# 5000 = min idle time (ms) before you can claim it
# This transfers ownership from crashed consumer to you
```

Our implementation handles this automatically: after each failed attempt, we sleep (backoff) and the consumer group will redeliver.

### Why Consumer Groups Instead of Plain XREAD?

| Plain XREAD | Consumer Group (XREADGROUP) |
|-------------|---------------------------|
| All consumers see all messages | Each message goes to ONE consumer |
| You track what's processed manually | Redis tracks it for you |
| No automatic retry | Automatic retry via delivery count |
| No crash recovery | Pending messages can be claimed |
| No scaling | Add consumers, they auto-share work |

### Visual: How Messages Flow Through Consumer Group

```
                    stt_jobs Stream
                    ┌─────────────────────────────────┐
                    │ Msg 1 │ Msg 2 │ Msg 3 │ Msg 4 │ Msg 5 │
                    └────┬───┴───┬───┴───────┴───┬───┴───────┘
                         │       │               │
         ┌───────────────┼───────┼───────────────┤
         │               │       │               │
         ▼               ▼       ▼               ▼
   ┌──────────┐   ┌──────────┐   ┌──────────┐
   │ Node-1   │   │ Node-2   │   │ Node-3   │
   │ Consumer │   │ Consumer │   │ Consumer │
   │          │   │          │   │          │
   │ Msg 1 ✓  │   │ Msg 2 ✓  │   │ Msg 3 ✓  │
   │ Msg 4 ✓  │   │ Msg 5 ⏳ │   │          │
   └──────────┘   └──────────┘   └──────────┘
         │               │
         │               └── Processing... (if crashes, Msg 5 becomes pending)
         │
         └── ✓ = XACK'd (processed successfully)
```

### Our Pipeline Streams

| Stream | Consumer Group | What Happens |
|--------|---------------|--------------|
| `stt_jobs` | `najim_workers` | Audio → Text |
| `llm_jobs` | `najim_workers` | Text → Response (may call tools) |
| `tts_jobs` | `najim_workers` | Response → Audio |
| `responses` | `najim_workers` | Audio → WebSocket send |

All use the same group name because they're independent streams (not competing for same messages).

---

## 20. Deep Dive: WebSocket Protocol

### Why WebSocket?

**HTTP** is request-response. Client asks, server answers. Done. Connection closes.

```
Client: GET /weather?city=Cairo
Server: {"temp": 32}
[Connection closes]
```

**WebSocket** is a persistent, two-way connection. Both sides can send at any time.

```
Client: Connect ws://server/connect
Server: Accept
[Connection stays open]

Client: Here's audio chunk 1
Server: Got it
Client: Here's audio chunk 2
Server: Here's interim transcript
Server: I need to call a tool
Client: Here's the tool result
Server: Here's the final audio
... (continues until either side closes)
```

### WebSocket Handshake

```
1. Client sends HTTP request with upgrade header:
   GET /api/v1/connect?token=eyJhbGci... HTTP/1.1
   Host: localhost:8080
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
   Sec-WebSocket-Version: 13

2. Server validates JWT from query parameter
   - If invalid → HTTP 401, connection rejected
   - If valid → proceed

3. Server sends HTTP 101 Switching Protocols:
   HTTP/1.1 101 Switching Protocols
   Upgrade: websocket
   Connection: Upgrade
   Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=

4. Now it's a WebSocket connection, not HTTP anymore.
   Both sides can send frames (messages).
```

### Message Types: Client → Server

| Type | When | Example |
|------|------|---------|
| `connect` | First message after handshake | `{"type": "connect", "capabilities": ["gps", "file_index"]}` |
| `heartbeat` | Every 30 seconds | `{"type": "heartbeat"}` |
| `audio` | User speaks | `{"type": "audio", "audio_data": "base64...", "language": "en"}` |
| `tool_response` | After executing a tool | `{"type": "tool_response", "correlation_id": "uuid", "result": {...}}` |
| `disconnect` | Graceful close | `{"type": "disconnect"}` |

### Message Types: Server → Client

| Type | When | Example |
|------|------|---------|
| `connected` | After `connect` message | `{"type": "connected", "device_id": "...", "node_id": "..."}` |
| `heartbeat_ack` | After `heartbeat` | `{"type": "heartbeat_ack", "timestamp": 1700000000}` |
| `accepted` | After receiving audio | `{"type": "accepted", "message": "Processing started"}` |
| `interim_transcript` | During STT | `{"type": "interim_transcript", "text": "What's the..."}` |
| `thinking` | During LLM | `{"type": "thinking", "text": "Let me check..."}` |
| `tool_request` | LLM needs tool | `{"type": "tool_request", "correlation_id": "...", "tool_name": "get_gps", "params": {...}}` |
| `audio_chunk` | Final response | `{"type": "audio_chunk", "audio_data": "base64...", "text": "The weather is..."}` |
| `error` | Something failed | `{"type": "error", "message": "STT failed"}` |

### The Message Loop (Server Side)

```python
# Simplified version of api/websocket.py
async def websocket_handler(websocket, token):
    # 1. Validate JWT
    claims = verify_jwt(token)
    device_id = claims["device_id"]
    
    # 2. Accept connection
    await websocket.accept()
    
    # 3. Wait for connect message (10s timeout)
    connect_msg = await asyncio.wait_for(websocket.receive_json(), timeout=10)
    capabilities = connect_msg.get("capabilities", [])
    
    # 4. Register device
    await device_registry.register(device_id, {"node_id": my_node_id, ...})
    
    # 5. Message loop
    heartbeat_missed = 0
    while True:
        try:
            msg = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=heartbeat_interval + 5
            )
            heartbeat_missed = 0  # Reset on any message
            
            if msg["type"] == "heartbeat":
                await websocket.send_json({"type": "heartbeat_ack"})
            
            elif msg["type"] == "audio":
                await handle_audio(device_id, msg)
            
            elif msg["type"] == "tool_response":
                await tool_bridge.handle_response(msg["correlation_id"], msg["result"])
            
            elif msg["type"] == "disconnect":
                break
        
        except asyncio.TimeoutError:
            heartbeat_missed += 1
            if heartbeat_missed >= 3:
                await websocket.close(code=4004, reason="Heartbeat timeout")
                break
            # Send probe heartbeat
            await websocket.send_json({"type": "heartbeat"})
    
    # 6. Cleanup
    await device_registry.set_status(device_id, "offline")
```

### WebSocket Close Codes

| Code | Meaning | When We Use It |
|------|---------|---------------|
| 1000 | Normal close | Graceful disconnect |
| 1001 | Going away | Server shutting down |
| 4001 | Missing token | No JWT in query param |
| 4002 | Invalid token | JWT verification failed |
| 4003 | Missing device_id | JWT doesn't have device_id |
| 4004 | Heartbeat timeout | Missed 3 heartbeats |

---

## 21. The Full Data Flow — A Complete Trace

Let's trace a single voice request from beginning to end. Phone A says "What's the weather in Cairo?" connected to Node-1.

```
  Phone A                    Node-1                        Redis                 Groq API
    │                          │                            │                      │
    │ (1) WS: audio data      │                            │                      │
    │ ───────────────────────►│                            │                      │
    │                          │                            │                      │
    │                          │ (2) base64 decode          │                      │
    │                          │ (3) Write temp WAV file    │                      │
    │                          │                            │                      │
    │                          │ (4) STT: Whisper thread    │                      │
    │                          │ ───────── waiting ────────►│                      │
    │                          │                            │                      │
    │                          │ (5) Transcribed:           │                      │
    │                          │     "What's the weather    │                      │
    │                          │      in Cairo?"            │                      │
    │                          │                            │                      │
    │ (6) WS: interim transcript │                          │                      │
    │ ◄─────────────────────────│                           │                      │
    │                          │                            │                      │
    │                          │ (7) Load conversation      │                      │
    │                          │     history for Phone A    │                      │
    │                          │ ──────────────────────────►│ conv:phone-a        │
    │                          │ ◄──────────────────────────│                      │
    │                          │                            │                      │
    │                          │ (8) Build messages:        │                      │
    │                          │     [system prompt,        │                      │
    │                          │      user: "What's the     │                      │
    │                          │            weather...?"]    │                      │
    │                          │                            │                      │
    │                          │ (9) LLM call               │                      │
    │                          │ ───────────────────────────────────────────────►│
    │                          │                            │                      │
    │                          │ (10) LLM responds:         │                      │
    │                          │     tool_call: get_weather │                      │
    │                          │     params: {loc: "Cairo"} │                     │
    │                          │ ◄───────────────────────────────────────────────│
    │                          │                            │                      │
    │                          │ (11) Router checks:        │                      │
    │                          │     "get_weather" is       │                      │
    │                          │     INTERNAL tool          │                      │
    │                          │                            │                      │
    │                          │ (12) Run get_weather       │                      │
    │                          │     Returns: {temp: 32}    │                      │
    │                          │                            │                      │
    │                          │ (13) Send tool result back │                      │
    │                          │     to LLM                 │                      │
    │                          │ ───────────────────────────────────────────────►│
    │                          │                            │                      │
    │                          │ (14) LLM final response:   │                      │
    │                          │     "The weather in Cairo  │                      │
    │                          │      is 32°C."            │                      │
    │                          │ ◄───────────────────────────────────────────────│
    │                          │                            │                      │
    │                          │ (15) Save to history       │                      │
    │                          │ ──────────────────────────►│ conv:phone-a        │
    │                          │                            │                      │
    │                          │ (16) TTS: Piper            │                      │
    │                          │     Synthesize WAV from    │                      │
    │                          │     "The weather in Cairo" │                      │
    │                          │                            │                      │
    │                          │ (17) base64 encode WAV     │                      │
    │                          │                            │                      │
    │ (18) WS: audio_chunk     │                            │                      │
    │ ◄─────────────────────────│                           │                      │
    │                          │                            │                      │
    │ (19) Phone plays audio   │                            │                      │
```

### What If a Remote Tool Was Needed?

Change step 12: instead of running `get_weather` locally, it would call `call_client_tool()` which generates a correlation ID, publishes to `najim:ws_send:node-X`, and waits for the phone's response.

---

## 20. Security Model

### Layers of Security

```
                    ┌─────────────────────────────────────┐
                    │          Firewall                   │
                    │  (only expose port 443, 8443)       │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         Load Balancer               │
                    │  (terminates TLS/SSL)                │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         JWT Authentication          │
                    │  (every request needs valid token)   │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │      Permission Checks              │
                    │  (per-device tool permissions)      │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         Rate Limiting               │
                    │  (60 req/min per device)            │
                    └─────────────────────────────────────┘
```

### What's Protected?

| Asset | Protection |
|-------|-----------|
| WebSocket endpoint | JWT token in query parameter |
| REST API | JWT token in Authorization header |
| Tool calls | PermissionStore per device |
| Redis | Password + TLS encryption |
| Cluster | Configurable node_id (not exposed externally) |

### What About the LLM API Key?

The Groq API key is stored in the config file. It never leaves the cluster nodes. Phones don't see it. Other nodes don't see it (it's in local config, not Redis).

---

## 21. Glossary

| Term | Meaning |
|------|---------|
| **STT** | Speech-to-Text. Converts audio to text (Whisper). |
| **TTS** | Text-to-Speech. Converts text to audio (Piper). |
| **LLM** | Large Language Model. The "brain" (Groq/llama). |
| **JWT** | JSON Web Token. A signed token for authentication. |
| **WebSocket** | Persistent two-way connection. |
| **Redis** | In-memory database, used as shared state + pub/sub. |
| **Pub/Sub** | Publish/Subscribe. Messages go to channels. |
| **Correlation ID** | UUID that ties a tool request to its response. |
| **Load Balancer** | Distributes incoming connections across nodes. |
| **Session** | Per-device state (language, config, active status). |
| **Conversation** | Message history between user and LLM. |
| **Internal Tool** | Tool that runs on the cluster node. |
| **Remote Tool** | Tool that runs on the Android phone. |
| **TTL** | Time To Live. Auto-delete Redis key after N seconds. |
| **Base64** | Binary-to-text encoding (used for audio in JSON). |
| **FastAPI** | Python async web framework (what we built on). |
| **ASGI** | Async Server Gateway Interface (uvicorn serves it). |
| **Node** | One computer in the cluster (we have 3). |
| **Pipeline** | Sequence: STT → LLM → TTS. |
| **Rate Limiting** | Max requests per time window per device. |
| **HMAC-SHA256** | Algorithm used to sign JWTs. |
| **asyncio** | Python's async/await concurrency. |
| **Tenacity** | Python library for retry logic. |

---

## Quick Reference: Redis Keys

```
session:{device_id}          Hash     Session metadata (TTL 24h)
conv:{device_id}             List     Conversation history
devices                      Hash     All registered devices
device_ws:{device_id}        String   Node ID where device is connected
perms:{device_id}            Hash     Per-device tool permissions
tool_corr:{correlation_id}   String   Pending tool call (TTL 35s)
tool_resp:{correlation_id}   String   Tool call response (BLPOP target)
stt_jobs                     Stream   Pipeline checkpoint: audio → STT
llm_jobs                     Stream   Pipeline checkpoint: transcript → LLM
tts_jobs                     Stream   Pipeline checkpoint: text → TTS
responses                    Stream   Pipeline checkpoint: audio → WS
najim:events                 Channel  Cluster-wide pub/sub
najim:ws_send:{node_id}      Channel  Node-specific message relay
```

---

## Quick Reference: Files

```
core/config.py           Pydantic settings (Redis, JWT, models, cluster)
core/redis_manager.py    Async Redis client with all operations
core/schemas.py          All Pydantic data models (Session, Message, etc.)
core/jwt_auth.py         JWT create, verify, auth dependencies
core/app_state.py        ML model storage (Whisper, TTS, LLM client)

sessions/session_registry.py    Session CRUD in Redis
sessions/conversation_store.py  Message history in Redis
sessions/device_registry.py     Device tracking and heartbeat
sessions/permissions.py         Per-device tool permissions

tools/registry.py               Central tool registry (internal + remote)
tools/call_client_tool.py       Remote tool bridge (correlation IDs)
tools/internal_tools.py         Built-in tool implementations
tools/router.py                 Tool call routing decision

pipeline/stt_buffer.py          STT audio queue + Whisper transcription
pipeline/llm_runner.py          LLM orchestration with tool loop
pipeline/tts_queue.py           TTS synthesis queue
pipeline/orchestrator.py        Full pipeline orchestration

api/websocket.py                WebSocket endpoint + connection management
api/sessions.py                 REST session CRUD endpoints
api/health.py                   Health, metrics, readiness endpoints

app.py                          FastAPI app, wires everything together
config.yaml                     All configuration
```

---

*End of tutorial. If something is still unclear, point me at the specific part and I'll go deeper.*

---

## 22. Frequently Asked Questions

### Q1: What if node-1 crashes mid-pipeline (STT? LLM? TTS? Tool call?)

**Short answer**: The checkpoint pipeline SAVES the work. Each stage writes to a Redis stream. If a node crashes, another node picks up the job from the stream. The user's turn is NOT lost (except in rare edge cases).

**This is IMPLEMENTED, not planned.**

### Detailed table — what happens at each stage:

| Crash during | What's in Redis | Recovery |
|---|---|---|
| Receiving audio via WebSocket | Audio already `XADD`'d to `stt_jobs` stream | Any node's STT worker picks it up |
| STT (Whisper transcribing) | Job is "pending" in `stt_jobs` consumer group | After visibility timeout, another node's STT worker claims it |
| STT done, result in `llm_jobs` | Transcript safely in `llm_jobs` stream | Any node's LLM worker picks it up |
| LLM awaiting Groq API response | Job is "pending" in `llm_jobs` | Consumer group will retry (exponential backoff) |
| LLM done, awaiting tool response from phone | `tool_corr:{id}` in Redis with TTL 35s, response goes to `tool_resp:{id}` | Any node waiting on `BLPOP tool_resp:{id}` gets it |
| TTS synthesizing | Response text in `tts_jobs` stream | Any node's TTS worker picks it up |
| TTS done, audio in `responses` | Audio in `responses` stream | WS Sender on any node picks it up |
| Sending to phone | Audio in `responses` + phone is on a specific node | If that node died, phone reconnects to another node, but this audio is "orphaned" |

### The Only Edge Case: Orphaned Responses

If TTS finishes and puts audio in `responses`, but the target node died:
- The audio sits in the stream
- WS Sender on the dead node can't deliver it
- WS Sender on other nodes check: "Is this phone connected to ME?" → No → Skip
- Phone reconnects to a new node and sends new audio

**Mitigation (future)**: Could store undelivered responses and replay on reconnect. For now, user just asks again.

### How Consumer Groups Enable Recovery

```
Timeline:
────────────────────────────────────────────────────────────────►

Node-1: XREADGROUP stt_jobs → gets job ID 123
        Processing... 
        [CRASH]

Job 123 is now "pending" in stt_jobs, owned by consumer "worker:node-1:stt-0"

After ~5 seconds (visibility timeout):
Node-2: XREADGROUP stt_jobs → sees pending job 123
        Claims it: "I'll take over this job"
        Processes it successfully
        XACK stt_jobs 123  → job marked as done
```

Redis tracks:
- Which consumer is processing which job
- How long it's been pending
- How many times it's been delivered

If delivery count exceeds `max_retries` (default 3), we acknowledge and discard it (give up).

### The `device_ws:{device_id}` Key

We also track which node a phone is connected to:

```
device_ws:phone-android-123 = "node-1"  (TTL: 35 seconds)
```

This TTL auto-expires if the node dies. The phone's heartbeat refreshes it every 30 seconds.

**If node-1 crashes:**
1. `device_ws:phone-123` TTL expires after 35s
2. Phone's WebSocket disconnects
3. Phone reconnects → load balancer sends to node-2
4. Node-2 sees: no `device_ws:phone-123` key → phone is "new" here
5. Phone's session data is still in Redis (`session:phone-123`, `conv:phone-123`)
6. Node-2 continues seamlessly

---

### Q2: How much Redis memory per request (average)?

| Item | Size | Notes |
|---|---|---|
| Session metadata | ~500 bytes | Per device, stored once |
| Audio base64 in STT queue | ~200KB | For 5 seconds of audio (only if using stream queue) |
| Transcribed text (message) | ~200 bytes | Per conversation turn |
| LLM response (message) | ~500 bytes | Per conversation turn |
| Tool call record | ~300 bytes | TTL 35s, auto-deleted |
| TTS audio | 0 | Returned directly, not persisted in Redis |

**Per turn**: ~201KB (mostly the base64 audio in the queue if streaming), or ~1KB if processing inline.
**Per session (50 turns)**: ~10.5MB with streaming, ~50KB without.

---

### Q3: How many tools can I call in a single turn?

Up to 5 iterations of the tool loop (configurable via `mcp.max_tool_loops`). In each iteration, the LLM can call multiple tools in parallel. So the theoretical max is **5 iterations × N parallel calls** (where N is up to the LLM's batch limit, usually 5-10).

Example:
```
User: "What's the weather in Cairo, Alexandria, and Luxor?"

Iteration 1:
  LLM calls get_weather(Cairo), get_weather(Alexandria), get_weather(Luxor) — in parallel
  Results come back

Iteration 2:
  LLM generates final response — no more tool calls
  Done
```

---

### Q4: How do I set up Redis TLS?

**Redis TLS is NOT like HTTPS.** HTTPS uses public Certificate Authorities (Let's Encrypt, etc.) that automatically issue trusted certificates. Redis requires **self-managed certificates** created with `openssl`.

**For a local LAN cluster (recommended):** Skip TLS, just use a password:

```yaml
redis:
  host: "192.168.1.100"
  port: 6379
  password: "your-strong-password-here"
  tls: false
```

**For production over the internet (complex):**

1. Generate a Certificate Authority (CA) cert:
   ```bash
   openssl req -new -x509 -days 365 -nodes -out ca.crt -keyout ca.key
   ```

2. Generate server cert for Redis:
   ```bash
   openssl req -new -nodes -out redis.csr -keyout redis.key
   openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis.crt
   ```

3. Configure Redis server (`redis.conf`):
   ```
   port 0
   tls-port 6379
   tls-cert-file /path/to/redis.crt
   tls-key-file /path/to/redis.key
   tls-ca-cert-file /path/to/ca.crt
   ```

4. Configure the client (your `config.yaml`):
   ```yaml
   redis:
     url: "rediss://:password@host:6379/0"
     tls: true
   ```
   (Note: `rediss://` with double `s` means Redis over TLS)

---

### Q5: What if I have a single node and a single Redis server?

Works perfectly. The architecture scales from 1 to N nodes. With 1 node:

- No load balancer needed — devices connect directly
- Pub/sub still works (the same node publishes and subscribes to itself)
- Session TTL, conversation history, tool correlation — all work identically

Set:
```yaml
cluster:
  node_id: "node-1"
```

---

### Q6: How do I set up Redis with Docker (simple)?

```yaml
# docker-compose.yml
version: "3"
services:
  redis:
    image: redis:7-alpine
    command: redis-server --requirepass mypassword --port 6379
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data

  node-1:
    build: .
    environment:
      REDIS_URL: "redis://:mypassword@redis:6379/0"
      JWT_SECRET: "your-jwt-secret"
      CLUSTER_NODE_ID: "node-1"
    depends_on:
      - redis

volumes:
  redis-data:
```

Set `REDIS_URL=redis://:mypassword@redis:6379/0` in the environment. Done. No certs, no extra config files. Just password + URL.

---

### Q7: How does a node get its serial/number?

By default, from the OS hostname:

```python
class ClusterSettings(BaseSettings):
    node_id: str = os.uname().nodename   # gets hostname
```

Override in config.yaml:
```yaml
cluster:
  node_id: "atom-node-1"
```

Or via environment variable:
```bash
export CLUSTER_NODE_ID="atom-node-1"
```

---

### Q8: What if my nodes are weak and processing takes 2+ minutes?

**Option A: Streaming (best)** — Send partial results as they come. The user hears the LLM start talking while it's still generating more. Requires LLM streaming (`stream=True`).

**Option B: "I'm thinking" response (safest)** — Recommended for weak nodes:

```
1. Phone sends audio via WebSocket
2. Node immediately sends back:
   {"type": "thinking", "text": "Give me a moment..."}
   Phone plays a short "processing" chime

3. Node processes in background:
   - XADD audio to stt_jobs stream → returns job_id to phone
   
4. Background workers:
   - STT worker picks up → transcribes
   - LLM worker picks up → generates response
   - TTS worker picks up → synthesizes audio

5. Node sends when ready:
   {"type": "response", "job_id": "...", "audio_data": "..."}
```

The user never feels like the connection dropped because they hear a "processing" sound immediately.

**Option C: Job queue pattern** — Phone uploads audio, gets a job ID, polls later or waits for a notification.

---

### Q9: How can I test without a thin client? Using curl?

**REST endpoints** (available via curl):

```bash
# 1. Health check (no auth needed)
curl http://localhost:8000/health

# 2. Readiness check
curl http://localhost:8000/ready

# 3. Liveness check  
curl http://localhost:8000/live

# 4. List sessions (with JWT)
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/sessions

# 5. Get device list
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/devices

# 6. Get conversation history for a device
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/conversations/device-123
```

**WebSocket** (need `wscat` or `websocat`):

```bash
# Install wscat
npm install -g wscat

# Connect (replace token with a real JWT)
wscat -c "ws://localhost:8000/api/v1/connect?token=<JWT>"

# Once connected, send messages:
{"type": "heartbeat"}
{"type": "connect", "capabilities": ["gps", "test"]}
```

---

### Q10: How does the node know what tools the phone has?

Two mechanisms:

1. **Phone announces capabilities on connect:**
   ```json
   {"type": "connect", "capabilities": ["gps", "file_index", "http_server"]}
   ```
   These are stored in Redis (`devices` hash → `capabilities` field).

2. **Backend has a ToolRegistry** with predefined remote tools:
   ```python
   self._remote["get_gps"] = ToolDefinition(name="get_gps", ...)
   self._remote["index_files"] = ToolDefinition(name="index_files", ...)
   ```

The router checks: is this tool registered? Does the device have the capability? Does the device have permission? If all yes → send tool request.

**Dynamic discovery (planned)**: The phone could send its full tool definitions on connect, and the backend would register them dynamically instead of having hardcoded remote tools.

---

### Q11: Does the old config.yaml auth still work? Rate limiting?

**Auth**: The old API key system was replaced by JWT. The new config has both:

```yaml
auth:
  jwt_only: true                    # JWT only (recommended)
  api_keys:                         # Legacy API keys (for fallback)
    "sk-dev-001": { name: "dev", rate_limit: 100 }
```

If `jwt_only: false`, the server tries JWT first, then falls back to API keys.

**Rate limiting**: Re-wired via `slowapi` in the new `app.py`. Configurable:

```yaml
api:
  rate_limit: "60/minute"           # Global default
```

Each REST endpoint has `@limiter.limit(...)` decorators (to be added per endpoint).

---

### Q12: What config values changed from the old version?

| Old config key | New config key | Notes |
|---|---|---|
| `stt.model_path` | `stt.model_dir` + `stt.model_name` | Now constructed as `{model_dir}/whisper-{model_name}` |
| `stt.hf_repo` | `stt.hf_repo` | Restored |
| `stt.device` | `stt.device` | Restored (was hardcoded to "cpu") |
| `stt.compute_type` | `stt.compute_type` | Restored (was hardcoded to "int8") |
| `tts.en.local_path` | `tts.voices.en.local_path` | Now nested under `voices` dict |
| `tts.ar.voice` | `tts.voices.ar.voice` | Same structure |
| `settings.volume` | `tts.synthesis.volume` | Moved under `tts.synthesis` |
| `settings.length_scale` | `tts.synthesis.length_scale` | Same |
| `settings.noise_scale` | `tts.synthesis.noise_scale` | Same |
| `settings.noise_w_scale` | `tts.synthesis.noise_w_scale` | Same |
| `llm.api_url` | `llm.api_base_url` | Renamed |
| `mcp.servers` | `mcp.servers` | Now defaults to empty list `[]` |
| `auth.api_keys` | `auth.api_keys` | Kept for backward compatibility |
| `redis` | `redis` | New section |
| `jwt` | `jwt` | New section |
| `cluster` | `cluster` | New section |
| `session` | `session` | New section |
| `tool` | `tool` | New section |

---

### Q13: Redis TLS setup in detail

**For a local LAN cluster — DON'T use TLS.** Just use a password:

```yaml
redis:
  url: "redis://:mypassword@192.168.1.100:6379/0"
  # or without URL:
  host: "192.168.1.100"
  port: 6379
  password: "mypassword"
  tls: false
```

This is safe on a local network not exposed to the internet. If your LAN is compromised, TLS won't help anyway.

**For production over internet — you need certificates:**

Step 1: Create a Certificate Authority and server certificates:

```bash
# CA
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt

# Redis server
openssl genrsa -out redis.key 2048
openssl req -new -key redis.key -out redis.csr
openssl x509 -req -in redis.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out redis.crt -days 365
```

Step 2: Configure Redis server (`redis.conf`):
```
port 0
tls-port 6379
tls-cert-file /etc/redis/redis.crt
tls-key-file /etc/redis/redis.key
tls-ca-cert-file /etc/redis/ca.crt
tls-auth-clients yes
requirepass mypassword
```

Step 3: Configure your app:
```yaml
redis:
  url: "rediss://:mypassword@redis-host:6379/0"
  # rediss:// = Redis over TLS
```

Your app's `redis_manager.py` uses `SSLConnection` when `tls: true`, which handles the client side of the TLS handshake. If using URL format with `rediss://`, the driver auto-detects TLS.

---

### Q14: I asked about the old config values. Which ones are still used?

All old config values are available. Here's the mapping explicitly:

**STT** — all restored:
```yaml
stt:
  model_name: "medium"        # was: part of model_path
  model_dir: "./models"       # base dir
  hf_repo: "Systran/..."      # RESTORED — for downloader
  device: "cpu"               # RESTORED — was hardcoded
  compute_type: "int8"        # RESTORED — was hardcoded
  beam_size: 5                # kept
  vad_filter: true            # kept
  language: null              # kept (null = auto-detect)
```

**TTS** — restored with synthesis config:
```yaml
tts:
  model_dir: "./models"
  voices:
    en:
      local_path: "TTS-CORI-EN"
      hf_repo: "rhasspy/piper-voices"
      voice: "en.en_GB.cori.high"
  synthesis:
    volume: 0.75              # RESTORED
    length_scale: 1.0         # RESTORED
    noise_scale: 0.75         # RESTORED
    noise_w_scale: 0.5        # RESTORED
    normalize_audio: true     # RESTORED
    nchannels: 1              # RESTORED
    sampwidth: 2              # RESTORED
    framerate: 22050          # RESTORED
```

**Auth** — both JWT and legacy API keys:
```yaml
auth:
  jwt_only: true              # true = JWT only, false = JWT + API keys
  api_keys:                   # kept for backward compat
    "sk-dev-001": { name: "dev", rate_limit: 100 }
```

**Rate limiting** — configurable:
```yaml
api:
  rate_limit: "60/minute"
```

---

### Q15: I have a single node. Does this still make sense without Redis?

**NO.** You need Redis even with a single node. Here's why:

| Feature | Without Redis | With Redis |
|---|---|---|
| Session TTL auto-expiry | Manual cleanup | Built-in |
| Conversation history persistence | Lost on restart | Survives restart |
| Tool correlation IDs | In-memory, lost on crash | Survives crash |
| Pub/Sub for future multi-node | Impossible | Ready to scale |
| Per-device permissions | In-memory dict | Persistent hash |

Redis is not optional. It's the backbone. Without it you just have a single-user in-memory app with no persistence and no scalability.

If you don't want a separate Redis process, you could embed it (redis with `--save ""` and `--appendonly no`), but a separate Redis Docker container is simpler.

---

### Q16: What does every config.yaml section look like now?

```yaml
# ─────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────
api:
  host: "0.0.0.0"
  port: 8080
  debug: false
  cors_origins: ["*"]
  rate_limit: "60/minute"

# ─────────────────────────────────────────────────
# REDIS
# ─────────────────────────────────────────────────
redis:
  url: "redis://:password@host:6379/0"      # Simplest: just a URL
  # OR separate fields:
  # host: "localhost"
  # port: 6379
  # password: ""
  tls: false
  pool_size: 20

# ─────────────────────────────────────────────────
# JWT
# ─────────────────────────────────────────────────
jwt:
  secret: "YOUR_RANDOM_SECRET_HERE"
  algorithm: "HS256"
  expiry_minutes: 1440

# ─────────────────────────────────────────────────
# CLUSTER
# ─────────────────────────────────────────────────
cluster:
  node_id: "node-1"
  node_role: "worker"
  pubsub_channel: "najim:events"

# ─────────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────────
auth:
  jwt_only: true
  api_keys: {}

# ─────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────
session:
  ttl_seconds: 86400
  max_history: 100
  heartbeat_interval: 30

# ─────────────────────────────────────────────────
# TOOL
# ─────────────────────────────────────────────────
tool:
  remote_timeout: 30.0
  internal_timeout: 10.0
  max_retries: 2

# ─────────────────────────────────────────────────
# STT
# ─────────────────────────────────────────────────
stt:
  model_name: "medium"
  model_dir: "./models"
  hf_repo: "Systran/faster-whisper-medium"
  device: "cpu"
  compute_type: "int8"
  beam_size: 5
  vad_filter: true
  language: null

# ─────────────────────────────────────────────────
# TTS
# ─────────────────────────────────────────────────
tts:
  model_dir: "./models"
  voices:
    en:
      local_path: "TTS-CORI-EN"
      hf_repo: "rhasspy/piper-voices"
      voice: "en.en_GB.cori.high"
      use_cuda: false
    ar:
      local_path: "TTS-KAREEM-ARABIC"
      hf_repo: "rhasspy/piper-voices"
      voice: "ar.ar_JO.kareem.medium"
      use_cuda: false
  default_voice: "en"
  max_length: 500
  synthesis:
    volume: 0.75
    length_scale: 1.0
    noise_scale: 0.75
    noise_w_scale: 0.5
    normalize_audio: true
    nchannels: 1
    sampwidth: 2
    framerate: 22050

# ─────────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────────
llm:
  api_base_url: "https://api.groq.com/openai/v1"
  api_key: "gsk_..."
  model: "llama-3.3-70b-versatile"
  timeout: 60.0
  max_retries: 2

# ─────────────────────────────────────────────────
# MCP (legacy external tools)
# ─────────────────────────────────────────────────
mcp:
  servers: []
  sse_read_timeout: 300.0
  tool_timeout: 30.0
  max_retries: 2
  max_tool_loops: 5
```
