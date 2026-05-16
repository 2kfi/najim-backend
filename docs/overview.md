# Najim Backend — Overview

> A voice assistant backend designed for a cluster of Intel Atom computers.  
> Android App → STT → LLM → Tool Calls → TTS. All state in Redis.

## One Sentence

Najim is a voice assistant backend where an Android app connects via WebSocket, sends audio, the backend runs Speech-to-Text → LLM → Tool Calls → Text-to-Speech, and streams audio back — using **Redis** as the shared brain so any cluster node can handle any request.

## System Diagram

```
                    ┌─────────────────┐
                    │  Load Balancer   │  ← phones connect here
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
    │  Node 1   │      │  Node 2   │      │  Node 3   │
    │  (FastAPI)│      │  (FastAPI)│      │  (FastAPI)│
    └─────┬─────┘      └─────┬─────┘      └─────┬─────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                    ┌────────▼────────┐
                    │     Redis       │
                    │  Sessions       │
                    │  Conversations  │
                    │  Tool Bridge    │
                    │  Pub/Sub        │
                    │  Checkpoints    │
                    └─────────────────┘
```

## Pipeline

```
WS Handler → [stt_jobs] → STT → [llm_jobs] → LLM → [tts_jobs] → TTS → [responses] → WS Sender
                                   ↗              ↖
                            Internal Tools    Phone Tools
```

Every stage writes to a Redis stream checkpoint before the next stage starts. If a node crashes mid-stage, another node picks up the pending job.

## Key Concepts

| Concept | Summary |
|---------|---------|
| **Shared-nothing** | No session data in local memory. All state in Redis. |
| **Stateless nodes** | Any node can handle any request. |
| **Checkpoint pipeline** | Redis streams between stages for crash recovery. |
| **Correlation IDs** | Async tool call routing via Redis BLPOP. |
| **JWT auth** | Per-device tokens with embedded permissions. |
| **Phone as tool server** | Android app runs tools on demand (GPS, file index, etc.). |

## Quick Start

```bash
docker compose up -d
```

See [deployment.md](deployment.md) for full setup.
