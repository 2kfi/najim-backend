# Clustering

## Architecture

3 Intel Atom nodes behind a load balancer. All nodes share one Redis instance.

```
                ┌─────────────────┐
                │  Load Balancer   │
                │  (nginx/haproxy) │
                └────────┬────────┘
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
┌─────────┐        ┌─────────┐        ┌─────────┐
│  node-1 │        │  node-2 │        │  node-3 │
│ port 8081│        │ port 8082│        │ port 8083│
└────┬────┘        └────┬────┘        └────┬────┘
     │                  │                  │
     └──────────────────┼──────────────────┘
                        │
                  ┌─────▼─────┐
                  │   Redis   │
                  │  :6379    │
                  └───────────┘
```

## Load Balancer

The load balancer distributes WebSocket connections across the 3 nodes.

| Strategy | Used? | Why |
|----------|-------|-----|
| Round-robin | Yes | Simple, even distribution for initial WS connect |
| Least connections | Better | If some users are more active |
| Source IP hash | No | Can cause uneven load |
| Sticky sessions | No | Not needed — all session state is in Redis |

**Important**: Once a phone connects to node-1, if node-1 crashes, the phone reconnects and gets node-2 or node-3. Because all session data is in Redis, node-2 can resume the conversation seamlessly.

## Crash Recovery

```
1. Phone A is connected to node-1 (stored in Redis as device_ws:phoneA = "node-1")
2. Node-1 crashes
3. Phone A's WebSocket disconnects
4. Phone A reconnects → load balancer sends to node-2
5. Node-2 reads from Redis: phoneA's session, conversation history available
6. Pipeline jobs for phoneA are in Redis streams — node-2's workers pick them up
7. Node-2 continues. Phone A never knew a node died.
```

## Cross-Node Communication

Nodes talk to each other via Redis Pub/Sub:

| Channel | Who Publishes | Who Subscribes | Purpose |
|---------|--------------|----------------|---------|
| `najim:events` | Any node | All nodes | Cluster-wide announcements |
| `najim:ws_send:{node_id}` | Any node | Only that node | Send WS message to device on that node |

**Example**: Node-1 needs to send a tool request to a phone connected to Node-2:

```python
# Node-1
phone_node = await device_registry.get_node_for_device("phone-android-123")
channel = f"najim:ws_send:{phone_node}"
await redis.publish(channel, {"type": "tool_request", ...})

# Node-2 (listening on najim:ws_send:node-2)
# Receives message, finds WebSocket for device, sends to phone
```

## Why Not Direct HTTP Between Nodes?

Nodes might not reach each other directly (firewalls, different networks). Redis acts as the central hub. Every node talks to Redis, and Redis routes messages.

## Single-Node Mode

Works the same way. No load balancer needed. Devices connect directly.

```yaml
cluster:
  node_id: "node-1"  # or use hostname
```

Pub/sub still works — the same node publishes and subscribes to itself. Session TTL, conversation history, tool correlation — all work identically.

## Startup

Each node:

1. Loads config from `config.yaml`
2. Connects to Redis (verify with PING)
3. Loads ML models (Whisper, Piper)
4. Initializes LLM HTTP client
5. Subscribes to `najim:ws_send:{node_id}` (other nodes can now route messages through this node)
6. Starts 4 pipeline workers (STT, LLM, TTS, WS Sender) as asyncio tasks
7. Starts serving WebSocket + REST endpoints

## Shutdown

1. Close all active WebSocket connections (send disconnect, mark offline)
2. Stop pipeline workers (finish current job, then exit)
3. Unsubscribe from pub/sub channels
4. Close Redis connection pool
5. Free ML model memory

## Docker Compose (Cluster)

```bash
docker compose -f docker-compose.cluster.yml up -d
```

Starts:
- 1 Redis container
- 3 app containers (node-1, node-2, node-3) on different host ports (8081, 8082, 8083)
