# Tools

The tool system lets the LLM perform actions beyond text generation. Tools are either **internal** (run on the cluster node) or **remote** (run on the Android phone).

## Tool Registry

`ToolRegistry` at `tools.registry` holds all known tools:

```python
registry = ToolRegistry()

# Register internal tool:
registry.register_internal_tool("get_weather", ToolDefinition(
    name="get_weather",
    description="Get weather for a location",
    handler=get_weather_fn,  # async callable
    input_schema={...}
))

# Register remote tool:
registry.register_remote_tool("get_gps", ToolDefinition(
    name="get_gps",
    description="Get phone GPS coordinates",
    input_schema={"type": "object", ...}
))
```

## Internal Tools

Run directly on the cluster node. No network needed.

| Tool | Description | Speed |
|------|-------------|-------|
| `get_time` | Returns current UTC time | <1ms |
| `get_weather` | Returns mock weather data for a city | ~50ms |
| `calculator` | Safe math expression evaluation | <1ms |

Adding a new internal tool:

```python
# 1. Write the handler (async function)
async def my_tool(params: dict) -> dict:
    return {"result": do_something(params["input"])}

# 2. Register it
registry.register_internal_tool("my_tool", ToolDefinition(
    name="my_tool",
    description="What my_tool does",
    handler=my_tool,
    input_schema={
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"]
    }
))
```

## Remote Tools

Run on the Android phone via cross-node bridge.

| Tool | Description |
|------|-------------|
| `index_files` | Index files on phone storage |
| `get_gps` | Get phone GPS coordinates |
| `start_http_server` | Start HTTP server on phone |

### Dynamic Registration via WebSocket

During the WebSocket handshake, the phone announces capabilities:

```json
{"type": "connect", "capabilities": ["gps", "file_index"]}
```

The server registers tool definitions from a predefined remote tool map for each capability the phone declares. This enables phones with different capabilities to expose different tools.

### Remote Tool Bridge

When the LLM calls a remote tool, the Tool Router:

1. Checks permissions (`perms:{device_id}` hash)
2. Generates a unique correlation ID (UUID)
3. Stores the pending call in Redis (`tool_corr:{id}` with TTL 35s)
4. Publishes to the pub/sub channel for the phone's connected node (`najim:ws_send:{node_id}`)
5. Uses `BLPOP tool_resp:{id}` to wait for the phone's response (timeout: 30s)
6. Returns the result to the LLM

```
LLM → Router → [tool_corr:{id}] → PUBLISH najim:ws_send:node-2 → Phone (on node-2)
                                                                      ↓
LLM ← result ← BLPOP tool_resp:{id} ←────── PUBLISH tool_resp:{id} ← Phone
```

## Tool Router

`ToolRouter.route_tool_call(name, device_id)`:

```
1. Check if name is in internal tools → run locally, return result
2. Check if name is in remote tools:
   a. Check permission store (perms:{device_id})
   b. If denied → return error
   c. If allowed → send via bridge, wait for response
3. If not found in either → raise UnknownToolError
```

## Permission Store

Controls which devices can call which tools.

```yaml
perms:phone-android-123:
  get_gps: "allow"
  index_files: "deny"
```

Default: **deny** (explicit allow required).

```bash
# Allow tool for device
curl -X PUT "http://localhost:8000/api/v1/permissions/phone-123/get_gps?allowed=true" \
  -H "Authorization: Bearer <token>"
```
