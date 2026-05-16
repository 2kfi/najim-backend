# Authentication

## JWT Tokens

Every WebSocket connection and REST API call requires a valid JWT token.

### Structure

```
Header:    { "alg": "HS256", "typ": "JWT" }
Payload:   {
             "device_id": "phone-android-123",
             "user_id": "user-abc",
             "permissions": ["index_files", "get_gps"],
             "iat": 1700000000,
             "exp": 1700086400
           }
Signature: HMAC-SHA256(base64url(header) + "." + base64url(payload), SECRET)
```

### Token Lifetime

Default: 24 hours (`jwt.expiry_minutes: 1440`). After expiry, the phone must get a new token (new login).

### Token Creation

```bash
# Option 1 — admin JWT from docker-entrypoint (DEBUG=true):
docker compose run -e DEBUG=true -e JWT_SECRET=$JWT_SECRET najim
# → [najim] Admin JWT: eyJhbGciOi...

# Option 2 — generate with Python:
python3 -c "
import os; os.environ['JWT_SECRET'] = 'your-secret-here'
from core.jwt_auth import get_jwt_manager
mgr = get_jwt_manager()
print(mgr.create_token('phone-android-123', 'user-abc', ['get_gps']))
"

Or create manually with any JWT library:

```python
from core.jwt_auth import create_token
token = create_token(
    device_id="phone-android-123",
    user_id="user-abc",
    permissions=["get_gps"]
)
# → "eyJhbGciOiJIUzI1NiIs..."
```

### Token Verification

```python
from core.jwt_auth import verify_token
payload = verify_token(token)  # raises JWTError if invalid/expired
# → {"device_id": "phone-android-123", "user_id": "user-abc", ...}
```

### WebSocket Auth

Token passed as query parameter:

```
ws://server/api/v1/connect?token=eyJhbGciOi...
```

The server validates on connection handshake. Invalid tokens get a 401 close frame.

### REST API Auth

Token passed as `Authorization` header:

```bash
curl -H "Authorization: Bearer eyJhbGciOi..." http://localhost:8000/api/v1/sessions
```

## API Key Fallback

For backward compatibility, the server supports legacy API keys:

```yaml
auth:
  jwt_only: false    # false = JWT first, then check API keys
  api_keys:
    "sk-dev-001": { name: "dev", rate_limit: 100 }
```

When `jwt_only: true` (default), API keys are ignored.

## Permission Checks

Even with a valid JWT, tool calls are subject to per-device permissions:

```
1. Phone connects → JWT says device_id = "phone-123"
2. Phone's tool permissions checked in Redis perms:phone-123
3. If get_gps is not explicitly allowed → denied
```

Set permissions:

```bash
curl -X PUT "http://localhost:8000/api/v1/permissions/phone-123/get_gps?allowed=true" \
  -H "Authorization: Bearer <admin-token>"
```

## Security Layers

```
Firewall (port 443, 8443 only)
  └── Load Balancer (TLS termination)
        └── JWT Authentication (every request)
              └── Permission Checks (per-device tool allow/deny)
                    └── Rate Limiting (60 req/min per device)
```
