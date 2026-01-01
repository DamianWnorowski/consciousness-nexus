# Consciousness Computing Suite - API Documentation

[![API Version](https://img.shields.io/badge/API-v2.0.0-blue.svg)]()
[![OpenAPI](https://img.shields.io/badge/OpenAPI-3.0-green.svg)]()
[![REST](https://img.shields.io/badge/Protocol-REST-orange.svg)]()

Complete API reference for the Consciousness Computing Suite REST API.

## Base URL

```
Production: https://api.consciousness-suite.com
Development: http://localhost:8000
```

## Authentication

All API requests (except `/health`, `/docs`, and `/`) require authentication via API key.

### API Key Header

```http
X-API-Key: your-api-key
```

### Query Parameter

```http
GET /endpoint?api_key=your-api-key
```

### Session-Based Authentication

After login, use the session ID for subsequent requests:

```http
POST /evolution/run
Content-Type: application/json
X-API-Key: your-api-key

{
  "session_id": "uuid-session-id",
  ...
}
```

## Response Format

All responses follow a standard format:

```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "session_id": "uuid-session-id",
  "execution_time": 0.123
}
```

### Error Response

```json
{
  "success": false,
  "data": null,
  "error": "Error message description",
  "execution_time": 0.001
}
```

## Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Server information |
| GET | `/health` | Health check |
| GET | `/status` | System status |
| GET | `/docs` | Interactive API documentation |
| GET | `/redoc` | ReDoc API documentation |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Authenticate user |
| GET | `/session/{id}` | Get session info |
| DELETE | `/session/{id}` | End session |

### Evolution Operations

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/evolution/run` | Run evolution operation |
| POST | `/evolution/run/stream` | Run evolution with streaming |

### Validation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/validation/run` | Run validation checks |

### Analysis

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/analysis/run` | Run analysis operations |

---

## Detailed Endpoint Reference

### GET /

Server information and documentation links.

**Response:**
```json
{
  "message": "Consciousness Computing Suite API",
  "version": "2.0.0",
  "status": "operational",
  "uptime": 3600.5,
  "documentation": "/docs"
}
```

---

### GET /health

Health check for monitoring and load balancers.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": 1703980800.0,
  "active_sessions": 42,
  "uptime": 86400.5
}
```

---

### GET /status

Comprehensive system status including all subsystems.

**Response:**
```json
{
  "api_server": {
    "status": "operational",
    "uptime": 86400.5,
    "active_sessions": 42
  },
  "consciousness_suite": {
    "initialized": true,
    "safety_level": "standard",
    "systems_status": {
      "auth": "operational",
      "validation": "operational",
      "evolution": "operational",
      "analysis": "operational"
    }
  },
  "timestamp": 1703980800.0
}
```

---

### POST /auth/login

Authenticate user and create session.

**Request:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": "username",
    "roles": ["admin", "developer"]
  },
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "execution_time": 0.05
}
```

**Errors:**
- `401`: Invalid credentials
- `500`: Authentication system unavailable

---

### GET /session/{session_id}

Get session information.

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "username",
  "created_at": 1703894400.0,
  "last_activity": 1703980800.0,
  "operations_count": 15
}
```

**Errors:**
- `404`: Session not found

---

### DELETE /session/{session_id}

End session (logout).

**Response:**
```json
{
  "message": "Session ended"
}
```

**Errors:**
- `404`: Session not found

---

### POST /evolution/run

Run an evolution operation.

**Request:**
```json
{
  "operation_type": "verified | recursive",
  "target_system": "string",
  "parameters": {
    "max_iterations": 50,
    "custom_param": "value"
  },
  "safety_level": "minimal | standard | strict | paranoid",
  "user_id": "string",
  "session_id": "string (optional)"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "evolution_id": "evo-550e8400-e29b-41d4",
    "status": "completed",
    "results": {
      "changes_applied": 15,
      "systems_evolved": ["system_a", "system_b"]
    },
    "metrics": {
      "fitness_score": 0.95,
      "execution_time": 45.2,
      "safety_checks": 23,
      "warnings": []
    }
  },
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "execution_time": 45.2
}
```

**Errors:**
- `400`: Invalid operation type
- `401`: Invalid session
- `500`: Evolution failed

---

### POST /evolution/run/stream

Run evolution with Server-Sent Events streaming.

**Request:** Same as `/evolution/run`

**Response:** `text/event-stream`

```
data: {"stage": "initialization", "progress": 0.1, "message": "Initializing evolution"}

data: {"stage": "analysis", "progress": 0.3, "message": "Analyzing target system"}

data: {"stage": "evolution", "progress": 0.7, "message": "Applying evolution cycles"}

data: {"complete": true, "result": {...}}
```

---

### POST /validation/run

Run validation checks on files.

**Request:**
```json
{
  "files": ["path/to/file1.py", "path/to/file2.py"],
  "validation_scope": "basic | full | comprehensive",
  "user_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "passed_checks": 45,
    "total_checks": 50,
    "issues": [
      {
        "severity": "low | medium | high | critical",
        "category": "security",
        "title": "Issue Title",
        "description": "Detailed description",
        "file": "path/to/file.py"
      }
    ],
    "warnings": 2,
    "fitness_score": 0.90
  },
  "execution_time": 5.3
}
```

---

### POST /analysis/run

Run analysis operations on data.

**Request:**
```json
{
  "data": {
    "metrics": {...},
    "custom_data": {...}
  },
  "analysis_type": "fitness | performance | security",
  "user_id": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis-123",
    "results": {
      "score": 0.85,
      "recommendations": [...],
      "predictions": {...}
    }
  },
  "execution_time": 2.1
}
```

**Errors:**
- `400`: Invalid analysis type
- `500`: Analysis failed

---

## Rate Limiting

| Tier | Requests/Minute | Requests/Day |
|------|----------------|--------------|
| Free | 60 | 1,000 |
| Standard | 300 | 10,000 |
| Enterprise | Unlimited | Unlimited |

Rate limit headers:
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1703980860
```

---

## Error Codes

| HTTP Code | Description |
|-----------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Invalid/missing API key or session |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## SDK Examples

### Python

```python
import requests

client = requests.Session()
client.headers['X-API-Key'] = 'your-api-key'

# Run evolution
response = client.post('http://localhost:8000/evolution/run', json={
    'operation_type': 'verified',
    'target_system': 'my_app',
    'safety_level': 'strict'
})
print(response.json())
```

### JavaScript

```javascript
const response = await fetch('http://localhost:8000/evolution/run', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'your-api-key'
  },
  body: JSON.stringify({
    operation_type: 'verified',
    target_system: 'my_app'
  })
});
const data = await response.json();
```

### cURL

```bash
curl -X POST http://localhost:8000/evolution/run \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{"operation_type": "verified", "target_system": "my_app"}'
```

---

## Webhooks (Coming Soon)

Configure webhooks to receive notifications about evolution events:

```json
{
  "url": "https://your-server.com/webhook",
  "events": ["evolution.completed", "evolution.failed", "validation.completed"],
  "secret": "webhook-secret"
}
```

---

## Related Documentation

- [Main README](../../README.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [JavaScript SDK](../../consciousness-sdk-js/README.md)
- [Go SDK](../../consciousness-sdk-go/README.md)
- [Rust SDK](../../consciousness-sdk-rust/README.md)
- [Vector Matrix Architecture](../vector_matrix_architecture.md)
