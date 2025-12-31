# 🔀 CONSCIOUSNESS SUITE - UNIQUE PORT MAPPING

## 🎲 Randomly Generated Ports (No Conflicts!)

All services use unique random port numbers to avoid conflicts with other applications.

### 🌐 External Ports (Access from Host)
| Service | Port | URL | Purpose |
|---------|------|-----|---------|
| **Consciousness API** | `18473` | http://localhost:18473 | Main REST API & Docs |
| **Prometheus** | `24789` | http://localhost:24789 | Metrics monitoring |
| **Grafana** | `31572` | http://localhost:31572 | Monitoring dashboards |
| **Web Dashboard** | `31573` | http://localhost:31573 | Interactive web UI |
| **Loki** | `42851` | http://localhost:42851 | Log aggregation |
| **Nginx** | `25746` | http://localhost:25746 | Reverse proxy (HTTP) |
| **Nginx SSL** | `36827` | https://localhost:36827 | Reverse proxy (HTTPS) |
| **PostgreSQL** | `17392` | localhost:17392 | Database (external access) |
| **Redis** | `29481` | localhost:29481 | Cache (external access) |

### 🐳 Internal Container Ports (Standard)
| Service | Internal Port | External Port | Notes |
|---------|---------------|---------------|-------|
| Consciousness API | `8000` | `18473` | FastAPI server |
| Prometheus | `9090` | `24789` | Standard Prometheus port |
| Grafana | `3000` | `31572` | Standard Grafana port |
| Web Dashboard | `3000` | `31573` | React application |
| Loki | `3100` | `42851` | Standard Loki port |
| Promtail | `9080` | N/A | Internal only |
| Nginx | `80/443` | `25746/36827` | Standard web ports |
| PostgreSQL | `5432` | `17392` | Standard PostgreSQL port |
| Redis | `6379` | `29481` | Standard Redis port |

## 🚀 Quick Access Commands

```bash
# API & Documentation
open http://localhost:18473        # Main API
open http://localhost:18473/docs   # Interactive API docs

# Monitoring & Dashboards
open http://localhost:31573        # Web Dashboard (PRIMARY UI)
open http://localhost:31572        # Grafana dashboards
open http://localhost:24789        # Prometheus metrics
open http://localhost:42851        # Loki logs

# CLI Usage
./consciousness-cli --url http://localhost:18473 health

# SDK Examples
# JavaScript
const client = new ConsciousnessClient({
  baseURL: 'http://localhost:18473'
});

// Go
client := consciousness.NewClient("http://localhost:18473", "api-key")

// Rust
let client = create_local_client(18473)?;
```

## 🔧 Environment Variables

```bash
# Override default ports
export CONSCIOUSNESS_API_PORT=18473
export CONSCIOUSNESS_PROMETHEUS_PORT=24789
export CONSCIOUSNESS_GRAFANA_PORT=31572

# Or use CLI flags
./consciousness-cli --url http://localhost:18473 health
```

## 🛡️ Security Benefits

- **No Port Conflicts**: Unique random ports prevent conflicts with other apps
- **Hardened Security**: Non-standard ports reduce automated attack surface
- **Multi-Tenant Safe**: Can run multiple instances without conflicts
- **Development Friendly**: Easy to remember and type unique numbers

## 📊 Service Health Checks

```bash
# Test all services
curl http://localhost:18473/health     # API
curl http://localhost:24789/-/healthy  # Prometheus
curl http://localhost:31572/api/health # Grafana
curl http://localhost:42851/ready      # Loki
```

## 🔄 Port Forwarding (Optional)

If you need standard ports, use port forwarding:

```bash
# Forward to standard ports
ssh -L 8000:localhost:18473 user@server  # API
ssh -L 3000:localhost:31572 user@server  # Grafana
ssh -L 9090:localhost:24789 user@server  # Prometheus
```

## 🎯 Why Random Unique Ports?

1. **Zero Conflicts**: No more "port already in use" errors
2. **Security**: Non-standard ports avoid automated scans
3. **Multi-Instance**: Run multiple deployments simultaneously
4. **Development**: Clear separation between services
5. **Production Ready**: Unique per deployment

**Your Consciousness Suite now runs on truly unique, conflict-free ports!** 🎲✨
