# Consciousness Computing Suite - Rust SDK

[![Crates.io](https://img.shields.io/crates/v/consciousness-suite-sdk.svg)](https://crates.io/crates/consciousness-suite-sdk)
[![Documentation](https://docs.rs/consciousness-suite-sdk/badge.svg)](https://docs.rs/consciousness-suite-sdk)
[![License](https://img.shields.io/crates/l/consciousness-suite-sdk.svg)](https://github.com/DAMIANWNOROWSKI/consciousness-suite/blob/main/LICENSE)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)

Rust SDK for accessing Consciousness Computing Suite from Rust applications. Makes enterprise-grade AI safety and evolution tools available in Rust applications, WebAssembly modules, system-level services, and high-performance computing environments.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
consciousness-suite-sdk = "2.0"
tokio = { version = "1.0", features = ["full"] }
```

## Quick Start

```rust
use consciousness_suite_sdk::prelude::*;

#[tokio::main]
async fn main() -> Result<(), ConsciousnessError> {
    // Create a local development client
    let client = create_local_client(8000)?;

    // Check health
    let health = client.get_health().await?;
    println!("System status: {:?}", health);

    Ok(())
}
```

## Usage Examples

### Creating Clients

```rust
use consciousness_suite_sdk::prelude::*;
use consciousness_suite_sdk::client::ClientConfig;

// Local development client
let local_client = create_local_client(8000)?;

// Production client with API key
let prod_client = create_production_client(
    "your-api-key".to_string(),
    "https://api.consciousness-suite.com".to_string()
)?;

// Custom configuration
let custom_client = ConsciousnessClient::new(ClientConfig {
    base_url: "http://localhost:8000".to_string(),
    api_key: Some("your-api-key".to_string()),
    timeout: std::time::Duration::from_secs(60),
    enable_websocket: true,
})?;
```

### Running Evolution Operations

```rust
use consciousness_suite_sdk::prelude::*;
use std::collections::HashMap;

// Run verified evolution
let result = client.run_evolution(
    EvolutionOperation::Verified,
    "my_application",
    None,
    Some(SafetyLevel::Strict)
).await?;

println!("Evolution ID: {}", result.evolution_id);
println!("Fitness Score: {}", result.metrics.fitness_score);

// Run with parameters
let mut params = HashMap::new();
params.insert("max_iterations".to_string(), serde_json::json!(50));

let result = client.run_evolution(
    EvolutionOperation::Recursive,
    "my_application",
    Some(params),
    Some(SafetyLevel::Standard)
).await?;
```

### Streaming Evolution Progress

```rust
use futures::StreamExt;

let stream = client.run_evolution_stream(
    EvolutionOperation::Recursive,
    "my_application",
    None,
    None
).await;

tokio::pin!(stream);

while let Some(progress) = stream.next().await {
    match progress {
        Ok(p) => {
            println!("Stage: {}, Progress: {:.1}%", p.stage, p.progress * 100.0);
            if p.complete.unwrap_or(false) {
                println!("Evolution complete!");
                break;
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
            break;
        }
    }
}
```

### Running Validation

```rust
let validation = client.run_validation(
    vec!["src/main.rs".to_string(), "src/lib.rs".to_string()],
    Some(ValidationScope::Comprehensive)
).await?;

println!("Valid: {}", validation.is_valid);
println!("Passed: {}/{}", validation.passed_checks, validation.total_checks);

for issue in &validation.issues {
    println!("  [{:?}] {}: {}", issue.severity, issue.title, issue.description);
}
```

### Running Analysis

```rust
use std::collections::HashMap;

let mut data = HashMap::new();
data.insert("system_metrics".to_string(), serde_json::json!({
    "cpu_usage": 45.2,
    "memory_usage": 62.8
}));

let analysis = client.run_analysis(
    AnalysisType::Fitness,
    data
).await?;

println!("Analysis results: {:?}", analysis);
```

### Authentication

```rust
// Login
let login_response = client.login("username", "password").await?;
println!("Session ID: {}", login_response.session_id);

// Operations are now authenticated via session

// Logout
client.logout().await?;
```

## API Reference

### ConsciousnessClient

Main client struct for interacting with the Consciousness API.

#### Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `new(config)` | Create new client | `Result<Self, ConsciousnessError>` |
| `login(username, password)` | Authenticate user | `Result<LoginResponse, ConsciousnessError>` |
| `logout()` | End session | `Result<(), ConsciousnessError>` |
| `run_evolution(op, target, params, safety)` | Run evolution | `Result<EvolutionResult, ConsciousnessError>` |
| `run_evolution_stream(op, target, params, safety)` | Stream evolution | `impl Stream<Item = Result<EvolutionProgress, ConsciousnessError>>` |
| `run_validation(files, scope)` | Validate files | `Result<ValidationResult, ConsciousnessError>` |
| `run_analysis(type, data)` | Run analysis | `Result<HashMap<String, Value>, ConsciousnessError>` |
| `get_health()` | Get health status | `Result<HealthStatus, ConsciousnessError>` |
| `get_system_status()` | Get system status | `Result<SystemStatus, ConsciousnessError>` |

### Enums

```rust
// Safety levels
pub enum SafetyLevel {
    Minimal,
    Standard,
    Strict,
    Paranoid,
}

// Evolution operations
pub enum EvolutionOperation {
    Verified,
    Recursive,
}

// Validation scopes
pub enum ValidationScope {
    Basic,
    Full,
    Comprehensive,
}

// Analysis types
pub enum AnalysisType {
    Fitness,
    Performance,
    Security,
}

// Validation severity
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}
```

### Structs

```rust
// Evolution result
pub struct EvolutionResult {
    pub evolution_id: String,
    pub status: String,
    pub results: HashMap<String, serde_json::Value>,
    pub metrics: EvolutionMetrics,
}

pub struct EvolutionMetrics {
    pub fitness_score: f64,
    pub execution_time: f64,
    pub safety_checks: u32,
    pub warnings: Vec<String>,
}

// Validation result
pub struct ValidationResult {
    pub is_valid: bool,
    pub total_checks: u32,
    pub passed_checks: u32,
    pub issues: Vec<ValidationIssue>,
    pub fitness_score: f64,
    pub warnings: Vec<String>,
}

// Health status
pub struct HealthStatus {
    pub status: String,
    pub timestamp: f64,
    pub uptime: f64,
    pub active_sessions: u32,
}
```

## Error Handling

```rust
use consciousness_suite_sdk::error::ConsciousnessError;

match client.run_evolution(EvolutionOperation::Verified, "app", None, None).await {
    Ok(result) => {
        println!("Success: {:?}", result);
    }
    Err(ConsciousnessError::Api(msg)) => {
        eprintln!("API error: {}", msg);
    }
    Err(ConsciousnessError::Http(e)) => {
        eprintln!("HTTP error: {}", e);
    }
    Err(ConsciousnessError::Json(e)) => {
        eprintln!("JSON parsing error: {}", e);
    }
}
```

## Features

- **Async/Await**: Full async support with Tokio
- **Streaming**: Server-Sent Events for long-running operations
- **Type Safety**: Strong typing with serde serialization
- **Error Handling**: Comprehensive error types with thiserror
- **WebAssembly**: Compatible with wasm32 targets

## Testing

```bash
# Run tests
cargo test

# Run tests with all features
cargo test --all-features

# Run specific test
cargo test test_evolution
```

## Building

```bash
# Debug build
cargo build

# Release build
cargo build --release

# Generate documentation
cargo doc --open
```

## WebAssembly Support

```bash
# Install wasm-pack
cargo install wasm-pack

# Build for WebAssembly
wasm-pack build --target web
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

## Related

- [Main Consciousness Suite](https://github.com/DAMIANWNOROWSKI/consciousness-suite)
- [Go SDK](../consciousness-sdk-go/README.md)
- [JavaScript SDK](../consciousness-sdk-js/README.md)
- [API Documentation](../docs/api/README.md)
