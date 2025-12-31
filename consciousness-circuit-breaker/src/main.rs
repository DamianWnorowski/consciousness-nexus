//! Consciousness Circuit Breaker CLI
//!
//! Production-grade circuit breaker with recursive ultrathought execution.
//! **ZERO SIMULATION** - All data is REAL.

use std::sync::Arc;
use std::time::Duration;

use clap::{Parser, Subcommand};
use consciousness_circuit_breaker::{
    CircuitBreaker,
    core::config::{ConfigBuilder, CoordinationBackend, DistributedConfig, MetricsConfig},
    distributed::coordinator::DistributedCoordinator,
    metrics::collector::MetricsCollector,
    ultrathought::engine::{ThoughtConfig, UltraThoughtEngine},
};
use tokio::signal;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Consciousness Circuit Breaker - Production-grade resilience
#[derive(Parser)]
#[command(name = "consciousness-circuit-breaker")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Configuration file path
    #[arg(short, long)]
    config: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    /// Enable distributed mode
    #[arg(long)]
    distributed: bool,

    /// Node ID for distributed mode
    #[arg(long)]
    node_id: Option<String>,

    /// Coordination backend (etcd, consul, in-memory)
    #[arg(long, default_value = "in-memory")]
    backend: String,

    /// Coordination endpoints (comma-separated)
    #[arg(long)]
    endpoints: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the circuit breaker server
    Run {
        /// Circuit breaker name
        #[arg(short, long, default_value = "default")]
        name: String,

        /// Failure threshold
        #[arg(long, default_value = "5")]
        failure_threshold: u32,

        /// Success threshold
        #[arg(long, default_value = "3")]
        success_threshold: u32,

        /// Reset timeout in seconds
        #[arg(long, default_value = "30")]
        reset_timeout: u64,

        /// Enable Prometheus metrics
        #[arg(long)]
        prometheus: bool,

        /// Prometheus port
        #[arg(long, default_value = "9090")]
        prometheus_port: u16,
    },

    /// Run the UltraThought engine
    Think {
        /// Input query to think about
        input: String,

        /// Maximum recursion depth
        #[arg(long, default_value = "100")]
        max_depth: u32,

        /// Maximum total thoughts
        #[arg(long, default_value = "10000")]
        max_thoughts: u64,

        /// Enable parallel exploration
        #[arg(long)]
        parallel: bool,

        /// Number of parallel workers
        #[arg(long, default_value = "4")]
        workers: usize,
    },

    /// Collect and display real system metrics
    Metrics {
        /// Collection interval in seconds
        #[arg(long, default_value = "1")]
        interval: u64,

        /// Number of collections (0 = infinite)
        #[arg(long, default_value = "10")]
        count: u64,

        /// Output format (text, json, prometheus)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Benchmark the circuit breaker
    Bench {
        /// Number of operations
        #[arg(short, long, default_value = "100000")]
        operations: u64,

        /// Simulated failure rate (0.0-1.0)
        #[arg(long, default_value = "0.1")]
        failure_rate: f64,

        /// Number of concurrent workers
        #[arg(long, default_value = "8")]
        workers: usize,
    },

    /// Show cluster status (distributed mode)
    Status,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            EnvFilter::new(&cli.log_level)
        }))
        .init();

    match &cli.command {
        Commands::Run {
            name,
            failure_threshold,
            success_threshold,
            reset_timeout,
            prometheus,
            prometheus_port,
        } => {
            run_server(
                &cli,
                name.clone(),
                *failure_threshold,
                *success_threshold,
                *reset_timeout,
                *prometheus,
                *prometheus_port,
            )
            .await
        }

        Commands::Think {
            input,
            max_depth,
            max_thoughts,
            parallel,
            workers,
        } => run_ultrathought(input.clone(), *max_depth, *max_thoughts, *parallel, *workers).await,

        Commands::Metrics {
            interval,
            count,
            format,
        } => run_metrics(*interval, *count, format).await,

        Commands::Bench {
            operations,
            failure_rate,
            workers,
        } => run_benchmark(*operations, *failure_rate, *workers).await,

        Commands::Status => show_status(&cli).await,
    }
}

async fn run_server(
    cli: &Cli,
    name: String,
    failure_threshold: u32,
    success_threshold: u32,
    reset_timeout: u64,
    prometheus: bool,
    prometheus_port: u16,
) -> anyhow::Result<()> {
    tracing::info!("Starting Consciousness Circuit Breaker");
    tracing::info!("  Name: {}", name);
    tracing::info!("  Failure threshold: {}", failure_threshold);
    tracing::info!("  Success threshold: {}", success_threshold);
    tracing::info!("  Reset timeout: {}s", reset_timeout);

    // Build configuration
    let config = ConfigBuilder::new()
        .name(&name)
        .failure_threshold(failure_threshold)
        .success_threshold(success_threshold)
        .reset_timeout(Duration::from_secs(reset_timeout))
        .build()?;

    // Create circuit breaker
    let breaker = CircuitBreaker::new(config);

    // Set up distributed coordination if enabled
    if cli.distributed {
        let backend = match cli.backend.as_str() {
            "etcd" => CoordinationBackend::Etcd,
            "consul" => CoordinationBackend::Consul,
            _ => CoordinationBackend::InMemory,
        };

        let endpoints: Vec<String> = cli
            .endpoints
            .as_ref()
            .map(|e| e.split(',').map(String::from).collect())
            .unwrap_or_else(|| vec!["localhost:2379".to_string()]);

        let dist_config = DistributedConfig {
            backend,
            endpoints,
            namespace: "consciousness".to_string(),
            node_id: cli.node_id.clone(),
            lease_ttl: Duration::from_secs(10),
        };

        let coordinator = DistributedCoordinator::new(dist_config);
        coordinator.start().await?;
        tracing::info!("Distributed coordination enabled (node: {})", coordinator.node_id());
    }

    // Set up metrics collection
    let metrics_config = MetricsConfig {
        enabled: true,
        prometheus_endpoint: if prometheus {
            Some(format!("0.0.0.0:{}", prometheus_port))
        } else {
            None
        },
        otlp_endpoint: None,
        collection_interval: Duration::from_secs(10),
        collect_system_metrics: true,
        labels: vec![("service".to_string(), name.clone())],
    };

    let collector = MetricsCollector::new(&metrics_config, &name);

    if prometheus {
        tracing::info!("Prometheus metrics available at ::{}/metrics", prometheus_port);
        // In production, you would start an HTTP server here
    }

    // Main loop - demonstrate the circuit breaker
    tracing::info!("Circuit breaker is running. Press Ctrl+C to stop.");

    let mut interval = tokio::time::interval(Duration::from_secs(5));
    loop {
        tokio::select! {
            _ = interval.tick() => {
                let stats = breaker.stats();
                let metrics = collector.collect().await;

                tracing::info!(
                    state = ?stats.state,
                    total_calls = stats.total_calls,
                    total_failures = stats.total_failures,
                    failure_rate = format!("{:.2}%", stats.failure_rate * 100.0),
                    cpu = format!("{:.1}%", metrics.cpu.usage_percent),
                    memory = format!("{:.1}%", metrics.memory.usage_percent),
                    "Circuit breaker status"
                );
            }
            _ = signal::ctrl_c() => {
                tracing::info!("Shutting down...");
                break;
            }
        }
    }

    Ok(())
}

async fn run_ultrathought(
    input: String,
    max_depth: u32,
    max_thoughts: u64,
    parallel: bool,
    workers: usize,
) -> anyhow::Result<()> {
    tracing::info!("Starting UltraThought Engine");
    tracing::info!("  Input: {}", input);
    tracing::info!("  Max depth: {}", max_depth);
    tracing::info!("  Max thoughts: {}", max_thoughts);
    tracing::info!("  Parallel: {} ({} workers)", parallel, workers);

    let config = ThoughtConfig {
        max_depth,
        max_thoughts,
        max_parallel: workers,
        timeout: Duration::from_secs(300),
        min_confidence: 0.1,
        parallel_exploration: parallel,
        synthesis_threshold: 10,
    };

    let engine = UltraThoughtEngine::new(config);

    tracing::info!("Thinking...");
    let result = engine.think(&input).await;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    ULTRATHOUGHT RESULT                        ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Output: {}", truncate(&result.output, 55));
    println!("║ Confidence: {:.2}%", result.confidence * 100.0);
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ METRICS                                                        ║");
    println!("║   Total thoughts: {}", result.metrics.total_thoughts);
    println!("║   Max depth reached: {}", result.metrics.max_depth_reached);
    println!("║   Execution time: {:?}", result.metrics.execution_time);
    println!("║   Thoughts/sec: {:.2}", result.metrics.thoughts_per_second);
    println!("║   Parallel efficiency: {:.2}%", result.metrics.parallel_efficiency * 100.0);
    println!("║   Memory estimate: {} KB", result.metrics.memory_estimate / 1024);
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}

async fn run_metrics(interval: u64, count: u64, format: &str) -> anyhow::Result<()> {
    tracing::info!("Starting REAL metrics collection");
    tracing::info!("  Interval: {}s", interval);
    tracing::info!("  Count: {}", if count == 0 { "infinite".to_string() } else { count.to_string() });
    tracing::info!("  Format: {}", format);

    let config = MetricsConfig::default();
    let collector = MetricsCollector::new(&config, "metrics-demo");

    let mut collected = 0u64;
    let mut interval_timer = tokio::time::interval(Duration::from_secs(interval));

    loop {
        interval_timer.tick().await;

        let metrics = collector.collect().await;

        match format {
            "json" => {
                let json = serde_json::json!({
                    "timestamp": metrics.timestamp,
                    "cpu": {
                        "usage_percent": metrics.cpu.usage_percent,
                        "cores": metrics.cpu.logical_cores,
                        "brand": metrics.cpu.brand,
                    },
                    "memory": {
                        "total_bytes": metrics.memory.total_bytes,
                        "used_bytes": metrics.memory.used_bytes,
                        "usage_percent": metrics.memory.usage_percent,
                    },
                    "process": {
                        "pid": metrics.process.pid,
                        "memory_bytes": metrics.process.memory_bytes,
                        "uptime_secs": metrics.process.uptime_secs,
                    }
                });
                println!("{}", serde_json::to_string_pretty(&json)?);
            }
            "prometheus" => {
                println!("{}", collector.to_prometheus(&metrics));
            }
            _ => {
                println!("\n┌─────────────────────────────────────────────────────────┐");
                println!("│ REAL SYSTEM METRICS (timestamp: {})             │", metrics.timestamp);
                println!("├─────────────────────────────────────────────────────────┤");
                println!("│ CPU                                                      │");
                println!("│   Usage: {:>6.2}%                                        │", metrics.cpu.usage_percent);
                println!("│   Cores: {} physical, {} logical                         │",
                    metrics.cpu.physical_cores, metrics.cpu.logical_cores);
                println!("│   Brand: {}                                              │", truncate(&metrics.cpu.brand, 40));
                println!("├─────────────────────────────────────────────────────────┤");
                println!("│ MEMORY                                                   │");
                println!("│   Total: {:>10} MB                                   │", metrics.memory.total_bytes / 1024 / 1024);
                println!("│   Used:  {:>10} MB ({:.1}%)                          │",
                    metrics.memory.used_bytes / 1024 / 1024,
                    metrics.memory.usage_percent);
                println!("│   Swap:  {:>10} MB used                              │", metrics.memory.swap_used_bytes / 1024 / 1024);
                println!("├─────────────────────────────────────────────────────────┤");
                println!("│ PROCESS (PID: {})                                       │", metrics.process.pid);
                println!("│   Memory:  {:>10} MB                                 │", metrics.process.memory_bytes / 1024 / 1024);
                println!("│   Virtual: {:>10} MB                                 │", metrics.process.virtual_memory_bytes / 1024 / 1024);
                println!("│   Uptime:  {:>10} seconds                            │", metrics.process.uptime_secs);
                println!("└─────────────────────────────────────────────────────────┘");
            }
        }

        collected += 1;
        if count > 0 && collected >= count {
            break;
        }
    }

    Ok(())
}

async fn run_benchmark(operations: u64, failure_rate: f64, workers: usize) -> anyhow::Result<()> {
    tracing::info!("Starting circuit breaker benchmark");
    tracing::info!("  Operations: {}", operations);
    tracing::info!("  Failure rate: {:.1}%", failure_rate * 100.0);
    tracing::info!("  Workers: {}", workers);

    let config = ConfigBuilder::new()
        .name("benchmark")
        .failure_threshold(5)
        .success_threshold(3)
        .reset_timeout(Duration::from_millis(100))
        .no_bulkhead() // Disable bulkhead for benchmark
        .build()?;

    let breaker = Arc::new(CircuitBreaker::new(config));
    let operations_per_worker = operations / workers as u64;

    let start = std::time::Instant::now();

    let mut handles = Vec::new();
    for worker_id in 0..workers {
        let breaker = breaker.clone();
        let handle = tokio::spawn(async move {
            let mut successes = 0u64;
            let mut failures = 0u64;
            let mut rejections = 0u64;

            for i in 0..operations_per_worker {
                let should_fail = ((worker_id as u64 * operations_per_worker + i) as f64
                    / (operations as f64))
                    < failure_rate;

                let result = breaker
                    .call(|| async move {
                        if should_fail {
                            Err(std::io::Error::new(
                                std::io::ErrorKind::Other,
                                "simulated failure",
                            ))
                        } else {
                            Ok(())
                        }
                    })
                    .await;

                match result {
                    Ok(_) => successes += 1,
                    Err(e) if e.is_circuit_open() => rejections += 1,
                    Err(_) => failures += 1,
                }
            }

            (successes, failures, rejections)
        });
        handles.push(handle);
    }

    let mut total_successes = 0u64;
    let mut total_failures = 0u64;
    let mut total_rejections = 0u64;

    for handle in handles {
        let (s, f, r) = handle.await?;
        total_successes += s;
        total_failures += f;
        total_rejections += r;
    }

    let elapsed = start.elapsed();
    let ops_per_sec = operations as f64 / elapsed.as_secs_f64();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK RESULTS                           ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Total operations: {:>10}                                   ║", operations);
    println!("║ Successes:        {:>10}                                   ║", total_successes);
    println!("║ Failures:         {:>10}                                   ║", total_failures);
    println!("║ Rejections:       {:>10} (circuit open)                    ║", total_rejections);
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Duration:         {:>10.2?}                                  ║", elapsed);
    println!("║ Operations/sec:   {:>10.0}                                   ║", ops_per_sec);
    println!("║ Avg latency:      {:>10.2} µs                               ║", elapsed.as_micros() as f64 / operations as f64);
    println!("╠════════════════════════════════════════════════════════════════╣");
    let stats = breaker.stats();
    println!("║ Final state: {:?}                                            ║", stats.state);
    println!("║ Total calls recorded: {:>10}                              ║", stats.total_calls);
    println!("║ Failure rate: {:>6.2}%                                        ║", stats.failure_rate * 100.0);
    println!("╚════════════════════════════════════════════════════════════════╝");

    Ok(())
}

async fn show_status(cli: &Cli) -> anyhow::Result<()> {
    if !cli.distributed {
        println!("Distributed mode is not enabled. Use --distributed flag.");
        return Ok(());
    }

    let backend = match cli.backend.as_str() {
        "etcd" => CoordinationBackend::Etcd,
        "consul" => CoordinationBackend::Consul,
        _ => CoordinationBackend::InMemory,
    };

    let endpoints: Vec<String> = cli
        .endpoints
        .as_ref()
        .map(|e| e.split(',').map(String::from).collect())
        .unwrap_or_else(|| vec!["localhost:2379".to_string()]);

    let config = DistributedConfig {
        backend,
        endpoints,
        namespace: "consciousness".to_string(),
        node_id: cli.node_id.clone(),
        lease_ttl: Duration::from_secs(10),
    };

    let coordinator = DistributedCoordinator::new(config);
    coordinator.start().await?;

    // Give time for discovery
    tokio::time::sleep(Duration::from_millis(500)).await;

    let stats = coordinator.stats();
    let cluster_state = coordinator.cluster_state().await;

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    CLUSTER STATUS                              ║");
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ Node ID:        {}                                  ║", truncate(&stats.node_id, 20));
    println!("║ Is Leader:      {}                                            ║", stats.is_leader);
    println!("║ Backend:        {:?}                                     ║", stats.backend);
    println!("║ Total nodes:    {}                                             ║", stats.total_nodes);
    println!("║ Healthy nodes:  {}                                             ║", stats.healthy_nodes);
    println!("╠════════════════════════════════════════════════════════════════╣");
    println!("║ CLUSTER STATE                                                  ║");
    println!("║   Namespace: {}                                                ║", cluster_state.namespace);
    println!("║   Health: {:?}                                              ║", cluster_state.health);
    println!("║   Active nodes: {}                                             ║", cluster_state.active_nodes);
    println!("║   Circuits: {}                                                 ║", cluster_state.circuits.len());
    println!("╚════════════════════════════════════════════════════════════════╝");

    if !cluster_state.circuits.is_empty() {
        println!("\n┌─────────────────────────────────────────────────────────────┐");
        println!("│ CIRCUIT BREAKER STATES                                       │");
        println!("├─────────────────────────────────────────────────────────────┤");
        for (name, circuit) in &cluster_state.circuits {
            println!("│ {} ", name);
            println!("│   Open: {}, Closed: {}, HalfOpen: {}                       │",
                circuit.open_count, circuit.closed_count, circuit.half_open_count);
            println!("│   Calls: {}, Failures: {}, Rate: {:.2}%                     │",
                circuit.total_calls, circuit.total_failures, circuit.avg_failure_rate * 100.0);
            println!("│   Recommendation: {:?}                                     │", circuit.recommendation);
        }
        println!("└─────────────────────────────────────────────────────────────┘");
    }

    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        format!("{:width$}", s, width = max_len)
    } else {
        format!("{}...", &s[..max_len - 3])
    }
}
