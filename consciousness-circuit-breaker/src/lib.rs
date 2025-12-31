//! # Consciousness Circuit Breaker
//!
//! Production-grade circuit breaker with recursive ultrathought execution.
//! **ZERO SIMULATION** - All metrics and data are REAL.
//!
//! ## Features
//!
//! - Lock-free circuit breaker state machine
//! - Real system metrics via sysinfo/procfs
//! - Prometheus metrics export
//! - Distributed coordination via etcd/consul
//! - Recursive ultrathought execution engine
//! - Trampoline-based deep recursion (no stack overflow)
//!
//! ## Example
//!
//! ```rust,no_run
//! use consciousness_circuit_breaker::{CircuitBreaker, Config};
//!
//! #[tokio::main]
//! async fn main() {
//!     let breaker = CircuitBreaker::new(Config::default());
//!
//!     let result = breaker.call(|| async {
//!         // Your fallible operation
//!         Ok::<_, std::io::Error>(42)
//!     }).await;
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms, clippy::all)]
#![allow(clippy::type_complexity)]

pub mod core;
pub mod metrics;
pub mod ultrathought;
pub mod distributed;

// Re-exports
pub use crate::core::breaker::{CircuitBreaker, CircuitBreakerBuilder};
pub use crate::core::state::{CircuitState, StateTransition};
pub use crate::core::config::Config;
pub use crate::core::error::{Error, Result};
pub use crate::metrics::collector::{MetricsCollector, RealMetrics};
pub use crate::ultrathought::engine::{UltraThoughtEngine, ThoughtResult};
pub use crate::distributed::coordinator::{DistributedCoordinator, ClusterState};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::breaker::*;
    pub use crate::core::state::*;
    pub use crate::core::config::*;
    pub use crate::core::error::*;
    pub use crate::metrics::collector::*;
    pub use crate::ultrathought::engine::*;
}
