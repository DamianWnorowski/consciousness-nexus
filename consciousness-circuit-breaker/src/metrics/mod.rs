//! Metrics module for real system metrics collection
//!
//! **ZERO SIMULATION** - All metrics are REAL system data.

pub mod collector;

pub use collector::{MetricsCollector, RealMetrics, SystemMetrics};
