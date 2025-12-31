//! Configuration for the circuit breaker
//!
//! All values are carefully tuned for production use.
//! No magic numbers - everything is configurable and validated.

use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::core::error::{Error, Result};

/// Main configuration for the circuit breaker
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Name identifier for this circuit breaker instance
    pub name: String,

    /// Failure threshold before opening the circuit
    pub failure_threshold: u32,

    /// Success threshold to close the circuit from half-open
    pub success_threshold: u32,

    /// Duration to wait before transitioning from open to half-open
    pub reset_timeout: Duration,

    /// Maximum time to wait for an operation
    pub call_timeout: Duration,

    /// Sliding window configuration
    pub sliding_window: SlidingWindowConfig,

    /// Bulkhead configuration for concurrency limiting
    pub bulkhead: BulkheadConfig,

    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,

    /// Metrics configuration
    pub metrics: MetricsConfig,

    /// Health check configuration
    pub health_check: HealthCheckConfig,

    /// Distributed coordination configuration
    pub distributed: Option<DistributedConfig>,
}

/// Sliding window configuration for failure rate calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SlidingWindowConfig {
    /// Type of sliding window
    pub window_type: SlidingWindowType,

    /// Size of the sliding window (count or duration in seconds)
    pub size: u64,

    /// Minimum number of calls before calculating failure rate
    pub minimum_calls: u32,

    /// Failure rate threshold (0.0 to 1.0)
    pub failure_rate_threshold: f64,

    /// Slow call rate threshold (0.0 to 1.0)
    pub slow_call_rate_threshold: f64,

    /// Duration threshold for slow calls
    pub slow_call_duration_threshold: Duration,
}

/// Type of sliding window
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlidingWindowType {
    /// Count-based window (last N calls)
    CountBased,
    /// Time-based window (calls in last N seconds)
    TimeBased,
}

/// Bulkhead configuration for limiting concurrent executions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BulkheadConfig {
    /// Whether bulkhead is enabled
    pub enabled: bool,

    /// Maximum concurrent calls
    pub max_concurrent_calls: usize,

    /// Maximum wait queue size
    pub max_wait_queue: usize,

    /// Maximum time to wait in queue
    pub max_wait_duration: Duration,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct RateLimitConfig {
    /// Whether rate limiting is enabled
    pub enabled: bool,

    /// Maximum requests per second
    pub requests_per_second: f64,

    /// Burst capacity
    pub burst_capacity: u32,
}

/// Metrics configuration - REAL metrics only
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MetricsConfig {
    /// Whether metrics collection is enabled
    pub enabled: bool,

    /// Prometheus endpoint for scraping (if using Prometheus)
    pub prometheus_endpoint: Option<String>,

    /// OpenTelemetry endpoint for traces
    pub otlp_endpoint: Option<String>,

    /// Metrics collection interval
    pub collection_interval: Duration,

    /// Whether to collect system metrics
    pub collect_system_metrics: bool,

    /// Labels to add to all metrics
    pub labels: Vec<(String, String)>,
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HealthCheckConfig {
    /// Whether health checks are enabled
    pub enabled: bool,

    /// Interval between health checks
    pub interval: Duration,

    /// Timeout for health check calls
    pub timeout: Duration,

    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
}

/// Distributed coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Coordination backend type
    pub backend: CoordinationBackend,

    /// Endpoints for the coordination service
    pub endpoints: Vec<String>,

    /// Cluster namespace/prefix
    pub namespace: String,

    /// Node ID (auto-generated if not provided)
    pub node_id: Option<String>,

    /// Lease/session TTL
    pub lease_ttl: Duration,
}

/// Coordination backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationBackend {
    /// etcd v3
    Etcd,
    /// HashiCorp Consul
    Consul,
    /// In-memory (single node only)
    InMemory,
}

// ─────────────────────────────────────────────────────────────────────────────
// Default implementations
// ─────────────────────────────────────────────────────────────────────────────

impl Default for Config {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            failure_threshold: 5,
            success_threshold: 3,
            reset_timeout: Duration::from_secs(30),
            call_timeout: Duration::from_secs(10),
            sliding_window: SlidingWindowConfig::default(),
            bulkhead: BulkheadConfig::default(),
            rate_limit: RateLimitConfig::default(),
            metrics: MetricsConfig::default(),
            health_check: HealthCheckConfig::default(),
            distributed: None,
        }
    }
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_type: SlidingWindowType::CountBased,
            size: 100,
            minimum_calls: 10,
            failure_rate_threshold: 0.5,
            slow_call_rate_threshold: 0.8,
            slow_call_duration_threshold: Duration::from_secs(2),
        }
    }
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_calls: 100,
            max_wait_queue: 50,
            max_wait_duration: Duration::from_secs(5),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            requests_per_second: 1000.0,
            burst_capacity: 100,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            prometheus_endpoint: Some("0.0.0.0:9090".to_string()),
            otlp_endpoint: None,
            collection_interval: Duration::from_secs(10),
            collect_system_metrics: true,
            labels: vec![],
        }
    }
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

impl Config {
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        if self.failure_threshold == 0 {
            return Err(Error::Configuration {
                field: "failure_threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if self.success_threshold == 0 {
            return Err(Error::Configuration {
                field: "success_threshold".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if self.reset_timeout.is_zero() {
            return Err(Error::Configuration {
                field: "reset_timeout".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        self.sliding_window.validate()?;
        self.bulkhead.validate()?;
        self.rate_limit.validate()?;

        Ok(())
    }

    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| Error::Configuration {
            field: "file".to_string(),
            reason: e.to_string(),
        })?;

        let config: Config = toml::from_str(&content).map_err(|e| Error::Configuration {
            field: "toml".to_string(),
            reason: e.to_string(),
        })?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        let mut config = Config::default();

        if let Ok(name) = std::env::var("CB_NAME") {
            config.name = name;
        }

        if let Ok(val) = std::env::var("CB_FAILURE_THRESHOLD") {
            config.failure_threshold = val.parse().map_err(|_| Error::Configuration {
                field: "CB_FAILURE_THRESHOLD".to_string(),
                reason: "must be a valid u32".to_string(),
            })?;
        }

        if let Ok(val) = std::env::var("CB_SUCCESS_THRESHOLD") {
            config.success_threshold = val.parse().map_err(|_| Error::Configuration {
                field: "CB_SUCCESS_THRESHOLD".to_string(),
                reason: "must be a valid u32".to_string(),
            })?;
        }

        if let Ok(val) = std::env::var("CB_RESET_TIMEOUT_SECS") {
            let secs: u64 = val.parse().map_err(|_| Error::Configuration {
                field: "CB_RESET_TIMEOUT_SECS".to_string(),
                reason: "must be a valid u64".to_string(),
            })?;
            config.reset_timeout = Duration::from_secs(secs);
        }

        config.validate()?;
        Ok(config)
    }
}

impl SlidingWindowConfig {
    /// Validate sliding window configuration
    pub fn validate(&self) -> Result<()> {
        if self.size == 0 {
            return Err(Error::Configuration {
                field: "sliding_window.size".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.failure_rate_threshold) {
            return Err(Error::Configuration {
                field: "sliding_window.failure_rate_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.slow_call_rate_threshold) {
            return Err(Error::Configuration {
                field: "sliding_window.slow_call_rate_threshold".to_string(),
                reason: "must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }
}

impl BulkheadConfig {
    /// Validate bulkhead configuration
    pub fn validate(&self) -> Result<()> {
        if self.enabled && self.max_concurrent_calls == 0 {
            return Err(Error::Configuration {
                field: "bulkhead.max_concurrent_calls".to_string(),
                reason: "must be greater than 0 when enabled".to_string(),
            });
        }

        Ok(())
    }
}

impl RateLimitConfig {
    /// Validate rate limit configuration
    pub fn validate(&self) -> Result<()> {
        if self.enabled && self.requests_per_second <= 0.0 {
            return Err(Error::Configuration {
                field: "rate_limit.requests_per_second".to_string(),
                reason: "must be greater than 0 when enabled".to_string(),
            });
        }

        Ok(())
    }
}

/// Builder for creating configurations fluently
#[derive(Debug, Clone, Default)]
pub struct ConfigBuilder {
    config: Config,
}

impl ConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the circuit breaker name
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.config.name = name.into();
        self
    }

    /// Set the failure threshold
    pub fn failure_threshold(mut self, threshold: u32) -> Self {
        self.config.failure_threshold = threshold;
        self
    }

    /// Set the success threshold
    pub fn success_threshold(mut self, threshold: u32) -> Self {
        self.config.success_threshold = threshold;
        self
    }

    /// Set the reset timeout
    pub fn reset_timeout(mut self, timeout: Duration) -> Self {
        self.config.reset_timeout = timeout;
        self
    }

    /// Set the call timeout
    pub fn call_timeout(mut self, timeout: Duration) -> Self {
        self.config.call_timeout = timeout;
        self
    }

    /// Configure bulkhead
    pub fn bulkhead(mut self, max_concurrent: usize, max_queue: usize) -> Self {
        self.config.bulkhead.enabled = true;
        self.config.bulkhead.max_concurrent_calls = max_concurrent;
        self.config.bulkhead.max_wait_queue = max_queue;
        self
    }

    /// Disable bulkhead
    pub fn no_bulkhead(mut self) -> Self {
        self.config.bulkhead.enabled = false;
        self
    }

    /// Configure rate limiting
    pub fn rate_limit(mut self, rps: f64, burst: u32) -> Self {
        self.config.rate_limit.enabled = true;
        self.config.rate_limit.requests_per_second = rps;
        self.config.rate_limit.burst_capacity = burst;
        self
    }

    /// Build and validate the configuration
    pub fn build(self) -> Result<Config> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder() {
        let config = ConfigBuilder::new()
            .name("test-breaker")
            .failure_threshold(10)
            .success_threshold(5)
            .reset_timeout(Duration::from_secs(60))
            .bulkhead(50, 25)
            .build()
            .unwrap();

        assert_eq!(config.name, "test-breaker");
        assert_eq!(config.failure_threshold, 10);
        assert_eq!(config.success_threshold, 5);
        assert!(config.bulkhead.enabled);
        assert_eq!(config.bulkhead.max_concurrent_calls, 50);
    }

    #[test]
    fn test_invalid_failure_threshold() {
        let mut config = Config::default();
        config.failure_threshold = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invalid_failure_rate() {
        let mut config = Config::default();
        config.sliding_window.failure_rate_threshold = 1.5;
        assert!(config.validate().is_err());
    }
}
