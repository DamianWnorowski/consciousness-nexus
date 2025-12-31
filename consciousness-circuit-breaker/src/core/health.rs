//! Health tracking for the circuit breaker
//!
//! Monitors the health of protected services with REAL metrics.
//! No simulation - all data reflects actual system state.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::core::config::HealthCheckConfig;

/// Health tracker for monitoring service health
#[derive(Debug)]
pub struct HealthTracker {
    /// Configuration
    enabled: bool,
    /// Health check interval
    interval: Duration,
    /// Health check timeout
    timeout: Duration,
    /// Failure threshold for unhealthy status
    failure_threshold: u32,

    /// Current consecutive failures
    consecutive_failures: AtomicU64,
    /// Current consecutive successes
    consecutive_successes: AtomicU64,
    /// Total health checks performed
    total_checks: AtomicU64,
    /// Total failures recorded
    total_failures: AtomicU64,
    /// Last health check timestamp (nanos since epoch)
    last_check_at: AtomicU64,
    /// Health status (0 = unknown, 1 = healthy, 2 = unhealthy, 3 = degraded)
    status: AtomicU64,
    /// Creation timestamp
    created_at: Instant,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Health status is unknown (no checks performed yet)
    Unknown,
    /// Service is healthy
    Healthy,
    /// Service is unhealthy (exceeded failure threshold)
    Unhealthy,
    /// Service is degraded (some failures but below threshold)
    Degraded,
}

impl From<u64> for HealthStatus {
    fn from(value: u64) -> Self {
        match value {
            1 => HealthStatus::Healthy,
            2 => HealthStatus::Unhealthy,
            3 => HealthStatus::Degraded,
            _ => HealthStatus::Unknown,
        }
    }
}

impl From<HealthStatus> for u64 {
    fn from(status: HealthStatus) -> Self {
        match status {
            HealthStatus::Unknown => 0,
            HealthStatus::Healthy => 1,
            HealthStatus::Unhealthy => 2,
            HealthStatus::Degraded => 3,
        }
    }
}

/// Detailed health report
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Current health status
    pub status: HealthStatus,
    /// Current consecutive failures
    pub consecutive_failures: u64,
    /// Current consecutive successes
    pub consecutive_successes: u64,
    /// Total health checks performed
    pub total_checks: u64,
    /// Total failures recorded
    pub total_failures: u64,
    /// Failure rate (0.0 to 1.0)
    pub failure_rate: f64,
    /// Time since last check
    pub time_since_last_check: Duration,
    /// Uptime of the tracker
    pub uptime: Duration,
    /// Is health checking enabled
    pub enabled: bool,
}

impl HealthTracker {
    /// Create a new health tracker with the given configuration
    pub fn new(config: &HealthCheckConfig) -> Self {
        Self {
            enabled: config.enabled,
            interval: config.interval,
            timeout: config.timeout,
            failure_threshold: config.failure_threshold,
            consecutive_failures: AtomicU64::new(0),
            consecutive_successes: AtomicU64::new(0),
            total_checks: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            last_check_at: AtomicU64::new(0),
            status: AtomicU64::new(0), // Unknown
            created_at: Instant::now(),
        }
    }

    /// Record a successful health check
    pub fn record_success(&self) {
        if !self.enabled {
            return;
        }

        self.total_checks.fetch_add(1, Ordering::Relaxed);
        self.consecutive_successes.fetch_add(1, Ordering::Relaxed);
        self.consecutive_failures.store(0, Ordering::Relaxed);

        // Update last check timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_check_at.store(now, Ordering::Relaxed);

        // Update status
        self.status.store(HealthStatus::Healthy.into(), Ordering::Relaxed);
    }

    /// Record a failed health check
    pub fn record_failure(&self) {
        if !self.enabled {
            return;
        }

        self.total_checks.fetch_add(1, Ordering::Relaxed);
        self.total_failures.fetch_add(1, Ordering::Relaxed);
        let failures = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        self.consecutive_successes.store(0, Ordering::Relaxed);

        // Update last check timestamp
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        self.last_check_at.store(now, Ordering::Relaxed);

        // Update status based on failure count
        let new_status = if failures >= self.failure_threshold as u64 {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };
        self.status.store(new_status.into(), Ordering::Relaxed);
    }

    /// Get current health status
    pub fn status(&self) -> HealthStatus {
        HealthStatus::from(self.status.load(Ordering::Relaxed))
    }

    /// Check if the service is healthy
    pub fn is_healthy(&self) -> bool {
        self.status() == HealthStatus::Healthy
    }

    /// Check if the service is unhealthy
    pub fn is_unhealthy(&self) -> bool {
        self.status() == HealthStatus::Unhealthy
    }

    /// Get detailed health report
    pub fn report(&self) -> HealthReport {
        let total_checks = self.total_checks.load(Ordering::Relaxed);
        let total_failures = self.total_failures.load(Ordering::Relaxed);

        let time_since_last = {
            let last = self.last_check_at.load(Ordering::Relaxed);
            if last == 0 {
                Duration::MAX
            } else {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
                Duration::from_nanos(now.saturating_sub(last))
            }
        };

        HealthReport {
            status: self.status(),
            consecutive_failures: self.consecutive_failures.load(Ordering::Relaxed),
            consecutive_successes: self.consecutive_successes.load(Ordering::Relaxed),
            total_checks,
            total_failures,
            failure_rate: if total_checks > 0 {
                total_failures as f64 / total_checks as f64
            } else {
                0.0
            },
            time_since_last_check: time_since_last,
            uptime: self.created_at.elapsed(),
            enabled: self.enabled,
        }
    }

    /// Reset the health tracker
    pub fn reset(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        self.consecutive_successes.store(0, Ordering::Relaxed);
        self.total_checks.store(0, Ordering::Relaxed);
        self.total_failures.store(0, Ordering::Relaxed);
        self.last_check_at.store(0, Ordering::Relaxed);
        self.status.store(HealthStatus::Unknown.into(), Ordering::Relaxed);
    }

    /// Check if a health check is due
    pub fn is_check_due(&self) -> bool {
        if !self.enabled {
            return false;
        }

        let last = self.last_check_at.load(Ordering::Relaxed);
        if last == 0 {
            return true;
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        let elapsed = Duration::from_nanos(now.saturating_sub(last));
        elapsed >= self.interval
    }

    /// Get the health check interval
    pub fn interval(&self) -> Duration {
        self.interval
    }

    /// Get the health check timeout
    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Get the failure threshold
    pub fn failure_threshold(&self) -> u32 {
        self.failure_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> HealthCheckConfig {
        HealthCheckConfig {
            enabled: true,
            interval: Duration::from_secs(10),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
        }
    }

    #[test]
    fn test_initial_status() {
        let tracker = HealthTracker::new(&test_config());
        assert_eq!(tracker.status(), HealthStatus::Unknown);
        assert!(!tracker.is_healthy());
        assert!(!tracker.is_unhealthy());
    }

    #[test]
    fn test_success_tracking() {
        let tracker = HealthTracker::new(&test_config());

        tracker.record_success();
        assert_eq!(tracker.status(), HealthStatus::Healthy);
        assert!(tracker.is_healthy());

        let report = tracker.report();
        assert_eq!(report.consecutive_successes, 1);
        assert_eq!(report.total_checks, 1);
    }

    #[test]
    fn test_failure_tracking() {
        let tracker = HealthTracker::new(&test_config());

        // First failure - degraded
        tracker.record_failure();
        assert_eq!(tracker.status(), HealthStatus::Degraded);

        // Second failure - still degraded
        tracker.record_failure();
        assert_eq!(tracker.status(), HealthStatus::Degraded);

        // Third failure - unhealthy
        tracker.record_failure();
        assert_eq!(tracker.status(), HealthStatus::Unhealthy);
        assert!(tracker.is_unhealthy());
    }

    #[test]
    fn test_recovery() {
        let tracker = HealthTracker::new(&test_config());

        // Become unhealthy
        for _ in 0..3 {
            tracker.record_failure();
        }
        assert!(tracker.is_unhealthy());

        // Single success should recover
        tracker.record_success();
        assert!(tracker.is_healthy());
        assert_eq!(tracker.report().consecutive_failures, 0);
    }

    #[test]
    fn test_disabled_tracker() {
        let mut config = test_config();
        config.enabled = false;
        let tracker = HealthTracker::new(&config);

        tracker.record_failure();
        tracker.record_failure();
        tracker.record_failure();

        // Should remain unknown when disabled
        assert_eq!(tracker.status(), HealthStatus::Unknown);
        assert!(!tracker.is_check_due());
    }

    #[test]
    fn test_failure_rate() {
        let tracker = HealthTracker::new(&test_config());

        // 3 successes, 1 failure = 25% failure rate
        tracker.record_success();
        tracker.record_success();
        tracker.record_success();
        tracker.record_failure();

        let report = tracker.report();
        assert_eq!(report.total_checks, 4);
        assert!((report.failure_rate - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_reset() {
        let tracker = HealthTracker::new(&test_config());

        tracker.record_success();
        tracker.record_failure();

        tracker.reset();

        let report = tracker.report();
        assert_eq!(report.status, HealthStatus::Unknown);
        assert_eq!(report.total_checks, 0);
        assert_eq!(report.consecutive_failures, 0);
        assert_eq!(report.consecutive_successes, 0);
    }

    #[test]
    fn test_check_due() {
        let mut config = test_config();
        config.interval = Duration::from_millis(10);
        let tracker = HealthTracker::new(&config);

        // First check should always be due
        assert!(tracker.is_check_due());

        tracker.record_success();

        // Immediately after, not due
        assert!(!tracker.is_check_due());
    }
}
