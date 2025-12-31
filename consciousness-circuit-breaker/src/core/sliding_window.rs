//! Sliding window implementation for failure rate calculation
//!
//! Supports both count-based and time-based windows.
//! All data is REAL - no simulation.

use std::collections::VecDeque;
use std::sync::RwLock;
use std::time::{Duration, Instant};

use crate::core::config::{SlidingWindowConfig, SlidingWindowType};

/// Sliding window for tracking call outcomes
#[derive(Debug)]
pub struct SlidingWindow {
    /// Window type configuration
    window_type: SlidingWindowType,
    /// Window size (count or duration in seconds)
    size: u64,
    /// Duration threshold for slow calls
    slow_call_threshold: Duration,
    /// Internal storage
    inner: RwLock<WindowInner>,
}

#[derive(Debug)]
struct WindowInner {
    /// Call records
    records: VecDeque<CallRecord>,
    /// Cached statistics (invalidated on each record)
    cached_stats: Option<WindowStats>,
}

/// Record of a single call
#[derive(Debug, Clone, Copy)]
struct CallRecord {
    /// When the call occurred
    timestamp: Instant,
    /// Whether the call succeeded
    success: bool,
    /// Duration of the call
    duration: Duration,
}

/// Statistics from the sliding window
#[derive(Debug, Clone, Copy)]
pub struct WindowStats {
    /// Total calls in window
    pub total_calls: u64,
    /// Successful calls in window
    pub successful_calls: u64,
    /// Failed calls in window
    pub failed_calls: u64,
    /// Failure rate (0.0 to 1.0)
    pub failure_rate: f64,
    /// Slow calls in window
    pub slow_calls: u64,
    /// Slow call rate (0.0 to 1.0)
    pub slow_call_rate: f64,
    /// Average duration
    pub average_duration: Duration,
    /// P50 duration (median)
    pub p50_duration: Duration,
    /// P95 duration
    pub p95_duration: Duration,
    /// P99 duration
    pub p99_duration: Duration,
    /// Minimum duration
    pub min_duration: Duration,
    /// Maximum duration
    pub max_duration: Duration,
}

impl Default for WindowStats {
    fn default() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            failure_rate: 0.0,
            slow_calls: 0,
            slow_call_rate: 0.0,
            average_duration: Duration::ZERO,
            p50_duration: Duration::ZERO,
            p95_duration: Duration::ZERO,
            p99_duration: Duration::ZERO,
            min_duration: Duration::ZERO,
            max_duration: Duration::ZERO,
        }
    }
}

impl SlidingWindow {
    /// Create a new sliding window with the given configuration
    pub fn new(config: &SlidingWindowConfig) -> Self {
        Self {
            window_type: config.window_type,
            size: config.size,
            slow_call_threshold: config.slow_call_duration_threshold,
            inner: RwLock::new(WindowInner {
                records: VecDeque::with_capacity(config.size as usize),
                cached_stats: None,
            }),
        }
    }

    /// Record a successful call
    pub fn record_success(&self, duration: Duration) {
        self.record(true, duration);
    }

    /// Record a failed call
    pub fn record_failure(&self, duration: Duration) {
        self.record(false, duration);
    }

    /// Record a call outcome
    fn record(&self, success: bool, duration: Duration) {
        let mut inner = self.inner.write().unwrap();

        // Add new record
        inner.records.push_back(CallRecord {
            timestamp: Instant::now(),
            success,
            duration,
        });

        // Evict old records based on window type
        self.evict_old_records(&mut inner);

        // Invalidate cache
        inner.cached_stats = None;
    }

    /// Evict records outside the window
    fn evict_old_records(&self, inner: &mut WindowInner) {
        match self.window_type {
            SlidingWindowType::CountBased => {
                // Keep only the last N records
                while inner.records.len() > self.size as usize {
                    inner.records.pop_front();
                }
            }
            SlidingWindowType::TimeBased => {
                // Keep only records within the time window
                let cutoff = Instant::now() - Duration::from_secs(self.size);
                while let Some(front) = inner.records.front() {
                    if front.timestamp < cutoff {
                        inner.records.pop_front();
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> WindowStats {
        let mut inner = self.inner.write().unwrap();

        // Evict old records first (for time-based windows)
        if self.window_type == SlidingWindowType::TimeBased {
            let cutoff = Instant::now() - Duration::from_secs(self.size);
            while let Some(front) = inner.records.front() {
                if front.timestamp < cutoff {
                    inner.records.pop_front();
                } else {
                    break;
                }
            }
            inner.cached_stats = None;
        }

        // Return cached stats if available
        if let Some(stats) = inner.cached_stats {
            return stats;
        }

        // Calculate fresh statistics
        let stats = self.calculate_stats(&inner.records);
        inner.cached_stats = Some(stats);
        stats
    }

    /// Calculate statistics from records
    fn calculate_stats(&self, records: &VecDeque<CallRecord>) -> WindowStats {
        if records.is_empty() {
            return WindowStats::default();
        }

        let total = records.len() as u64;
        let successful = records.iter().filter(|r| r.success).count() as u64;
        let failed = total - successful;
        let slow = records
            .iter()
            .filter(|r| r.duration >= self.slow_call_threshold)
            .count() as u64;

        // Calculate duration statistics
        let mut durations: Vec<Duration> = records.iter().map(|r| r.duration).collect();
        durations.sort();

        let total_duration: Duration = durations.iter().sum();
        let average = total_duration / total as u32;

        let p50 = percentile(&durations, 50);
        let p95 = percentile(&durations, 95);
        let p99 = percentile(&durations, 99);

        WindowStats {
            total_calls: total,
            successful_calls: successful,
            failed_calls: failed,
            failure_rate: if total > 0 {
                failed as f64 / total as f64
            } else {
                0.0
            },
            slow_calls: slow,
            slow_call_rate: if total > 0 {
                slow as f64 / total as f64
            } else {
                0.0
            },
            average_duration: average,
            p50_duration: p50,
            p95_duration: p95,
            p99_duration: p99,
            min_duration: *durations.first().unwrap_or(&Duration::ZERO),
            max_duration: *durations.last().unwrap_or(&Duration::ZERO),
        }
    }

    /// Reset the sliding window
    pub fn reset(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.records.clear();
        inner.cached_stats = None;
    }

    /// Get the number of records in the window
    pub fn len(&self) -> usize {
        self.inner.read().unwrap().records.len()
    }

    /// Check if the window is empty
    pub fn is_empty(&self) -> bool {
        self.inner.read().unwrap().records.is_empty()
    }
}

/// Calculate percentile from sorted durations
fn percentile(sorted: &[Duration], p: u32) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() - 1) as f64 * (p as f64 / 100.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> SlidingWindowConfig {
        SlidingWindowConfig {
            window_type: SlidingWindowType::CountBased,
            size: 10,
            minimum_calls: 5,
            failure_rate_threshold: 0.5,
            slow_call_rate_threshold: 0.8,
            slow_call_duration_threshold: Duration::from_millis(100),
        }
    }

    #[test]
    fn test_count_based_window() {
        let window = SlidingWindow::new(&test_config());

        // Add 15 records, should keep only last 10
        for i in 0..15 {
            window.record_success(Duration::from_millis(i * 10));
        }

        assert_eq!(window.len(), 10);

        let stats = window.stats();
        assert_eq!(stats.total_calls, 10);
        assert_eq!(stats.failure_rate, 0.0);
    }

    #[test]
    fn test_failure_rate_calculation() {
        let window = SlidingWindow::new(&test_config());

        // 3 successes, 7 failures = 70% failure rate
        for _ in 0..3 {
            window.record_success(Duration::from_millis(10));
        }
        for _ in 0..7 {
            window.record_failure(Duration::from_millis(10));
        }

        let stats = window.stats();
        assert_eq!(stats.total_calls, 10);
        assert_eq!(stats.successful_calls, 3);
        assert_eq!(stats.failed_calls, 7);
        assert!((stats.failure_rate - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_slow_call_detection() {
        let mut config = test_config();
        config.slow_call_duration_threshold = Duration::from_millis(50);
        let window = SlidingWindow::new(&config);

        // 5 fast calls, 5 slow calls
        for _ in 0..5 {
            window.record_success(Duration::from_millis(10));
        }
        for _ in 0..5 {
            window.record_success(Duration::from_millis(100));
        }

        let stats = window.stats();
        assert_eq!(stats.slow_calls, 5);
        assert!((stats.slow_call_rate - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_percentile_calculation() {
        let window = SlidingWindow::new(&test_config());

        // Add records with known durations
        for i in 1..=10 {
            window.record_success(Duration::from_millis(i * 10));
        }

        let stats = window.stats();
        assert_eq!(stats.min_duration, Duration::from_millis(10));
        assert_eq!(stats.max_duration, Duration::from_millis(100));
        assert_eq!(stats.p50_duration, Duration::from_millis(50));
    }

    #[test]
    fn test_reset() {
        let window = SlidingWindow::new(&test_config());

        for _ in 0..5 {
            window.record_success(Duration::from_millis(10));
        }

        assert_eq!(window.len(), 5);

        window.reset();

        assert!(window.is_empty());
        assert_eq!(window.stats().total_calls, 0);
    }

    #[test]
    fn test_time_based_window() {
        let mut config = test_config();
        config.window_type = SlidingWindowType::TimeBased;
        config.size = 1; // 1 second window

        let window = SlidingWindow::new(&config);

        // Add some records
        for _ in 0..5 {
            window.record_success(Duration::from_millis(10));
        }

        assert_eq!(window.len(), 5);

        // Records should still be there immediately
        let stats = window.stats();
        assert_eq!(stats.total_calls, 5);
    }

    #[test]
    fn test_empty_window_stats() {
        let window = SlidingWindow::new(&test_config());
        let stats = window.stats();

        assert_eq!(stats.total_calls, 0);
        assert_eq!(stats.failure_rate, 0.0);
        assert_eq!(stats.average_duration, Duration::ZERO);
    }
}
