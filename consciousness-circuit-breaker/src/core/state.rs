//! Circuit breaker state machine
//!
//! Lock-free state transitions using atomic operations.
//! All state changes are recorded with real timestamps - NO SIMULATION.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CircuitState {
    /// Circuit is closed - requests flow through normally
    Closed = 0,

    /// Circuit is open - requests are rejected immediately
    Open = 1,

    /// Circuit is half-open - limited requests allowed to test recovery
    HalfOpen = 2,

    /// Circuit is disabled - all requests pass through without tracking
    Disabled = 3,

    /// Circuit is forced open - manually opened, stays open until manual close
    ForcedOpen = 4,
}

impl CircuitState {
    /// Convert from u8
    #[inline]
    pub fn from_u8(value: u8) -> Self {
        match value {
            0 => CircuitState::Closed,
            1 => CircuitState::Open,
            2 => CircuitState::HalfOpen,
            3 => CircuitState::Disabled,
            4 => CircuitState::ForcedOpen,
            _ => CircuitState::Closed, // Default to closed
        }
    }

    /// Check if requests are allowed
    #[inline]
    pub fn allows_requests(&self) -> bool {
        matches!(self, CircuitState::Closed | CircuitState::HalfOpen | CircuitState::Disabled)
    }

    /// Check if circuit is rejecting requests
    #[inline]
    pub fn is_rejecting(&self) -> bool {
        matches!(self, CircuitState::Open | CircuitState::ForcedOpen)
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            CircuitState::Closed => "CLOSED",
            CircuitState::Open => "OPEN",
            CircuitState::HalfOpen => "HALF_OPEN",
            CircuitState::Disabled => "DISABLED",
            CircuitState::ForcedOpen => "FORCED_OPEN",
        }
    }
}

impl std::fmt::Display for CircuitState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// State transition record with REAL timestamps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateTransition {
    /// Previous state
    pub from: CircuitState,

    /// New state
    pub to: CircuitState,

    /// Reason for transition
    pub reason: TransitionReason,

    /// REAL timestamp (not simulated)
    pub timestamp: u64,

    /// Monotonic timestamp for duration calculations
    #[serde(skip)]
    pub instant: Option<Instant>,

    /// Failure count at time of transition
    pub failure_count: u64,

    /// Success count at time of transition
    pub success_count: u64,

    /// Calculated failure rate
    pub failure_rate: f64,
}

/// Reasons for state transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransitionReason {
    /// Failure threshold exceeded
    FailureThresholdExceeded,

    /// Failure rate exceeded threshold
    FailureRateExceeded,

    /// Success threshold reached in half-open
    SuccessThresholdReached,

    /// Reset timeout expired (open -> half-open)
    ResetTimeoutExpired,

    /// Half-open trial failed
    HalfOpenTrialFailed,

    /// Manual state change
    Manual,

    /// Manual reset
    ManualReset,

    /// Forced open by operator
    ForcedOpen,

    /// Health check failed
    HealthCheckFailed,

    /// Health check passed
    HealthCheckPassed,

    /// Slow call rate exceeded
    SlowCallRateExceeded,

    /// Distributed state sync
    DistributedSync,
}

impl std::fmt::Display for TransitionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransitionReason::FailureThresholdExceeded => write!(f, "failure_threshold_exceeded"),
            TransitionReason::FailureRateExceeded => write!(f, "failure_rate_exceeded"),
            TransitionReason::SuccessThresholdReached => write!(f, "success_threshold_reached"),
            TransitionReason::ResetTimeoutExpired => write!(f, "reset_timeout_expired"),
            TransitionReason::HalfOpenTrialFailed => write!(f, "half_open_trial_failed"),
            TransitionReason::Manual => write!(f, "manual"),
            TransitionReason::ManualReset => write!(f, "manual_reset"),
            TransitionReason::ForcedOpen => write!(f, "forced_open"),
            TransitionReason::HealthCheckFailed => write!(f, "health_check_failed"),
            TransitionReason::HealthCheckPassed => write!(f, "health_check_passed"),
            TransitionReason::SlowCallRateExceeded => write!(f, "slow_call_rate_exceeded"),
            TransitionReason::DistributedSync => write!(f, "distributed_sync"),
        }
    }
}

/// Atomic state machine for lock-free state management
#[derive(Debug)]
pub struct AtomicStateMachine {
    /// Current state (atomic for lock-free access)
    state: AtomicU8,

    /// Timestamp when state was last changed (Unix timestamp in millis)
    state_changed_at: AtomicU64,

    /// Timestamp when circuit opened (for reset timeout calculation)
    opened_at: AtomicU64,

    /// Consecutive failure count
    failure_count: AtomicU64,

    /// Consecutive success count (in half-open state)
    success_count: AtomicU64,

    /// Total calls tracked
    total_calls: AtomicU64,

    /// Total failures tracked
    total_failures: AtomicU64,

    /// Monotonic instant for timing
    #[allow(dead_code)]
    created_at: Instant,
}

impl AtomicStateMachine {
    /// Create a new state machine in closed state
    pub fn new() -> Self {
        let now = Self::current_timestamp();
        Self {
            state: AtomicU8::new(CircuitState::Closed as u8),
            state_changed_at: AtomicU64::new(now),
            opened_at: AtomicU64::new(0),
            failure_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            total_calls: AtomicU64::new(0),
            total_failures: AtomicU64::new(0),
            created_at: Instant::now(),
        }
    }

    /// Get REAL current timestamp in milliseconds
    #[inline]
    fn current_timestamp() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Time went backwards")
            .as_millis() as u64
    }

    /// Get current state
    #[inline]
    pub fn state(&self) -> CircuitState {
        CircuitState::from_u8(self.state.load(Ordering::Acquire))
    }

    /// Get current state (alias for state())
    #[inline]
    pub fn current_state(&self) -> CircuitState {
        self.state()
    }

    /// Get uptime since creation
    #[inline]
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get failure count
    #[inline]
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(Ordering::Acquire)
    }

    /// Get success count
    #[inline]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(Ordering::Acquire)
    }

    /// Get total calls
    #[inline]
    pub fn total_calls(&self) -> u64 {
        self.total_calls.load(Ordering::Acquire)
    }

    /// Get total failures
    #[inline]
    pub fn total_failures(&self) -> u64 {
        self.total_failures.load(Ordering::Acquire)
    }

    /// Calculate current failure rate (REAL calculation, not simulated)
    #[inline]
    pub fn failure_rate(&self) -> f64 {
        let total = self.total_calls.load(Ordering::Acquire);
        if total == 0 {
            return 0.0;
        }
        let failures = self.total_failures.load(Ordering::Acquire);
        failures as f64 / total as f64
    }

    /// Get time since circuit opened
    pub fn time_since_opened(&self) -> Option<Duration> {
        let opened_at = self.opened_at.load(Ordering::Acquire);
        if opened_at == 0 {
            return None;
        }
        let now = Self::current_timestamp();
        let diff = now.saturating_sub(opened_at);
        Some(Duration::from_millis(diff))
    }

    /// Get time until reset (None if not open)
    pub fn time_until_reset(&self, reset_timeout: Duration) -> Option<Duration> {
        let time_since = self.time_since_opened()?;
        if time_since >= reset_timeout {
            Some(Duration::ZERO)
        } else {
            Some(reset_timeout - time_since)
        }
    }

    /// Record a successful call
    pub fn record_success(&self) -> Option<StateTransition> {
        self.total_calls.fetch_add(1, Ordering::AcqRel);

        let state = self.state();

        match state {
            CircuitState::Closed => {
                // Reset consecutive failure count on success
                self.failure_count.store(0, Ordering::Release);
                None
            }
            CircuitState::HalfOpen => {
                // Increment success count
                let successes = self.success_count.fetch_add(1, Ordering::AcqRel) + 1;
                Some(StateTransition {
                    from: state,
                    to: state, // Stay in half-open until threshold
                    reason: TransitionReason::SuccessThresholdReached,
                    timestamp: Self::current_timestamp(),
                    instant: Some(Instant::now()),
                    failure_count: self.failure_count.load(Ordering::Acquire),
                    success_count: successes,
                    failure_rate: self.failure_rate(),
                })
            }
            _ => None,
        }
    }

    /// Record a failed call
    pub fn record_failure(&self) -> StateTransition {
        self.total_calls.fetch_add(1, Ordering::AcqRel);
        self.total_failures.fetch_add(1, Ordering::AcqRel);

        let failures = self.failure_count.fetch_add(1, Ordering::AcqRel) + 1;
        let state = self.state();

        StateTransition {
            from: state,
            to: state, // State change happens via transition methods
            reason: TransitionReason::FailureThresholdExceeded,
            timestamp: Self::current_timestamp(),
            instant: Some(Instant::now()),
            failure_count: failures,
            success_count: self.success_count.load(Ordering::Acquire),
            failure_rate: self.failure_rate(),
        }
    }

    /// Transition to open state
    pub fn transition_to_open(&self, reason: TransitionReason) -> StateTransition {
        let from = self.state();
        let now = Self::current_timestamp();

        self.state.store(CircuitState::Open as u8, Ordering::Release);
        self.state_changed_at.store(now, Ordering::Release);
        self.opened_at.store(now, Ordering::Release);
        self.success_count.store(0, Ordering::Release);

        StateTransition {
            from,
            to: CircuitState::Open,
            reason,
            timestamp: now,
            instant: Some(Instant::now()),
            failure_count: self.failure_count.load(Ordering::Acquire),
            success_count: 0,
            failure_rate: self.failure_rate(),
        }
    }

    /// Transition to half-open state
    pub fn transition_to_half_open(&self) -> StateTransition {
        let from = self.state();
        let now = Self::current_timestamp();

        self.state.store(CircuitState::HalfOpen as u8, Ordering::Release);
        self.state_changed_at.store(now, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);

        StateTransition {
            from,
            to: CircuitState::HalfOpen,
            reason: TransitionReason::ResetTimeoutExpired,
            timestamp: now,
            instant: Some(Instant::now()),
            failure_count: 0,
            success_count: 0,
            failure_rate: self.failure_rate(),
        }
    }

    /// Transition to closed state
    pub fn transition_to_closed(&self, reason: TransitionReason) -> StateTransition {
        let from = self.state();
        let now = Self::current_timestamp();

        self.state.store(CircuitState::Closed as u8, Ordering::Release);
        self.state_changed_at.store(now, Ordering::Release);
        self.opened_at.store(0, Ordering::Release);
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);

        StateTransition {
            from,
            to: CircuitState::Closed,
            reason,
            timestamp: now,
            instant: Some(Instant::now()),
            failure_count: 0,
            success_count: 0,
            failure_rate: self.failure_rate(),
        }
    }

    /// Check if should transition from open to half-open
    pub fn should_attempt_reset(&self, reset_timeout: Duration) -> bool {
        if self.state() != CircuitState::Open {
            return false;
        }

        match self.time_since_opened() {
            Some(elapsed) => elapsed >= reset_timeout,
            None => false,
        }
    }

    /// Force transition to a specific state
    pub fn force_state(&self, new_state: CircuitState) -> StateTransition {
        let from = self.state();
        let now = Self::current_timestamp();

        self.state.store(new_state as u8, Ordering::Release);
        self.state_changed_at.store(now, Ordering::Release);

        if new_state == CircuitState::Open || new_state == CircuitState::ForcedOpen {
            self.opened_at.store(now, Ordering::Release);
        } else {
            self.opened_at.store(0, Ordering::Release);
        }

        StateTransition {
            from,
            to: new_state,
            reason: TransitionReason::Manual,
            timestamp: now,
            instant: Some(Instant::now()),
            failure_count: self.failure_count.load(Ordering::Acquire),
            success_count: self.success_count.load(Ordering::Acquire),
            failure_rate: self.failure_rate(),
        }
    }

    /// Get snapshot of current state for serialization
    pub fn snapshot(&self) -> StateSnapshot {
        StateSnapshot {
            state: self.state(),
            state_changed_at: self.state_changed_at.load(Ordering::Acquire),
            opened_at: self.opened_at.load(Ordering::Acquire),
            failure_count: self.failure_count.load(Ordering::Acquire),
            success_count: self.success_count.load(Ordering::Acquire),
            total_calls: self.total_calls.load(Ordering::Acquire),
            total_failures: self.total_failures.load(Ordering::Acquire),
            failure_rate: self.failure_rate(),
        }
    }

    /// Restore from snapshot (for distributed sync)
    pub fn restore_from_snapshot(&self, snapshot: &StateSnapshot) {
        self.state.store(snapshot.state as u8, Ordering::Release);
        self.state_changed_at.store(snapshot.state_changed_at, Ordering::Release);
        self.opened_at.store(snapshot.opened_at, Ordering::Release);
        self.failure_count.store(snapshot.failure_count, Ordering::Release);
        self.success_count.store(snapshot.success_count, Ordering::Release);
        self.total_calls.store(snapshot.total_calls, Ordering::Release);
        self.total_failures.store(snapshot.total_failures, Ordering::Release);
    }

    /// Reset all counters (useful for testing or manual reset)
    pub fn reset_counters(&self) {
        self.failure_count.store(0, Ordering::Release);
        self.success_count.store(0, Ordering::Release);
        self.total_calls.store(0, Ordering::Release);
        self.total_failures.store(0, Ordering::Release);
    }
}

impl Default for AtomicStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Serializable snapshot of state machine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// Current state
    pub state: CircuitState,

    /// When state changed (Unix timestamp millis)
    pub state_changed_at: u64,

    /// When circuit opened (Unix timestamp millis, 0 if not open)
    pub opened_at: u64,

    /// Consecutive failure count
    pub failure_count: u64,

    /// Success count in half-open
    pub success_count: u64,

    /// Total calls
    pub total_calls: u64,

    /// Total failures
    pub total_failures: u64,

    /// Calculated failure rate
    pub failure_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_is_closed() {
        let sm = AtomicStateMachine::new();
        assert_eq!(sm.state(), CircuitState::Closed);
        assert!(sm.state().allows_requests());
    }

    #[test]
    fn test_record_success_resets_failure_count() {
        let sm = AtomicStateMachine::new();
        sm.record_failure();
        sm.record_failure();
        assert_eq!(sm.failure_count(), 2);

        sm.record_success();
        assert_eq!(sm.failure_count(), 0);
    }

    #[test]
    fn test_transition_to_open() {
        let sm = AtomicStateMachine::new();
        let transition = sm.transition_to_open(TransitionReason::FailureThresholdExceeded);

        assert_eq!(transition.from, CircuitState::Closed);
        assert_eq!(transition.to, CircuitState::Open);
        assert_eq!(sm.state(), CircuitState::Open);
        assert!(!sm.state().allows_requests());
    }

    #[test]
    fn test_transition_to_half_open() {
        let sm = AtomicStateMachine::new();
        sm.transition_to_open(TransitionReason::FailureThresholdExceeded);
        let transition = sm.transition_to_half_open();

        assert_eq!(transition.from, CircuitState::Open);
        assert_eq!(transition.to, CircuitState::HalfOpen);
        assert_eq!(sm.state(), CircuitState::HalfOpen);
        assert!(sm.state().allows_requests());
    }

    #[test]
    fn test_failure_rate_calculation() {
        let sm = AtomicStateMachine::new();

        // Record 10 calls, 3 failures
        for _ in 0..7 {
            sm.record_success();
        }
        for _ in 0..3 {
            sm.record_failure();
        }

        let rate = sm.failure_rate();
        assert!((rate - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_snapshot_and_restore() {
        let sm1 = AtomicStateMachine::new();
        sm1.record_failure();
        sm1.record_failure();
        sm1.transition_to_open(TransitionReason::FailureThresholdExceeded);

        let snapshot = sm1.snapshot();

        let sm2 = AtomicStateMachine::new();
        sm2.restore_from_snapshot(&snapshot);

        assert_eq!(sm2.state(), CircuitState::Open);
        assert_eq!(sm2.failure_count(), snapshot.failure_count);
    }

    #[test]
    fn test_time_since_opened() {
        let sm = AtomicStateMachine::new();
        assert!(sm.time_since_opened().is_none());

        sm.transition_to_open(TransitionReason::FailureThresholdExceeded);
        std::thread::sleep(Duration::from_millis(10));

        let elapsed = sm.time_since_opened().unwrap();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
