//! Error types for the circuit breaker
//!
//! All errors are real - no simulated or fake error conditions.

use std::time::Duration;
use thiserror::Error;

/// Result type alias for circuit breaker operations
pub type Result<T> = std::result::Result<T, Error>;

/// Circuit breaker error types
#[derive(Error, Debug, Clone)]
pub enum Error {
    /// Circuit is currently open, rejecting requests
    #[error("Circuit breaker is OPEN - request rejected (reset in {reset_after:?})")]
    CircuitOpen {
        /// Time until the circuit breaker will attempt to half-open
        reset_after: Duration,
        /// Number of consecutive failures that caused the open state
        failure_count: u64,
        /// Last recorded error message
        last_error: Option<String>,
    },

    /// Operation timed out
    #[error("Operation timed out after {elapsed:?} (limit: {limit:?})")]
    Timeout {
        /// How long the operation ran before timeout
        elapsed: Duration,
        /// The configured timeout limit
        limit: Duration,
    },

    /// Too many concurrent requests (bulkhead pattern)
    #[error("Bulkhead limit reached: {current}/{max} concurrent requests")]
    BulkheadFull {
        /// Current number of in-flight requests
        current: usize,
        /// Maximum allowed concurrent requests
        max: usize,
    },

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {requests_per_second:.2} req/s (limit: {limit:.2})")]
    RateLimitExceeded {
        /// Current request rate
        requests_per_second: f64,
        /// Configured limit
        limit: f64,
    },

    /// Metrics collection failed
    #[error("Failed to collect real metrics: {message}")]
    MetricsCollection {
        /// Error message
        message: String,
    },

    /// Distributed coordination error
    #[error("Distributed coordination failed: {message}")]
    Coordination {
        /// Error message
        message: String,
        /// Node ID that experienced the error
        node_id: Option<String>,
    },

    /// Configuration error
    #[error("Invalid configuration: {field} - {reason}")]
    Configuration {
        /// Field that has invalid value
        field: String,
        /// Reason why it's invalid
        reason: String,
    },

    /// Internal error (should never happen in production)
    #[error("Internal circuit breaker error: {0}")]
    Internal(String),

    /// Wrapped operation error
    #[error("Operation failed: {0}")]
    Operation(String),
}

impl Error {
    /// Returns true if this error indicates the circuit is open
    #[inline]
    pub fn is_circuit_open(&self) -> bool {
        matches!(self, Error::CircuitOpen { .. })
    }

    /// Returns true if this error is retriable
    #[inline]
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Error::Timeout { .. }
                | Error::BulkheadFull { .. }
                | Error::RateLimitExceeded { .. }
        )
    }

    /// Returns the reset duration if circuit is open
    #[inline]
    pub fn reset_after(&self) -> Option<Duration> {
        match self {
            Error::CircuitOpen { reset_after, .. } => Some(*reset_after),
            _ => None,
        }
    }

    /// Create a circuit open error with context
    pub fn circuit_open(reset_after: Duration, failure_count: u64, last_error: Option<String>) -> Self {
        Error::CircuitOpen {
            reset_after,
            failure_count,
            last_error,
        }
    }

    /// Create a timeout error
    pub fn timeout(elapsed: Duration, limit: Duration) -> Self {
        Error::Timeout { elapsed, limit }
    }

    /// Create a bulkhead full error
    pub fn bulkhead_full(current: usize, max: usize) -> Self {
        Error::BulkheadFull { current, max }
    }
}

/// Extension trait for converting errors
pub trait IntoCircuitError {
    /// Convert into a circuit breaker error
    fn into_circuit_error(self) -> Error;
}

impl<E: std::error::Error> IntoCircuitError for E {
    fn into_circuit_error(self) -> Error {
        Error::Operation(self.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_open_error() {
        let err = Error::circuit_open(
            Duration::from_secs(30),
            5,
            Some("connection refused".to_string()),
        );

        assert!(err.is_circuit_open());
        assert!(!err.is_retriable());
        assert_eq!(err.reset_after(), Some(Duration::from_secs(30)));
    }

    #[test]
    fn test_retriable_errors() {
        assert!(Error::timeout(Duration::from_secs(1), Duration::from_secs(5)).is_retriable());
        assert!(Error::bulkhead_full(10, 10).is_retriable());
        assert!(!Error::Internal("test".to_string()).is_retriable());
    }
}
