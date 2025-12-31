//! Core circuit breaker implementation
//!
//! This module contains the fundamental building blocks:
//! - State machine with lock-free transitions
//! - Configuration management
//! - Error types
//! - Health tracking
//! - Sliding window for failure rate calculation

pub mod breaker;
pub mod config;
pub mod error;
pub mod health;
pub mod sliding_window;
pub mod state;
