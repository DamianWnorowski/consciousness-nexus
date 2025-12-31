//! Error types for Consciousness SDK

use thiserror::Error;

/// Main error type for Consciousness operations
#[derive(Error, Debug)]
pub enum ConsciousnessError {
    #[error("HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON serialization/deserialization failed: {0}")]
    Json(#[from] serde_json::Error),

    #[error("API error: {0}")]
    Api(String),

    #[error("URL parsing failed: {0}")]
    Url(#[from] url::ParseError),

    #[error("WebSocket error: {0}")]
    WebSocket(String),

    #[error("Authentication failed")]
    Authentication,

    #[error("Session expired or invalid")]
    SessionExpired,

    #[error("Operation timeout")]
    Timeout,

    #[error("Invalid configuration: {0}")]
    Config(String),
}

/// Result type alias for convenience
pub type ConsciousnessResult<T> = Result<T, ConsciousnessError>;
