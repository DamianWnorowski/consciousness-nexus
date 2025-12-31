//! 🛡️ CONSCIOUSNESS COMPUTING SUITE - RUST SDK
//!
//! Universal SDK for accessing Consciousness Computing Suite from Rust applications.
//! Makes enterprise-grade AI safety and evolution tools available in:
//! - Rust applications and libraries
//! - WebAssembly (WASM) modules
//! - System-level services
//! - High-performance computing environments

pub mod client;
pub mod error;
pub mod models;

pub use client::ConsciousnessClient;
pub use error::ConsciousnessError;
pub use models::*;

/// Quick setup functions for common configurations
pub mod prelude {
    pub use super::client::ConsciousnessClient;
    pub use super::error::ConsciousnessError;
    pub use super::models::*;

    use super::client::ClientConfig;

    /// Create a client for local development
    pub fn create_local_client(port: u16) -> Result<ConsciousnessClient, ConsciousnessError> {
        ConsciousnessClient::new(ClientConfig {
            base_url: format!("http://localhost:{}", port),
            api_key: None,
            timeout: std::time::Duration::from_secs(30),
            enable_websocket: true,
        })
    }

    /// Create a client for production use
    pub fn create_production_client(
        api_key: String,
        base_url: String,
    ) -> Result<ConsciousnessClient, ConsciousnessError> {
        ConsciousnessClient::new(ClientConfig {
            base_url,
            api_key: Some(api_key),
            timeout: std::time::Duration::from_secs(60),
            enable_websocket: false,
        })
    }
}
