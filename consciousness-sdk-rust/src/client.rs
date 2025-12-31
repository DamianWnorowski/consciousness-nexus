//! Consciousness API Client Implementation

use crate::error::ConsciousnessError;
use crate::models::*;
use async_stream::stream;
use futures::{Stream, StreamExt};
use reqwest::{Client as HttpClient, RequestBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for the Consciousness client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL of the Consciousness API server
    pub base_url: String,
    /// Optional API key for authentication
    pub api_key: Option<String>,
    /// Request timeout
    pub timeout: std::time::Duration,
    /// Enable WebSocket support for streaming
    pub enable_websocket: bool,
}

/// Main client for interacting with Consciousness Computing Suite
pub struct ConsciousnessClient {
    config: ClientConfig,
    http_client: HttpClient,
    session_id: Option<String>,
}

impl ConsciousnessClient {
    /// Create a new client with the given configuration
    pub fn new(config: ClientConfig) -> Result<Self, ConsciousnessError> {
        let http_client = HttpClient::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| ConsciousnessError::Http(e))?;

        Ok(Self {
            config,
            http_client,
            session_id: None,
        })
    }

    /// Authenticate user and establish session
    pub async fn login(
        &mut self,
        username: &str,
        password: &str,
    ) -> Result<LoginResponse, ConsciousnessError> {
        let request = LoginRequest {
            username: username.to_string(),
            password: password.to_string(),
        };

        let response: ApiResponse<LoginData> = self
            .post("/auth/login", &request)
            .await?;

        if response.success {
            if let Some(data) = response.data {
                self.session_id = Some(data.session_id.clone());
                Ok(LoginResponse {
                    session_id: data.session_id,
                    roles: data.roles,
                })
            } else {
                Err(ConsciousnessError::Api("No data in login response".to_string()))
            }
        } else {
            Err(ConsciousnessError::Api(
                response.error.unwrap_or_else(|| "Login failed".to_string())
            ))
        }
    }

    /// Run evolution operation
    pub async fn run_evolution(
        &self,
        operation_type: EvolutionOperation,
        target_system: &str,
        parameters: Option<HashMap<String, serde_json::Value>>,
        safety_level: Option<SafetyLevel>,
    ) -> Result<EvolutionResult, ConsciousnessError> {
        let request = EvolutionRequest {
            operation_type,
            target_system: target_system.to_string(),
            parameters: parameters.unwrap_or_default(),
            safety_level: safety_level.unwrap_or(SafetyLevel::Standard),
            user_id: "rust_sdk_user".to_string(),
            session_id: self.session_id.clone(),
        };

        let response: ApiResponse<EvolutionResult> = self
            .post("/evolution/run", &request)
            .await?;

        if response.success {
            response.data.ok_or_else(|| ConsciousnessError::Api("No evolution result".to_string()))
        } else {
            Err(ConsciousnessError::Api(
                response.error.unwrap_or_else(|| "Evolution failed".to_string())
            ))
        }
    }

    /// Run evolution with streaming progress updates
    pub fn run_evolution_stream(
        &self,
        operation_type: EvolutionOperation,
        target_system: &str,
        parameters: Option<HashMap<String, serde_json::Value>>,
        safety_level: Option<SafetyLevel>,
    ) -> impl Stream<Item = Result<EvolutionProgress, ConsciousnessError>> {
        let request = EvolutionRequest {
            operation_type,
            target_system: target_system.to_string(),
            parameters: parameters.unwrap_or_default(),
            safety_level: safety_level.unwrap_or(SafetyLevel::Standard),
            user_id: "rust_sdk_user".to_string(),
            session_id: self.session_id.clone(),
        };

        // Clone values needed inside the stream to avoid borrowing self
        let client = self.http_client.clone();
        let base_url = self.config.base_url.clone();

        stream! {
            let response = match client
                .post(&format!("{}/evolution/run/stream", base_url))
                .json(&request)
                .send()
                .await
            {
                Ok(resp) => resp,
                Err(e) => {
                    yield Err(ConsciousnessError::Http(e));
                    return;
                }
            };

            let mut byte_stream = response.bytes_stream();

            while let Some(chunk_result) = byte_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        for line in chunk_str.lines() {
                            if line.starts_with("data: ") {
                                let json_str = &line[6..];
                                match serde_json::from_str::<EvolutionProgress>(json_str) {
                                    Ok(progress) => yield Ok(progress),
                                    Err(e) => yield Err(ConsciousnessError::Json(e)),
                                }
                            }
                        }
                    }
                    Err(e) => yield Err(ConsciousnessError::Http(e)),
                }
            }
        }
    }

    /// Run validation on files
    pub async fn run_validation(
        &self,
        files: Vec<String>,
        validation_scope: Option<ValidationScope>,
    ) -> Result<ValidationResult, ConsciousnessError> {
        let request = ValidationRequest {
            files,
            validation_scope: validation_scope.unwrap_or(ValidationScope::Full),
            user_id: "rust_sdk_user".to_string(),
        };

        let response: ApiResponse<ValidationResult> = self
            .post("/validation/run", &request)
            .await?;

        if response.success {
            response.data.ok_or_else(|| ConsciousnessError::Api("No validation result".to_string()))
        } else {
            Err(ConsciousnessError::Api(
                response.error.unwrap_or_else(|| "Validation failed".to_string())
            ))
        }
    }

    /// Run analysis operations
    pub async fn run_analysis(
        &self,
        analysis_type: AnalysisType,
        data: HashMap<String, serde_json::Value>,
    ) -> Result<HashMap<String, serde_json::Value>, ConsciousnessError> {
        let request = AnalysisRequest {
            data,
            analysis_type,
            user_id: "rust_sdk_user".to_string(),
        };

        let response: ApiResponse<HashMap<String, serde_json::Value>> = self
            .post("/analysis/run", &request)
            .await?;

        if response.success {
            response.data.ok_or_else(|| ConsciousnessError::Api("No analysis result".to_string()))
        } else {
            Err(ConsciousnessError::Api(
                response.error.unwrap_or_else(|| "Analysis failed".to_string())
            ))
        }
    }

    /// Get system health status
    pub async fn get_health(&self) -> Result<HealthStatus, ConsciousnessError> {
        let response = self.get("/health").await?;
        Ok(response)
    }

    /// Get comprehensive system status
    pub async fn get_system_status(&self) -> Result<SystemStatus, ConsciousnessError> {
        let response = self.get("/status").await?;
        Ok(response)
    }

    /// Logout and end session
    pub async fn logout(&mut self) -> Result<(), ConsciousnessError> {
        if let Some(session_id) = &self.session_id {
            let _: serde_json::Value = self
                .delete(&format!("/session/{}", session_id))
                .await?;
            self.session_id = None;
        }
        Ok(())
    }

    // Private helper methods
    fn prepare_request(&self, method: reqwest::Method, endpoint: &str) -> RequestBuilder {
        let url = format!("{}{}", self.config.base_url, endpoint);
        let mut request = self.http_client.request(method, &url);

        if let Some(api_key) = &self.config.api_key {
            request = request.header("X-API-Key", api_key);
        }

        request.header("Content-Type", "application/json")
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, endpoint: &str) -> Result<T, ConsciousnessError> {
        let response = self.prepare_request(reqwest::Method::GET, endpoint)
            .send()
            .await
            .map_err(ConsciousnessError::Http)?;

        response.json().await.map_err(ConsciousnessError::Http)
    }

    async fn post<T: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        endpoint: &str,
        data: &T,
    ) -> Result<R, ConsciousnessError> {
        let response = self.prepare_request(reqwest::Method::POST, endpoint)
            .json(data)
            .send()
            .await
            .map_err(ConsciousnessError::Http)?;

        response.json().await.map_err(ConsciousnessError::Http)
    }

    async fn delete<T: for<'de> Deserialize<'de>>(&self, endpoint: &str) -> Result<T, ConsciousnessError> {
        let response = self.prepare_request(reqwest::Method::DELETE, endpoint)
            .send()
            .await
            .map_err(ConsciousnessError::Http)?;

        response.json().await.map_err(ConsciousnessError::Http)
    }
}
