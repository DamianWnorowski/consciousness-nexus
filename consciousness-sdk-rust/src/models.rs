//! Data models for Consciousness API

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Generic API response wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub session_id: Option<String>,
    pub execution_time: f64,
}

/// Safety levels for operations
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SafetyLevel {
    Minimal,
    Standard,
    Strict,
    Paranoid,
}

/// Evolution operation types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvolutionOperation {
    Verified,
    Recursive,
}

/// Validation scopes
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationScope {
    Basic,
    Full,
    Comprehensive,
}

/// Analysis types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnalysisType {
    Fitness,
    Performance,
    Security,
}

/// Login request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginRequest {
    pub username: String,
    pub password: String,
}

/// Login response data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginData {
    pub session_id: String,
    pub roles: Vec<String>,
}

/// Login response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoginResponse {
    pub session_id: String,
    pub roles: Vec<String>,
}

/// Evolution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionRequest {
    pub operation_type: EvolutionOperation,
    pub target_system: String,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
    pub safety_level: SafetyLevel,
    pub user_id: String,
    pub session_id: Option<String>,
}

/// Evolution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub evolution_id: String,
    pub status: String,
    pub results: HashMap<String, serde_json::Value>,
    pub metrics: EvolutionMetrics,
}

/// Evolution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionMetrics {
    pub fitness_score: f64,
    pub execution_time: f64,
    pub safety_checks: u32,
    pub warnings: Vec<String>,
}

/// Evolution progress update
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionProgress {
    pub stage: String,
    pub progress: f64,
    pub message: String,
    pub complete: Option<bool>,
    pub error: Option<String>,
}

/// Validation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRequest {
    pub files: Vec<String>,
    pub validation_scope: ValidationScope,
    pub user_id: String,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub total_checks: u32,
    pub passed_checks: u32,
    pub issues: Vec<ValidationIssue>,
    pub fitness_score: f64,
    pub warnings: Vec<String>,
}

/// Validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub category: String,
    pub title: String,
    pub description: String,
    pub file: String,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ValidationSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Analysis request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisRequest {
    pub data: HashMap<String, serde_json::Value>,
    pub analysis_type: AnalysisType,
    pub user_id: String,
}

/// Health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub timestamp: f64,
    pub uptime: f64,
    pub active_sessions: u32,
}

/// System status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub api_server: ApiServerStatus,
    pub consciousness_suite: ConsciousnessSuiteStatus,
    pub timestamp: f64,
}

/// API server status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiServerStatus {
    pub status: String,
    pub uptime: f64,
    pub active_sessions: u32,
}

/// Consciousness suite status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessSuiteStatus {
    pub initialized: bool,
    pub safety_level: String,
    pub systems_status: HashMap<String, String>,
}
