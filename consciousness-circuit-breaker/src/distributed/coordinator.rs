//! Distributed coordinator for cluster-wide circuit breaker state
//!
//! Provides:
//! - Node discovery and registration
//! - State synchronization across nodes
//! - Leader election for coordination
//! - Health monitoring of cluster

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};

use crate::core::config::{CoordinationBackend, DistributedConfig};
use crate::core::error::Result;
use crate::core::state::CircuitState;

/// Distributed coordinator for circuit breaker clusters
#[derive(Debug)]
pub struct DistributedCoordinator {
    /// Configuration
    config: DistributedConfig,
    /// This node's ID
    node_id: String,
    /// Known nodes in the cluster
    nodes: DashMap<String, NodeInfo>,
    /// Cluster state
    cluster_state: Arc<RwLock<ClusterState>>,
    /// Whether this node is the leader
    is_leader: AtomicBool,
    /// Current leader ID
    leader_id: RwLock<Option<String>>,
    /// Event broadcaster
    event_tx: broadcast::Sender<CoordinatorEvent>,
    /// Start time
    started_at: Instant,
    /// Heartbeat counter
    heartbeat_count: AtomicU64,
}

/// Information about a node in the cluster
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    /// Node unique identifier
    pub id: String,
    /// Node address (host:port)
    pub address: String,
    /// When the node joined
    pub joined_at: u64,
    /// Last heartbeat received
    pub last_heartbeat: u64,
    /// Node health status
    pub health: NodeHealth,
    /// Circuit breaker states on this node
    pub circuit_states: HashMap<String, CircuitStateInfo>,
    /// Node metadata
    pub metadata: HashMap<String, String>,
}

/// Node health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeHealth {
    /// Node is healthy
    Healthy,
    /// Node is suspected (missed heartbeats)
    Suspect,
    /// Node is unreachable
    Unreachable,
    /// Node has left the cluster
    Left,
}

/// Circuit breaker state info for synchronization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStateInfo {
    /// Circuit breaker name
    pub name: String,
    /// Current state
    pub state: String,
    /// Total calls
    pub total_calls: u64,
    /// Total failures
    pub total_failures: u64,
    /// Failure rate
    pub failure_rate: f64,
    /// Last state change timestamp
    pub last_state_change: u64,
    /// Last updated timestamp
    pub last_updated: u64,
}

/// Overall cluster state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClusterState {
    /// Cluster namespace
    pub namespace: String,
    /// Number of active nodes
    pub active_nodes: usize,
    /// Current leader ID
    pub leader_id: Option<String>,
    /// Cluster health
    pub health: ClusterHealth,
    /// Aggregated circuit states
    pub circuits: HashMap<String, AggregatedCircuitState>,
    /// Last sync timestamp
    pub last_sync: u64,
}

/// Cluster health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ClusterHealth {
    /// All nodes healthy
    #[default]
    Healthy,
    /// Some nodes degraded
    Degraded,
    /// Cluster is unhealthy
    Unhealthy,
    /// No quorum
    NoQuorum,
}

/// Aggregated circuit state across all nodes
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedCircuitState {
    /// Circuit breaker name
    pub name: String,
    /// Nodes where this circuit is OPEN
    pub open_count: u32,
    /// Nodes where this circuit is CLOSED
    pub closed_count: u32,
    /// Nodes where this circuit is HALF_OPEN
    pub half_open_count: u32,
    /// Total calls across all nodes
    pub total_calls: u64,
    /// Total failures across all nodes
    pub total_failures: u64,
    /// Average failure rate
    pub avg_failure_rate: f64,
    /// Recommendation: should the circuit be open?
    pub recommendation: CircuitRecommendation,
}

/// Recommendation for circuit state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CircuitRecommendation {
    /// Keep current state
    #[default]
    KeepCurrent,
    /// Should open the circuit
    ShouldOpen,
    /// Should close the circuit
    ShouldClose,
    /// Should try half-open
    ShouldHalfOpen,
}

/// Events from the coordinator
#[derive(Debug, Clone)]
pub enum CoordinatorEvent {
    /// Node joined the cluster
    NodeJoined(String),
    /// Node left the cluster
    NodeLeft(String),
    /// Leadership changed
    LeaderChanged { old: Option<String>, new: String },
    /// Circuit state changed on a node
    CircuitStateChanged {
        node_id: String,
        circuit: String,
        old_state: String,
        new_state: String,
    },
    /// Cluster health changed
    ClusterHealthChanged(ClusterHealth),
}

impl DistributedCoordinator {
    /// Create a new distributed coordinator
    pub fn new(config: DistributedConfig) -> Self {
        let node_id = config
            .node_id
            .clone()
            .unwrap_or_else(|| format!("node-{}", uuid_v4()));

        let (event_tx, _) = broadcast::channel(1000);

        Self {
            config,
            node_id: node_id.clone(),
            nodes: DashMap::new(),
            cluster_state: Arc::new(RwLock::new(ClusterState::default())),
            is_leader: AtomicBool::new(false),
            leader_id: RwLock::new(None),
            event_tx,
            started_at: Instant::now(),
            heartbeat_count: AtomicU64::new(0),
        }
    }

    /// Start the coordinator
    pub async fn start(&self) -> Result<()> {
        // Register this node
        self.register_self().await?;

        // Start background tasks
        self.start_heartbeat_task();
        self.start_health_check_task();

        // Initial leader election
        self.try_become_leader().await;

        tracing::info!(
            node_id = %self.node_id,
            namespace = %self.config.namespace,
            "Distributed coordinator started"
        );

        Ok(())
    }

    /// Register this node in the cluster
    async fn register_self(&self) -> Result<()> {
        let now = current_timestamp();

        let node_info = NodeInfo {
            id: self.node_id.clone(),
            address: self.config.endpoints.first().cloned().unwrap_or_default(),
            joined_at: now,
            last_heartbeat: now,
            health: NodeHealth::Healthy,
            circuit_states: HashMap::new(),
            metadata: HashMap::new(),
        };

        self.nodes.insert(self.node_id.clone(), node_info);

        Ok(())
    }

    /// Start heartbeat background task
    fn start_heartbeat_task(&self) {
        let node_id = self.node_id.clone();
        let nodes = self.nodes.clone();
        let heartbeat_count = Arc::new(AtomicU64::new(self.heartbeat_count.load(Ordering::Relaxed)));
        let interval = self.config.lease_ttl / 3;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                let now = current_timestamp();

                // Update own heartbeat
                if let Some(mut node) = nodes.get_mut(&node_id) {
                    node.last_heartbeat = now;
                }

                heartbeat_count.fetch_add(1, Ordering::Relaxed);
            }
        });
    }

    /// Start health check background task
    fn start_health_check_task(&self) {
        let nodes = self.nodes.clone();
        let cluster_state = self.cluster_state.clone();
        let event_tx = self.event_tx.clone();
        let lease_ttl = self.config.lease_ttl;
        let namespace = self.config.namespace.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(lease_ttl / 2);

            loop {
                interval.tick().await;

                let now = current_timestamp();
                let suspect_threshold = lease_ttl.as_secs();
                let unreachable_threshold = lease_ttl.as_secs() * 3;

                let mut healthy_count = 0;
                let mut suspect_count = 0;
                let mut unreachable_count = 0;

                // Check each node's health
                for mut entry in nodes.iter_mut() {
                    let elapsed = now.saturating_sub(entry.last_heartbeat);

                    let new_health = if elapsed > unreachable_threshold {
                        unreachable_count += 1;
                        NodeHealth::Unreachable
                    } else if elapsed > suspect_threshold {
                        suspect_count += 1;
                        NodeHealth::Suspect
                    } else {
                        healthy_count += 1;
                        NodeHealth::Healthy
                    };

                    if entry.health != new_health {
                        entry.health = new_health;
                    }
                }

                // Update cluster state
                let cluster_health = if unreachable_count > healthy_count {
                    ClusterHealth::Unhealthy
                } else if suspect_count > 0 || unreachable_count > 0 {
                    ClusterHealth::Degraded
                } else {
                    ClusterHealth::Healthy
                };

                let mut state = cluster_state.write().await;
                let old_health = state.health;
                state.namespace = namespace.clone();
                state.active_nodes = healthy_count;
                state.health = cluster_health;
                state.last_sync = now;

                if old_health != cluster_health {
                    let _ = event_tx.send(CoordinatorEvent::ClusterHealthChanged(cluster_health));
                }
            }
        });
    }

    /// Try to become the leader
    async fn try_become_leader(&self) {
        // Simple leader election: node with lowest ID becomes leader
        let mut lowest_id: Option<String> = None;

        for entry in self.nodes.iter() {
            if entry.health == NodeHealth::Healthy {
                if lowest_id.is_none() || entry.id < *lowest_id.as_ref().unwrap() {
                    lowest_id = Some(entry.id.clone());
                }
            }
        }

        if let Some(leader) = lowest_id {
            let is_leader = leader == self.node_id;
            let was_leader = self.is_leader.swap(is_leader, Ordering::SeqCst);

            let mut leader_id = self.leader_id.write().await;
            let old_leader = leader_id.clone();

            if *leader_id != Some(leader.clone()) {
                *leader_id = Some(leader.clone());

                let _ = self.event_tx.send(CoordinatorEvent::LeaderChanged {
                    old: old_leader,
                    new: leader.clone(),
                });

                if is_leader && !was_leader {
                    tracing::info!(node_id = %self.node_id, "This node is now the leader");
                }
            }
        }
    }

    /// Update circuit state from this node
    pub async fn update_circuit_state(
        &self,
        name: &str,
        state: CircuitState,
        total_calls: u64,
        total_failures: u64,
        failure_rate: f64,
    ) -> Result<()> {
        let now = current_timestamp();

        let state_info = CircuitStateInfo {
            name: name.to_string(),
            state: format!("{:?}", state),
            total_calls,
            total_failures,
            failure_rate,
            last_state_change: now,
            last_updated: now,
        };

        // Update local node info
        if let Some(mut node) = self.nodes.get_mut(&self.node_id) {
            let old_state = node
                .circuit_states
                .get(name)
                .map(|s| s.state.clone())
                .unwrap_or_default();

            node.circuit_states.insert(name.to_string(), state_info);

            // Emit event if state changed
            if old_state != format!("{:?}", state) {
                let _ = self.event_tx.send(CoordinatorEvent::CircuitStateChanged {
                    node_id: self.node_id.clone(),
                    circuit: name.to_string(),
                    old_state,
                    new_state: format!("{:?}", state),
                });
            }
        }

        // Aggregate states across cluster
        self.aggregate_circuit_states(name).await;

        Ok(())
    }

    /// Aggregate circuit states across all nodes
    async fn aggregate_circuit_states(&self, circuit_name: &str) {
        let mut open_count = 0u32;
        let mut closed_count = 0u32;
        let mut half_open_count = 0u32;
        let mut total_calls = 0u64;
        let mut total_failures = 0u64;
        let mut failure_rate_sum = 0.0f64;
        let mut node_count = 0u32;

        for entry in self.nodes.iter() {
            if entry.health != NodeHealth::Healthy {
                continue;
            }

            if let Some(state) = entry.circuit_states.get(circuit_name) {
                match state.state.as_str() {
                    "Open" => open_count += 1,
                    "Closed" => closed_count += 1,
                    "HalfOpen" => half_open_count += 1,
                    _ => {}
                }
                total_calls += state.total_calls;
                total_failures += state.total_failures;
                failure_rate_sum += state.failure_rate;
                node_count += 1;
            }
        }

        let avg_failure_rate = if node_count > 0 {
            failure_rate_sum / node_count as f64
        } else {
            0.0
        };

        // Determine recommendation
        let recommendation = if open_count > (closed_count + half_open_count) {
            CircuitRecommendation::ShouldOpen
        } else if closed_count > (open_count * 2) && avg_failure_rate < 0.1 {
            CircuitRecommendation::ShouldClose
        } else if half_open_count > 0 {
            CircuitRecommendation::ShouldHalfOpen
        } else {
            CircuitRecommendation::KeepCurrent
        };

        let aggregated = AggregatedCircuitState {
            name: circuit_name.to_string(),
            open_count,
            closed_count,
            half_open_count,
            total_calls,
            total_failures,
            avg_failure_rate,
            recommendation,
        };

        let mut state = self.cluster_state.write().await;
        state.circuits.insert(circuit_name.to_string(), aggregated);
    }

    /// Get the current cluster state
    pub async fn cluster_state(&self) -> ClusterState {
        self.cluster_state.read().await.clone()
    }

    /// Check if this node is the leader
    pub fn is_leader(&self) -> bool {
        self.is_leader.load(Ordering::Relaxed)
    }

    /// Get the current leader ID
    pub async fn leader_id(&self) -> Option<String> {
        self.leader_id.read().await.clone()
    }

    /// Get this node's ID
    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    /// Get all nodes
    pub fn nodes(&self) -> Vec<NodeInfo> {
        self.nodes.iter().map(|e| e.value().clone()).collect()
    }

    /// Get healthy nodes
    pub fn healthy_nodes(&self) -> Vec<NodeInfo> {
        self.nodes
            .iter()
            .filter(|e| e.health == NodeHealth::Healthy)
            .map(|e| e.value().clone())
            .collect()
    }

    /// Subscribe to coordinator events
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinatorEvent> {
        self.event_tx.subscribe()
    }

    /// Get coordinator statistics
    pub fn stats(&self) -> CoordinatorStats {
        CoordinatorStats {
            node_id: self.node_id.clone(),
            is_leader: self.is_leader.load(Ordering::Relaxed),
            total_nodes: self.nodes.len(),
            healthy_nodes: self.nodes.iter().filter(|e| e.health == NodeHealth::Healthy).count(),
            heartbeat_count: self.heartbeat_count.load(Ordering::Relaxed),
            uptime: self.started_at.elapsed(),
            backend: self.config.backend,
        }
    }
}

/// Coordinator statistics
#[derive(Debug, Clone)]
pub struct CoordinatorStats {
    /// This node's ID
    pub node_id: String,
    /// Whether this node is the leader
    pub is_leader: bool,
    /// Total nodes in cluster
    pub total_nodes: usize,
    /// Healthy nodes in cluster
    pub healthy_nodes: usize,
    /// Heartbeats sent
    pub heartbeat_count: u64,
    /// Coordinator uptime
    pub uptime: Duration,
    /// Backend type
    pub backend: CoordinationBackend,
}

/// Generate a simple UUID v4
fn uuid_v4() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", now)
}

/// Get current timestamp in seconds
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> DistributedConfig {
        DistributedConfig {
            backend: CoordinationBackend::InMemory,
            endpoints: vec!["localhost:2379".to_string()],
            namespace: "test".to_string(),
            node_id: Some("test-node-1".to_string()),
            lease_ttl: Duration::from_secs(10),
        }
    }

    #[tokio::test]
    async fn test_coordinator_creation() {
        let coordinator = DistributedCoordinator::new(test_config());
        assert_eq!(coordinator.node_id(), "test-node-1");
    }

    #[tokio::test]
    async fn test_coordinator_start() {
        let coordinator = DistributedCoordinator::new(test_config());
        let result = coordinator.start().await;
        assert!(result.is_ok());
        assert_eq!(coordinator.nodes().len(), 1);
    }

    #[tokio::test]
    async fn test_circuit_state_update() {
        let coordinator = DistributedCoordinator::new(test_config());
        coordinator.start().await.unwrap();

        coordinator
            .update_circuit_state("test-breaker", CircuitState::Closed, 100, 5, 0.05)
            .await
            .unwrap();

        let state = coordinator.cluster_state().await;
        assert!(state.circuits.contains_key("test-breaker"));
    }

    #[tokio::test]
    async fn test_single_node_is_leader() {
        let coordinator = DistributedCoordinator::new(test_config());
        coordinator.start().await.unwrap();

        // Give time for leader election
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Single node should be leader
        assert!(coordinator.is_leader());
    }

    #[test]
    fn test_coordinator_stats() {
        let coordinator = DistributedCoordinator::new(test_config());
        let stats = coordinator.stats();

        assert_eq!(stats.node_id, "test-node-1");
        assert_eq!(stats.backend, CoordinationBackend::InMemory);
    }
}
