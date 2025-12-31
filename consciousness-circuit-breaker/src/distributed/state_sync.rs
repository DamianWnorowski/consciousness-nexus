//! State synchronization for distributed circuit breakers
//!
//! Handles real-time synchronization of circuit breaker state
//! across cluster nodes.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};

use crate::core::state::CircuitState;
use crate::distributed::coordinator::DistributedCoordinator;

/// State synchronization handler
#[derive(Debug)]
pub struct StateSync {
    /// Reference to the coordinator
    coordinator: Arc<DistributedCoordinator>,
    /// Pending updates queue
    pending_updates: Arc<RwLock<Vec<StateUpdate>>>,
    /// Event sender
    event_tx: broadcast::Sender<SyncEvent>,
    /// Sync interval
    sync_interval: Duration,
    /// Started flag
    started: Arc<std::sync::atomic::AtomicBool>,
}

/// A state update to be synchronized
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    /// Circuit breaker name
    pub circuit_name: String,
    /// New state
    pub state: String,
    /// Total calls at time of update
    pub total_calls: u64,
    /// Total failures at time of update
    pub total_failures: u64,
    /// Failure rate at time of update
    pub failure_rate: f64,
    /// Timestamp of update
    pub timestamp: u64,
    /// Source node ID
    pub source_node: String,
    /// Update sequence number
    pub sequence: u64,
}

/// Synchronization events
#[derive(Debug, Clone)]
pub enum SyncEvent {
    /// State was synchronized successfully
    StateSynced {
        circuit_name: String,
        from_node: String,
        state: String,
    },
    /// Sync conflict detected
    ConflictDetected {
        circuit_name: String,
        local_state: String,
        remote_state: String,
        resolution: ConflictResolution,
    },
    /// Full sync completed
    FullSyncCompleted {
        circuits_synced: usize,
        duration: Duration,
    },
    /// Sync failed
    SyncFailed {
        circuit_name: Option<String>,
        error: String,
    },
}

/// How a conflict was resolved
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictResolution {
    /// Used local state (more recent)
    UsedLocal,
    /// Used remote state (more recent)
    UsedRemote,
    /// Used most conservative (OPEN wins)
    UsedConservative,
    /// Leader decided
    LeaderDecision,
}

impl StateSync {
    /// Create a new state synchronizer
    pub fn new(coordinator: Arc<DistributedCoordinator>, sync_interval: Duration) -> Self {
        let (event_tx, _) = broadcast::channel(1000);

        Self {
            coordinator,
            pending_updates: Arc::new(RwLock::new(Vec::new())),
            event_tx,
            sync_interval,
            started: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Start the synchronization loop
    pub async fn start(&self) {
        use std::sync::atomic::Ordering;

        if self.started.swap(true, Ordering::SeqCst) {
            return; // Already started
        }

        let pending = self.pending_updates.clone();
        let coordinator = self.coordinator.clone();
        let event_tx = self.event_tx.clone();
        let interval = self.sync_interval;

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                interval_timer.tick().await;

                // Process pending updates
                let updates = {
                    let mut pending = pending.write().await;
                    std::mem::take(&mut *pending)
                };

                if !updates.is_empty() {
                    for update in updates {
                        if let Err(e) = Self::apply_update(&coordinator, &update).await {
                            let _ = event_tx.send(SyncEvent::SyncFailed {
                                circuit_name: Some(update.circuit_name),
                                error: e.to_string(),
                            });
                        }
                    }
                }
            }
        });
    }

    /// Queue a state update for synchronization
    pub async fn queue_update(&self, update: StateUpdate) {
        let mut pending = self.pending_updates.write().await;
        pending.push(update);
    }

    /// Apply a state update
    async fn apply_update(
        coordinator: &DistributedCoordinator,
        update: &StateUpdate,
    ) -> Result<(), String> {
        let state = match update.state.as_str() {
            "Open" => CircuitState::Open,
            "Closed" => CircuitState::Closed,
            "HalfOpen" => CircuitState::HalfOpen,
            _ => return Err(format!("Unknown state: {}", update.state)),
        };

        coordinator
            .update_circuit_state(
                &update.circuit_name,
                state,
                update.total_calls,
                update.total_failures,
                update.failure_rate,
            )
            .await
            .map_err(|e| e.to_string())
    }

    /// Force a full sync with the cluster
    pub async fn full_sync(&self) -> Result<SyncResult, String> {
        let start = Instant::now();
        let cluster_state = self.coordinator.cluster_state().await;

        let mut circuits_synced = 0;
        let mut conflicts = Vec::new();

        for (name, aggregated) in cluster_state.circuits.iter() {
            // Resolve any conflicts
            if aggregated.open_count > 0 && aggregated.closed_count > 0 {
                // Conflict: some nodes have circuit open, others closed
                conflicts.push(SyncConflict {
                    circuit_name: name.clone(),
                    open_nodes: aggregated.open_count,
                    closed_nodes: aggregated.closed_count,
                    half_open_nodes: aggregated.half_open_count,
                    resolution: aggregated.recommendation,
                });
            }

            circuits_synced += 1;
        }

        let duration = start.elapsed();

        let _ = self.event_tx.send(SyncEvent::FullSyncCompleted {
            circuits_synced,
            duration,
        });

        Ok(SyncResult {
            circuits_synced,
            conflicts,
            duration,
        })
    }

    /// Resolve a state conflict
    pub fn resolve_conflict(
        local_state: CircuitState,
        remote_state: CircuitState,
        local_timestamp: u64,
        remote_timestamp: u64,
    ) -> (CircuitState, ConflictResolution) {
        // Conservative resolution: OPEN always wins for safety
        match (local_state, remote_state) {
            (CircuitState::Open, _) | (_, CircuitState::Open) => {
                (CircuitState::Open, ConflictResolution::UsedConservative)
            }
            (CircuitState::HalfOpen, CircuitState::Closed)
            | (CircuitState::Closed, CircuitState::HalfOpen) => {
                // Use more recent
                if local_timestamp >= remote_timestamp {
                    (local_state, ConflictResolution::UsedLocal)
                } else {
                    (remote_state, ConflictResolution::UsedRemote)
                }
            }
            _ => {
                // Same state or timestamp-based
                if local_timestamp >= remote_timestamp {
                    (local_state, ConflictResolution::UsedLocal)
                } else {
                    (remote_state, ConflictResolution::UsedRemote)
                }
            }
        }
    }

    /// Subscribe to sync events
    pub fn subscribe(&self) -> broadcast::Receiver<SyncEvent> {
        self.event_tx.subscribe()
    }

    /// Get pending update count
    pub async fn pending_count(&self) -> usize {
        self.pending_updates.read().await.len()
    }
}

/// Result of a full sync operation
#[derive(Debug, Clone)]
pub struct SyncResult {
    /// Number of circuits synchronized
    pub circuits_synced: usize,
    /// Conflicts detected and resolved
    pub conflicts: Vec<SyncConflict>,
    /// Time taken
    pub duration: Duration,
}

/// A detected sync conflict
#[derive(Debug, Clone)]
pub struct SyncConflict {
    /// Circuit breaker name
    pub circuit_name: String,
    /// Nodes with circuit open
    pub open_nodes: u32,
    /// Nodes with circuit closed
    pub closed_nodes: u32,
    /// Nodes with circuit half-open
    pub half_open_nodes: u32,
    /// How it was resolved
    pub resolution: crate::distributed::coordinator::CircuitRecommendation,
}

/// Vector clock for causal ordering
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VectorClock {
    /// Timestamps per node
    timestamps: HashMap<String, u64>,
}

impl VectorClock {
    /// Create a new vector clock
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the clock for a node
    pub fn increment(&mut self, node_id: &str) {
        let counter = self.timestamps.entry(node_id.to_string()).or_insert(0);
        *counter += 1;
    }

    /// Merge with another clock (take max of each entry)
    pub fn merge(&mut self, other: &VectorClock) {
        for (node, timestamp) in &other.timestamps {
            let entry = self.timestamps.entry(node.clone()).or_insert(0);
            *entry = (*entry).max(*timestamp);
        }
    }

    /// Check if this clock happened before another
    pub fn happened_before(&self, other: &VectorClock) -> bool {
        let mut dominated = false;

        for (node, &timestamp) in &self.timestamps {
            let other_timestamp = other.timestamps.get(node).copied().unwrap_or(0);
            if timestamp > other_timestamp {
                return false;
            }
            if timestamp < other_timestamp {
                dominated = true;
            }
        }

        for (node, &timestamp) in &other.timestamps {
            if !self.timestamps.contains_key(node) && timestamp > 0 {
                dominated = true;
            }
        }

        dominated
    }

    /// Check if clocks are concurrent (neither happened before the other)
    pub fn is_concurrent(&self, other: &VectorClock) -> bool {
        !self.happened_before(other) && !other.happened_before(self)
    }

    /// Get timestamp for a node
    pub fn get(&self, node_id: &str) -> u64 {
        self.timestamps.get(node_id).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{CoordinationBackend, DistributedConfig};

    fn test_coordinator() -> Arc<DistributedCoordinator> {
        Arc::new(DistributedCoordinator::new(DistributedConfig {
            backend: CoordinationBackend::InMemory,
            endpoints: vec!["localhost:2379".to_string()],
            namespace: "test".to_string(),
            node_id: Some("test-node".to_string()),
            lease_ttl: Duration::from_secs(10),
        }))
    }

    #[tokio::test]
    async fn test_state_sync_creation() {
        let coordinator = test_coordinator();
        let sync = StateSync::new(coordinator, Duration::from_secs(1));

        assert_eq!(sync.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_queue_update() {
        let coordinator = test_coordinator();
        let sync = StateSync::new(coordinator, Duration::from_secs(1));

        let update = StateUpdate {
            circuit_name: "test".to_string(),
            state: "Open".to_string(),
            total_calls: 100,
            total_failures: 50,
            failure_rate: 0.5,
            timestamp: 12345,
            source_node: "node-1".to_string(),
            sequence: 1,
        };

        sync.queue_update(update).await;
        assert_eq!(sync.pending_count().await, 1);
    }

    #[test]
    fn test_conflict_resolution_conservative() {
        // OPEN should always win
        let (resolved, resolution) = StateSync::resolve_conflict(
            CircuitState::Closed,
            CircuitState::Open,
            100,
            50,
        );

        assert_eq!(resolved, CircuitState::Open);
        assert_eq!(resolution, ConflictResolution::UsedConservative);
    }

    #[test]
    fn test_conflict_resolution_timestamp() {
        let (resolved, resolution) = StateSync::resolve_conflict(
            CircuitState::Closed,
            CircuitState::HalfOpen,
            100, // local is newer
            50,
        );

        assert_eq!(resolved, CircuitState::Closed);
        assert_eq!(resolution, ConflictResolution::UsedLocal);
    }

    #[test]
    fn test_vector_clock() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("node1");
        clock1.increment("node1");

        clock2.increment("node2");

        assert!(!clock1.happened_before(&clock2));
        assert!(!clock2.happened_before(&clock1));
        assert!(clock1.is_concurrent(&clock2));

        clock1.merge(&clock2);
        assert!(clock2.happened_before(&clock1));
    }

    #[test]
    fn test_vector_clock_ordering() {
        let mut clock1 = VectorClock::new();
        let mut clock2 = VectorClock::new();

        clock1.increment("node1");
        clock2.timestamps = clock1.timestamps.clone();
        clock2.increment("node1");

        assert!(clock1.happened_before(&clock2));
        assert!(!clock2.happened_before(&clock1));
    }
}
