//! Distributed coordination module
//!
//! Enables cluster-wide circuit breaker state synchronization.
//! Supports etcd, Consul, and in-memory backends.

pub mod coordinator;
pub mod state_sync;

pub use coordinator::{DistributedCoordinator, ClusterState, NodeInfo};
pub use state_sync::{StateSync, SyncEvent};
