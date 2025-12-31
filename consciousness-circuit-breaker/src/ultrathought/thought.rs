//! Thought representation and chains
//!
//! Core data structures for the UltraThought engine.
//! All thoughts are REAL computational units - no simulation.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// Unique thought identifier generator
static THOUGHT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique thought ID
fn next_thought_id() -> u64 {
    THOUGHT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// A single thought unit in the computation chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Thought {
    /// Unique identifier
    pub id: u64,
    /// Depth in the thought chain
    pub depth: u32,
    /// Parent thought ID (None for root thoughts)
    pub parent_id: Option<u64>,
    /// Child thought IDs
    pub children: Vec<u64>,
    /// Current state
    pub state: ThoughtState,
    /// Thought content/payload
    pub content: ThoughtContent,
    /// Metrics for this thought
    pub metrics: ThoughtMetrics,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// State of a thought
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThoughtState {
    /// Thought is pending execution
    Pending,
    /// Thought is currently being processed
    Processing,
    /// Thought completed successfully
    Completed,
    /// Thought failed
    Failed,
    /// Thought was cancelled
    Cancelled,
    /// Thought is waiting for child thoughts
    WaitingForChildren,
}

/// Content of a thought
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtContent {
    /// The input to this thought
    pub input: String,
    /// The output/result of this thought
    pub output: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Metrics for a thought
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ThoughtMetrics {
    /// Time taken to process (microseconds)
    pub processing_time_us: u64,
    /// Memory used (bytes, if measurable)
    pub memory_bytes: Option<u64>,
    /// Number of sub-computations
    pub sub_computations: u64,
    /// Recursion depth reached
    pub recursion_depth: u32,
}

impl Thought {
    /// Create a new root thought
    pub fn new_root(input: impl Into<String>) -> Self {
        Self {
            id: next_thought_id(),
            depth: 0,
            parent_id: None,
            children: Vec::new(),
            state: ThoughtState::Pending,
            content: ThoughtContent {
                input: input.into(),
                output: None,
                error: None,
                confidence: 0.0,
                tags: Vec::new(),
            },
            metrics: ThoughtMetrics::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create a child thought
    pub fn spawn_child(&mut self, input: impl Into<String>) -> Thought {
        let child = Thought {
            id: next_thought_id(),
            depth: self.depth + 1,
            parent_id: Some(self.id),
            children: Vec::new(),
            state: ThoughtState::Pending,
            content: ThoughtContent {
                input: input.into(),
                output: None,
                error: None,
                confidence: 0.0,
                tags: Vec::new(),
            },
            metrics: ThoughtMetrics::default(),
            metadata: HashMap::new(),
        };
        self.children.push(child.id);
        child
    }

    /// Mark thought as processing
    pub fn start_processing(&mut self) {
        self.state = ThoughtState::Processing;
    }

    /// Complete the thought with output
    pub fn complete(&mut self, output: impl Into<String>, confidence: f64, processing_time: Duration) {
        self.state = ThoughtState::Completed;
        self.content.output = Some(output.into());
        self.content.confidence = confidence.clamp(0.0, 1.0);
        self.metrics.processing_time_us = processing_time.as_micros() as u64;
    }

    /// Mark thought as failed
    pub fn fail(&mut self, error: impl Into<String>) {
        self.state = ThoughtState::Failed;
        self.content.error = Some(error.into());
    }

    /// Check if thought is terminal (no more processing needed)
    pub fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            ThoughtState::Completed | ThoughtState::Failed | ThoughtState::Cancelled
        )
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: impl Into<String>) {
        self.content.tags.push(tag.into());
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// A chain of connected thoughts
#[derive(Debug, Clone)]
pub struct ThoughtChain {
    /// All thoughts in the chain, keyed by ID
    thoughts: HashMap<u64, Thought>,
    /// Root thought IDs
    roots: Vec<u64>,
    /// Chain statistics
    stats: ChainStats,
    /// Creation time
    created_at: Instant,
}

/// Statistics for a thought chain
#[derive(Debug, Clone, Default)]
pub struct ChainStats {
    /// Total thoughts in chain
    pub total_thoughts: u64,
    /// Completed thoughts
    pub completed: u64,
    /// Failed thoughts
    pub failed: u64,
    /// Pending thoughts
    pub pending: u64,
    /// Maximum depth reached
    pub max_depth: u32,
    /// Total processing time (microseconds)
    pub total_processing_time_us: u64,
    /// Average confidence
    pub average_confidence: f64,
}

impl ThoughtChain {
    /// Create a new thought chain
    pub fn new() -> Self {
        Self {
            thoughts: HashMap::new(),
            roots: Vec::new(),
            stats: ChainStats::default(),
            created_at: Instant::now(),
        }
    }

    /// Add a root thought
    pub fn add_root(&mut self, thought: Thought) -> u64 {
        let id = thought.id;
        self.roots.push(id);
        self.thoughts.insert(id, thought);
        self.update_stats();
        id
    }

    /// Add a thought to the chain
    pub fn add_thought(&mut self, thought: Thought) -> u64 {
        let id = thought.id;
        self.thoughts.insert(id, thought);
        self.update_stats();
        id
    }

    /// Get a thought by ID
    pub fn get(&self, id: u64) -> Option<&Thought> {
        self.thoughts.get(&id)
    }

    /// Get a mutable reference to a thought
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Thought> {
        self.thoughts.get_mut(&id)
    }

    /// Get all root thoughts
    pub fn roots(&self) -> Vec<&Thought> {
        self.roots
            .iter()
            .filter_map(|id| self.thoughts.get(id))
            .collect()
    }

    /// Get children of a thought
    pub fn children(&self, parent_id: u64) -> Vec<&Thought> {
        self.thoughts
            .get(&parent_id)
            .map(|parent| {
                parent
                    .children
                    .iter()
                    .filter_map(|id| self.thoughts.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all pending thoughts
    pub fn pending(&self) -> Vec<&Thought> {
        self.thoughts
            .values()
            .filter(|t| t.state == ThoughtState::Pending)
            .collect()
    }

    /// Get all completed thoughts
    pub fn completed(&self) -> Vec<&Thought> {
        self.thoughts
            .values()
            .filter(|t| t.state == ThoughtState::Completed)
            .collect()
    }

    /// Get all leaf thoughts (no children)
    pub fn leaves(&self) -> Vec<&Thought> {
        self.thoughts
            .values()
            .filter(|t| t.children.is_empty())
            .collect()
    }

    /// Get thoughts at a specific depth
    pub fn at_depth(&self, depth: u32) -> Vec<&Thought> {
        self.thoughts
            .values()
            .filter(|t| t.depth == depth)
            .collect()
    }

    /// Update chain statistics
    fn update_stats(&mut self) {
        let mut stats = ChainStats::default();
        let mut total_confidence = 0.0;
        let mut confidence_count = 0u64;

        for thought in self.thoughts.values() {
            stats.total_thoughts += 1;

            match thought.state {
                ThoughtState::Completed => {
                    stats.completed += 1;
                    total_confidence += thought.content.confidence;
                    confidence_count += 1;
                }
                ThoughtState::Failed => stats.failed += 1,
                ThoughtState::Pending => stats.pending += 1,
                _ => {}
            }

            if thought.depth > stats.max_depth {
                stats.max_depth = thought.depth;
            }

            stats.total_processing_time_us += thought.metrics.processing_time_us;
        }

        stats.average_confidence = if confidence_count > 0 {
            total_confidence / confidence_count as f64
        } else {
            0.0
        };

        self.stats = stats;
    }

    /// Get current statistics
    pub fn stats(&self) -> &ChainStats {
        &self.stats
    }

    /// Get chain age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get total thought count
    pub fn len(&self) -> usize {
        self.thoughts.len()
    }

    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.thoughts.is_empty()
    }

    /// Check if all thoughts are complete
    pub fn is_complete(&self) -> bool {
        self.thoughts.values().all(|t| t.is_terminal())
    }

    /// Get completion percentage
    pub fn completion_percentage(&self) -> f64 {
        if self.thoughts.is_empty() {
            return 100.0;
        }

        let terminal = self.thoughts.values().filter(|t| t.is_terminal()).count();
        (terminal as f64 / self.thoughts.len() as f64) * 100.0
    }

    /// Traverse the chain depth-first
    pub fn traverse_depth_first<F>(&self, mut visitor: F)
    where
        F: FnMut(&Thought, u32),
    {
        fn visit<F>(chain: &ThoughtChain, id: u64, depth: u32, visitor: &mut F)
        where
            F: FnMut(&Thought, u32),
        {
            if let Some(thought) = chain.thoughts.get(&id) {
                visitor(thought, depth);
                for child_id in &thought.children {
                    visit(chain, *child_id, depth + 1, visitor);
                }
            }
        }

        for root_id in &self.roots {
            visit(self, *root_id, 0, &mut visitor);
        }
    }

    /// Traverse the chain breadth-first
    pub fn traverse_breadth_first<F>(&self, mut visitor: F)
    where
        F: FnMut(&Thought, u32),
    {
        use std::collections::VecDeque;

        let mut queue: VecDeque<(u64, u32)> = self.roots.iter().map(|&id| (id, 0)).collect();

        while let Some((id, depth)) = queue.pop_front() {
            if let Some(thought) = self.thoughts.get(&id) {
                visitor(thought, depth);
                for child_id in &thought.children {
                    queue.push_back((*child_id, depth + 1));
                }
            }
        }
    }

    /// Serialize chain to JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        let serializable: HashMap<u64, &Thought> = self.thoughts.iter().map(|(k, v)| (*k, v)).collect();
        serde_json::to_string_pretty(&serializable)
    }
}

impl Default for ThoughtChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_creation() {
        let thought = Thought::new_root("test input");
        assert_eq!(thought.depth, 0);
        assert_eq!(thought.state, ThoughtState::Pending);
        assert!(thought.parent_id.is_none());
    }

    #[test]
    fn test_spawn_child() {
        let mut parent = Thought::new_root("parent");
        let child = parent.spawn_child("child");

        assert_eq!(child.depth, 1);
        assert_eq!(child.parent_id, Some(parent.id));
        assert!(parent.children.contains(&child.id));
    }

    #[test]
    fn test_thought_completion() {
        let mut thought = Thought::new_root("test");
        thought.start_processing();
        assert_eq!(thought.state, ThoughtState::Processing);

        thought.complete("output", 0.95, Duration::from_millis(100));
        assert_eq!(thought.state, ThoughtState::Completed);
        assert_eq!(thought.content.confidence, 0.95);
        assert!(thought.is_terminal());
    }

    #[test]
    fn test_thought_chain() {
        let mut chain = ThoughtChain::new();

        let mut root = Thought::new_root("root");
        let child1 = root.spawn_child("child1");
        let child2 = root.spawn_child("child2");

        chain.add_root(root);
        chain.add_thought(child1);
        chain.add_thought(child2);

        assert_eq!(chain.len(), 3);
        assert_eq!(chain.stats().max_depth, 1);
        assert_eq!(chain.leaves().len(), 2);
    }

    #[test]
    fn test_chain_traversal() {
        let mut chain = ThoughtChain::new();

        let mut root = Thought::new_root("root");
        let child = root.spawn_child("child");
        let root_id = root.id;

        chain.add_root(root);
        chain.add_thought(child);

        let mut visited = Vec::new();
        chain.traverse_depth_first(|thought, depth| {
            visited.push((thought.id, depth));
        });

        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0].1, 0); // root at depth 0
        assert_eq!(visited[1].1, 1); // child at depth 1
    }

    #[test]
    fn test_chain_completion() {
        let mut chain = ThoughtChain::new();

        let mut root = Thought::new_root("root");
        root.complete("done", 1.0, Duration::from_millis(10));

        chain.add_root(root);

        assert!(chain.is_complete());
        assert_eq!(chain.completion_percentage(), 100.0);
    }
}
