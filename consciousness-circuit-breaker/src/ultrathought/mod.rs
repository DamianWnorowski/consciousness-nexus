//! UltraThought recursive execution engine
//!
//! Implements deep recursive thinking with:
//! - Trampoline pattern to avoid stack overflow
//! - Real thought chains (no simulation)
//! - Parallel thought exploration
//! - Emergent pattern synthesis

pub mod engine;
pub mod trampoline;
pub mod thought;

pub use engine::{UltraThoughtEngine, ThoughtResult, ThoughtConfig};
pub use trampoline::Trampoline;
pub use thought::{Thought, ThoughtState, ThoughtChain};
