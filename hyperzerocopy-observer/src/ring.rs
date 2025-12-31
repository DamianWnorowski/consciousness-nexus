//! Zero-Copy Ring Buffer - Lock-Free Shared Memory
//!
//! REVOLUTIONARY: Direct memory-mapped ring buffer with atomic cursors.
//! Writers and readers operate on the SAME physical memory - ZERO copies.
//!
//! Architecture:
//! ```
//!  ┌─────────────────────────────────────────────────────────────┐
//!  │                    SHARED MEMORY REGION                      │
//!  ├──────────┬──────────┬──────────────────────────────────────┤
//!  │  Header  │ Metadata │           Ring Buffer Data            │
//!  │ (64 bytes)│(64 bytes)│         (configurable size)          │
//!  └──────────┴──────────┴──────────────────────────────────────┘
//!       │           │                      │
//!       │           │                      └─► Event slots (fixed size)
//!       │           └─► Stats, counters, timestamps
//!       └─► Magic, version, write_cursor (atomic), read_cursor (atomic)
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use bytemuck::{Pod, Zeroable};

/// Magic number to validate shared memory
pub const RING_MAGIC: u32 = 0x48_5A_43_4F; // "HZCO" - HyperZeroCopy Observer

/// Version for compatibility checking
pub const RING_VERSION: u32 = 1;

/// Default ring buffer size (1MB = ~16K events)
pub const DEFAULT_RING_SIZE: usize = 1024 * 1024;

/// Single event slot size (64 bytes - cache line aligned)
pub const EVENT_SLOT_SIZE: usize = 64;

/// Header at start of shared memory (64 bytes, cache-line aligned)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct RingHeader {
    /// Magic number for validation
    pub magic: u32,
    /// Version number
    pub version: u32,
    /// Total ring buffer size in bytes
    pub ring_size: u32,
    /// Number of event slots
    pub slot_count: u32,
    /// Write cursor (atomic, wraps around)
    pub write_cursor: u64,
    /// Read cursor (atomic, for consumers)
    pub read_cursor: u64,
    /// Total events ever written (monotonic)
    pub total_writes: u64,
    /// Total events ever read (monotonic)
    pub total_reads: u64,
    /// Timestamp of last write (unix millis)
    pub last_write_ts: u64,
    /// Reserved for future use
    _reserved: [u8; 8],
}

/// Metadata section (64 bytes)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct RingMetadata {
    /// Session start timestamp
    pub session_start: u64,
    /// Current context usage (tokens)
    pub context_tokens: u64,
    /// Context window max
    pub context_max: u64,
    /// Active background tasks
    pub active_tasks: u32,
    /// Completed tasks this session
    pub completed_tasks: u32,
    /// Total tool calls
    pub total_tool_calls: u64,
    /// Errors encountered
    pub error_count: u32,
    /// Current state flags
    pub state_flags: u32,
    /// Reserved
    _reserved: [u8; 16],
}

/// State flags for current operation
pub mod state_flags {
    pub const IDLE: u32 = 0;
    pub const THINKING: u32 = 1 << 0;
    pub const TOOL_CALL: u32 = 1 << 1;
    pub const WAITING_USER: u32 = 1 << 2;
    pub const BACKGROUND_OP: u32 = 1 << 3;
    pub const PARALLEL_EXEC: u32 = 1 << 4;
    pub const ERROR: u32 = 1 << 5;
}

/// Event types for the ring buffer
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventType {
    None = 0,
    ToolCall = 1,
    ToolResult = 2,
    TaskStart = 3,
    TaskEnd = 4,
    UserMessage = 5,
    AssistantMessage = 6,
    Error = 7,
    StateChange = 8,
    Metric = 9,
}

/// Single event in the ring buffer (64 bytes - cache line)
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
pub struct RingEvent {
    /// Sequence number (for ordering)
    pub seq: u64,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Event type
    pub event_type: u8,
    /// Event flags
    pub flags: u8,
    /// Duration in microseconds (for completed events)
    pub duration_us: u16,
    /// Associated numeric value
    pub value: u32,
    /// Short identifier (tool name, task id, etc) - 16 bytes
    pub ident: [u8; 16],
    /// Payload data - 24 bytes
    pub payload: [u8; 24],
}

impl RingEvent {
    /// Create a new event
    pub fn new(event_type: EventType, ident: &str, payload: &str) -> Self {
        let mut event = Self::zeroed();
        event.event_type = event_type as u8;
        event.timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        // Copy identifier (truncate if needed)
        let ident_bytes = ident.as_bytes();
        let ident_len = ident_bytes.len().min(16);
        event.ident[..ident_len].copy_from_slice(&ident_bytes[..ident_len]);

        // Copy payload (truncate if needed)
        let payload_bytes = payload.as_bytes();
        let payload_len = payload_bytes.len().min(24);
        event.payload[..payload_len].copy_from_slice(&payload_bytes[..payload_len]);

        event
    }

    /// Get identifier as string
    pub fn ident_str(&self) -> &str {
        let len = self.ident.iter().position(|&b| b == 0).unwrap_or(16);
        std::str::from_utf8(&self.ident[..len]).unwrap_or("")
    }

    /// Get payload as string
    pub fn payload_str(&self) -> &str {
        let len = self.payload.iter().position(|&b| b == 0).unwrap_or(24);
        std::str::from_utf8(&self.payload[..len]).unwrap_or("")
    }

    /// Check if event is empty/unused
    pub fn is_empty(&self) -> bool {
        self.event_type == EventType::None as u8
    }
}

/// Zero-copy ring buffer operations
pub struct RingBuffer {
    /// Base pointer to shared memory
    base: *mut u8,
    /// Total size of shared memory
    size: usize,
}

// Safety: We use atomics for all concurrent access
unsafe impl Send for RingBuffer {}
unsafe impl Sync for RingBuffer {}

impl RingBuffer {
    /// Create a new ring buffer from raw memory
    ///
    /// # Safety
    /// Caller must ensure memory is valid and properly sized
    pub unsafe fn from_raw(ptr: *mut u8, size: usize) -> Self {
        Self { base: ptr, size }
    }

    /// Initialize the ring buffer (called by daemon/writer)
    pub fn init(&self) {
        let header = self.header_mut();
        header.magic = RING_MAGIC;
        header.version = RING_VERSION;
        header.ring_size = (self.size - 128) as u32; // Subtract header + metadata
        header.slot_count = header.ring_size / EVENT_SLOT_SIZE as u32;
        header.write_cursor = 0;
        header.read_cursor = 0;
        header.total_writes = 0;
        header.total_reads = 0;
        header.last_write_ts = 0;

        let metadata = self.metadata_mut();
        metadata.session_start = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        metadata.context_max = 200_000; // Default context window
    }

    /// Validate the ring buffer is properly initialized
    pub fn validate(&self) -> bool {
        let header = self.header();
        header.magic == RING_MAGIC && header.version == RING_VERSION
    }

    /// Get header (read-only)
    pub fn header(&self) -> &RingHeader {
        unsafe { &*(self.base as *const RingHeader) }
    }

    /// Get header (mutable)
    fn header_mut(&self) -> &mut RingHeader {
        unsafe { &mut *(self.base as *mut RingHeader) }
    }

    /// Get metadata (read-only)
    pub fn metadata(&self) -> &RingMetadata {
        unsafe { &*((self.base as usize + 64) as *const RingMetadata) }
    }

    /// Get metadata (mutable)
    pub fn metadata_mut(&self) -> &mut RingMetadata {
        unsafe { &mut *((self.base as usize + 64) as *mut RingMetadata) }
    }

    /// Get atomic write cursor
    fn write_cursor_atomic(&self) -> &AtomicU64 {
        unsafe {
            let ptr = &self.header().write_cursor as *const u64 as *const AtomicU64;
            &*ptr
        }
    }

    /// Get atomic read cursor
    fn read_cursor_atomic(&self) -> &AtomicU64 {
        unsafe {
            let ptr = &self.header().read_cursor as *const u64 as *const AtomicU64;
            &*ptr
        }
    }

    /// Get event slot by index (zero-copy access)
    fn event_slot(&self, index: u32) -> &RingEvent {
        let offset = 128 + (index as usize * EVENT_SLOT_SIZE);
        unsafe { &*(self.base.add(offset) as *const RingEvent) }
    }

    /// Get mutable event slot by index
    fn event_slot_mut(&self, index: u32) -> &mut RingEvent {
        let offset = 128 + (index as usize * EVENT_SLOT_SIZE);
        unsafe { &mut *(self.base.add(offset) as *mut RingEvent) }
    }

    /// Write an event (lock-free)
    pub fn write(&self, mut event: RingEvent) -> u64 {
        let header = self.header();
        let slot_count = header.slot_count as u64;

        // Atomically claim a slot
        let seq = self.write_cursor_atomic().fetch_add(1, Ordering::AcqRel);
        let slot_idx = (seq % slot_count) as u32;

        // Fill in sequence number
        event.seq = seq;

        // Write to slot (this is a single cache-line write)
        *self.event_slot_mut(slot_idx) = event;

        // Update stats
        let header_mut = self.header_mut();
        header_mut.total_writes = header_mut.total_writes.wrapping_add(1);
        header_mut.last_write_ts = event.timestamp;

        seq
    }

    /// Read events since last read cursor (zero-copy iteration)
    pub fn read_new(&self) -> impl Iterator<Item = &RingEvent> {
        let header = self.header();
        let slot_count = header.slot_count as u64;
        let write_pos = self.write_cursor_atomic().load(Ordering::Acquire);
        let read_pos = self.read_cursor_atomic().load(Ordering::Acquire);

        // Calculate how many new events
        let available = write_pos.saturating_sub(read_pos);
        let to_read = available.min(slot_count); // Don't read more than buffer size

        let base_read = if available > slot_count {
            // Buffer wrapped, start from oldest available
            write_pos - slot_count
        } else {
            read_pos
        };

        // Update read cursor
        self.read_cursor_atomic().store(write_pos, Ordering::Release);
        self.header_mut().total_reads = self.header().total_reads.wrapping_add(to_read);

        (0..to_read).map(move |i| {
            let idx = ((base_read + i) % slot_count) as u32;
            self.event_slot(idx)
        })
    }

    /// Peek at recent events without advancing cursor
    pub fn peek_recent(&self, count: usize) -> Vec<&RingEvent> {
        let header = self.header();
        let slot_count = header.slot_count as u64;
        let write_pos = self.write_cursor_atomic().load(Ordering::Acquire);

        let to_read = (count as u64).min(write_pos).min(slot_count);
        let start = write_pos.saturating_sub(to_read);

        (0..to_read)
            .map(|i| {
                let idx = ((start + i) % slot_count) as u32;
                self.event_slot(idx)
            })
            .collect()
    }

    /// Get current stats
    pub fn stats(&self) -> RingStats {
        let header = self.header();
        let metadata = self.metadata();

        RingStats {
            total_writes: header.total_writes,
            total_reads: header.total_reads,
            pending: header.write_cursor.saturating_sub(header.read_cursor),
            slot_count: header.slot_count,
            context_tokens: metadata.context_tokens,
            context_pct: (metadata.context_tokens as f64 / metadata.context_max as f64 * 100.0) as u32,
            active_tasks: metadata.active_tasks,
            tool_calls: metadata.total_tool_calls,
            state_flags: metadata.state_flags,
        }
    }
}

/// Ring buffer statistics
#[derive(Debug, Clone)]
pub struct RingStats {
    pub total_writes: u64,
    pub total_reads: u64,
    pub pending: u64,
    pub slot_count: u32,
    pub context_tokens: u64,
    pub context_pct: u32,
    pub active_tasks: u32,
    pub tool_calls: u64,
    pub state_flags: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_size() {
        assert_eq!(std::mem::size_of::<RingEvent>(), EVENT_SLOT_SIZE);
        assert_eq!(std::mem::size_of::<RingHeader>(), 64);
        assert_eq!(std::mem::size_of::<RingMetadata>(), 64);
    }

    #[test]
    fn test_event_creation() {
        let event = RingEvent::new(EventType::ToolCall, "Read", "/path/to/file");
        assert_eq!(event.event_type, EventType::ToolCall as u8);
        assert_eq!(event.ident_str(), "Read");
        assert!(event.payload_str().starts_with("/path"));
    }
}
