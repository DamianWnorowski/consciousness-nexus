//! HyperZeroCopy Daemon - Shared Memory Writer
//!
//! Runs as background process, receives events via:
//! 1. Named pipe (from hooks)
//! 2. File watching (for JSON event files)
//! 3. Direct stdin (for testing)
//!
//! Writes directly to shared memory - ZERO COPY to readers.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use memmap2::MmapMut;
use notify::{Watcher, RecursiveMode, Event as NotifyEvent};
use serde::Deserialize;
use tokio::sync::mpsc;

mod ring;
use ring::*;

/// HyperZeroCopy Observer Daemon
#[derive(Parser, Debug)]
#[command(name = "hzc-daemon")]
#[command(about = "Zero-copy shared memory observer daemon")]
struct Args {
    /// Shared memory file path
    #[arg(short, long, default_value = "C:\\Users\\Ouroboros\\.claude\\hzc_ring.mem")]
    memory_file: PathBuf,

    /// Ring buffer size in bytes
    #[arg(short, long, default_value = "1048576")]
    size: usize,

    /// Watch directory for JSON event files
    #[arg(short, long, default_value = "C:\\Users\\Ouroboros\\.claude\\events")]
    watch_dir: PathBuf,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

/// Event from hook (JSON format)
#[derive(Debug, Deserialize)]
struct HookEvent {
    #[serde(rename = "type")]
    event_type: String,
    tool: Option<String>,
    task_id: Option<String>,
    message: Option<String>,
    tokens: Option<u64>,
    duration_ms: Option<u64>,
    error: Option<String>,
}

impl HookEvent {
    fn to_ring_event(&self) -> RingEvent {
        let event_type = match self.event_type.as_str() {
            "tool_call" => EventType::ToolCall,
            "tool_result" => EventType::ToolResult,
            "task_start" => EventType::TaskStart,
            "task_end" => EventType::TaskEnd,
            "user_message" => EventType::UserMessage,
            "assistant_message" => EventType::AssistantMessage,
            "error" => EventType::Error,
            "state_change" => EventType::StateChange,
            "metric" => EventType::Metric,
            _ => EventType::None,
        };

        let ident = self.tool.as_deref()
            .or(self.task_id.as_deref())
            .unwrap_or("");

        let payload = self.message.as_deref()
            .or(self.error.as_deref())
            .unwrap_or("");

        let mut event = RingEvent::new(event_type, ident, payload);

        if let Some(dur) = self.duration_ms {
            event.duration_us = (dur * 1000).min(u16::MAX as u64) as u16;
        }

        if let Some(tokens) = self.tokens {
            event.value = tokens.min(u32::MAX as u64) as u32;
        }

        event
    }
}

/// Process all JSON files in a directory
fn process_existing_files(dir: &PathBuf, tx: &mpsc::UnboundedSender<RingEvent>, verbose: bool) -> usize {
    let mut count = 0;
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "json").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    if let Ok(hook_event) = serde_json::from_str::<HookEvent>(&content) {
                        let ring_event = hook_event.to_ring_event();
                        let _ = tx.send(ring_event);
                        count += 1;
                        if verbose {
                            println!("  ← {:?} (existing)", hook_event.event_type);
                        }
                    }
                    // Delete processed file
                    let _ = fs::remove_file(&path);
                }
            }
        }
    }
    count
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║     HYPERZEROCOPY OBSERVER DAEMON - ZERO OVERHEAD         ║");
    println!("╠═══════════════════════════════════════════════════════════╣");
    println!("║  Memory: {:44} ║", args.memory_file.display());
    println!("║  Size:   {:44} ║", format!("{} bytes ({} events)", args.size, args.size / 64));
    println!("║  Watch:  {:44} ║", args.watch_dir.display());
    println!("╚═══════════════════════════════════════════════════════════╝");

    // Create/open shared memory file
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&args.memory_file)?;

    file.set_len(args.size as u64)?;

    // Memory map it
    let mut mmap = unsafe { MmapMut::map_mut(&file)? };

    // Initialize ring buffer
    let ring = unsafe { RingBuffer::from_raw(mmap.as_mut_ptr(), args.size) };
    ring.init();

    println!("\n✓ Shared memory initialized");
    println!("  Magic: 0x{:08X}", ring.header().magic);
    println!("  Slots: {}", ring.header().slot_count);

    // Create events directory if needed
    fs::create_dir_all(&args.watch_dir)?;

    // Channel for events
    let (tx, mut rx) = mpsc::unbounded_channel::<RingEvent>();

    // Process existing files first
    let existing = process_existing_files(&args.watch_dir, &tx, args.verbose);
    if existing > 0 {
        println!("✓ Processed {} existing events", existing);
    }

    // Start polling task (more reliable than notify on Windows)
    let tx_poll = tx.clone();
    let poll_dir = args.watch_dir.clone();
    let poll_verbose = args.verbose;
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
        loop {
            interval.tick().await;
            process_existing_files(&poll_dir, &tx_poll, poll_verbose);
        }
    });

    println!("✓ Polling for events (100ms)");

    // Stdin reader for direct input (testing)
    let tx_stdin = tx.clone();
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        let reader = BufReader::new(stdin.lock());
        for line in reader.lines().flatten() {
            if let Ok(hook_event) = serde_json::from_str::<HookEvent>(&line) {
                let ring_event = hook_event.to_ring_event();
                let _ = tx_stdin.send(ring_event);
            }
        }
    });

    println!("\n▶ Daemon running. Events → Shared Memory (zero-copy)");
    println!("  Press Ctrl+C to stop\n");

    // Write startup event
    ring.write(RingEvent::new(
        EventType::StateChange,
        "daemon",
        "HZC daemon started"
    ));

    // Main event loop
    let mut event_count = 0u64;
    while let Some(event) = rx.recv().await {
        let seq = ring.write(event);
        event_count += 1;

        if args.verbose {
            println!("  [{}] seq={} type={}",
                event_count,
                seq,
                event.ident_str()
            );
        }

        // Update metadata based on event
        let metadata = ring.metadata_mut();
        metadata.total_tool_calls += 1;

        // Flush periodically
        if event_count % 100 == 0 {
            mmap.flush_async()?;
        }
    }

    Ok(())
}
