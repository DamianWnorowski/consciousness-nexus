//! HyperZeroCopy Watcher - Interactive Real-time TUI Dashboard
//!
//! ZERO-COPY: Maps the same shared memory as daemon.
//! FULL CONTROL: Pause, filter, inspect, inject prompts.
//!
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ HYPERZEROCOPY OBSERVER [LIVE]     ctx: 45% ████░░░░  12.3k tok  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ STATE: ⚡ TOOL_CALL              TASKS: 3 active  127 done      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ EVENTS (newest first)                               seq: 1,234  │
//! │ ─────────────────────────────────────────────────────────────── │
//! │ > 12:34:56.789 │ Read     │ config.rs          │      2ms   ◄──│
//! │   12:34:56.123 │ Grep     │ pattern: "fn main" │     15ms      │
//! │   12:34:55.999 │ TaskEnd  │ agent-a4ee80b      │  1,234ms      │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ [p]ause [/]filter [s]tats [i]nject [Enter]details [q]uit        │
//! └─────────────────────────────────────────────────────────────────┘

use std::io::{self, Write as IoWrite};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use std::fs;

use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use memmap2::Mmap;
use ratatui::{
    prelude::*,
    widgets::*,
};

mod ring;
use ring::*;

/// HyperZeroCopy Watcher TUI - Interactive Edition
#[derive(Parser, Debug)]
#[command(name = "hzc-watch")]
#[command(about = "Zero-copy real-time TUI observer with full interactivity")]
struct Args {
    /// Shared memory file path
    #[arg(short, long, default_value = "C:\\Users\\Ouroboros\\.claude\\hzc_ring.mem")]
    memory_file: PathBuf,

    /// Refresh rate in milliseconds
    #[arg(short, long, default_value = "50")]
    refresh_ms: u64,

    /// Number of events to display
    #[arg(short, long, default_value = "20")]
    event_count: usize,

    /// Events directory for prompt injection
    #[arg(long, default_value = "C:\\Users\\Ouroboros\\.claude\\events")]
    events_dir: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Mode {
    Normal,
    Paused,
    Filter,
    Stats,
    Details,
    Inject,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum ToolFilter {
    All,
    Read,
    Write,
    Edit,
    Bash,
    Grep,
    Glob,
    Task,
    Todo,
    Error,
}

impl ToolFilter {
    fn matches(&self, event_type: u8, ident: &str) -> bool {
        match self {
            ToolFilter::All => true,
            ToolFilter::Read => ident.starts_with("Read"),
            ToolFilter::Write => ident.starts_with("Write"),
            ToolFilter::Edit => ident.starts_with("Edit"),
            ToolFilter::Bash => ident.starts_with("Bash"),
            ToolFilter::Grep => ident.starts_with("Grep"),
            ToolFilter::Glob => ident.starts_with("Glob"),
            ToolFilter::Task => ident.starts_with("Task") || event_type == EventType::TaskStart as u8 || event_type == EventType::TaskEnd as u8,
            ToolFilter::Todo => ident.starts_with("Todo"),
            ToolFilter::Error => event_type == EventType::Error as u8,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            ToolFilter::All => "ALL",
            ToolFilter::Read => "Read",
            ToolFilter::Write => "Write",
            ToolFilter::Edit => "Edit",
            ToolFilter::Bash => "Bash",
            ToolFilter::Grep => "Grep",
            ToolFilter::Glob => "Glob",
            ToolFilter::Task => "Task",
            ToolFilter::Todo => "Todo",
            ToolFilter::Error => "Error",
        }
    }

    fn next(&self) -> Self {
        match self {
            ToolFilter::All => ToolFilter::Read,
            ToolFilter::Read => ToolFilter::Write,
            ToolFilter::Write => ToolFilter::Edit,
            ToolFilter::Edit => ToolFilter::Bash,
            ToolFilter::Bash => ToolFilter::Grep,
            ToolFilter::Grep => ToolFilter::Glob,
            ToolFilter::Glob => ToolFilter::Task,
            ToolFilter::Task => ToolFilter::Todo,
            ToolFilter::Todo => ToolFilter::Error,
            ToolFilter::Error => ToolFilter::All,
        }
    }
}

struct App {
    ring: RingBuffer,
    last_stats: RingStats,
    events_per_sec: f64,
    last_update: Instant,
    last_total: u64,
    mode: Mode,
    filter: ToolFilter,
    selected_index: usize,
    paused_events: Vec<CapturedEvent>,
    inject_buffer: String,
    events_dir: PathBuf,
    throughput_history: Vec<f64>,
    show_help: bool,
}

#[derive(Clone)]
struct CapturedEvent {
    seq: u64,
    timestamp: u64,
    event_type: u8,
    duration_us: u16,
    ident: String,
    payload: String,
}

impl App {
    fn new(ring: RingBuffer, events_dir: PathBuf) -> Self {
        let stats = ring.stats();
        Self {
            last_total: stats.total_writes,
            last_stats: stats,
            ring,
            events_per_sec: 0.0,
            last_update: Instant::now(),
            mode: Mode::Normal,
            filter: ToolFilter::All,
            selected_index: 0,
            paused_events: Vec::new(),
            inject_buffer: String::new(),
            events_dir,
            throughput_history: vec![0.0; 60],
            show_help: false,
        }
    }

    fn update(&mut self) {
        if self.mode == Mode::Paused {
            return;
        }

        let stats = self.ring.stats();
        let elapsed = self.last_update.elapsed().as_secs_f64();

        if elapsed > 0.0 {
            let new_events = stats.total_writes.saturating_sub(self.last_total);
            self.events_per_sec = self.events_per_sec * 0.9 + (new_events as f64 / elapsed) * 0.1;

            // Update throughput history
            self.throughput_history.remove(0);
            self.throughput_history.push(self.events_per_sec);
        }

        self.last_total = stats.total_writes;
        self.last_stats = stats;
        self.last_update = Instant::now();
    }

    fn toggle_pause(&mut self) {
        match self.mode {
            Mode::Paused => {
                self.mode = Mode::Normal;
                self.paused_events.clear();
            }
            Mode::Normal => {
                self.mode = Mode::Paused;
                // Capture current events
                self.paused_events = self.get_filtered_events(50);
            }
            _ => {}
        }
    }

    fn get_filtered_events(&self, count: usize) -> Vec<CapturedEvent> {
        self.ring.peek_recent(count * 2)
            .iter()
            .filter(|e| self.filter.matches(e.event_type, e.ident_str()))
            .take(count)
            .map(|e| CapturedEvent {
                seq: e.seq,
                timestamp: e.timestamp,
                event_type: e.event_type,
                duration_us: e.duration_us,
                ident: e.ident_str().to_string(),
                payload: e.payload_str().to_string(),
            })
            .collect()
    }

    fn inject_prompt(&mut self, prompt: &str) {
        let event = serde_json::json!({
            "type": "user_message",
            "message": prompt,
            "timestamp": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64
        });

        let filename = format!("inject_{}.json", uuid::Uuid::new_v4().to_string().split('-').next().unwrap_or("0000"));
        let path = self.events_dir.join(filename);

        if let Ok(_) = fs::write(&path, event.to_string()) {
            self.inject_buffer.clear();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Open shared memory (read-only)
    let file = std::fs::File::open(&args.memory_file)?;
    let mmap = unsafe { Mmap::map(&file)? };

    // Create ring buffer view
    let ring = unsafe { RingBuffer::from_raw(mmap.as_ptr() as *mut u8, mmap.len()) };

    if !ring.validate() {
        eprintln!("ERROR: Invalid shared memory. Is the daemon running?");
        eprintln!("       Expected magic: 0x{:08X}", RING_MAGIC);
        return Ok(());
    }

    // Setup terminal
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;

    let mut app = App::new(ring, args.events_dir);
    let tick_rate = Duration::from_millis(args.refresh_ms);

    loop {
        app.update();

        terminal.draw(|f| ui(f, &app, args.event_count))?;

        if event::poll(tick_rate)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match app.mode {
                        Mode::Inject => {
                            match key.code {
                                KeyCode::Esc => {
                                    app.mode = Mode::Normal;
                                    app.inject_buffer.clear();
                                }
                                KeyCode::Enter => {
                                    if !app.inject_buffer.is_empty() {
                                        let prompt = app.inject_buffer.clone();
                                        app.inject_prompt(&prompt);
                                        app.mode = Mode::Normal;
                                    }
                                }
                                KeyCode::Backspace => {
                                    app.inject_buffer.pop();
                                }
                                KeyCode::Char(c) => {
                                    app.inject_buffer.push(c);
                                }
                                _ => {}
                            }
                        }
                        _ => {
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Esc => break,
                                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                                KeyCode::Char('p') => app.toggle_pause(),
                                KeyCode::Char('/') | KeyCode::Char('f') => {
                                    app.filter = app.filter.next();
                                }
                                KeyCode::Char('s') => {
                                    app.mode = if app.mode == Mode::Stats { Mode::Normal } else { Mode::Stats };
                                }
                                KeyCode::Char('i') => {
                                    app.mode = Mode::Inject;
                                    app.inject_buffer.clear();
                                }
                                KeyCode::Char('?') | KeyCode::Char('h') => {
                                    app.show_help = !app.show_help;
                                }
                                KeyCode::Up => {
                                    if app.selected_index > 0 {
                                        app.selected_index -= 1;
                                    }
                                }
                                KeyCode::Down => {
                                    app.selected_index += 1;
                                }
                                KeyCode::Enter => {
                                    app.mode = if app.mode == Mode::Details { Mode::Normal } else { Mode::Details };
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
    }

    // Cleanup
    disable_raw_mode()?;
    io::stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn ui(frame: &mut Frame, app: &App, event_count: usize) {
    let stats = &app.last_stats;

    // Main layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Header
            Constraint::Length(3),  // State bar
            Constraint::Min(10),    // Events or Stats
            Constraint::Length(3),  // Footer/Input
        ])
        .split(frame.size());

    // Mode indicator
    let mode_str = match app.mode {
        Mode::Normal => "[LIVE]",
        Mode::Paused => "[PAUSED]",
        Mode::Filter => "[FILTER]",
        Mode::Stats => "[STATS]",
        Mode::Details => "[DETAILS]",
        Mode::Inject => "[INJECT]",
    };

    let mode_color = match app.mode {
        Mode::Normal => Color::Green,
        Mode::Paused => Color::Yellow,
        Mode::Stats => Color::Blue,
        Mode::Inject => Color::Magenta,
        _ => Color::Cyan,
    };

    // Header
    let ctx_pct = stats.context_pct.min(100);
    let ctx_bar: String = format!(
        "{}{}",
        "█".repeat((ctx_pct / 5) as usize),
        "░".repeat((20 - ctx_pct / 5) as usize)
    );

    let header = Paragraph::new(format!(
        " HYPERZEROCOPY OBSERVER {}     ctx: {}% {} {:>6} tok",
        mode_str, ctx_pct, ctx_bar, stats.context_tokens
    ))
    .style(Style::default().fg(mode_color).bold())
    .block(Block::default().borders(Borders::ALL).border_style(Style::default().fg(mode_color)));
    frame.render_widget(header, chunks[0]);

    // State bar with filter indicator
    let state = match stats.state_flags {
        f if f & state_flags::TOOL_CALL != 0 => ("⚡ TOOL_CALL", Color::Yellow),
        f if f & state_flags::THINKING != 0 => ("🧠 THINKING", Color::Blue),
        f if f & state_flags::PARALLEL_EXEC != 0 => ("🔀 PARALLEL", Color::Magenta),
        f if f & state_flags::BACKGROUND_OP != 0 => ("⏳ BACKGROUND", Color::Green),
        f if f & state_flags::WAITING_USER != 0 => ("⏸ WAITING", Color::Gray),
        f if f & state_flags::ERROR != 0 => ("❌ ERROR", Color::Red),
        _ => ("● IDLE", Color::DarkGray),
    };

    let filter_str = if app.filter != ToolFilter::All {
        format!(" [Filter: {}]", app.filter.name())
    } else {
        String::new()
    };

    let state_bar = Paragraph::new(format!(
        " STATE: {:20} TASKS: {} active  {} total  TOOLS: {}{}",
        state.0, stats.active_tasks, stats.total_writes, stats.tool_calls, filter_str
    ))
    .style(Style::default().fg(state.1))
    .block(Block::default().borders(Borders::ALL));
    frame.render_widget(state_bar, chunks[1]);

    // Main content area
    match app.mode {
        Mode::Stats => {
            render_stats(frame, app, chunks[2]);
        }
        Mode::Details => {
            render_details(frame, app, chunks[2], event_count);
        }
        _ => {
            render_events(frame, app, chunks[2], event_count);
        }
    }

    // Footer
    match app.mode {
        Mode::Inject => {
            let input = Paragraph::new(format!(" > {}_", app.inject_buffer))
                .style(Style::default().fg(Color::Magenta))
                .block(Block::default()
                    .title(" INJECT PROMPT (Enter=send, Esc=cancel) ")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::Magenta)));
            frame.render_widget(input, chunks[3]);
        }
        _ => {
            let footer = Paragraph::new(format!(
                " THROUGHPUT: {:.0} evt/s   PENDING: {}   SLOTS: {}/{}   [p]ause [/]filter [s]tats [i]nject [?]help [q]uit",
                app.events_per_sec,
                stats.pending,
                stats.total_writes % stats.slot_count as u64,
                stats.slot_count
            ))
            .style(Style::default().fg(Color::DarkGray))
            .block(Block::default().borders(Borders::ALL));
            frame.render_widget(footer, chunks[3]);
        }
    }

    // Help overlay
    if app.show_help {
        render_help(frame);
    }
}

fn render_events(frame: &mut Frame, app: &App, area: Rect, event_count: usize) {
    let events = if app.mode == Mode::Paused {
        app.paused_events.iter().take(event_count).cloned().collect::<Vec<_>>()
    } else {
        app.get_filtered_events(event_count)
    };

    let rows: Vec<Row> = events.iter().enumerate().map(|(i, e)| {
        let ts = chrono::DateTime::from_timestamp_millis(e.timestamp as i64)
            .map(|dt| dt.format("%H:%M:%S%.3f").to_string())
            .unwrap_or_else(|| "??:??:??.???".to_string());

        let event_type = match e.event_type {
            x if x == EventType::ToolCall as u8 => "Tool→",
            x if x == EventType::ToolResult as u8 => "←Result",
            x if x == EventType::TaskStart as u8 => "Task▶",
            x if x == EventType::TaskEnd as u8 => "Task■",
            x if x == EventType::Error as u8 => "ERROR",
            x if x == EventType::StateChange as u8 => "State",
            x if x == EventType::UserMessage as u8 => "User",
            _ => "•",
        };

        let mut style = match e.event_type {
            x if x == EventType::Error as u8 => Style::default().fg(Color::Red),
            x if x == EventType::ToolCall as u8 => Style::default().fg(Color::Yellow),
            x if x == EventType::TaskStart as u8 => Style::default().fg(Color::Green),
            x if x == EventType::UserMessage as u8 => Style::default().fg(Color::Magenta),
            _ => Style::default(),
        };

        // Highlight selected row
        if i == app.selected_index {
            style = style.add_modifier(Modifier::REVERSED);
        }

        let selector = if i == app.selected_index { ">" } else { " " };

        Row::new(vec![
            Cell::from(selector),
            Cell::from(ts),
            Cell::from(event_type),
            Cell::from(e.ident.clone()),
            Cell::from(e.payload.clone()),
            Cell::from(format!("{}μs", e.duration_us)),
        ]).style(style)
    }).collect();

    let table = Table::new(
        rows,
        [
            Constraint::Length(1),
            Constraint::Length(14),
            Constraint::Length(8),
            Constraint::Length(16),
            Constraint::Fill(1),
            Constraint::Length(10),
        ],
    )
    .header(Row::new(vec![" ", "Time", "Type", "Ident", "Payload", "Duration"])
        .style(Style::default().bold().fg(Color::Cyan)))
    .block(Block::default()
        .title(format!(" EVENTS (seq: {}) ", app.last_stats.total_writes))
        .borders(Borders::ALL));
    frame.render_widget(table, area);
}

fn render_stats(frame: &mut Frame, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Throughput sparkline
    let sparkline_data: Vec<u64> = app.throughput_history.iter()
        .map(|&v| v as u64)
        .collect();

    let sparkline = Sparkline::default()
        .block(Block::default().title(" Throughput (60s) ").borders(Borders::ALL))
        .data(&sparkline_data)
        .style(Style::default().fg(Color::Cyan));
    frame.render_widget(sparkline, chunks[0]);

    // Stats summary
    let stats_text = format!(
        r#"
 Total Events:     {:>12}
 Events/sec:       {:>12.1}
 Pending:          {:>12}
 Active Tasks:     {:>12}
 Tool Calls:       {:>12}
 Context Tokens:   {:>12}
 Context Usage:    {:>11}%
 Slot Utilization: {:>11}%
"#,
        app.last_stats.total_writes,
        app.events_per_sec,
        app.last_stats.pending,
        app.last_stats.active_tasks,
        app.last_stats.tool_calls,
        app.last_stats.context_tokens,
        app.last_stats.context_pct,
        (app.last_stats.total_writes % app.last_stats.slot_count as u64) * 100 / app.last_stats.slot_count as u64
    );

    let stats_widget = Paragraph::new(stats_text)
        .block(Block::default().title(" Statistics ").borders(Borders::ALL))
        .style(Style::default().fg(Color::White));
    frame.render_widget(stats_widget, chunks[1]);
}

fn render_details(frame: &mut Frame, app: &App, area: Rect, event_count: usize) {
    let events = app.get_filtered_events(event_count);

    if let Some(event) = events.get(app.selected_index) {
        let detail_text = format!(
            r#"
 Sequence:    {}
 Timestamp:   {} ({})
 Type:        {} ({})
 Duration:    {} μs

 Identifier:
   {}

 Payload:
   {}

 Raw bytes (ident):
   {:?}

 Raw bytes (payload):
   {:?}
"#,
            event.seq,
            event.timestamp,
            chrono::DateTime::from_timestamp_millis(event.timestamp as i64)
                .map(|dt| dt.format("%Y-%m-%d %H:%M:%S%.3f").to_string())
                .unwrap_or_default(),
            event.event_type,
            match event.event_type {
                x if x == EventType::ToolCall as u8 => "ToolCall",
                x if x == EventType::ToolResult as u8 => "ToolResult",
                x if x == EventType::TaskStart as u8 => "TaskStart",
                x if x == EventType::TaskEnd as u8 => "TaskEnd",
                x if x == EventType::Error as u8 => "Error",
                x if x == EventType::StateChange as u8 => "StateChange",
                x if x == EventType::UserMessage as u8 => "UserMessage",
                _ => "Unknown",
            },
            event.duration_us,
            event.ident,
            event.payload,
            event.ident.as_bytes(),
            event.payload.as_bytes(),
        );

        let details = Paragraph::new(detail_text)
            .block(Block::default()
                .title(format!(" EVENT DETAILS [{}] ", event.seq))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Yellow)))
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        frame.render_widget(details, area);
    } else {
        let no_event = Paragraph::new(" No event selected. Use ↑/↓ to select, Enter to view details.")
            .block(Block::default().title(" EVENT DETAILS ").borders(Borders::ALL));
        frame.render_widget(no_event, area);
    }
}

fn render_help(frame: &mut Frame) {
    let area = centered_rect(60, 70, frame.size());

    let help_text = r#"
    HYPERZEROCOPY OBSERVER - KEYBOARD SHORTCUTS
    ═══════════════════════════════════════════

    NAVIGATION
    ──────────
    ↑/↓         Select event
    Enter       Toggle event details view

    MODES
    ─────
    p           Pause/Resume live updates
    /  or  f    Cycle through tool filters
    s           Toggle statistics view
    i           Enter prompt injection mode

    GENERAL
    ───────
    ?  or  h    Toggle this help
    q  or  Esc  Quit
    Ctrl+C      Quit

    INJECT MODE
    ───────────
    Type        Enter prompt text
    Enter       Send prompt as event
    Esc         Cancel and return to normal

    ─────────────────────────────────────────────
    Press any key to close this help
"#;

    let help = Paragraph::new(help_text)
        .block(Block::default()
            .title(" HELP ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::Cyan)))
        .style(Style::default().fg(Color::White));

    frame.render_widget(Clear, area);
    frame.render_widget(help, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
