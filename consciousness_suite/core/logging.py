"""
Consciousness Computing Logging System
======================================

Advanced logging system with structured JSON output, correlation IDs,
performance monitoring, and consciousness-aware log analysis.
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class ConsciousnessLogger:
    """
    Advanced logging system for consciousness computing operations.

    Features:
    - Structured JSON logging with correlation IDs
    - Performance monitoring and metrics
    - Consciousness-aware log analysis
    - Multi-level logging with context
    - Async logging with buffering
    - Log analysis and pattern recognition
    """

    def __init__(self, name: str, log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create loggers for different levels
        self._setup_loggers()

        # Context and correlation tracking
        self.context_stack = []
        self.correlation_ids = set()

        # Performance tracking
        self.operation_start_times = {}
        self.metrics_buffer = []

        # Async logging
        self.log_queue = asyncio.Queue()
        self.log_worker_task = None

        # Pattern analysis
        self.log_patterns = defaultdict(int)
        self.error_patterns = defaultdict(int)

    def _setup_loggers(self):
        """Setup logging handlers and formatters"""

        # Main logger
        self.logger = logging.getLogger(f"consciousness.{self.name}")
        self.logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        self.logger.handlers.clear()

        # JSON formatter for structured logging
        json_formatter = JSONFormatter()

        # File handler
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)

        # Console handler for development
        if self._is_development():
            console_handler = logging.StreamHandler()
            console_formatter = ConsciousnessFormatter()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)
            self.logger.addHandler(console_handler)

    def _is_development(self) -> bool:
        """Check if running in development environment"""
        import os
        return os.getenv('CONSCIOUSNESS_ENV', 'development') == 'development'

    async def start_async_logging(self):
        """Start async logging worker"""
        if self.log_worker_task is None:
            self.log_worker_task = asyncio.create_task(self._async_log_worker())

    async def stop_async_logging(self):
        """Stop async logging worker"""
        if self.log_worker_task:
            self.log_queue.put_nowait(None)  # Sentinel value
            await self.log_worker_task
            self.log_worker_task = None

    async def _async_log_worker(self):
        """Async logging worker task"""
        while True:
            try:
                log_entry = await self.log_queue.get()
                if log_entry is None:  # Sentinel value
                    break

                # Process log entry
                await self._process_log_entry(log_entry)

            except Exception as e:
                print(f"Async logging error: {e}")

    async def _process_log_entry(self, log_entry: Dict[str, Any]):
        """Process a log entry asynchronously"""
        # Analyze patterns
        await self._analyze_log_patterns(log_entry)

        # Store metrics
        if 'metrics' in log_entry:
            self.metrics_buffer.append(log_entry['metrics'])

        # Write to additional outputs if needed
        await self._write_additional_outputs(log_entry)

    def start_operation(self, operation_name: str, correlation_id: Optional[str] = None) -> str:
        """Start tracking an operation"""
        if correlation_id is None:
            correlation_id = str(uuid.uuid4())

        self.correlation_ids.add(correlation_id)
        self.operation_start_times[correlation_id] = time.time()

        # Push context
        self.context_stack.append({
            'operation': operation_name,
            'correlation_id': correlation_id,
            'start_time': datetime.now()
        })

        self.debug("Operation started", {
            'operation': operation_name,
            'correlation_id': correlation_id
        })

        return correlation_id

    def end_operation(self, correlation_id: str, success: bool = True, result: Any = None):
        """End tracking an operation"""
        if correlation_id not in self.operation_start_times:
            self.warning("Operation end without start", {
                'correlation_id': correlation_id
            })
            return

        start_time = self.operation_start_times.pop(correlation_id)
        duration = time.time() - start_time

        # Pop context
        if self.context_stack:
            self.context_stack.pop()

        self.info("Operation completed", {
            'correlation_id': correlation_id,
            'duration': duration,
            'success': success,
            'result_type': type(result).__name__ if result is not None else None
        })

        # Clean up correlation ID
        self.correlation_ids.discard(correlation_id)

        return duration

    def trace(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log trace level message"""
        self._log('TRACE', message, extra)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug level message"""
        self._log('DEBUG', message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info level message"""
        self._log('INFO', message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning level message"""
        self._log('WARNING', message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error level message"""
        self._log('ERROR', message, extra)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical level message"""
        self._log('CRITICAL', message, extra)

    def _log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Internal logging method"""
        extra = extra or {}

        # Add context information
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'logger': self.name,
            'message': message,
            'correlation_id': self._get_current_correlation_id(),
            'context_stack': len(self.context_stack),
            'thread_id': threading.get_ident(),
            **extra
        }

        # Add performance metrics if available
        if hasattr(self, '_get_performance_metrics'):
            log_entry['performance'] = self._get_performance_metrics()

        # Queue for async processing
        if self.log_worker_task and not self.log_worker_task.done():
            try:
                self.log_queue.put_nowait(log_entry)
            except asyncio.QueueFull:
                # Fallback to synchronous logging
                self._log_synchronously(log_entry)
        else:
            # Synchronous logging
            self._log_synchronously(log_entry)

    def _log_synchronously(self, log_entry: Dict[str, Any]):
        """Log synchronously when async worker not available"""
        level_map = {
            'TRACE': logging.DEBUG,
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        log_level = level_map.get(log_entry['level'], logging.INFO)
        # Remove 'message' from extra to avoid conflict with standard logging
        extra_data = {k: v for k, v in log_entry.items() if k != 'message'}
        self.logger.log(log_level, log_entry['message'], extra=extra_data)

    def _get_current_correlation_id(self) -> Optional[str]:
        """Get current correlation ID from context stack"""
        if self.context_stack:
            return self.context_stack[-1]['correlation_id']
        return None

    async def _analyze_log_patterns(self, log_entry: Dict[str, Any]):
        """Analyze patterns in log entries"""
        # Track message patterns
        message_key = log_entry.get('message', '').lower()
        self.log_patterns[message_key] += 1

        # Track error patterns
        if log_entry.get('level') in ['ERROR', 'CRITICAL']:
            error_key = f"{log_entry.get('message', '')}:{log_entry.get('error', '')}"
            self.error_patterns[error_key] += 1

    async def _write_additional_outputs(self, log_entry: Dict[str, Any]):
        """Write to additional outputs (metrics, monitoring, etc.)"""
        # Could write to external monitoring systems, databases, etc.
        pass

    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of logging activity"""
        return {
            'total_correlation_ids': len(self.correlation_ids),
            'active_operations': len(self.operation_start_times),
            'context_stack_depth': len(self.context_stack),
            'common_patterns': dict(sorted(self.log_patterns.items(),
                                         key=lambda x: x[1], reverse=True)[:10]),
            'error_patterns': dict(sorted(self.error_patterns.items(),
                                        key=lambda x: x[1], reverse=True)[:5]),
            'metrics_buffered': len(self.metrics_buffer)
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            'active_operations': len(self.operation_start_times),
            'queued_logs': self.log_queue.qsize() if self.log_worker_task else 0,
            'correlation_ids': len(self.correlation_ids),
            'metrics_buffered': len(self.metrics_buffer)
        }

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record) -> str:
        # Extract standard log record fields
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }

        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno',
                             'pathname', 'filename', 'module', 'exc_info',
                             'exc_text', 'stack_info', 'lineno', 'funcName',
                             'created', 'msecs', 'relativeCreated', 'thread',
                             'threadName', 'processName', 'process', 'message']:
                    if not key.startswith('_'):
                        log_entry[key] = value

        return json.dumps(log_entry, default=str)

class ConsciousnessFormatter(logging.Formatter):
    """Human-readable formatter for consciousness computing logs"""

    def format(self, record) -> str:
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        level = record.levelname[:1]  # First letter only
        logger = record.name.split('.')[-1]  # Last part of logger name

        message = f"[{timestamp}] {level} {logger}: {record.getMessage()}"

        # Add correlation ID if available
        if hasattr(record, 'correlation_id') and record.correlation_id:
            message += f" [CID:{record.correlation_id[:8]}]"

        # Add key metrics if available
        if hasattr(record, 'performance') and record.performance:
            perf = record.performance
            if 'duration_ms' in perf:
                message += f" ({perf['duration_ms']:.1f}ms)"

        return message

class LogAnalyzer:
    """
    Analyze log patterns and extract insights from logging data
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)

    def analyze_log_file(self, log_file: str) -> Dict[str, Any]:
        """Analyze a log file for patterns and insights"""

        log_path = self.log_dir / log_file
        if not log_path.exists():
            return {'error': f'Log file not found: {log_file}'}

        patterns = defaultdict(int)
        errors = []
        performance_metrics = []
        operations = defaultdict(list)

        with open(log_path, encoding='utf-8') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.strip())

                    # Analyze patterns
                    level = log_entry.get('level', 'UNKNOWN')
                    message = log_entry.get('message', '').lower()
                    patterns[f"{level}:{message}"] += 1

                    # Collect errors
                    if level in ['ERROR', 'CRITICAL']:
                        errors.append({
                            'timestamp': log_entry.get('timestamp'),
                            'message': log_entry.get('message'),
                            'correlation_id': log_entry.get('correlation_id')
                        })

                    # Collect performance metrics
                    if 'performance' in log_entry:
                        performance_metrics.append(log_entry['performance'])

                    # Track operations
                    if 'operation' in log_entry:
                        operations[log_entry['operation']].append(log_entry)

                except json.JSONDecodeError:
                    continue  # Skip malformed lines

        return {
            'total_entries': sum(patterns.values()),
            'pattern_analysis': dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:20]),
            'error_analysis': {
                'total_errors': len(errors),
                'error_rate': len(errors) / max(sum(patterns.values()), 1),
                'recent_errors': errors[-5:] if errors else []
            },
            'performance_analysis': self._analyze_performance_metrics(performance_metrics),
            'operation_analysis': {
                op: len(entries) for op, entries in operations.items()
            }
        }

    def _analyze_performance_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics from logs"""

        if not metrics:
            return {'error': 'No performance metrics found'}

        # Extract numeric metrics
        durations = [m.get('duration_ms', 0) for m in metrics if 'duration_ms' in m]
        memory_usage = [m.get('memory_mb', 0) for m in metrics if 'memory_mb' in m]
        cpu_usage = [m.get('cpu_percent', 0) for m in metrics if 'cpu_percent' in m]

        analysis = {}

        if durations:
            analysis['duration_stats'] = {
                'avg': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'p95': sorted(durations)[int(len(durations) * 0.95)]
            }

        if memory_usage:
            analysis['memory_stats'] = {
                'avg': sum(memory_usage) / len(memory_usage),
                'peak': max(memory_usage)
            }

        if cpu_usage:
            analysis['cpu_stats'] = {
                'avg': sum(cpu_usage) / len(cpu_usage),
                'peak': max(cpu_usage)
            }

        return analysis

# Global logger instance
_default_logger = None

def get_logger(name: str) -> ConsciousnessLogger:
    """Get or create a consciousness logger instance"""
    global _default_logger
    if _default_logger is None:
        _default_logger = ConsciousnessLogger(name)
    return _default_logger
