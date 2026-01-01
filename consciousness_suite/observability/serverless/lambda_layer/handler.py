"""Lambda Extension Handler

AWS Lambda extension for telemetry collection:
- Registers with Lambda Extensions API
- Handles INVOKE and SHUTDOWN events
- Manages telemetry buffer flushing
- Integrates with OTLP exporter
"""

from __future__ import annotations

import os
import json
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
from contextlib import contextmanager
import urllib.request
import urllib.error

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)

# Lambda Extensions API base URL
LAMBDA_EXTENSION_API = os.environ.get("AWS_LAMBDA_RUNTIME_API", "")
EXTENSION_NAME = "consciousness-observability"


class ExtensionState(str, Enum):
    """Lambda extension lifecycle states."""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    SHUTTING_DOWN = "shutting_down"
    TERMINATED = "terminated"


class EventType(str, Enum):
    """Lambda extension event types."""
    INVOKE = "INVOKE"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class ExtensionConfig:
    """Configuration for Lambda extension."""
    extension_name: str = EXTENSION_NAME
    # Telemetry settings
    flush_interval_ms: int = 1000
    max_batch_size: int = 100
    max_buffer_size: int = 10000
    # Timeout settings
    register_timeout_ms: int = 5000
    event_timeout_ms: int = 0  # 0 = wait indefinitely
    # OTLP settings
    otlp_endpoint: str = "http://localhost:4317"
    otlp_protocol: str = "grpc"  # grpc or http
    # Feature flags
    enable_metrics: bool = True
    enable_traces: bool = True
    enable_logs: bool = True
    # X-Ray integration
    xray_enabled: bool = True
    # Environment
    function_name: str = field(default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_NAME", "unknown"))
    function_version: str = field(default_factory=lambda: os.environ.get("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST"))
    region: str = field(default_factory=lambda: os.environ.get("AWS_REGION", "us-east-1"))


@dataclass
class ExtensionEvent:
    """Lambda extension event."""
    event_type: EventType
    deadline_ms: int
    request_id: Optional[str] = None
    invoked_function_arn: Optional[str] = None
    shutdown_reason: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtensionEvent":
        """Create from API response."""
        event_type = EventType(data.get("eventType", "INVOKE"))
        return cls(
            event_type=event_type,
            deadline_ms=data.get("deadlineMs", 0),
            request_id=data.get("requestId"),
            invoked_function_arn=data.get("invokedFunctionArn"),
            shutdown_reason=data.get("shutdownReason"),
        )


class LambdaExtensionHandler:
    """AWS Lambda extension handler for observability.

    Implements the Lambda Extensions API to:
    - Register the extension during Lambda init phase
    - Handle INVOKE events to collect per-invocation telemetry
    - Handle SHUTDOWN events for graceful cleanup
    - Buffer and flush telemetry asynchronously

    Usage:
        extension = LambdaExtensionHandler()
        extension.start()

        # Extension runs in background, collecting telemetry
        # Automatically flushes on INVOKE completion and SHUTDOWN
    """

    def __init__(
        self,
        config: Optional[ExtensionConfig] = None,
        on_invoke: Optional[Callable[[ExtensionEvent], None]] = None,
        on_shutdown: Optional[Callable[[ExtensionEvent], None]] = None,
        namespace: str = "consciousness",
    ):
        self.config = config or ExtensionConfig()
        self.on_invoke = on_invoke
        self.on_shutdown = on_shutdown
        self.namespace = namespace

        self._state = ExtensionState.INITIALIZING
        self._extension_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._telemetry_buffer: List[Dict[str, Any]] = []
        self._buffer_lock = threading.Lock()

        # Metrics
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics."""
        self.extension_events = Counter(
            f"{self.namespace}_lambda_extension_events_total",
            "Lambda extension events processed",
            ["event_type", "function_name"],
        )

        self.extension_errors = Counter(
            f"{self.namespace}_lambda_extension_errors_total",
            "Lambda extension errors",
            ["error_type", "function_name"],
        )

        self.extension_state = Gauge(
            f"{self.namespace}_lambda_extension_state",
            "Lambda extension state (0=init, 1=ready, 2=processing, 3=shutdown)",
            ["function_name"],
        )

        self.buffer_size = Gauge(
            f"{self.namespace}_lambda_extension_buffer_size",
            "Telemetry buffer size",
            ["function_name"],
        )

        self.flush_duration = Histogram(
            f"{self.namespace}_lambda_extension_flush_duration_seconds",
            "Telemetry flush duration",
            ["function_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

    @property
    def state(self) -> ExtensionState:
        """Get current extension state."""
        return self._state

    @state.setter
    def state(self, value: ExtensionState):
        """Set extension state and update metric."""
        self._state = value
        state_values = {
            ExtensionState.INITIALIZING: 0,
            ExtensionState.READY: 1,
            ExtensionState.PROCESSING: 2,
            ExtensionState.SHUTTING_DOWN: 3,
            ExtensionState.TERMINATED: 4,
        }
        self.extension_state.labels(
            function_name=self.config.function_name
        ).set(state_values.get(value, 0))

    def _api_request(
        self,
        path: str,
        method: str = "GET",
        data: Optional[Dict] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_ms: int = 5000,
    ) -> Dict[str, Any]:
        """Make request to Lambda Extensions API."""
        if not LAMBDA_EXTENSION_API:
            raise RuntimeError("AWS_LAMBDA_RUNTIME_API not set")

        url = f"http://{LAMBDA_EXTENSION_API}{path}"
        req_headers = headers or {}

        if self._extension_id:
            req_headers["Lambda-Extension-Identifier"] = self._extension_id

        req_data = json.dumps(data).encode() if data else None

        request = urllib.request.Request(
            url,
            data=req_data,
            headers=req_headers,
            method=method,
        )

        try:
            timeout = timeout_ms / 1000.0 if timeout_ms > 0 else None
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_data = json.loads(response.read().decode())
                return response_data
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            logger.error(f"Extension API error: {e.code} - {error_body}")
            raise
        except urllib.error.URLError as e:
            logger.error(f"Extension API connection error: {e}")
            raise

    def register(self) -> str:
        """Register the extension with Lambda.

        Returns:
            Extension ID
        """
        logger.info(f"Registering extension: {self.config.extension_name}")

        events = []
        if self.config.enable_metrics or self.config.enable_traces:
            events.append("INVOKE")
        events.append("SHUTDOWN")

        response = self._api_request(
            "/2020-01-01/extension/register",
            method="POST",
            data={"events": events},
            headers={
                "Lambda-Extension-Name": self.config.extension_name,
                "Content-Type": "application/json",
            },
            timeout_ms=self.config.register_timeout_ms,
        )

        # Extension ID is returned in header, but also check response
        self._extension_id = response.get("extensionId")
        if not self._extension_id:
            # Try to get from response body if available
            self._extension_id = response.get("Lambda-Extension-Identifier", "")

        logger.info(f"Extension registered: {self._extension_id}")
        self.state = ExtensionState.READY

        return self._extension_id

    def next_event(self) -> ExtensionEvent:
        """Wait for and return the next extension event.

        Returns:
            ExtensionEvent
        """
        response = self._api_request(
            "/2020-01-01/extension/event/next",
            method="GET",
            timeout_ms=self.config.event_timeout_ms,
        )

        return ExtensionEvent.from_dict(response)

    def _process_event(self, event: ExtensionEvent):
        """Process an extension event."""
        self.extension_events.labels(
            event_type=event.event_type.value,
            function_name=self.config.function_name,
        ).inc()

        if event.event_type == EventType.INVOKE:
            self._handle_invoke(event)
        elif event.event_type == EventType.SHUTDOWN:
            self._handle_shutdown(event)

    def _handle_invoke(self, event: ExtensionEvent):
        """Handle INVOKE event."""
        self.state = ExtensionState.PROCESSING
        logger.debug(f"Processing invocation: {event.request_id}")

        # Call custom handler if provided
        if self.on_invoke:
            try:
                self.on_invoke(event)
            except Exception as e:
                logger.error(f"Error in on_invoke handler: {e}")
                self.extension_errors.labels(
                    error_type="invoke_handler",
                    function_name=self.config.function_name,
                ).inc()

        # Flush telemetry after invocation
        self._flush_telemetry()
        self.state = ExtensionState.READY

    def _handle_shutdown(self, event: ExtensionEvent):
        """Handle SHUTDOWN event."""
        self.state = ExtensionState.SHUTTING_DOWN
        logger.info(f"Shutdown requested: {event.shutdown_reason}")

        # Call custom handler if provided
        if self.on_shutdown:
            try:
                self.on_shutdown(event)
            except Exception as e:
                logger.error(f"Error in on_shutdown handler: {e}")

        # Final flush
        self._flush_telemetry(force=True)
        self.state = ExtensionState.TERMINATED

    def add_telemetry(self, data: Dict[str, Any]):
        """Add telemetry data to buffer.

        Args:
            data: Telemetry data to buffer
        """
        with self._buffer_lock:
            if len(self._telemetry_buffer) < self.config.max_buffer_size:
                self._telemetry_buffer.append({
                    **data,
                    "timestamp": time.time(),
                    "function_name": self.config.function_name,
                    "function_version": self.config.function_version,
                })
            else:
                logger.warning("Telemetry buffer full, dropping data")
                self.extension_errors.labels(
                    error_type="buffer_overflow",
                    function_name=self.config.function_name,
                ).inc()

        self.buffer_size.labels(
            function_name=self.config.function_name
        ).set(len(self._telemetry_buffer))

    def _flush_telemetry(self, force: bool = False):
        """Flush telemetry buffer.

        Args:
            force: Force flush regardless of batch size
        """
        with self._buffer_lock:
            if not self._telemetry_buffer:
                return

            if not force and len(self._telemetry_buffer) < self.config.max_batch_size:
                return

            batch = self._telemetry_buffer[:self.config.max_batch_size]
            self._telemetry_buffer = self._telemetry_buffer[self.config.max_batch_size:]

        start_time = time.perf_counter()

        try:
            # Export telemetry (implementation in exporter.py)
            self._export_batch(batch)
        except Exception as e:
            logger.error(f"Failed to export telemetry: {e}")
            self.extension_errors.labels(
                error_type="export_failure",
                function_name=self.config.function_name,
            ).inc()
            # Re-add to buffer if export fails (best effort)
            with self._buffer_lock:
                self._telemetry_buffer = batch + self._telemetry_buffer
        finally:
            duration = time.perf_counter() - start_time
            self.flush_duration.labels(
                function_name=self.config.function_name
            ).observe(duration)

        self.buffer_size.labels(
            function_name=self.config.function_name
        ).set(len(self._telemetry_buffer))

    def _export_batch(self, batch: List[Dict[str, Any]]):
        """Export a batch of telemetry data.

        This is a placeholder - actual implementation uses LambdaOTLPExporter.

        Args:
            batch: List of telemetry records
        """
        # Import here to avoid circular dependency
        from .exporter import LambdaOTLPExporter, ExportConfig

        exporter = LambdaOTLPExporter(
            config=ExportConfig(
                endpoint=self.config.otlp_endpoint,
                protocol=self.config.otlp_protocol,
            )
        )

        exporter.export_batch(batch)

    def _event_loop(self):
        """Main event loop for the extension."""
        try:
            self.register()

            while not self._stop_event.is_set():
                try:
                    event = self.next_event()
                    self._process_event(event)

                    if event.event_type == EventType.SHUTDOWN:
                        break
                except Exception as e:
                    logger.error(f"Error in event loop: {e}")
                    self.extension_errors.labels(
                        error_type="event_loop",
                        function_name=self.config.function_name,
                    ).inc()
                    # Brief sleep before retrying
                    time.sleep(0.1)

        except Exception as e:
            logger.error(f"Fatal extension error: {e}")
            self.state = ExtensionState.TERMINATED

    def start(self):
        """Start the extension in a background thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Extension already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._event_loop,
            name=f"{self.config.extension_name}-thread",
            daemon=True,
        )
        self._thread.start()
        logger.info("Extension started")

    def stop(self, timeout: float = 5.0):
        """Stop the extension.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        self._stop_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Extension thread did not stop gracefully")

        self.state = ExtensionState.TERMINATED
        logger.info("Extension stopped")

    @contextmanager
    def invocation_context(self, request_id: str):
        """Context manager for tracking a Lambda invocation.

        Args:
            request_id: AWS request ID

        Yields:
            Dict for adding custom telemetry
        """
        invocation_data = {
            "request_id": request_id,
            "start_time": time.time(),
        }

        try:
            yield invocation_data
        except Exception as e:
            invocation_data["error"] = str(e)
            raise
        finally:
            invocation_data["end_time"] = time.time()
            invocation_data["duration_ms"] = (
                invocation_data["end_time"] - invocation_data["start_time"]
            ) * 1000
            self.add_telemetry(invocation_data)


# Singleton for global access
_extension: Optional[LambdaExtensionHandler] = None


def register_extension(
    config: Optional[ExtensionConfig] = None,
    on_invoke: Optional[Callable[[ExtensionEvent], None]] = None,
    on_shutdown: Optional[Callable[[ExtensionEvent], None]] = None,
) -> LambdaExtensionHandler:
    """Register and start the Lambda extension.

    Args:
        config: Extension configuration
        on_invoke: Callback for INVOKE events
        on_shutdown: Callback for SHUTDOWN events

    Returns:
        LambdaExtensionHandler instance
    """
    global _extension

    if _extension is not None:
        return _extension

    _extension = LambdaExtensionHandler(
        config=config,
        on_invoke=on_invoke,
        on_shutdown=on_shutdown,
    )

    # Only start if running in Lambda environment
    if LAMBDA_EXTENSION_API:
        _extension.start()
    else:
        logger.info("Not running in Lambda environment, extension not started")

    return _extension
