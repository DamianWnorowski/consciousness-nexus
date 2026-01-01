"""
Logging Bridge to OpenTelemetry

Connects Python's standard logging to OpenTelemetry:
- Exports logs to OTLP endpoint
- Adds trace context to log records
- Preserves existing ConsciousnessLogger functionality
- Provides structured logging with span correlation
"""

from typing import Any, Dict, Optional
import logging
import sys
import json
from datetime import datetime

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor, ConsoleLogExporter
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.sdk.resources import Resource


class OTelLoggingHandler(LoggingHandler):
    """
    Custom logging handler that bridges Python logging to OpenTelemetry.

    Extends the standard OTel LoggingHandler to:
    - Add trace context (trace_id, span_id) to all logs
    - Support structured logging with extra fields
    - Preserve consciousness-specific attributes
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record with trace context."""
        # Add trace context if available
        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            record.trace_id = format(ctx.trace_id, "032x")
            record.span_id = format(ctx.span_id, "016x")
            record.trace_flags = ctx.trace_flags
        else:
            record.trace_id = "0" * 32
            record.span_id = "0" * 16
            record.trace_flags = 0

        super().emit(record)


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs logs in a format suitable for:
    - Loki ingestion
    - CloudWatch Logs
    - Datadog log management
    - Any JSON log aggregator
    """

    def __init__(
        self,
        service_name: str = "consciousness-nexus",
        include_trace_context: bool = True,
    ):
        super().__init__()
        self.service_name = service_name
        self.include_trace_context = include_trace_context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
        }

        # Add source location
        log_data["source"] = {
            "file": record.pathname,
            "line": record.lineno,
            "function": record.funcName,
        }

        # Add trace context if available
        if self.include_trace_context:
            trace_id = getattr(record, "trace_id", None)
            span_id = getattr(record, "span_id", None)
            if trace_id and trace_id != "0" * 32:
                log_data["trace_id"] = trace_id
                log_data["span_id"] = span_id

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "thread", "threadName", "exc_info", "exc_text",
            "message", "trace_id", "span_id", "trace_flags",
        }
        extra = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }
        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data)


class ConsciousnessLoggingBridge:
    """
    Bridge between ConsciousnessLogger and OpenTelemetry.

    Provides:
    - OTLP log export
    - Console output with structured formatting
    - Trace context injection
    - Integration with existing logging setup
    """

    def __init__(
        self,
        service_name: str = "consciousness-nexus",
        service_version: str = "1.0.0",
        otlp_endpoint: Optional[str] = None,
        log_level: str = "INFO",
        enable_console: bool = True,
        enable_otlp: bool = True,
        structured_format: bool = True,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.otlp_endpoint = otlp_endpoint
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.enable_console = enable_console
        self.enable_otlp = enable_otlp
        self.structured_format = structured_format

        self._logger_provider: Optional[LoggerProvider] = None
        self._handlers: list = []

    def setup(self, resource: Optional[Resource] = None) -> None:
        """
        Initialize the logging bridge.

        Args:
            resource: OpenTelemetry resource for log attribution
        """
        if resource is None:
            from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                SERVICE_VERSION: self.service_version,
            })

        # Create logger provider
        self._logger_provider = LoggerProvider(resource=resource)

        # Add OTLP exporter if enabled
        if self.enable_otlp and self.otlp_endpoint:
            try:
                otlp_exporter = OTLPLogExporter(
                    endpoint=self.otlp_endpoint,
                    insecure=True,
                )
                self._logger_provider.add_log_record_processor(
                    BatchLogRecordProcessor(otlp_exporter)
                )
            except Exception as e:
                logging.warning(f"Failed to setup OTLP log exporter: {e}")

        # Add console exporter for development
        if self.enable_console:
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(ConsoleLogExporter())
            )

        set_logger_provider(self._logger_provider)

        # Configure root logger
        self._configure_root_logger()

    def _configure_root_logger(self) -> None:
        """Configure the root logger with OTel handler."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add OTel handler
        otel_handler = OTelLoggingHandler(
            level=self.log_level,
            logger_provider=self._logger_provider,
        )
        root_logger.addHandler(otel_handler)
        self._handlers.append(otel_handler)

        # Add console handler with structured formatting
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)

            if self.structured_format:
                console_handler.setFormatter(
                    StructuredFormatter(service_name=self.service_name)
                )
            else:
                console_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s | %(levelname)-8s | %(name)s | "
                        "[trace_id=%(trace_id)s span_id=%(span_id)s] | %(message)s",
                        defaults={"trace_id": "0" * 32, "span_id": "0" * 16},
                    )
                )

            root_logger.addHandler(console_handler)
            self._handlers.append(console_handler)

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger instance that integrates with OTel.

        Args:
            name: Logger name (typically __name__)

        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        return TraceContextLogger(logger)

    def shutdown(self) -> None:
        """Shutdown the logging bridge and flush pending logs."""
        if self._logger_provider:
            self._logger_provider.shutdown()


class TraceContextLogger:
    """
    Logger wrapper that automatically adds trace context.

    Provides the same interface as logging.Logger but
    automatically injects trace_id and span_id.
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger

    def _inject_context(self, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Inject trace context into extra dict."""
        extra = extra or {}

        span = trace.get_current_span()
        if span.is_recording():
            ctx = span.get_span_context()
            extra["trace_id"] = format(ctx.trace_id, "032x")
            extra["span_id"] = format(ctx.span_id, "016x")

        return extra

    def debug(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.debug(msg, *args, extra=self._inject_context(extra), **kwargs)

    def info(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.info(msg, *args, extra=self._inject_context(extra), **kwargs)

    def warning(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.warning(msg, *args, extra=self._inject_context(extra), **kwargs)

    def error(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.error(msg, *args, extra=self._inject_context(extra), **kwargs)

    def critical(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.critical(msg, *args, extra=self._inject_context(extra), **kwargs)

    def exception(self, msg: str, *args, extra: Optional[Dict] = None, **kwargs):
        self._logger.exception(msg, *args, extra=self._inject_context(extra), **kwargs)

    # Property passthrough
    @property
    def name(self) -> str:
        return self._logger.name

    @property
    def level(self) -> int:
        return self._logger.level

    def setLevel(self, level) -> None:
        self._logger.setLevel(level)

    def isEnabledFor(self, level: int) -> bool:
        return self._logger.isEnabledFor(level)


# Global bridge instance
_bridge: Optional[ConsciousnessLoggingBridge] = None


def setup_logging_bridge(
    service_name: str = "consciousness-nexus",
    otlp_endpoint: Optional[str] = None,
    log_level: str = "INFO",
) -> ConsciousnessLoggingBridge:
    """
    Initialize the global logging bridge.

    Args:
        service_name: Service name for log attribution
        otlp_endpoint: OTLP endpoint for log export
        log_level: Minimum log level

    Returns:
        Configured logging bridge
    """
    global _bridge
    _bridge = ConsciousnessLoggingBridge(
        service_name=service_name,
        otlp_endpoint=otlp_endpoint,
        log_level=log_level,
    )
    _bridge.setup()
    return _bridge


def get_otel_logger(name: str) -> TraceContextLogger:
    """
    Get a logger that automatically includes trace context.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger with trace context injection
    """
    global _bridge
    if _bridge is None:
        setup_logging_bridge()
    return _bridge.get_logger(name)
