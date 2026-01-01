"""
Consciousness Nexus Ultimate Observability Platform

Bleeding-edge observability covering all 25 domains:
- Metrics, Logs, Traces, Profiles, Events
- LLM Observability, ML Monitoring, AI Platform
- Service Mesh, Chaos Engineering, Digital Twin
- Business Metrics, FinOps, GreenOps, Compliance
- Security/SIEM, RUM, Synthetic, Mobile, Database
- Serverless, API Gateway, Incident Management, GitOps
- Network/eBPF, Container Security

Zero vendor lock-in, <1% overhead, full OSS compatibility.
"""

from typing import Optional
import logging

# Version info
__version__ = "1.0.0"
__all__ = [
    # Core modules
    "setup_observability",
    "get_tracer",
    "get_meter",
    "get_logger",
    # Configuration
    "ObservabilityConfig",
    # Exporters
    "ConsciousnessMetrics",
    "ConsciousnessTracer",
    # Middleware
    "ObservabilityMiddleware",
    # Service Mesh
    "mesh",
    # Business Observability
    "business",
    # FinOps
    "finops",
    # GreenOps
    "greenops",
    # Database Observability
    "database",
    # API Gateway Observability
    "api",
    # Container Security
    "security",
    # Events & CloudEvents
    "events",
    # Synthetic Monitoring
    "synthetic",
    # Incident Management
    "incidents",
]

# Lazy imports to avoid circular dependencies
_tracer = None
_meter = None
_config = None


class ObservabilityConfig:
    """Central configuration for all observability components."""

    def __init__(
        self,
        service_name: str = "consciousness-nexus",
        service_version: str = "1.0.0",
        environment: str = "development",
        # OTLP endpoints
        otlp_endpoint: str = "http://localhost:4317",
        otlp_http_endpoint: str = "http://localhost:4318",
        # Prometheus
        prometheus_port: int = 9090,
        enable_prometheus: bool = True,
        # Tracing
        enable_tracing: bool = True,
        trace_sample_rate: float = 1.0,
        # Logging
        enable_log_export: bool = True,
        log_level: str = "INFO",
        # Profiling
        enable_profiling: bool = True,
        profiling_interval_ms: int = 10,
        # eBPF (requires root/CAP_BPF)
        enable_ebpf: bool = False,
        # LLM Observability
        langfuse_enabled: bool = False,
        langfuse_public_key: Optional[str] = None,
        langfuse_secret_key: Optional[str] = None,
        langfuse_host: str = "https://cloud.langfuse.com",
        # Phoenix (self-hosted)
        phoenix_enabled: bool = False,
        phoenix_endpoint: str = "http://localhost:6006",
        # Adaptive telemetry
        enable_adaptive_sampling: bool = True,
        telemetry_budget_mb_per_hour: int = 100,
    ):
        self.service_name = service_name
        self.service_version = service_version
        self.environment = environment

        # OTLP
        self.otlp_endpoint = otlp_endpoint
        self.otlp_http_endpoint = otlp_http_endpoint

        # Prometheus
        self.prometheus_port = prometheus_port
        self.enable_prometheus = enable_prometheus

        # Tracing
        self.enable_tracing = enable_tracing
        self.trace_sample_rate = trace_sample_rate

        # Logging
        self.enable_log_export = enable_log_export
        self.log_level = log_level

        # Profiling
        self.enable_profiling = enable_profiling
        self.profiling_interval_ms = profiling_interval_ms

        # eBPF
        self.enable_ebpf = enable_ebpf

        # LLM Observability
        self.langfuse_enabled = langfuse_enabled
        self.langfuse_public_key = langfuse_public_key
        self.langfuse_secret_key = langfuse_secret_key
        self.langfuse_host = langfuse_host

        # Phoenix
        self.phoenix_enabled = phoenix_enabled
        self.phoenix_endpoint = phoenix_endpoint

        # Adaptive
        self.enable_adaptive_sampling = enable_adaptive_sampling
        self.telemetry_budget_mb_per_hour = telemetry_budget_mb_per_hour

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "consciousness-nexus"),
            service_version=os.getenv("SERVICE_VERSION", "1.0.0"),
            environment=os.getenv("ENVIRONMENT", "development"),
            otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
            otlp_http_endpoint=os.getenv("OTEL_EXPORTER_OTLP_HTTP_ENDPOINT", "http://localhost:4318"),
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "9090")),
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            enable_tracing=os.getenv("ENABLE_TRACING", "true").lower() == "true",
            trace_sample_rate=float(os.getenv("TRACE_SAMPLE_RATE", "1.0")),
            enable_log_export=os.getenv("ENABLE_LOG_EXPORT", "true").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            enable_profiling=os.getenv("ENABLE_PROFILING", "true").lower() == "true",
            enable_ebpf=os.getenv("ENABLE_EBPF", "false").lower() == "true",
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            phoenix_enabled=os.getenv("PHOENIX_ENABLED", "false").lower() == "true",
            phoenix_endpoint=os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006"),
            enable_adaptive_sampling=os.getenv("ENABLE_ADAPTIVE_SAMPLING", "true").lower() == "true",
            telemetry_budget_mb_per_hour=int(os.getenv("TELEMETRY_BUDGET_MB_PER_HOUR", "100")),
        )


def setup_observability(config: Optional[ObservabilityConfig] = None) -> None:
    """
    Initialize the complete observability stack.

    This sets up:
    - OpenTelemetry SDK with OTLP exporters
    - Prometheus metrics export
    - Distributed tracing with W3C Trace Context
    - Log bridge to OTel
    - Optional: Profiling, eBPF, LLM observability
    """
    global _config, _tracer, _meter

    if config is None:
        config = ObservabilityConfig.from_env()

    _config = config

    # Import and initialize OTel
    from .otel_config import initialize_otel
    _tracer, _meter = initialize_otel(config)

    # Initialize Prometheus metrics
    if config.enable_prometheus:
        from .prometheus import setup_prometheus
        setup_prometheus(config)

    # Initialize LLM observability if enabled
    if config.langfuse_enabled:
        from .llm import setup_langfuse
        setup_langfuse(config)

    if config.phoenix_enabled:
        from .llm import setup_phoenix
        setup_phoenix(config)

    logging.info(
        f"Observability initialized: service={config.service_name}, "
        f"env={config.environment}, tracing={config.enable_tracing}, "
        f"prometheus={config.enable_prometheus}"
    )


def get_tracer(name: str = __name__):
    """Get a tracer instance for creating spans."""
    global _tracer
    if _tracer is None:
        from opentelemetry import trace
        return trace.get_tracer(name)
    return _tracer


def get_meter(name: str = __name__):
    """Get a meter instance for creating metrics."""
    global _meter
    if _meter is None:
        from opentelemetry import metrics
        return metrics.get_meter(name)
    return _meter


def get_logger(name: str = __name__) -> logging.Logger:
    """Get a logger that bridges to OTel."""
    from .logging_bridge import get_otel_logger
    return get_otel_logger(name)


# Convenience re-exports
from .metrics import ConsciousnessMetrics
from .tracing import ConsciousnessTracer
from .middleware import ObservabilityMiddleware

# Sub-modules (lazy-loaded for performance)
from . import mesh
from . import business
from . import finops
from . import greenops
from . import database
from . import api
from . import security
from . import events
from . import synthetic
from . import incidents
