"""
OpenTelemetry SDK Configuration

Complete OTel setup with:
- TracerProvider with OTLP gRPC/HTTP export
- MeterProvider with Prometheus + OTLP export
- LoggerProvider with OTLP export
- Resource detection and attributes
- Batch processors for optimal performance
"""

from typing import Tuple, Optional
import logging
import os

from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider, SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader, ConsoleMetricExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.semconv.resource import ResourceAttributes

# OTLP Exporters
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

# Prometheus exporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader

# Propagators
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.composite import CompositePropagator
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.baggage.propagation import W3CBaggagePropagator

# Samplers
from opentelemetry.sdk.trace.sampling import (
    TraceIdRatioBased,
    ParentBased,
    ALWAYS_ON,
    ALWAYS_OFF,
)

logger = logging.getLogger(__name__)


def create_resource(config) -> Resource:
    """
    Create OpenTelemetry Resource with service metadata.

    Resources describe the entity producing telemetry.
    """
    return Resource.create({
        SERVICE_NAME: config.service_name,
        SERVICE_VERSION: config.service_version,
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: config.environment,
        ResourceAttributes.SERVICE_NAMESPACE: "consciousness-nexus",
        ResourceAttributes.SERVICE_INSTANCE_ID: os.getenv("HOSTNAME", os.getenv("COMPUTERNAME", "unknown")),
        # Custom attributes
        "consciousness.version": "1.0.0",
        "consciousness.deployment": config.environment,
    })


def create_sampler(config):
    """
    Create trace sampler based on configuration.

    Uses ParentBased sampler to respect parent span decisions
    with configurable ratio for root spans.
    """
    if config.trace_sample_rate >= 1.0:
        root_sampler = ALWAYS_ON
    elif config.trace_sample_rate <= 0.0:
        root_sampler = ALWAYS_OFF
    else:
        root_sampler = TraceIdRatioBased(config.trace_sample_rate)

    return ParentBased(root=root_sampler)


def setup_propagators():
    """
    Configure W3C Trace Context propagation.

    Enables distributed tracing across service boundaries
    using standard W3C headers: traceparent, tracestate, baggage.
    """
    propagator = CompositePropagator([
        TraceContextTextMapPropagator(),  # traceparent, tracestate
        W3CBaggagePropagator(),            # baggage
    ])
    set_global_textmap(propagator)
    logger.info("W3C Trace Context propagation configured")


def setup_tracer_provider(config, resource: Resource) -> TracerProvider:
    """
    Create and configure TracerProvider.

    Sets up:
    - OTLP gRPC exporter for production
    - Console exporter for development
    - Batch processor for performance
    """
    provider = TracerProvider(
        resource=resource,
        sampler=create_sampler(config),
    )

    # OTLP gRPC exporter (primary)
    if config.enable_tracing:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=config.otlp_endpoint,
                insecure=True,  # Set False in production with TLS
            )
            processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                export_timeout_millis=30000,
                schedule_delay_millis=5000,
            )
            provider.add_span_processor(processor)
            logger.info(f"OTLP span exporter configured: {config.otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")

    # Console exporter for development
    if config.environment == "development":
        console_processor = BatchSpanProcessor(ConsoleSpanExporter())
        provider.add_span_processor(console_processor)

    trace.set_tracer_provider(provider)
    return provider


def setup_meter_provider(config, resource: Resource) -> MeterProvider:
    """
    Create and configure MeterProvider.

    Sets up:
    - Prometheus metric reader for /metrics endpoint
    - OTLP exporter for metrics pipeline
    """
    readers = []

    # Prometheus reader for scraping
    if config.enable_prometheus:
        prometheus_reader = PrometheusMetricReader()
        readers.append(prometheus_reader)
        logger.info("Prometheus metric reader configured")

    # OTLP metric exporter
    try:
        otlp_metric_exporter = OTLPMetricExporter(
            endpoint=config.otlp_endpoint,
            insecure=True,
        )
        otlp_reader = PeriodicExportingMetricReader(
            otlp_metric_exporter,
            export_interval_millis=60000,  # Export every 60s
        )
        readers.append(otlp_reader)
        logger.info(f"OTLP metric exporter configured: {config.otlp_endpoint}")
    except Exception as e:
        logger.warning(f"Failed to configure OTLP metric exporter: {e}")

    # Console exporter for development
    if config.environment == "development":
        console_reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=30000,
        )
        readers.append(console_reader)

    provider = MeterProvider(
        resource=resource,
        metric_readers=readers,
    )

    metrics.set_meter_provider(provider)
    return provider


def initialize_otel(config) -> Tuple:
    """
    Initialize the complete OpenTelemetry SDK.

    Returns:
        Tuple of (tracer, meter) for application use
    """
    logger.info(f"Initializing OpenTelemetry for {config.service_name}")

    # Create resource
    resource = create_resource(config)

    # Setup propagators first
    setup_propagators()

    # Setup providers
    tracer_provider = setup_tracer_provider(config, resource)
    meter_provider = setup_meter_provider(config, resource)

    # Get instances
    tracer = trace.get_tracer(
        config.service_name,
        config.service_version,
    )
    meter = metrics.get_meter(
        config.service_name,
        config.service_version,
    )

    logger.info("OpenTelemetry initialization complete")

    return tracer, meter


def shutdown_otel():
    """
    Gracefully shutdown OpenTelemetry SDK.

    Flushes all pending telemetry before shutdown.
    """
    logger.info("Shutting down OpenTelemetry...")

    tracer_provider = trace.get_tracer_provider()
    if hasattr(tracer_provider, 'shutdown'):
        tracer_provider.shutdown()

    meter_provider = metrics.get_meter_provider()
    if hasattr(meter_provider, 'shutdown'):
        meter_provider.shutdown()

    logger.info("OpenTelemetry shutdown complete")
