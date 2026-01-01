"""Lambda Layer Package

AWS Lambda extension and observability layer:
- Lambda extension for telemetry collection
- OTLP exporter for async telemetry flush
- Integration with AWS X-Ray and CloudWatch
"""

from .handler import (
    LambdaExtensionHandler,
    ExtensionConfig,
    ExtensionState,
    register_extension,
)
from .exporter import (
    LambdaOTLPExporter,
    ExportConfig,
    ExportResult,
    BatchConfig,
)

__all__ = [
    # Extension Handler
    "LambdaExtensionHandler",
    "ExtensionConfig",
    "ExtensionState",
    "register_extension",
    # OTLP Exporter
    "LambdaOTLPExporter",
    "ExportConfig",
    "ExportResult",
    "BatchConfig",
]
