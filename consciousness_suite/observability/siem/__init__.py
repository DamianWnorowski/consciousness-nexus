"""SIEM Integration Module

Security Information and Event Management integrations:
- Security event generation with ECS/CEF/LEEF formats
- Threat pattern detection and anomaly scoring
- Elastic SIEM integration
- Splunk HEC integration
- Sumo Logic integration
"""

from .security_events import (
    # Enums
    EventSeverity,
    EventCategory,
    EventOutcome,
    EventKind,
    # Data classes
    EventSource,
    EventDestination,
    ThreatIndicator,
    SecurityEvent,
    # Generator
    SecurityEventGenerator,
    # Convenience functions
    create_authentication_event,
    create_authorization_event,
    create_intrusion_event,
    create_malware_event,
)

from .threat_detection import (
    # Enums
    ThreatType,
    PatternMatchType,
    # Data classes
    ThreatPattern,
    ThreatMatch,
    AnomalyScore,
    # Detector
    ThreatDetector,
)

from .elastic_siem import (
    # Enums
    ElasticAuthType,
    IndexStrategy,
    # Data classes
    ElasticConfig,
    BulkResult,
    SearchResult,
    AlertRule,
    # Client
    ElasticSIEMClient,
    # Factory functions
    create_elastic_cloud_client,
    create_local_client,
)

from .splunk_bridge import (
    # Enums
    HECEndpoint,
    SourceType,
    # Data classes
    HECConfig,
    HECResponse,
    BatchResult as SplunkBatchResult,
    HECEvent,
    # Clients
    SplunkHECClient,
    SplunkSearchClient,
    # Factory functions
    create_hec_client,
    create_cloud_hec_client,
)

from .sumo_logic import (
    # Enums
    SumoSourceType,
    SumoLogFormat,
    SumoCategory,
    # Data classes
    SumoConfig,
    SumoResponse,
    BatchResult as SumoBatchResult,
    SumoLogEntry,
    # Clients
    SumoLogicClient,
    SumoCloudSIEMClient,
    # Factory functions
    create_sumo_client,
    create_cloud_siem_client,
)

__all__ = [
    # Security Events
    "EventSeverity",
    "EventCategory",
    "EventOutcome",
    "EventKind",
    "EventSource",
    "EventDestination",
    "ThreatIndicator",
    "SecurityEvent",
    "SecurityEventGenerator",
    "create_authentication_event",
    "create_authorization_event",
    "create_intrusion_event",
    "create_malware_event",
    # Threat Detection
    "ThreatType",
    "PatternMatchType",
    "ThreatPattern",
    "ThreatMatch",
    "AnomalyScore",
    "ThreatDetector",
    # Elastic SIEM
    "ElasticAuthType",
    "IndexStrategy",
    "ElasticConfig",
    "BulkResult",
    "SearchResult",
    "AlertRule",
    "ElasticSIEMClient",
    "create_elastic_cloud_client",
    "create_local_client",
    # Splunk
    "HECEndpoint",
    "SourceType",
    "HECConfig",
    "HECResponse",
    "SplunkBatchResult",
    "HECEvent",
    "SplunkHECClient",
    "SplunkSearchClient",
    "create_hec_client",
    "create_cloud_hec_client",
    # Sumo Logic
    "SumoSourceType",
    "SumoLogFormat",
    "SumoCategory",
    "SumoConfig",
    "SumoResponse",
    "SumoBatchResult",
    "SumoLogEntry",
    "SumoLogicClient",
    "SumoCloudSIEMClient",
    "create_sumo_client",
    "create_cloud_siem_client",
]
