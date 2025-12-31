"""
Data Models for Consciousness Computing Suite
=============================================

Common data structures and models used across all consciousness computing systems.
Provides type safety, validation, and serialization capabilities.
"""

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


@dataclass
class VectorData:
    """Represents vector data with metadata"""
    vectors: List[List[float]]
    ids: List[str] = field(default_factory=list)
    metadata: List[Dict[str, Any]] = field(default_factory=list)
    dimensions: int = 0

    def __post_init__(self):
        if self.vectors and not self.dimensions:
            self.dimensions = len(self.vectors[0]) if self.vectors[0] else 0

        if not self.ids:
            self.ids = [str(uuid.uuid4()) for _ in range(len(self.vectors))]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorData':
        return cls(**data)

@dataclass
class ConversationData:
    """Represents conversation data structure"""
    conversation_id: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationData':
        data_copy = data.copy()
        if 'timestamp' in data_copy:
            data_copy['timestamp'] = datetime.fromisoformat(data_copy['timestamp'])
        return cls(**data_copy)

@dataclass
class AnalysisLayer:
    """Represents a single analysis layer result"""
    layer_id: str
    layer_name: str
    input_data: Any
    output_data: Any
    processing_time: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class StackedAnalysisResult:
    """Result from stacked analysis workflow"""
    analysis_id: str
    layers: List[AnalysisLayer]
    final_result: Any
    total_processing_time: float
    overall_confidence: float
    layer_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class APICall:
    """Represents an API call with metadata"""
    call_id: str
    provider: str
    endpoint: str
    method: str
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    status_code: Optional[int] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class APIMetrics:
    """API usage metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    providers_used: Set[str] = field(default_factory=set)

    def update_from_call(self, call: APICall):
        """Update metrics from an API call"""
        self.total_calls += 1
        self.providers_used.add(call.provider)

        if call.error or (call.status_code and call.status_code >= 400):
            self.failed_calls += 1
        else:
            self.successful_calls += 1

        if call.tokens_used:
            self.total_tokens += call.tokens_used

        if call.cost:
            self.total_cost += call.cost

        # Recalculate derived metrics
        self.error_rate = self.failed_calls / max(self.total_calls, 1)

        # Simple moving average for latency
        if call.latency_ms:
            self.average_latency = (self.average_latency * (self.total_calls - 1) + call.latency_ms) / self.total_calls

@dataclass
class OrchestrationPlan:
    """Plan for orchestrating multiple components"""
    plan_id: str
    components: List[Dict[str, Any]]
    execution_order: List[str]
    dependencies: Dict[str, List[str]]
    resource_requirements: Dict[str, Any]
    estimated_duration: float
    risk_assessment: Dict[str, Any]
    success_criteria: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class OrchestrationResult:
    """Result from orchestration execution"""
    execution_id: str
    plan: OrchestrationPlan
    component_results: Dict[str, Any]
    overall_success: bool
    total_duration: float
    resource_usage: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base"""
    entry_id: str
    title: str
    content: str
    category: str
    tags: List[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "unknown"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KnowledgeEntry':
        data_copy = data.copy()
        data_copy['created_at'] = datetime.fromisoformat(data_copy['created_at'])
        data_copy['updated_at'] = datetime.fromisoformat(data_copy['updated_at'])
        return cls(**data_copy)

@dataclass
class KnowledgeQuery:
    """Query for knowledge base"""
    query_id: str
    query_text: str
    filters: Dict[str, Any] = field(default_factory=dict)
    limit: int = 10
    include_metadata: bool = False
    semantic_search: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class KnowledgeQueryResult:
    """Result from knowledge base query"""
    query: KnowledgeQuery
    entries: List[KnowledgeEntry]
    total_found: int
    search_time: float
    relevance_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['entries'] = [entry.to_dict() for entry in self.entries]
        return data

@dataclass
class QueueTask:
    """Task for the queue system"""
    task_id: str
    description: str
    command: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    estimated_duration: int = 30  # minutes
    required_resources: Dict[str, Any] = field(default_factory=dict)
    success_criteria: List[str] = field(default_factory=list)
    failure_handling: str = "retry"
    expires_at: Optional[datetime] = None
    confidence_threshold: float = 0.75
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueueTask':
        data_copy = data.copy()
        data_copy['created_at'] = datetime.fromisoformat(data_copy['created_at'])
        if 'expires_at' in data_copy and data_copy['expires_at']:
            data_copy['expires_at'] = datetime.fromisoformat(data_copy['expires_at'])
        return cls(**data_copy)

@dataclass
class AbyssalTemplate:
    """Template for ABYSSAL execution"""
    name: str
    description: str
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    nested_templates: List[Dict[str, Any]] = field(default_factory=list)
    estimated_duration: int = 30
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    execution_mode: str = "parallel"  # parallel, sequential

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AbyssalTemplate':
        return cls(**data)

@dataclass
class AbyssalExecution:
    """ABYSSAL template execution"""
    execution_id: str
    template: AbyssalTemplate
    parameters: Dict[str, Any]
    execution_tree: Dict[str, Any]
    results: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    progress: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

@dataclass
class ProcessingContext:
    """Context for processing operations"""
    session_id: str
    correlation_id: str
    start_time: float
    user_id: Optional[str] = None
    system_version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ConfidenceScore:
    """Represents confidence scoring with factors"""
    value: float  # 0.0 to 1.0
    factors: List[str] = field(default_factory=list)
    uncertainty_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ProcessingMetadata:
    """Metadata for processing operations"""
    processor_name: str
    operation_type: str
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    processing_steps: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AnalysisResult:
    """Result from analysis operations"""
    success: bool
    data: Any
    confidence: ConfidenceScore
    metadata: ProcessingMetadata
    context: ProcessingContext
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['data'] = str(self.data)  # Convert to string for serialization
        return data

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    total_analyses: int = 0
    average_processing_time: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    last_analysis_time: Optional[datetime] = None
    system_health: str = "healthy"
    active_connections: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.last_analysis_time:
            data['last_analysis_time'] = self.last_analysis_time.isoformat()
        return data

@dataclass
class MetaAnalysis:
    """Meta-analysis result from sublayer meta-parser"""
    analysis_id: str
    quantum_patterns: Dict[str, Any]
    emergence_patterns: Dict[str, Any]
    intent_crystallization: Dict[str, Any]
    implementation_analysis: Dict[str, Any]
    self_analysis: Dict[str, Any]
    predictive_patterns: List[Dict[str, Any]]
    consciousness_index: float
    implementation_fidelity: float
    generative_unconscious_map: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Serialization utilities
class DataModelJSONEncoder(json.JSONEncoder):
    """JSON encoder for data models"""
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def serialize_data_model(obj: Any) -> str:
    """Serialize data model to JSON string"""
    return json.dumps(obj, cls=DataModelJSONEncoder, indent=2)

def deserialize_data_model(data: str, model_class: type) -> Any:
    """Deserialize JSON string to data model"""
    parsed = json.loads(data)
    if hasattr(model_class, 'from_dict'):
        return model_class.from_dict(parsed)
    else:
        return model_class(**parsed)
