"""Query Analyzer

Slow query detection and query performance analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Pattern
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import re
import hashlib

from prometheus_client import Gauge, Counter, Histogram, Summary

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of SQL queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"
    OTHER = "other"


class QuerySeverity(str, Enum):
    """Severity levels for slow queries."""
    NORMAL = "normal"
    SLOW = "slow"
    VERY_SLOW = "very_slow"
    CRITICAL = "critical"


@dataclass
class QueryPlan:
    """Query execution plan."""
    plan_text: str
    estimated_cost: float
    actual_time_ms: float
    rows_estimated: int
    rows_actual: int
    shared_blocks_hit: int = 0
    shared_blocks_read: int = 0
    temp_blocks_written: int = 0
    planning_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    index_scans: int = 0
    sequential_scans: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        """Calculate buffer cache hit ratio."""
        total = self.shared_blocks_hit + self.shared_blocks_read
        return self.shared_blocks_hit / total if total > 0 else 0.0

    @property
    def row_estimate_accuracy(self) -> float:
        """How accurate was the row estimate."""
        if self.rows_actual == 0:
            return 1.0 if self.rows_estimated == 0 else 0.0
        return min(self.rows_estimated, self.rows_actual) / max(self.rows_estimated, self.rows_actual)


@dataclass
class QueryMetrics:
    """Metrics for a single query execution."""
    query_hash: str
    query_normalized: str
    query_type: QueryType
    duration_ms: float
    rows_affected: int
    timestamp: datetime = field(default_factory=datetime.now)
    database: str = ""
    user: str = ""
    application: str = ""
    plan: Optional[QueryPlan] = None
    error: Optional[str] = None

    @property
    def severity(self) -> QuerySeverity:
        """Determine query severity based on duration."""
        if self.duration_ms > 10000:
            return QuerySeverity.CRITICAL
        elif self.duration_ms > 5000:
            return QuerySeverity.VERY_SLOW
        elif self.duration_ms > 1000:
            return QuerySeverity.SLOW
        return QuerySeverity.NORMAL


@dataclass
class SlowQuery:
    """A slow query record."""
    query_hash: str
    query_normalized: str
    sample_query: str
    avg_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    execution_count: int
    total_time_ms: float
    first_seen: datetime
    last_seen: datetime
    severity: QuerySeverity
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QueryPattern:
    """A detected query pattern."""
    pattern_id: str
    pattern_regex: str
    query_type: QueryType
    sample_queries: List[str]
    avg_duration_ms: float
    execution_count: int
    tables_accessed: List[str]


class QueryAnalyzer:
    """Analyzes query performance and detects slow queries.

    Usage:
        analyzer = QueryAnalyzer(
            slow_threshold_ms=1000,
            very_slow_threshold_ms=5000,
        )

        # Record a query
        metrics = analyzer.record_query(
            query="SELECT * FROM users WHERE id = 123",
            duration_ms=50,
            rows_affected=1,
        )

        # Get slow queries
        slow = analyzer.get_slow_queries()

        # Get query statistics
        stats = analyzer.get_statistics()
    """

    def __init__(
        self,
        namespace: str = "consciousness",
        slow_threshold_ms: float = 1000,
        very_slow_threshold_ms: float = 5000,
        critical_threshold_ms: float = 10000,
        max_query_samples: int = 100,
    ):
        self.namespace = namespace
        self.slow_threshold_ms = slow_threshold_ms
        self.very_slow_threshold_ms = very_slow_threshold_ms
        self.critical_threshold_ms = critical_threshold_ms
        self.max_query_samples = max_query_samples

        self._queries: Dict[str, List[QueryMetrics]] = {}
        self._slow_callbacks: List[Callable[[SlowQuery], None]] = []
        self._lock = threading.Lock()

        # Prometheus metrics
        self.query_duration = Histogram(
            f"{namespace}_db_query_duration_seconds",
            "Query duration in seconds",
            ["query_type", "database"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.query_count = Counter(
            f"{namespace}_db_query_total",
            "Total queries executed",
            ["query_type", "database", "severity"],
        )

        self.query_errors = Counter(
            f"{namespace}_db_query_errors_total",
            "Total query errors",
            ["query_type", "database", "error_type"],
        )

        self.rows_processed = Counter(
            f"{namespace}_db_rows_processed_total",
            "Total rows processed",
            ["query_type", "operation"],
        )

        self.slow_queries_total = Gauge(
            f"{namespace}_db_slow_queries_total",
            "Number of slow queries",
            ["severity"],
        )

        self.cache_hit_ratio = Gauge(
            f"{namespace}_db_cache_hit_ratio",
            "Buffer cache hit ratio",
            ["database"],
        )

    def record_query(
        self,
        query: str,
        duration_ms: float,
        rows_affected: int = 0,
        database: str = "default",
        user: str = "",
        application: str = "",
        plan: Optional[QueryPlan] = None,
        error: Optional[str] = None,
    ) -> QueryMetrics:
        """Record a query execution.

        Args:
            query: SQL query (will be normalized)
            duration_ms: Query duration
            rows_affected: Number of rows affected
            database: Database name
            user: User who executed
            application: Application name
            plan: Query execution plan
            error: Error message if failed

        Returns:
            QueryMetrics for this execution
        """
        # Normalize and hash query
        normalized = self._normalize_query(query)
        query_hash = hashlib.md5(normalized.encode()).hexdigest()[:16]
        query_type = self._detect_query_type(query)

        metrics = QueryMetrics(
            query_hash=query_hash,
            query_normalized=normalized,
            query_type=query_type,
            duration_ms=duration_ms,
            rows_affected=rows_affected,
            database=database,
            user=user,
            application=application,
            plan=plan,
            error=error,
        )

        # Store query
        with self._lock:
            if query_hash not in self._queries:
                self._queries[query_hash] = []

            self._queries[query_hash].append(metrics)

            # Trim old samples
            if len(self._queries[query_hash]) > self.max_query_samples:
                self._queries[query_hash] = self._queries[query_hash][-self.max_query_samples:]

        # Update Prometheus metrics
        self.query_duration.labels(
            query_type=query_type.value,
            database=database,
        ).observe(duration_ms / 1000)

        self.query_count.labels(
            query_type=query_type.value,
            database=database,
            severity=metrics.severity.value,
        ).inc()

        if error:
            error_type = self._classify_error(error)
            self.query_errors.labels(
                query_type=query_type.value,
                database=database,
                error_type=error_type,
            ).inc()

        self.rows_processed.labels(
            query_type=query_type.value,
            operation="read" if query_type == QueryType.SELECT else "write",
        ).inc(rows_affected)

        if plan:
            self.cache_hit_ratio.labels(database=database).set(plan.cache_hit_ratio)

        # Check if slow and trigger callbacks
        if metrics.severity != QuerySeverity.NORMAL:
            slow_query = self._create_slow_query(query_hash, query)
            for callback in self._slow_callbacks:
                try:
                    callback(slow_query)
                except Exception as e:
                    logger.error(f"Slow query callback error: {e}")

        return metrics

    def _normalize_query(self, query: str) -> str:
        """Normalize a query by replacing literals with placeholders."""
        # Remove extra whitespace
        normalized = " ".join(query.split())

        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)

        # Replace numeric literals
        normalized = re.sub(r"\b\d+\b", "?", normalized)

        # Replace IN lists
        normalized = re.sub(r"IN\s*\([^)]+\)", "IN (?)", normalized, flags=re.IGNORECASE)

        # Lowercase keywords
        keywords = ["SELECT", "FROM", "WHERE", "AND", "OR", "INSERT", "UPDATE",
                   "DELETE", "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON",
                   "GROUP", "BY", "ORDER", "LIMIT", "OFFSET", "HAVING"]
        for kw in keywords:
            normalized = re.sub(rf"\b{kw}\b", kw, normalized, flags=re.IGNORECASE)

        return normalized

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of SQL query."""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif any(query_upper.startswith(kw) for kw in ["CREATE", "ALTER", "DROP", "TRUNCATE"]):
            return QueryType.DDL

        return QueryType.OTHER

    def _classify_error(self, error: str) -> str:
        """Classify error type."""
        error_lower = error.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "deadlock" in error_lower:
            return "deadlock"
        elif "constraint" in error_lower:
            return "constraint_violation"
        elif "connection" in error_lower:
            return "connection"
        elif "syntax" in error_lower:
            return "syntax"

        return "unknown"

    def _create_slow_query(self, query_hash: str, sample_query: str) -> SlowQuery:
        """Create a SlowQuery record from stored metrics."""
        with self._lock:
            metrics_list = self._queries.get(query_hash, [])

        if not metrics_list:
            return SlowQuery(
                query_hash=query_hash,
                query_normalized="",
                sample_query=sample_query,
                avg_duration_ms=0,
                max_duration_ms=0,
                min_duration_ms=0,
                execution_count=0,
                total_time_ms=0,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                severity=QuerySeverity.NORMAL,
            )

        durations = [m.duration_ms for m in metrics_list]
        avg_duration = sum(durations) / len(durations)

        # Determine severity
        if avg_duration > self.critical_threshold_ms:
            severity = QuerySeverity.CRITICAL
        elif avg_duration > self.very_slow_threshold_ms:
            severity = QuerySeverity.VERY_SLOW
        elif avg_duration > self.slow_threshold_ms:
            severity = QuerySeverity.SLOW
        else:
            severity = QuerySeverity.NORMAL

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_list)

        return SlowQuery(
            query_hash=query_hash,
            query_normalized=metrics_list[0].query_normalized,
            sample_query=sample_query,
            avg_duration_ms=avg_duration,
            max_duration_ms=max(durations),
            min_duration_ms=min(durations),
            execution_count=len(metrics_list),
            total_time_ms=sum(durations),
            first_seen=metrics_list[0].timestamp,
            last_seen=metrics_list[-1].timestamp,
            severity=severity,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, metrics_list: List[QueryMetrics]) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Check for plans
        plans = [m.plan for m in metrics_list if m.plan]

        if plans:
            avg_cache_hit = sum(p.cache_hit_ratio for p in plans) / len(plans)
            if avg_cache_hit < 0.9:
                recommendations.append("Low cache hit ratio - consider increasing shared_buffers")

            seq_scans = sum(p.sequential_scans for p in plans)
            idx_scans = sum(p.index_scans for p in plans)
            if seq_scans > idx_scans * 2:
                recommendations.append("High sequential scan ratio - consider adding indexes")

            temp_writes = sum(p.temp_blocks_written for p in plans)
            if temp_writes > 0:
                recommendations.append("Query uses temp storage - consider increasing work_mem")

            # Check row estimate accuracy
            avg_accuracy = sum(p.row_estimate_accuracy for p in plans) / len(plans)
            if avg_accuracy < 0.5:
                recommendations.append("Poor row estimates - consider running ANALYZE")

        # Check query patterns
        if metrics_list:
            query = metrics_list[0].query_normalized.upper()

            if "SELECT *" in query:
                recommendations.append("Avoid SELECT * - specify needed columns")

            if query.count("JOIN") > 3:
                recommendations.append("Many JOINs - consider query restructuring or materialized views")

            if "LIKE '%'" in query or "LIKE '%" in query:
                recommendations.append("Leading wildcard in LIKE - cannot use index efficiently")

            if "OR" in query and "WHERE" in query:
                recommendations.append("OR in WHERE clause - consider UNION or restructuring")

        return recommendations

    def on_slow_query(self, callback: Callable[[SlowQuery], None]):
        """Register a callback for slow queries.

        Args:
            callback: Function to call with SlowQuery
        """
        self._slow_callbacks.append(callback)

    def get_slow_queries(
        self,
        min_severity: QuerySeverity = QuerySeverity.SLOW,
        limit: int = 100,
    ) -> List[SlowQuery]:
        """Get slow queries.

        Args:
            min_severity: Minimum severity level
            limit: Maximum results

        Returns:
            List of slow queries
        """
        severity_order = [QuerySeverity.NORMAL, QuerySeverity.SLOW,
                        QuerySeverity.VERY_SLOW, QuerySeverity.CRITICAL]
        min_idx = severity_order.index(min_severity)

        slow_queries = []

        with self._lock:
            for query_hash, metrics_list in self._queries.items():
                if metrics_list:
                    durations = [m.duration_ms for m in metrics_list]
                    avg_duration = sum(durations) / len(durations)

                    if avg_duration > self.slow_threshold_ms:
                        sq = self._create_slow_query(query_hash, metrics_list[-1].query_normalized)

                        if severity_order.index(sq.severity) >= min_idx:
                            slow_queries.append(sq)

        # Sort by total time (impact)
        slow_queries.sort(key=lambda q: q.total_time_ms, reverse=True)

        # Update gauge
        by_severity = {}
        for sq in slow_queries:
            by_severity[sq.severity.value] = by_severity.get(sq.severity.value, 0) + 1

        for severity in QuerySeverity:
            self.slow_queries_total.labels(severity=severity.value).set(
                by_severity.get(severity.value, 0)
            )

        return slow_queries[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get query statistics.

        Returns:
            Statistics dictionary
        """
        with self._lock:
            total_queries = sum(len(m) for m in self._queries.values())
            unique_queries = len(self._queries)

            all_durations = []
            by_type = {}

            for metrics_list in self._queries.values():
                for m in metrics_list:
                    all_durations.append(m.duration_ms)
                    by_type[m.query_type.value] = by_type.get(m.query_type.value, 0) + 1

        if all_durations:
            return {
                "total_queries": total_queries,
                "unique_queries": unique_queries,
                "avg_duration_ms": sum(all_durations) / len(all_durations),
                "max_duration_ms": max(all_durations),
                "min_duration_ms": min(all_durations),
                "p50_duration_ms": sorted(all_durations)[len(all_durations) // 2],
                "p95_duration_ms": sorted(all_durations)[int(len(all_durations) * 0.95)],
                "p99_duration_ms": sorted(all_durations)[int(len(all_durations) * 0.99)],
                "by_type": by_type,
                "slow_query_count": len(self.get_slow_queries()),
            }

        return {
            "total_queries": 0,
            "unique_queries": 0,
            "by_type": {},
        }


class QueryPatternAnalyzer:
    """Analyzes query patterns for optimization opportunities.

    Usage:
        analyzer = QueryPatternAnalyzer()

        # Add queries
        analyzer.add_query("SELECT * FROM users WHERE id = 1")
        analyzer.add_query("SELECT * FROM users WHERE id = 2")

        # Get patterns
        patterns = analyzer.get_patterns()
    """

    def __init__(self):
        self._patterns: Dict[str, QueryPattern] = {}
        self._lock = threading.Lock()

    def add_query(
        self,
        query: str,
        duration_ms: float = 0,
    ):
        """Add a query for pattern analysis.

        Args:
            query: SQL query
            duration_ms: Execution duration
        """
        # Generate pattern key
        pattern_key = self._extract_pattern(query)

        with self._lock:
            if pattern_key not in self._patterns:
                tables = self._extract_tables(query)
                query_type = self._detect_type(query)

                self._patterns[pattern_key] = QueryPattern(
                    pattern_id=hashlib.md5(pattern_key.encode()).hexdigest()[:12],
                    pattern_regex=pattern_key,
                    query_type=query_type,
                    sample_queries=[],
                    avg_duration_ms=0,
                    execution_count=0,
                    tables_accessed=tables,
                )

            pattern = self._patterns[pattern_key]

            # Update pattern
            pattern.execution_count += 1
            pattern.avg_duration_ms = (
                (pattern.avg_duration_ms * (pattern.execution_count - 1) + duration_ms)
                / pattern.execution_count
            )

            # Keep sample queries
            if len(pattern.sample_queries) < 5:
                pattern.sample_queries.append(query[:500])

    def _extract_pattern(self, query: str) -> str:
        """Extract pattern from query."""
        # Normalize whitespace
        pattern = " ".join(query.split())

        # Replace literals
        pattern = re.sub(r"'[^']*'", "'?'", pattern)
        pattern = re.sub(r"\b\d+\b", "?", pattern)
        pattern = re.sub(r"IN\s*\([^)]+\)", "IN (?)", pattern, flags=re.IGNORECASE)

        return pattern

    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        tables = []

        # FROM clause
        from_match = re.search(r"FROM\s+([^\s,;()]+)", query, re.IGNORECASE)
        if from_match:
            tables.append(from_match.group(1))

        # JOIN clauses
        join_matches = re.findall(r"JOIN\s+([^\s,;()]+)", query, re.IGNORECASE)
        tables.extend(join_matches)

        # INSERT INTO
        insert_match = re.search(r"INSERT\s+INTO\s+([^\s(]+)", query, re.IGNORECASE)
        if insert_match:
            tables.append(insert_match.group(1))

        # UPDATE
        update_match = re.search(r"UPDATE\s+([^\s]+)", query, re.IGNORECASE)
        if update_match:
            tables.append(update_match.group(1))

        return list(set(tables))

    def _detect_type(self, query: str) -> QueryType:
        """Detect query type."""
        query_upper = query.strip().upper()

        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE

        return QueryType.OTHER

    def get_patterns(
        self,
        min_executions: int = 10,
        order_by: str = "total_time",
    ) -> List[QueryPattern]:
        """Get discovered query patterns.

        Args:
            min_executions: Minimum executions to include
            order_by: Sort field (total_time, count, avg_duration)

        Returns:
            List of query patterns
        """
        with self._lock:
            patterns = [
                p for p in self._patterns.values()
                if p.execution_count >= min_executions
            ]

        if order_by == "total_time":
            patterns.sort(
                key=lambda p: p.avg_duration_ms * p.execution_count,
                reverse=True
            )
        elif order_by == "count":
            patterns.sort(key=lambda p: p.execution_count, reverse=True)
        elif order_by == "avg_duration":
            patterns.sort(key=lambda p: p.avg_duration_ms, reverse=True)

        return patterns

    def get_table_hotspots(self) -> Dict[str, Dict[str, Any]]:
        """Get tables with most query activity.

        Returns:
            Table hotspot analysis
        """
        table_stats: Dict[str, Dict[str, Any]] = {}

        with self._lock:
            for pattern in self._patterns.values():
                for table in pattern.tables_accessed:
                    if table not in table_stats:
                        table_stats[table] = {
                            "total_queries": 0,
                            "total_time_ms": 0,
                            "query_types": {},
                        }

                    stats = table_stats[table]
                    stats["total_queries"] += pattern.execution_count
                    stats["total_time_ms"] += pattern.avg_duration_ms * pattern.execution_count

                    qt = pattern.query_type.value
                    stats["query_types"][qt] = stats["query_types"].get(qt, 0) + pattern.execution_count

        return table_stats
