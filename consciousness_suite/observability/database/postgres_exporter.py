"""PostgreSQL Exporter

PostgreSQL-specific metrics collection and export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading
import logging
import asyncio

from prometheus_client import Gauge, Counter, Info

logger = logging.getLogger(__name__)


class ReplicationState(str, Enum):
    """PostgreSQL replication states."""
    STREAMING = "streaming"
    CATCHUP = "catchup"
    SYNC = "sync"
    BACKUP = "backup"
    DISCONNECTED = "disconnected"


@dataclass
class TableStats:
    """Statistics for a database table."""
    schema_name: str
    table_name: str
    # Size
    total_bytes: int = 0
    table_bytes: int = 0
    index_bytes: int = 0
    toast_bytes: int = 0
    # Activity
    seq_scan: int = 0
    seq_tup_read: int = 0
    idx_scan: int = 0
    idx_tup_fetch: int = 0
    n_tup_ins: int = 0
    n_tup_upd: int = 0
    n_tup_del: int = 0
    n_live_tup: int = 0
    n_dead_tup: int = 0
    # Maintenance
    last_vacuum: Optional[datetime] = None
    last_autovacuum: Optional[datetime] = None
    last_analyze: Optional[datetime] = None
    last_autoanalyze: Optional[datetime] = None
    vacuum_count: int = 0
    autovacuum_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def dead_tuple_ratio(self) -> float:
        """Ratio of dead tuples to total tuples."""
        total = self.n_live_tup + self.n_dead_tup
        return self.n_dead_tup / total if total > 0 else 0.0

    @property
    def index_usage_ratio(self) -> float:
        """Ratio of index scans to total scans."""
        total = self.seq_scan + self.idx_scan
        return self.idx_scan / total if total > 0 else 0.0


@dataclass
class IndexStats:
    """Statistics for a database index."""
    schema_name: str
    table_name: str
    index_name: str
    index_type: str
    index_size_bytes: int = 0
    idx_scan: int = 0
    idx_tup_read: int = 0
    idx_tup_fetch: int = 0
    is_unique: bool = False
    is_primary: bool = False
    is_valid: bool = True
    definition: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def full_name(self) -> str:
        return f"{self.schema_name}.{self.index_name}"

    @property
    def is_unused(self) -> bool:
        """Check if index is potentially unused."""
        return self.idx_scan == 0 and not self.is_primary and not self.is_unique


@dataclass
class ReplicationStatus:
    """PostgreSQL replication status."""
    slot_name: str
    client_addr: Optional[str]
    state: ReplicationState
    sent_lsn: str = ""
    write_lsn: str = ""
    flush_lsn: str = ""
    replay_lsn: str = ""
    write_lag_bytes: int = 0
    flush_lag_bytes: int = 0
    replay_lag_bytes: int = 0
    write_lag_seconds: float = 0
    flush_lag_seconds: float = 0
    replay_lag_seconds: float = 0
    sync_state: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PostgresMetrics:
    """PostgreSQL database metrics snapshot."""
    database: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Connection stats
    active_connections: int = 0
    idle_connections: int = 0
    idle_in_transaction: int = 0
    max_connections: int = 0
    reserved_connections: int = 0

    # Database size
    database_size_bytes: int = 0

    # Transaction stats
    xact_commit: int = 0
    xact_rollback: int = 0
    blks_read: int = 0
    blks_hit: int = 0

    # Tuple stats
    tup_returned: int = 0
    tup_fetched: int = 0
    tup_inserted: int = 0
    tup_updated: int = 0
    tup_deleted: int = 0

    # Conflicts
    conflicts: int = 0
    deadlocks: int = 0

    # Temp files
    temp_files: int = 0
    temp_bytes: int = 0

    # Checkpoints
    checkpoints_timed: int = 0
    checkpoints_req: int = 0
    checkpoint_write_time_ms: float = 0
    checkpoint_sync_time_ms: float = 0

    # WAL
    wal_records: int = 0
    wal_bytes: int = 0

    # Autovacuum
    autovacuum_workers: int = 0
    tables_need_vacuum: int = 0
    tables_need_analyze: int = 0

    @property
    def cache_hit_ratio(self) -> float:
        """Buffer cache hit ratio."""
        total = self.blks_read + self.blks_hit
        return self.blks_hit / total if total > 0 else 0.0

    @property
    def commit_ratio(self) -> float:
        """Transaction commit ratio."""
        total = self.xact_commit + self.xact_rollback
        return self.xact_commit / total if total > 0 else 1.0

    @property
    def connection_utilization(self) -> float:
        """Connection utilization."""
        available = self.max_connections - self.reserved_connections
        total = self.active_connections + self.idle_connections + self.idle_in_transaction
        return total / available if available > 0 else 0.0


class PostgresExporter:
    """Exports PostgreSQL metrics to Prometheus.

    Usage:
        exporter = PostgresExporter(
            namespace="consciousness",
        )

        # Update metrics from database stats
        exporter.update_database_metrics(metrics)
        exporter.update_table_stats(table_stats)
        exporter.update_replication_status(replication)

        # Or fetch directly (requires connection)
        await exporter.collect_metrics(connection)
    """

    def __init__(self, namespace: str = "consciousness"):
        self.namespace = namespace

        self._databases: Dict[str, PostgresMetrics] = {}
        self._tables: Dict[str, TableStats] = {}
        self._indexes: Dict[str, IndexStats] = {}
        self._replication: Dict[str, ReplicationStatus] = {}
        self._lock = threading.Lock()

        # Database info
        self.pg_info = Info(
            f"{namespace}_pg",
            "PostgreSQL server information",
        )

        # Connection metrics
        self.pg_connections = Gauge(
            f"{namespace}_pg_connections",
            "PostgreSQL connections",
            ["database", "state"],
        )

        self.pg_max_connections = Gauge(
            f"{namespace}_pg_max_connections",
            "Maximum connections",
        )

        self.pg_connection_utilization = Gauge(
            f"{namespace}_pg_connection_utilization_percent",
            "Connection utilization percentage",
            ["database"],
        )

        # Database size
        self.pg_database_size = Gauge(
            f"{namespace}_pg_database_size_bytes",
            "Database size in bytes",
            ["database"],
        )

        # Transaction metrics
        self.pg_transactions = Counter(
            f"{namespace}_pg_transactions_total",
            "Total transactions",
            ["database", "type"],
        )

        self.pg_commit_ratio = Gauge(
            f"{namespace}_pg_commit_ratio",
            "Transaction commit ratio",
            ["database"],
        )

        # Buffer cache
        self.pg_buffer_cache_hit_ratio = Gauge(
            f"{namespace}_pg_buffer_cache_hit_ratio",
            "Buffer cache hit ratio",
            ["database"],
        )

        self.pg_blocks = Counter(
            f"{namespace}_pg_blocks_total",
            "Blocks read/hit",
            ["database", "type"],
        )

        # Tuple metrics
        self.pg_tuples = Counter(
            f"{namespace}_pg_tuples_total",
            "Tuples processed",
            ["database", "operation"],
        )

        # Lock metrics
        self.pg_deadlocks = Counter(
            f"{namespace}_pg_deadlocks_total",
            "Total deadlocks",
            ["database"],
        )

        # Temp files
        self.pg_temp_files = Counter(
            f"{namespace}_pg_temp_files_total",
            "Temp files created",
            ["database"],
        )

        self.pg_temp_bytes = Counter(
            f"{namespace}_pg_temp_bytes_total",
            "Temp file bytes written",
            ["database"],
        )

        # Checkpoint metrics
        self.pg_checkpoints = Counter(
            f"{namespace}_pg_checkpoints_total",
            "Checkpoints performed",
            ["type"],
        )

        # Table metrics
        self.pg_table_size = Gauge(
            f"{namespace}_pg_table_size_bytes",
            "Table size in bytes",
            ["schema", "table", "type"],
        )

        self.pg_table_rows = Gauge(
            f"{namespace}_pg_table_rows",
            "Estimated row counts",
            ["schema", "table", "state"],
        )

        self.pg_table_scans = Counter(
            f"{namespace}_pg_table_scans_total",
            "Table scan counts",
            ["schema", "table", "type"],
        )

        self.pg_table_dead_tuple_ratio = Gauge(
            f"{namespace}_pg_table_dead_tuple_ratio",
            "Dead tuple ratio",
            ["schema", "table"],
        )

        self.pg_table_index_usage = Gauge(
            f"{namespace}_pg_table_index_usage_ratio",
            "Index usage ratio",
            ["schema", "table"],
        )

        # Index metrics
        self.pg_index_size = Gauge(
            f"{namespace}_pg_index_size_bytes",
            "Index size in bytes",
            ["schema", "table", "index"],
        )

        self.pg_index_scans = Counter(
            f"{namespace}_pg_index_scans_total",
            "Index scan counts",
            ["schema", "table", "index"],
        )

        self.pg_unused_indexes = Gauge(
            f"{namespace}_pg_unused_indexes_count",
            "Count of potentially unused indexes",
            ["schema"],
        )

        # Replication metrics
        self.pg_replication_lag_bytes = Gauge(
            f"{namespace}_pg_replication_lag_bytes",
            "Replication lag in bytes",
            ["slot", "type"],
        )

        self.pg_replication_lag_seconds = Gauge(
            f"{namespace}_pg_replication_lag_seconds",
            "Replication lag in seconds",
            ["slot", "type"],
        )

        self.pg_replication_state = Gauge(
            f"{namespace}_pg_replication_state",
            "Replication state (1=streaming, 0=other)",
            ["slot"],
        )

        # Autovacuum metrics
        self.pg_autovacuum_workers = Gauge(
            f"{namespace}_pg_autovacuum_workers",
            "Active autovacuum workers",
        )

        self.pg_tables_need_vacuum = Gauge(
            f"{namespace}_pg_tables_need_vacuum",
            "Tables needing vacuum",
        )

        self.pg_tables_need_analyze = Gauge(
            f"{namespace}_pg_tables_need_analyze",
            "Tables needing analyze",
        )

    def update_server_info(
        self,
        version: str,
        server_encoding: str = "UTF8",
        is_in_recovery: bool = False,
    ):
        """Update PostgreSQL server info.

        Args:
            version: PostgreSQL version
            server_encoding: Server encoding
            is_in_recovery: Whether server is in recovery
        """
        self.pg_info.info({
            "version": version,
            "encoding": server_encoding,
            "in_recovery": str(is_in_recovery).lower(),
        })

    def update_database_metrics(self, metrics: PostgresMetrics):
        """Update database-level metrics.

        Args:
            metrics: PostgresMetrics snapshot
        """
        db = metrics.database

        with self._lock:
            self._databases[db] = metrics

        # Connections
        self.pg_connections.labels(database=db, state="active").set(metrics.active_connections)
        self.pg_connections.labels(database=db, state="idle").set(metrics.idle_connections)
        self.pg_connections.labels(database=db, state="idle_in_transaction").set(metrics.idle_in_transaction)
        self.pg_max_connections.set(metrics.max_connections)
        self.pg_connection_utilization.labels(database=db).set(metrics.connection_utilization * 100)

        # Database size
        self.pg_database_size.labels(database=db).set(metrics.database_size_bytes)

        # Transactions
        self.pg_commit_ratio.labels(database=db).set(metrics.commit_ratio)

        # Buffer cache
        self.pg_buffer_cache_hit_ratio.labels(database=db).set(metrics.cache_hit_ratio)

        # Deadlocks
        # Note: These are counters so we should use .inc() but we don't have deltas
        # In practice, you'd track deltas between calls

        # Autovacuum
        self.pg_autovacuum_workers.set(metrics.autovacuum_workers)
        self.pg_tables_need_vacuum.set(metrics.tables_need_vacuum)
        self.pg_tables_need_analyze.set(metrics.tables_need_analyze)

    def update_table_stats(self, stats: TableStats):
        """Update table statistics.

        Args:
            stats: TableStats for a table
        """
        with self._lock:
            self._tables[stats.full_name] = stats

        schema = stats.schema_name
        table = stats.table_name

        # Sizes
        self.pg_table_size.labels(schema=schema, table=table, type="total").set(stats.total_bytes)
        self.pg_table_size.labels(schema=schema, table=table, type="table").set(stats.table_bytes)
        self.pg_table_size.labels(schema=schema, table=table, type="index").set(stats.index_bytes)
        self.pg_table_size.labels(schema=schema, table=table, type="toast").set(stats.toast_bytes)

        # Row counts
        self.pg_table_rows.labels(schema=schema, table=table, state="live").set(stats.n_live_tup)
        self.pg_table_rows.labels(schema=schema, table=table, state="dead").set(stats.n_dead_tup)

        # Dead tuple ratio
        self.pg_table_dead_tuple_ratio.labels(schema=schema, table=table).set(stats.dead_tuple_ratio)

        # Index usage
        self.pg_table_index_usage.labels(schema=schema, table=table).set(stats.index_usage_ratio)

    def update_index_stats(self, stats: IndexStats):
        """Update index statistics.

        Args:
            stats: IndexStats for an index
        """
        with self._lock:
            self._indexes[stats.full_name] = stats

        schema = stats.schema_name
        table = stats.table_name
        index = stats.index_name

        self.pg_index_size.labels(schema=schema, table=table, index=index).set(stats.index_size_bytes)

        # Calculate unused indexes per schema
        unused_count = sum(
            1 for idx in self._indexes.values()
            if idx.schema_name == schema and idx.is_unused
        )
        self.pg_unused_indexes.labels(schema=schema).set(unused_count)

    def update_replication_status(self, status: ReplicationStatus):
        """Update replication status.

        Args:
            status: ReplicationStatus
        """
        with self._lock:
            self._replication[status.slot_name] = status

        slot = status.slot_name

        # Lag in bytes
        self.pg_replication_lag_bytes.labels(slot=slot, type="write").set(status.write_lag_bytes)
        self.pg_replication_lag_bytes.labels(slot=slot, type="flush").set(status.flush_lag_bytes)
        self.pg_replication_lag_bytes.labels(slot=slot, type="replay").set(status.replay_lag_bytes)

        # Lag in seconds
        self.pg_replication_lag_seconds.labels(slot=slot, type="write").set(status.write_lag_seconds)
        self.pg_replication_lag_seconds.labels(slot=slot, type="flush").set(status.flush_lag_seconds)
        self.pg_replication_lag_seconds.labels(slot=slot, type="replay").set(status.replay_lag_seconds)

        # State
        self.pg_replication_state.labels(slot=slot).set(
            1 if status.state == ReplicationState.STREAMING else 0
        )

    def get_health_issues(self) -> List[Dict[str, Any]]:
        """Get health issues based on metrics.

        Returns:
            List of health issues
        """
        issues = []

        with self._lock:
            databases = dict(self._databases)
            tables = dict(self._tables)
            indexes = dict(self._indexes)
            replication = dict(self._replication)

        # Check database metrics
        for db_name, metrics in databases.items():
            # Cache hit ratio
            if metrics.cache_hit_ratio < 0.95:
                issues.append({
                    "severity": "warning" if metrics.cache_hit_ratio >= 0.9 else "critical",
                    "database": db_name,
                    "issue": f"Low cache hit ratio: {metrics.cache_hit_ratio:.2%}",
                    "recommendation": "Increase shared_buffers or optimize queries",
                })

            # Connection utilization
            if metrics.connection_utilization > 0.8:
                issues.append({
                    "severity": "warning" if metrics.connection_utilization < 0.9 else "critical",
                    "database": db_name,
                    "issue": f"High connection utilization: {metrics.connection_utilization:.1%}",
                    "recommendation": "Consider connection pooling or increasing max_connections",
                })

            # Commit ratio
            if metrics.commit_ratio < 0.95:
                issues.append({
                    "severity": "warning",
                    "database": db_name,
                    "issue": f"High rollback rate: {1 - metrics.commit_ratio:.2%}",
                    "recommendation": "Investigate application transaction handling",
                })

        # Check table stats
        for table_name, stats in tables.items():
            # Dead tuples
            if stats.dead_tuple_ratio > 0.2:
                issues.append({
                    "severity": "warning",
                    "table": table_name,
                    "issue": f"High dead tuple ratio: {stats.dead_tuple_ratio:.1%}",
                    "recommendation": "Consider manual VACUUM or tune autovacuum",
                })

            # Sequential scans on large tables
            if stats.n_live_tup > 10000 and stats.index_usage_ratio < 0.5:
                issues.append({
                    "severity": "warning",
                    "table": table_name,
                    "issue": f"Low index usage: {stats.index_usage_ratio:.1%}",
                    "recommendation": "Add indexes for frequently queried columns",
                })

        # Check indexes
        for idx_name, idx_stats in indexes.items():
            if idx_stats.is_unused and idx_stats.index_size_bytes > 10 * 1024 * 1024:  # >10MB
                issues.append({
                    "severity": "info",
                    "index": idx_name,
                    "issue": f"Potentially unused index ({idx_stats.index_size_bytes / 1024 / 1024:.1f}MB)",
                    "recommendation": "Consider removing unused index to save space",
                })

        # Check replication
        for slot_name, rep_status in replication.items():
            if rep_status.state != ReplicationState.STREAMING:
                issues.append({
                    "severity": "critical",
                    "slot": slot_name,
                    "issue": f"Replication not streaming: {rep_status.state.value}",
                    "recommendation": "Check replica connectivity",
                })

            if rep_status.replay_lag_seconds > 60:
                issues.append({
                    "severity": "warning" if rep_status.replay_lag_seconds < 300 else "critical",
                    "slot": slot_name,
                    "issue": f"High replication lag: {rep_status.replay_lag_seconds:.1f}s",
                    "recommendation": "Check replica performance or network",
                })

        return issues

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary.

        Returns:
            Summary dictionary
        """
        with self._lock:
            databases = dict(self._databases)
            tables = dict(self._tables)
            indexes = dict(self._indexes)
            replication = dict(self._replication)

        total_size = sum(m.database_size_bytes for m in databases.values())
        total_connections = sum(
            m.active_connections + m.idle_connections
            for m in databases.values()
        )

        unused_indexes = [idx for idx in indexes.values() if idx.is_unused]
        unused_index_size = sum(idx.index_size_bytes for idx in unused_indexes)

        issues = self.get_health_issues()
        critical_issues = [i for i in issues if i.get("severity") == "critical"]

        return {
            "databases": len(databases),
            "total_size_bytes": total_size,
            "total_connections": total_connections,
            "tables_monitored": len(tables),
            "indexes_monitored": len(indexes),
            "unused_indexes": len(unused_indexes),
            "unused_index_size_bytes": unused_index_size,
            "replication_slots": len(replication),
            "health_issues": len(issues),
            "critical_issues": len(critical_issues),
            "avg_cache_hit_ratio": (
                sum(m.cache_hit_ratio for m in databases.values()) / len(databases)
                if databases else 0
            ),
        }

    async def collect_from_connection(
        self,
        connection,
        database: str = "postgres",
    ):
        """Collect metrics directly from a database connection.

        Args:
            connection: asyncpg or psycopg connection
            database: Database name
        """
        # This is a template - actual implementation depends on connection type
        # Example for asyncpg:
        try:
            # Database stats
            row = await connection.fetchrow("""
                SELECT
                    numbackends as active_connections,
                    xact_commit,
                    xact_rollback,
                    blks_read,
                    blks_hit,
                    tup_returned,
                    tup_fetched,
                    tup_inserted,
                    tup_updated,
                    tup_deleted,
                    conflicts,
                    deadlocks,
                    temp_files,
                    temp_bytes
                FROM pg_stat_database
                WHERE datname = $1
            """, database)

            if row:
                metrics = PostgresMetrics(
                    database=database,
                    active_connections=row["active_connections"] or 0,
                    xact_commit=row["xact_commit"] or 0,
                    xact_rollback=row["xact_rollback"] or 0,
                    blks_read=row["blks_read"] or 0,
                    blks_hit=row["blks_hit"] or 0,
                    tup_returned=row["tup_returned"] or 0,
                    tup_fetched=row["tup_fetched"] or 0,
                    tup_inserted=row["tup_inserted"] or 0,
                    tup_updated=row["tup_updated"] or 0,
                    tup_deleted=row["tup_deleted"] or 0,
                    conflicts=row["conflicts"] or 0,
                    deadlocks=row["deadlocks"] or 0,
                    temp_files=row["temp_files"] or 0,
                    temp_bytes=row["temp_bytes"] or 0,
                )
                self.update_database_metrics(metrics)

        except Exception as e:
            logger.error(f"Failed to collect PostgreSQL metrics: {e}")
