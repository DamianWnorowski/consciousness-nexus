#!/usr/bin/env python3
"""
📊 RESOURCE QUOTAS & MONITORING SYSTEM
======================================

Prevent resource exhaustion during evolution with intelligent quotas and monitoring.
"""

import asyncio
import json
import time
import os
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

class ResourceType(Enum):
    """Types of resources to monitor and quota"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    GPU = "gpu"
    DATABASE_CONNECTIONS = "db_connections"
    FILE_HANDLES = "file_handles"
    THREADS = "threads"
    PROCESSES = "processes"

class QuotaAction(Enum):
    """Actions to take when quota is exceeded"""
    WARN = "warn"
    THROTTLE = "throttle"
    BLOCK = "block"
    TERMINATE = "terminate"
    ROLLBACK = "rollback"

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class ResourceQuota:
    """Quota definition for a resource"""
    resource_type: ResourceType
    max_usage: float
    warning_threshold: float = 0.8  # 80% of max
    critical_threshold: float = 0.95  # 95% of max
    action: QuotaAction = QuotaAction.WARN
    time_window: int = 60  # seconds
    burst_allowance: float = 0.1  # 10% burst allowance

@dataclass
class ResourceUsage:
    """Current resource usage"""
    resource_type: ResourceType
    current_usage: float
    max_usage: float
    usage_percentage: float
    timestamp: float
    trend: str = "stable"  # increasing, decreasing, stable

@dataclass
class ResourceAlert:
    """Resource usage alert"""
    resource_type: ResourceType
    severity: AlertSeverity
    message: str
    current_usage: float
    threshold: float
    timestamp: float
    recommended_action: str

@dataclass
class EvolutionResourceProfile:
    """Resource profile for evolution operations"""
    operation_type: str
    expected_resources: Dict[ResourceType, float]
    max_duration: int = 300  # 5 minutes
    priority: int = 1  # 1-5, higher is more important
    can_be_interrupted: bool = True

class ResourceQuotaManager:
    """Intelligent resource quota and monitoring system"""

    def __init__(self, config_file: str = "resource_quotas.json"):
        self.config_file = config_file
        self.quotas: Dict[ResourceType, ResourceQuota] = {}
        self.current_usage: Dict[ResourceType, List[ResourceUsage]] = {}
        self.active_alerts: List[ResourceAlert] = []
        self.monitoring_active = False
        self.evolution_profiles = self._load_evolution_profiles()

        # Setup logging
        self.logger = logging.getLogger('ResourceQuotaManager')
        self.logger.setLevel(logging.INFO)

        handler = logging.FileHandler('logs/resource_quotas.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)

        # Load configuration
        self._load_config()

        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitoring_worker, daemon=True)
        self.monitor_thread.start()

    def _load_config(self):
        """Load resource quota configuration"""
        default_quotas = {
            ResourceType.CPU: ResourceQuota(
                resource_type=ResourceType.CPU,
                max_usage=80.0,  # 80% CPU
                warning_threshold=0.7,
                critical_threshold=0.9,
                action=QuotaAction.THROTTLE
            ),
            ResourceType.MEMORY: ResourceQuota(
                resource_type=ResourceType.MEMORY,
                max_usage=85.0,  # 85% memory
                warning_threshold=0.75,
                critical_threshold=0.95,
                action=QuotaAction.BLOCK
            ),
            ResourceType.DISK: ResourceQuota(
                resource_type=ResourceType.DISK,
                max_usage=90.0,  # 90% disk
                warning_threshold=0.8,
                critical_threshold=0.95,
                action=QuotaAction.WARN
            ),
            ResourceType.DATABASE_CONNECTIONS: ResourceQuota(
                resource_type=ResourceType.DATABASE_CONNECTIONS,
                max_usage=100,  # 100 connections
                warning_threshold=80,
                critical_threshold=95,
                action=QuotaAction.THROTTLE
            ),
            ResourceType.THREADS: ResourceQuota(
                resource_type=ResourceType.THREADS,
                max_usage=50,  # 50 threads
                warning_threshold=40,
                critical_threshold=45,
                action=QuotaAction.BLOCK
            )
        }

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    quotas_config = config.get('quotas', {})

                    for res_type_str, quota_config in quotas_config.items():
                        res_type = ResourceType(res_type_str)
                        quota = ResourceQuota(
                            resource_type=res_type,
                            max_usage=quota_config['max_usage'],
                            warning_threshold=quota_config.get('warning_threshold', 0.8),
                            critical_threshold=quota_config.get('critical_threshold', 0.95),
                            action=QuotaAction(quota_config.get('action', 'warn')),
                            time_window=quota_config.get('time_window', 60),
                            burst_allowance=quota_config.get('burst_allowance', 0.1)
                        )
                        default_quotas[res_type] = quota

            except Exception as e:
                self.logger.error(f"Failed to load config: {e}")

        self.quotas = default_quotas

        # Initialize usage tracking
        for res_type in self.quotas.keys():
            self.current_usage[res_type] = []

    def _load_evolution_profiles(self) -> Dict[str, EvolutionResourceProfile]:
        """Load evolution operation resource profiles"""
        return {
            'auto_evolve': EvolutionResourceProfile(
                operation_type='auto_evolve',
                expected_resources={
                    ResourceType.CPU: 60.0,
                    ResourceType.MEMORY: 70.0,
                    ResourceType.DATABASE_CONNECTIONS: 5
                },
                max_duration=600,  # 10 minutes
                priority=3,
                can_be_interrupted=False
            ),
            'chain_orchestration': EvolutionResourceProfile(
                operation_type='chain_orchestration',
                expected_resources={
                    ResourceType.CPU: 40.0,
                    ResourceType.MEMORY: 50.0,
                    ResourceType.DATABASE_CONNECTIONS: 3
                },
                max_duration=300,
                priority=2,
                can_be_interrupted=True
            ),
            'validation_run': EvolutionResourceProfile(
                operation_type='validation_run',
                expected_resources={
                    ResourceType.CPU: 30.0,
                    ResourceType.MEMORY: 40.0
                },
                max_duration=180,
                priority=1,
                can_be_interrupted=True
            ),
            'backup_operation': EvolutionResourceProfile(
                operation_type='backup_operation',
                expected_resources={
                    ResourceType.DISK: 50.0,
                    ResourceType.CPU: 20.0
                },
                max_duration=120,
                priority=4,
                can_be_interrupted=False
            )
        }

    def _monitoring_worker(self):
        """Background monitoring thread"""
        self.monitoring_active = True

        while self.monitoring_active:
            try:
                # Collect current resource usage
                self._collect_resource_usage()

                # Check quotas and generate alerts
                self._check_quotas()

                # Clean old usage data
                self._cleanup_old_data()

                time.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(5)

    def _collect_resource_usage(self):
        """Collect current resource usage"""
        timestamp = time.time()

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_usage(ResourceType.CPU, cpu_percent, timestamp)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self._record_usage(ResourceType.MEMORY, memory_percent, timestamp)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self._record_usage(ResourceType.DISK, disk_percent, timestamp)

            # Thread count
            thread_count = threading.active_count()
            self._record_usage(ResourceType.THREADS, thread_count, timestamp)

            # Process count
            process_count = len(psutil.pids())
            self._record_usage(ResourceType.PROCESSES, process_count, timestamp)

        except Exception as e:
            self.logger.error(f"Resource collection error: {e}")

    def _record_usage(self, resource_type: ResourceType, current_usage: float, timestamp: float):
        """Record resource usage data"""
        quota = self.quotas.get(resource_type)
        if not quota:
            return

        max_usage = quota.max_usage
        usage_percentage = (current_usage / max_usage) * 100 if max_usage > 0 else 0

        # Calculate trend
        trend = "stable"
        usage_history = self.current_usage.get(resource_type, [])
        if len(usage_history) >= 3:
            recent = [u.usage_percentage for u in usage_history[-3:]]
            if recent[-1] > recent[0] + 5:
                trend = "increasing"
            elif recent[-1] < recent[0] - 5:
                trend = "decreasing"

        usage = ResourceUsage(
            resource_type=resource_type,
            current_usage=current_usage,
            max_usage=max_usage,
            usage_percentage=usage_percentage,
            timestamp=timestamp,
            trend=trend
        )

        self.current_usage[resource_type].append(usage)

        # Keep only last 100 readings
        if len(self.current_usage[resource_type]) > 100:
            self.current_usage[resource_type] = self.current_usage[resource_type][-100:]

    def _check_quotas(self):
        """Check resource quotas and generate alerts"""
        for resource_type, quota in self.quotas.items():
            usage_history = self.current_usage.get(resource_type, [])
            if not usage_history:
                continue

            current_usage = usage_history[-1]

            # Check warning threshold
            if current_usage.usage_percentage >= quota.warning_threshold * 100:
                if current_usage.usage_percentage >= quota.critical_threshold * 100:
                    severity = AlertSeverity.CRITICAL
                    message = f"CRITICAL: {resource_type.value} usage at {current_usage.usage_percentage:.1f}% (limit: {quota.critical_threshold*100:.1f}%)"
                    action = self._get_recommended_action(quota, AlertSeverity.CRITICAL)
                else:
                    severity = AlertSeverity.WARNING
                    message = f"WARNING: {resource_type.value} usage at {current_usage.usage_percentage:.1f}% (limit: {quota.warning_threshold*100:.1f}%)"
                    action = "Monitor closely"

                alert = ResourceAlert(
                    resource_type=resource_type,
                    severity=severity,
                    message=message,
                    current_usage=current_usage.usage_percentage,
                    threshold=quota.critical_threshold * 100 if severity == AlertSeverity.CRITICAL else quota.warning_threshold * 100,
                    timestamp=time.time(),
                    recommended_action=action
                )

                # Only add if not already active
                if not self._is_alert_active(alert):
                    self.active_alerts.append(alert)
                    self.logger.warning(f"RESOURCE ALERT: {message}")
                    self._handle_quota_exceedance(quota, alert)

    def _is_alert_active(self, alert: ResourceAlert) -> bool:
        """Check if similar alert is already active"""
        for active_alert in self.active_alerts:
            if (active_alert.resource_type == alert.resource_type and
                active_alert.severity == alert.severity):
                # Update timestamp
                active_alert.timestamp = alert.timestamp
                return True
        return False

    def _get_recommended_action(self, quota: ResourceQuota, severity: AlertSeverity) -> str:
        """Get recommended action for quota exceedance"""
        if severity == AlertSeverity.CRITICAL:
            action_map = {
                QuotaAction.WARN: "Immediate attention required",
                QuotaAction.THROTTLE: "Reduce operation intensity",
                QuotaAction.BLOCK: "Stop current operations",
                QuotaAction.TERMINATE: "Terminate problematic processes",
                QuotaAction.ROLLBACK: "Rollback recent changes"
            }
        else:
            action_map = {
                QuotaAction.WARN: "Monitor resource usage",
                QuotaAction.THROTTLE: "Consider reducing load",
                QuotaAction.BLOCK: "Prepare contingency plans",
                QuotaAction.TERMINATE: "Identify resource-intensive processes",
                QuotaAction.ROLLBACK: "Review recent deployments"
            }

        return action_map.get(quota.action, "Investigate resource usage")

    def _handle_quota_exceedance(self, quota: ResourceQuota, alert: ResourceAlert):
        """Handle quota exceedance based on configured action"""
        if alert.severity != AlertSeverity.CRITICAL:
            return

        action = quota.action

        if action == QuotaAction.THROTTLE:
            self._throttle_operations(alert.resource_type)
        elif action == QuotaAction.BLOCK:
            self._block_operations(alert.resource_type)
        elif action == QuotaAction.TERMINATE:
            self._terminate_processes(alert.resource_type)
        elif action == QuotaAction.ROLLBACK:
            self._trigger_rollback(alert.resource_type)

    def _throttle_operations(self, resource_type: ResourceType):
        """Throttle operations to reduce resource usage"""
        self.logger.info(f"THROTTLING operations for {resource_type.value}")

        # Implementation would depend on the specific resource
        # For CPU: reduce thread priority, add delays
        # For memory: trigger garbage collection
        # For network: reduce request rate

        print(f"[!] Throttling operations due to {resource_type.value} usage")

    def _block_operations(self, resource_type: ResourceType):
        """Block new operations"""
        self.logger.warning(f"BLOCKING new operations for {resource_type.value}")
        print(f"[!] Blocking new operations due to {resource_type.value} usage")

    def _terminate_processes(self, resource_type: ResourceType):
        """Terminate resource-intensive processes"""
        self.logger.error(f"TERMINATING processes for {resource_type.value}")
        print(f"[!] Terminating resource-intensive processes for {resource_type.value}")

    def _trigger_rollback(self, resource_type: ResourceType):
        """Trigger rollback of recent changes"""
        self.logger.critical(f"TRIGGERING ROLLBACK for {resource_type.value}")
        print(f"[!] Triggering rollback due to {resource_type.value} exhaustion")

    def _cleanup_old_data(self):
        """Clean up old usage data"""
        cutoff_time = time.time() - 3600  # Keep last hour

        for resource_type in self.current_usage:
            self.current_usage[resource_type] = [
                usage for usage in self.current_usage[resource_type]
                if usage.timestamp > cutoff_time
            ]

    def check_operation_feasibility(self, operation_type: str,
                                  estimated_resources: Dict[ResourceType, float]) -> Dict[str, Any]:
        """Check if an operation can run given current resource usage"""

        result = {
            'feasible': True,
            'blockers': [],
            'warnings': [],
            'recommendations': []
        }

        for resource_type, estimated_usage in estimated_resources.items():
            quota = self.quotas.get(resource_type)
            if not quota:
                continue

            usage_history = self.current_usage.get(resource_type, [])
            if not usage_history:
                continue

            current_usage = usage_history[-1]

            # Check if operation would exceed quota
            projected_usage = current_usage.usage_percentage + (estimated_usage / quota.max_usage) * 100

            if projected_usage >= quota.critical_threshold * 100:
                result['feasible'] = False
                result['blockers'].append(f"{resource_type.value} would exceed critical threshold")
            elif projected_usage >= quota.warning_threshold * 100:
                result['warnings'].append(f"{resource_type.value} would approach warning threshold")

        # Add recommendations
        if not result['feasible']:
            result['recommendations'].append("Wait for resource usage to decrease or scale infrastructure")
        elif result['warnings']:
            result['recommendations'].append("Monitor closely during operation execution")

        return result

    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status"""
        status = {
            'timestamp': time.time(),
            'resources': {},
            'alerts': [],
            'system_health': 'healthy'
        }

        # Resource status
        for resource_type, usage_history in self.current_usage.items():
            if usage_history:
                current = usage_history[-1]
                quota = self.quotas.get(resource_type)

                status['resources'][resource_type.value] = {
                    'current_usage': current.current_usage,
                    'max_usage': current.max_usage,
                    'usage_percentage': current.usage_percentage,
                    'trend': current.trend,
                    'quota_exceeded': current.usage_percentage > 95 if quota else False
                }

        # Active alerts
        status['alerts'] = [
            {
                'resource': alert.resource_type.value,
                'severity': alert.severity.value,
                'message': alert.message,
                'action': alert.recommended_action,
                'timestamp': alert.timestamp
            }
            for alert in self.active_alerts[-10:]  # Last 10 alerts
        ]

        # Overall health
        critical_alerts = [a for a in status['alerts'] if a['severity'] == 'critical']
        if critical_alerts:
            status['system_health'] = 'critical'
        elif status['alerts']:
            status['system_health'] = 'warning'

        return status

    def allocate_resources(self, operation_type: str, process_id: str) -> Dict[str, Any]:
        """Allocate resources for an operation"""
        profile = self.evolution_profiles.get(operation_type)
        if not profile:
            return {'allocated': False, 'reason': 'Unknown operation type'}

        # Check feasibility
        feasibility = self.check_operation_feasibility(operation_type, profile.expected_resources)

        if not feasibility['feasible']:
            return {
                'allocated': False,
                'reason': 'Resource constraints',
                'blockers': feasibility['blockers']
            }

        # Reserve resources (simplified - in real implementation would use semaphores/locks)
        allocation = {
            'allocated': True,
            'operation_type': operation_type,
            'process_id': process_id,
            'resources_allocated': profile.expected_resources,
            'max_duration': profile.max_duration,
            'priority': profile.priority,
            'can_interrupt': profile.can_be_interrupted,
            'allocation_time': time.time()
        }

        self.logger.info(f"Resources allocated for {operation_type} (process: {process_id})")
        return allocation

    def release_resources(self, process_id: str):
        """Release resources allocated to a process"""
        self.logger.info(f"Resources released for process: {process_id}")

    def shutdown(self):
        """Shutdown the resource quota manager"""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Resource Quota Manager shutdown")

def main():
    """CLI interface for resource quota management"""
    import argparse

    parser = argparse.ArgumentParser(description="Resource Quota Manager")
    parser.add_argument("--status", action="store_true", help="Show resource status")
    parser.add_argument("--check-operation", metavar="OPERATION",
                       help="Check if operation can run")
    parser.add_argument("--allocate", nargs=2, metavar=('OPERATION', 'PROCESS_ID'),
                       help="Allocate resources for operation")
    parser.add_argument("--alerts", action="store_true", help="Show active alerts")
    parser.add_argument("--monitor", action="store_true", help="Start monitoring mode")

    args = parser.parse_args()

    manager = ResourceQuotaManager()

    try:
        if args.status:
            status = manager.get_resource_status()
            print("📊 RESOURCE STATUS")
            print("=" * 40)

            print(f"System Health: {status['system_health'].upper()}")
            print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(status['timestamp']))}")
            print()

            print("Resource Usage:")
            for res_name, res_data in status['resources'].items():
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(res_data['trend'], "❓")
                status_emoji = "🔴" if res_data['quota_exceeded'] else "🟢"
                print(f"  {status_emoji} {res_name}: {res_data['usage_percentage']:.1f}% {trend_emoji}")

            print(f"\nActive Alerts: {len(status['alerts'])}")
            for alert in status['alerts'][-3:]:  # Show last 3
                severity_emoji = {"critical": "💥", "warning": "⚠️", "info": "ℹ️"}.get(alert['severity'], "❓")
                print(f"  {severity_emoji} {alert['resource']}: {alert['message']}")

        elif args.check_operation:
            profile = manager.evolution_profiles.get(args.check_operation)
            if not profile:
                print(f"[-] Unknown operation: {args.check_operation}")
                return

            feasibility = manager.check_operation_feasibility(args.check_operation, profile.expected_resources)

            print(f"🔍 OPERATION FEASIBILITY: {args.check_operation}")
            print("=" * 40)
            print(f"Feasible: {'✅ YES' if feasibility['feasible'] else '❌ NO'}")

            if feasibility['blockers']:
                print("Blockers:")
                for blocker in feasibility['blockers']:
                    print(f"  • {blocker}")

            if feasibility['warnings']:
                print("Warnings:")
                for warning in feasibility['warnings']:
                    print(f"  • {warning}")

            if feasibility['recommendations']:
                print("Recommendations:")
                for rec in feasibility['recommendations']:
                    print(f"  • {rec}")

        elif args.allocate:
            operation, process_id = args.allocate
            result = manager.allocate_resources(operation, process_id)

            print(f"🎯 RESOURCE ALLOCATION: {operation}")
            print("=" * 40)

            if result['allocated']:
                print("✅ Resources allocated successfully")
                print(f"Process ID: {process_id}")
                print(f"Max Duration: {result['max_duration']}s")
                print(f"Priority: {result['priority']}")
                print(f"Can Interrupt: {result['can_interrupt']}")
                print("Allocated Resources:")
                for res_type, amount in result['resources_allocated'].items():
                    print(f"  • {res_type.value}: {amount}")
            else:
                print("❌ Resource allocation failed")
                print(f"Reason: {result['reason']}")
                if 'blockers' in result:
                    for blocker in result['blockers']:
                        print(f"  • {blocker}")

        elif args.alerts:
            status = manager.get_resource_status()
            alerts = status['alerts']

            print(f"🚨 ACTIVE ALERTS ({len(alerts)})")
            print("=" * 40)

            if not alerts:
                print("✅ No active alerts")
            else:
                for alert in alerts:
                    severity_emoji = {"critical": "💥", "warning": "⚠️", "info": "ℹ️"}.get(alert['severity'], "❓")
                    print(f"{severity_emoji} {alert['resource'].upper()}")
                    print(f"   {alert['message']}")
                    print(f"   Recommended: {alert['action']}")
                    print(f"   Time: {time.strftime('%H:%M:%S', time.localtime(alert['timestamp']))}")
                    print()

        elif args.monitor:
            print("📊 MONITORING MODE - Press Ctrl+C to stop")
            print("=" * 40)

            try:
                while True:
                    status = manager.get_resource_status()
                    alerts = [a for a in status['alerts'] if a['severity'] in ['critical', 'warning']]

                    # Clear screen and show status
                    print(f"\rSystem: {status['system_health'].upper()} | "
                          f"Alerts: {len(alerts)} | "
                          f"Time: {time.strftime('%H:%M:%S')}", end='', flush=True)

                    time.sleep(2)

            except KeyboardInterrupt:
                print("\n[*] Monitoring stopped")

        else:
            parser.print_help()

    finally:
        manager.shutdown()

if __name__ == "__main__":
    main()
