"""Network Policy Monitoring

Monitor and analyze Cilium/Kubernetes network policies and their enforcement.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import threading
import logging
import json

from prometheus_client import Counter, Gauge, Histogram

from .hubble_client import HubbleClient, HubbleFlow, FlowFilter

logger = logging.getLogger(__name__)


class PolicyVerdict(str, Enum):
    """Network policy enforcement verdict."""
    ALLOWED = "allowed"
    DENIED = "denied"
    AUDIT = "audit"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass
class PolicyEndpoint:
    """Endpoint specification for network policy.

    Attributes:
        namespace: Kubernetes namespace
        pod_selector: Pod label selector
        ip_block: IP CIDR block
        ports: Allowed ports
        labels: Endpoint labels
    """
    namespace: Optional[str] = None
    pod_selector: Dict[str, str] = field(default_factory=dict)
    ip_block: Optional[str] = None
    ports: List[int] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)

    def matches_flow(self, flow: HubbleFlow, is_source: bool = True) -> bool:
        """Check if endpoint matches a flow."""
        if is_source:
            ns = flow.source_namespace
            labels = flow.source_labels
            ip = flow.source_ip
        else:
            ns = flow.destination_namespace
            labels = flow.destination_labels
            ip = flow.destination_ip

        # Namespace match
        if self.namespace and ns != self.namespace:
            return False

        # Label selector match
        if self.pod_selector:
            for key, value in self.pod_selector.items():
                if labels.get(key) != value:
                    return False

        # IP block match (simplified)
        if self.ip_block and ip:
            # In production, use ipaddress module for proper CIDR matching
            if not ip.startswith(self.ip_block.split("/")[0].rsplit(".", 1)[0]):
                return False

        # Port match
        if self.ports:
            port = flow.destination_port if not is_source else flow.source_port
            if port not in self.ports:
                return False

        return True


@dataclass
class PolicyRule:
    """Single rule within a network policy.

    Attributes:
        rule_id: Unique rule identifier
        direction: ingress or egress
        action: allow or deny
        from_endpoints: Source endpoints (for ingress)
        to_endpoints: Destination endpoints (for egress)
        ports: Allowed/denied ports
        protocols: Allowed/denied protocols
        priority: Rule priority (higher = more precedence)
    """
    rule_id: str
    direction: str = "ingress"
    action: str = "allow"
    from_endpoints: List[PolicyEndpoint] = field(default_factory=list)
    to_endpoints: List[PolicyEndpoint] = field(default_factory=list)
    ports: List[Dict[str, Any]] = field(default_factory=list)
    protocols: List[str] = field(default_factory=list)
    priority: int = 0

    def matches_flow(self, flow: HubbleFlow) -> bool:
        """Check if rule matches a flow."""
        # Direction check
        if self.direction == "ingress":
            if flow.traffic_direction != "INGRESS":
                return False
        elif self.direction == "egress":
            if flow.traffic_direction != "EGRESS":
                return False

        # Endpoint matching
        if self.direction == "ingress" and self.from_endpoints:
            if not any(ep.matches_flow(flow, is_source=True) for ep in self.from_endpoints):
                return False
        if self.direction == "egress" and self.to_endpoints:
            if not any(ep.matches_flow(flow, is_source=False) for ep in self.to_endpoints):
                return False

        # Port matching
        if self.ports:
            port_matched = False
            for port_spec in self.ports:
                if "port" in port_spec:
                    if flow.destination_port == port_spec["port"]:
                        port_matched = True
                        break
                if "portRange" in port_spec:
                    start, end = port_spec["portRange"].split("-")
                    if int(start) <= flow.destination_port <= int(end):
                        port_matched = True
                        break
            if not port_matched:
                return False

        # Protocol matching
        if self.protocols:
            if flow.protocol not in self.protocols:
                return False

        return True


@dataclass
class NetworkPolicy:
    """Kubernetes/Cilium network policy.

    Attributes:
        name: Policy name
        namespace: Policy namespace
        selector: Pod selector for policy target
        ingress_rules: Ingress rules
        egress_rules: Egress rules
        policy_type: cilium, kubernetes, or calico
        created_at: Policy creation time
        annotations: Policy annotations
        spec_hash: Hash of policy spec for change detection
    """
    name: str
    namespace: str
    selector: Dict[str, str] = field(default_factory=dict)
    ingress_rules: List[PolicyRule] = field(default_factory=list)
    egress_rules: List[PolicyRule] = field(default_factory=list)
    policy_type: str = "kubernetes"
    created_at: Optional[datetime] = None
    annotations: Dict[str, str] = field(default_factory=dict)
    spec_hash: Optional[str] = None

    @property
    def full_name(self) -> str:
        return f"{self.namespace}/{self.name}"

    def evaluate_flow(self, flow: HubbleFlow) -> PolicyVerdict:
        """Evaluate a flow against this policy.

        Args:
            flow: Flow to evaluate

        Returns:
            PolicyVerdict
        """
        # Check if policy applies to this pod
        if not self._applies_to_flow(flow):
            return PolicyVerdict.NONE

        # Check ingress rules
        if flow.traffic_direction == "INGRESS":
            for rule in self.ingress_rules:
                if rule.matches_flow(flow):
                    if rule.action == "allow":
                        return PolicyVerdict.ALLOWED
                    else:
                        return PolicyVerdict.DENIED

            # Default deny if ingress rules exist but none matched
            if self.ingress_rules:
                return PolicyVerdict.DENIED

        # Check egress rules
        if flow.traffic_direction == "EGRESS":
            for rule in self.egress_rules:
                if rule.matches_flow(flow):
                    if rule.action == "allow":
                        return PolicyVerdict.ALLOWED
                    else:
                        return PolicyVerdict.DENIED

            # Default deny if egress rules exist but none matched
            if self.egress_rules:
                return PolicyVerdict.DENIED

        return PolicyVerdict.NONE

    def _applies_to_flow(self, flow: HubbleFlow) -> bool:
        """Check if policy applies to the flow's destination."""
        if flow.destination_namespace != self.namespace:
            return False

        if not self.selector:
            return True

        for key, value in self.selector.items():
            if flow.destination_labels.get(key) != value:
                return False

        return True


@dataclass
class PolicyMatch:
    """Record of a policy match for a flow.

    Attributes:
        flow_id: Flow identifier
        policy_name: Matched policy name
        policy_namespace: Policy namespace
        rule_id: Matched rule ID
        verdict: Enforcement verdict
        timestamp: Match timestamp
        direction: ingress or egress
        action_taken: Action taken (forward, drop, audit)
    """
    flow_id: str
    policy_name: str
    policy_namespace: str
    rule_id: str
    verdict: PolicyVerdict
    timestamp: datetime
    direction: str = "ingress"
    action_taken: str = "forward"


@dataclass
class PolicyStats:
    """Statistics for a network policy.

    Attributes:
        policy_name: Policy name
        namespace: Policy namespace
        total_matches: Total flow matches
        allowed_count: Allowed flows
        denied_count: Denied flows
        audit_count: Audited flows
        last_match: Last match timestamp
        top_sources: Top matched sources
        top_destinations: Top matched destinations
    """
    policy_name: str
    namespace: str
    total_matches: int = 0
    allowed_count: int = 0
    denied_count: int = 0
    audit_count: int = 0
    last_match: Optional[datetime] = None
    top_sources: List[Tuple[str, int]] = field(default_factory=list)
    top_destinations: List[Tuple[str, int]] = field(default_factory=list)


class PolicyMonitor:
    """Monitor network policy enforcement.

    Tracks policy matches, violations, and provides analytics.

    Usage:
        client = HubbleClient(config)
        monitor = PolicyMonitor(client)

        # Add policies to monitor
        policy = NetworkPolicy(name="allow-ingress", namespace="production")
        monitor.add_policy(policy)

        # Process flows
        for flow in client.get_flows():
            match = monitor.evaluate_flow(flow)
            if match.verdict == PolicyVerdict.DENIED:
                alert(f"Flow denied by {match.policy_name}")

        # Get policy statistics
        stats = monitor.get_policy_stats("production/allow-ingress")
    """

    def __init__(
        self,
        client: Optional[HubbleClient] = None,
        namespace: str = "consciousness",
    ):
        self.client = client
        self.namespace = namespace
        self._lock = threading.Lock()

        # Policy storage
        self._policies: Dict[str, NetworkPolicy] = {}
        self._matches: List[PolicyMatch] = []
        self._stats: Dict[str, PolicyStats] = {}
        self._max_matches = 50000

        # Track violations
        self._violations: Dict[str, List[PolicyMatch]] = defaultdict(list)

        # Prometheus metrics
        self.policy_matches = Counter(
            f"{namespace}_network_policy_matches_total",
            "Total policy matches",
            ["policy", "namespace", "verdict", "direction"],
        )

        self.policy_violations = Counter(
            f"{namespace}_network_policy_violations_total",
            "Policy violations (denied flows)",
            ["policy", "namespace", "source_namespace", "destination_namespace"],
        )

        self.active_policies = Gauge(
            f"{namespace}_network_policies_active",
            "Number of active network policies",
            ["namespace"],
        )

        self.policy_evaluation_time = Histogram(
            f"{namespace}_network_policy_evaluation_seconds",
            "Policy evaluation latency",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05],
        )

    def add_policy(self, policy: NetworkPolicy):
        """Add a network policy to monitor.

        Args:
            policy: NetworkPolicy to add
        """
        with self._lock:
            self._policies[policy.full_name] = policy
            self._stats[policy.full_name] = PolicyStats(
                policy_name=policy.name,
                namespace=policy.namespace,
            )

        # Update metrics
        namespace_count = sum(
            1 for p in self._policies.values()
            if p.namespace == policy.namespace
        )
        self.active_policies.labels(namespace=policy.namespace).set(namespace_count)

        logger.info(f"Added network policy: {policy.full_name}")

    def remove_policy(self, policy_name: str, namespace: str):
        """Remove a network policy.

        Args:
            policy_name: Policy name
            namespace: Policy namespace
        """
        full_name = f"{namespace}/{policy_name}"

        with self._lock:
            self._policies.pop(full_name, None)
            self._stats.pop(full_name, None)

        logger.info(f"Removed network policy: {full_name}")

    def evaluate_flow(self, flow: HubbleFlow) -> Optional[PolicyMatch]:
        """Evaluate a flow against all policies.

        Args:
            flow: HubbleFlow to evaluate

        Returns:
            PolicyMatch if any policy matched, None otherwise
        """
        start_time = time.perf_counter()
        best_match: Optional[PolicyMatch] = None
        best_priority = -1

        with self._lock:
            policies = list(self._policies.values())

        for policy in policies:
            verdict = policy.evaluate_flow(flow)

            if verdict != PolicyVerdict.NONE:
                # Track which rule matched
                matched_rule_id = "default"
                rules = (
                    policy.ingress_rules
                    if flow.traffic_direction == "INGRESS"
                    else policy.egress_rules
                )
                for rule in rules:
                    if rule.matches_flow(flow):
                        matched_rule_id = rule.rule_id
                        if rule.priority > best_priority:
                            best_priority = rule.priority
                            best_match = PolicyMatch(
                                flow_id=flow.flow_id,
                                policy_name=policy.name,
                                policy_namespace=policy.namespace,
                                rule_id=matched_rule_id,
                                verdict=verdict,
                                timestamp=datetime.now(),
                                direction=flow.traffic_direction.lower(),
                                action_taken="forward" if verdict == PolicyVerdict.ALLOWED else "drop",
                            )
                        break

                # If no specific rule matched but policy applies
                if not best_match:
                    best_match = PolicyMatch(
                        flow_id=flow.flow_id,
                        policy_name=policy.name,
                        policy_namespace=policy.namespace,
                        rule_id="default",
                        verdict=verdict,
                        timestamp=datetime.now(),
                        direction=flow.traffic_direction.lower(),
                        action_taken="forward" if verdict == PolicyVerdict.ALLOWED else "drop",
                    )

        # Record metrics
        eval_time = time.perf_counter() - start_time
        self.policy_evaluation_time.observe(eval_time)

        if best_match:
            self._record_match(best_match, flow)

        return best_match

    def _record_match(self, match: PolicyMatch, flow: HubbleFlow):
        """Record a policy match."""
        with self._lock:
            # Store match
            self._matches.append(match)
            if len(self._matches) > self._max_matches:
                self._matches = self._matches[-self._max_matches // 2:]

            # Update stats
            full_name = f"{match.policy_namespace}/{match.policy_name}"
            if full_name in self._stats:
                stats = self._stats[full_name]
                stats.total_matches += 1
                stats.last_match = match.timestamp

                if match.verdict == PolicyVerdict.ALLOWED:
                    stats.allowed_count += 1
                elif match.verdict == PolicyVerdict.DENIED:
                    stats.denied_count += 1
                elif match.verdict == PolicyVerdict.AUDIT:
                    stats.audit_count += 1

        # Record Prometheus metrics
        self.policy_matches.labels(
            policy=match.policy_name,
            namespace=match.policy_namespace,
            verdict=match.verdict.value,
            direction=match.direction,
        ).inc()

        if match.verdict == PolicyVerdict.DENIED:
            self._violations[full_name].append(match)
            self.policy_violations.labels(
                policy=match.policy_name,
                namespace=match.policy_namespace,
                source_namespace=flow.source_namespace or "unknown",
                destination_namespace=flow.destination_namespace or "unknown",
            ).inc()

    def get_policy_stats(self, full_name: str) -> Optional[PolicyStats]:
        """Get statistics for a policy.

        Args:
            full_name: Full policy name (namespace/name)

        Returns:
            PolicyStats or None
        """
        with self._lock:
            return self._stats.get(full_name)

    def get_all_stats(self) -> Dict[str, PolicyStats]:
        """Get statistics for all policies.

        Returns:
            Dict mapping policy names to stats
        """
        with self._lock:
            return dict(self._stats)

    def get_violations(
        self,
        policy_name: Optional[str] = None,
        since: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[PolicyMatch]:
        """Get policy violations.

        Args:
            policy_name: Filter by policy name
            since: Get violations since this time ago
            limit: Maximum violations to return

        Returns:
            List of violation PolicyMatches
        """
        cutoff = datetime.now() - since if since else None

        with self._lock:
            if policy_name:
                violations = list(self._violations.get(policy_name, []))
            else:
                violations = []
                for v_list in self._violations.values():
                    violations.extend(v_list)

        # Filter by time
        if cutoff:
            violations = [v for v in violations if v.timestamp >= cutoff]

        # Sort by timestamp descending
        violations.sort(key=lambda v: v.timestamp, reverse=True)

        return violations[:limit]

    def get_policy_coverage(self) -> Dict[str, Any]:
        """Get policy coverage analysis.

        Returns:
            Coverage analysis dict
        """
        with self._lock:
            policies = list(self._policies.values())
            stats = dict(self._stats)

        # Analyze coverage
        namespaces_with_ingress: Set[str] = set()
        namespaces_with_egress: Set[str] = set()
        total_ingress_rules = 0
        total_egress_rules = 0

        for policy in policies:
            if policy.ingress_rules:
                namespaces_with_ingress.add(policy.namespace)
                total_ingress_rules += len(policy.ingress_rules)
            if policy.egress_rules:
                namespaces_with_egress.add(policy.namespace)
                total_egress_rules += len(policy.egress_rules)

        return {
            "total_policies": len(policies),
            "namespaces_covered": len(namespaces_with_ingress | namespaces_with_egress),
            "namespaces_with_ingress": list(namespaces_with_ingress),
            "namespaces_with_egress": list(namespaces_with_egress),
            "total_ingress_rules": total_ingress_rules,
            "total_egress_rules": total_egress_rules,
            "policies_with_matches": sum(1 for s in stats.values() if s.total_matches > 0),
            "policies_with_violations": sum(1 for s in stats.values() if s.denied_count > 0),
        }

    def get_unprotected_services(
        self,
        known_services: List[Tuple[str, str]],
    ) -> List[Tuple[str, str]]:
        """Find services without network policies.

        Args:
            known_services: List of (namespace, service) tuples

        Returns:
            List of unprotected (namespace, service) tuples
        """
        with self._lock:
            policies = list(self._policies.values())

        protected: Set[Tuple[str, str]] = set()

        for policy in policies:
            if policy.selector:
                # Extract app/service label
                app = (
                    policy.selector.get("app") or
                    policy.selector.get("app.kubernetes.io/name") or
                    policy.name
                )
                protected.add((policy.namespace, app))

        return [s for s in known_services if s not in protected]

    def sync_from_kubernetes(self, kubeconfig: Optional[str] = None):
        """Sync policies from Kubernetes API.

        Args:
            kubeconfig: Path to kubeconfig file
        """
        try:
            from kubernetes import client, config

            if kubeconfig:
                config.load_kube_config(config_file=kubeconfig)
            else:
                try:
                    config.load_incluster_config()
                except Exception:
                    config.load_kube_config()

            networking_v1 = client.NetworkingV1Api()

            # List all NetworkPolicies
            policies = networking_v1.list_network_policy_for_all_namespaces()

            for item in policies.items:
                policy = self._convert_k8s_policy(item)
                self.add_policy(policy)

            logger.info(f"Synced {len(policies.items)} policies from Kubernetes")

        except ImportError:
            logger.warning("kubernetes package not available for sync")
        except Exception as e:
            logger.error(f"Failed to sync from Kubernetes: {e}")

    def _convert_k8s_policy(self, k8s_policy: Any) -> NetworkPolicy:
        """Convert Kubernetes NetworkPolicy to our format."""
        metadata = k8s_policy.metadata
        spec = k8s_policy.spec

        # Extract pod selector
        selector = {}
        if spec.pod_selector and spec.pod_selector.match_labels:
            selector = dict(spec.pod_selector.match_labels)

        # Convert ingress rules
        ingress_rules = []
        if spec.ingress:
            for i, ing in enumerate(spec.ingress):
                from_endpoints = []
                if ing._from:
                    for f in ing._from:
                        ep = PolicyEndpoint()
                        if f.namespace_selector:
                            ep.namespace = f.namespace_selector.match_labels.get("name")
                        if f.pod_selector:
                            ep.pod_selector = dict(f.pod_selector.match_labels or {})
                        if f.ip_block:
                            ep.ip_block = f.ip_block.cidr
                        from_endpoints.append(ep)

                ports = []
                if ing.ports:
                    for p in ing.ports:
                        port_spec = {"port": p.port}
                        if p.protocol:
                            port_spec["protocol"] = p.protocol
                        ports.append(port_spec)

                ingress_rules.append(PolicyRule(
                    rule_id=f"ingress-{i}",
                    direction="ingress",
                    action="allow",
                    from_endpoints=from_endpoints,
                    ports=ports,
                ))

        # Convert egress rules
        egress_rules = []
        if spec.egress:
            for i, egr in enumerate(spec.egress):
                to_endpoints = []
                if egr.to:
                    for t in egr.to:
                        ep = PolicyEndpoint()
                        if t.namespace_selector:
                            ep.namespace = t.namespace_selector.match_labels.get("name")
                        if t.pod_selector:
                            ep.pod_selector = dict(t.pod_selector.match_labels or {})
                        if t.ip_block:
                            ep.ip_block = t.ip_block.cidr
                        to_endpoints.append(ep)

                ports = []
                if egr.ports:
                    for p in egr.ports:
                        port_spec = {"port": p.port}
                        if p.protocol:
                            port_spec["protocol"] = p.protocol
                        ports.append(port_spec)

                egress_rules.append(PolicyRule(
                    rule_id=f"egress-{i}",
                    direction="egress",
                    action="allow",
                    to_endpoints=to_endpoints,
                    ports=ports,
                ))

        return NetworkPolicy(
            name=metadata.name,
            namespace=metadata.namespace,
            selector=selector,
            ingress_rules=ingress_rules,
            egress_rules=egress_rules,
            policy_type="kubernetes",
            created_at=metadata.creation_timestamp,
            annotations=dict(metadata.annotations or {}),
        )

    def export_policies(self) -> str:
        """Export all policies as JSON.

        Returns:
            JSON string of policies
        """
        with self._lock:
            policies = list(self._policies.values())

        export_data = []
        for policy in policies:
            export_data.append({
                "name": policy.name,
                "namespace": policy.namespace,
                "selector": policy.selector,
                "policy_type": policy.policy_type,
                "ingress_rule_count": len(policy.ingress_rules),
                "egress_rule_count": len(policy.egress_rules),
            })

        return json.dumps(export_data, indent=2)
