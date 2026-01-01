"""Steadybit Custom Actions

Defines reusable chaos actions for Steadybit experiments.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ActionCategory(str, Enum):
    """Categories of chaos actions."""
    NETWORK = "network"
    RESOURCE = "resource"
    STATE = "state"
    KUBERNETES = "kubernetes"
    APPLICATION = "application"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class ActionParameter:
    """Parameter definition for an action."""
    name: str
    type: str  # string, integer, number, boolean, select
    description: str
    required: bool = True
    default: Optional[Any] = None
    options: Optional[List[str]] = None  # For select type
    min_value: Optional[float] = None
    max_value: Optional[float] = None


@dataclass
class SteadybitAction(ABC):
    """Base class for Steadybit actions.

    Actions define chaos attacks that can be executed against targets.
    """
    id: str
    name: str
    description: str
    category: ActionCategory
    version: str = "1.0.0"
    icon: str = "action"
    documentation_url: Optional[str] = None
    parameters: List[ActionParameter] = field(default_factory=list)
    target_types: List[str] = field(default_factory=list)

    def to_definition(self) -> Dict[str, Any]:
        """Convert to Steadybit action definition format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "icon": self.icon,
            "documentationUrl": self.documentation_url,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                    **({"options": p.options} if p.options else {}),
                    **({"min": p.min_value} if p.min_value is not None else {}),
                    **({"max": p.max_value} if p.max_value is not None else {}),
                }
                for p in self.parameters
            ],
            "targetTypes": self.target_types,
        }

    @abstractmethod
    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the action for execution.

        Args:
            target: Target to attack
            config: Action configuration

        Returns:
            Prepared state for execution
        """
        pass

    @abstractmethod
    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Start the action.

        Args:
            state: State from prepare

        Returns:
            Running state
        """
        pass

    @abstractmethod
    async def stop(self, state: Dict[str, Any]):
        """Stop the action.

        Args:
            state: State from start
        """
        pass


# Predefined Actions

class LatencyAction(SteadybitAction):
    """Network latency injection action."""

    def __init__(self):
        super().__init__(
            id="consciousness-network-delay",
            name="Network Delay",
            description="Inject network latency to slow down traffic",
            category=ActionCategory.NETWORK,
            parameters=[
                ActionParameter(
                    name="delay",
                    type="integer",
                    description="Delay in milliseconds",
                    default=100,
                    min_value=1,
                    max_value=60000,
                ),
                ActionParameter(
                    name="jitter",
                    type="integer",
                    description="Jitter in milliseconds",
                    default=0,
                    required=False,
                    min_value=0,
                    max_value=10000,
                ),
                ActionParameter(
                    name="port",
                    type="integer",
                    description="Target port (0 for all)",
                    default=0,
                    required=False,
                ),
                ActionParameter(
                    name="interface",
                    type="string",
                    description="Network interface",
                    default="eth0",
                    required=False,
                ),
            ],
            target_types=["container", "host", "kubernetes-pod"],
        )

    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": target,
            "delay_ms": config.get("delay", 100),
            "jitter_ms": config.get("jitter", 0),
            "port": config.get("port", 0),
            "interface": config.get("interface", "eth0"),
            "prepared_at": datetime.now().isoformat(),
        }

    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # In real implementation, this would use tc (traffic control)
        logger.info(
            f"Starting network delay: {state['delay_ms']}ms +/- {state['jitter_ms']}ms "
            f"on {state['interface']}"
        )
        return {**state, "started_at": datetime.now().isoformat(), "active": True}

    async def stop(self, state: Dict[str, Any]):
        logger.info(f"Stopping network delay on {state['interface']}")


class ErrorAction(SteadybitAction):
    """HTTP error injection action."""

    def __init__(self):
        super().__init__(
            id="consciousness-http-error",
            name="HTTP Error",
            description="Inject HTTP errors into responses",
            category=ActionCategory.NETWORK,
            parameters=[
                ActionParameter(
                    name="statusCode",
                    type="integer",
                    description="HTTP status code to return",
                    default=500,
                    min_value=400,
                    max_value=599,
                ),
                ActionParameter(
                    name="percentage",
                    type="number",
                    description="Percentage of requests to affect",
                    default=100.0,
                    min_value=0.1,
                    max_value=100.0,
                ),
                ActionParameter(
                    name="path",
                    type="string",
                    description="URL path pattern to match",
                    default="*",
                    required=False,
                ),
            ],
            target_types=["container", "application"],
        )

    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": target,
            "status_code": config.get("statusCode", 500),
            "percentage": config.get("percentage", 100.0),
            "path": config.get("path", "*"),
        }

    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            f"Starting HTTP error injection: {state['status_code']} "
            f"for {state['percentage']}% of requests"
        )
        return {**state, "active": True}

    async def stop(self, state: Dict[str, Any]):
        logger.info("Stopping HTTP error injection")


class ResourceAction(SteadybitAction):
    """Resource stress action."""

    def __init__(self, resource_type: str = "cpu"):
        self.resource_type = resource_type
        super().__init__(
            id=f"consciousness-stress-{resource_type}",
            name=f"{resource_type.upper()} Stress",
            description=f"Stress {resource_type.upper()} resources",
            category=ActionCategory.RESOURCE,
            parameters=[
                ActionParameter(
                    name="load",
                    type="integer",
                    description=f"{resource_type.upper()} load percentage",
                    default=80,
                    min_value=1,
                    max_value=100,
                ),
                ActionParameter(
                    name="cores",
                    type="integer",
                    description="Number of cores (CPU only)",
                    default=0,
                    required=False,
                ),
            ],
            target_types=["container", "host", "kubernetes-pod"],
        )

    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": target,
            "resource_type": self.resource_type,
            "load": config.get("load", 80),
            "cores": config.get("cores", 0),
        }

    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(
            f"Starting {state['resource_type']} stress: {state['load']}%"
        )
        return {**state, "active": True}

    async def stop(self, state: Dict[str, Any]):
        logger.info(f"Stopping {state['resource_type']} stress")


class NetworkAction(SteadybitAction):
    """Network partition/blackhole action."""

    def __init__(self):
        super().__init__(
            id="consciousness-network-blackhole",
            name="Network Blackhole",
            description="Drop all network traffic to specified hosts",
            category=ActionCategory.NETWORK,
            parameters=[
                ActionParameter(
                    name="hosts",
                    type="string",
                    description="Comma-separated list of hosts to block",
                    required=True,
                ),
                ActionParameter(
                    name="port",
                    type="integer",
                    description="Port to block (0 for all)",
                    default=0,
                    required=False,
                ),
            ],
            target_types=["container", "host", "kubernetes-pod"],
        )

    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        hosts = config.get("hosts", "")
        if isinstance(hosts, str):
            hosts = [h.strip() for h in hosts.split(",")]

        return {
            "target": target,
            "hosts": hosts,
            "port": config.get("port", 0),
        }

    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting network blackhole for hosts: {state['hosts']}")
        return {**state, "active": True}

    async def stop(self, state: Dict[str, Any]):
        logger.info("Stopping network blackhole")


class KubernetesAction(SteadybitAction):
    """Kubernetes-specific chaos action."""

    def __init__(self, action_type: str = "pod-delete"):
        self.action_type = action_type
        action_name = action_type.replace("-", " ").title()

        params = []
        if action_type == "pod-delete":
            params = [
                ActionParameter(
                    name="gracePeriodSeconds",
                    type="integer",
                    description="Grace period before force kill",
                    default=30,
                    min_value=0,
                    max_value=300,
                ),
            ]
        elif action_type == "scale-down":
            params = [
                ActionParameter(
                    name="replicas",
                    type="integer",
                    description="Target replica count",
                    default=0,
                    min_value=0,
                ),
            ]
        elif action_type == "drain-node":
            params = [
                ActionParameter(
                    name="deleteLocalData",
                    type="boolean",
                    description="Delete local data when draining",
                    default=False,
                ),
            ]

        super().__init__(
            id=f"consciousness-k8s-{action_type}",
            name=f"Kubernetes {action_name}",
            description=f"Kubernetes {action_name} action",
            category=ActionCategory.KUBERNETES,
            parameters=params,
            target_types=[
                "kubernetes-pod" if "pod" in action_type else "kubernetes-deployment",
                "kubernetes-node" if "node" in action_type else "kubernetes-pod",
            ],
        )

    async def prepare(self, target: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target": target,
            "action_type": self.action_type,
            "config": config,
        }

    async def start(self, state: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting Kubernetes {state['action_type']} on {state['target']}")
        return {**state, "active": True}

    async def stop(self, state: Dict[str, Any]):
        logger.info(f"Stopping Kubernetes {state['action_type']}")


class ActionRegistry:
    """Registry for Steadybit actions.

    Usage:
        registry = ActionRegistry()
        registry.register(LatencyAction())
        registry.register(ErrorAction())

        # Get action by ID
        action = registry.get("consciousness-network-delay")

        # List all actions
        actions = registry.list_actions()
    """

    def __init__(self):
        self._actions: Dict[str, SteadybitAction] = {}

    def register(self, action: SteadybitAction):
        """Register an action.

        Args:
            action: Action to register
        """
        self._actions[action.id] = action
        logger.debug(f"Registered action: {action.id}")

    def unregister(self, action_id: str):
        """Unregister an action.

        Args:
            action_id: ID of action to remove
        """
        if action_id in self._actions:
            del self._actions[action_id]

    def get(self, action_id: str) -> Optional[SteadybitAction]:
        """Get an action by ID.

        Args:
            action_id: Action ID

        Returns:
            Action or None
        """
        return self._actions.get(action_id)

    def list_actions(
        self,
        category: Optional[ActionCategory] = None,
    ) -> List[SteadybitAction]:
        """List registered actions.

        Args:
            category: Filter by category

        Returns:
            List of actions
        """
        actions = list(self._actions.values())

        if category:
            actions = [a for a in actions if a.category == category]

        return actions

    def to_catalog(self) -> Dict[str, Any]:
        """Export as action catalog.

        Returns:
            Catalog in Steadybit format
        """
        return {
            "actions": [
                action.to_definition()
                for action in self._actions.values()
            ]
        }

    @classmethod
    def default(cls) -> ActionRegistry:
        """Create registry with default actions.

        Returns:
            Populated registry
        """
        registry = cls()

        # Network actions
        registry.register(LatencyAction())
        registry.register(ErrorAction())
        registry.register(NetworkAction())

        # Resource actions
        registry.register(ResourceAction("cpu"))
        registry.register(ResourceAction("memory"))
        registry.register(ResourceAction("io"))

        # Kubernetes actions
        registry.register(KubernetesAction("pod-delete"))
        registry.register(KubernetesAction("scale-down"))
        registry.register(KubernetesAction("drain-node"))

        return registry
