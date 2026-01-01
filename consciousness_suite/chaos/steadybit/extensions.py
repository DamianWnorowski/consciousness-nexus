"""Steadybit Extension Framework

Build custom Steadybit extensions for target discovery and attacks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type
from datetime import datetime
from enum import Enum
import asyncio
import logging
import json

logger = logging.getLogger(__name__)


class TargetAttributeType(str, Enum):
    """Types of target attributes."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    LABEL = "label"
    KEY_VALUE = "key-value"


@dataclass
class TargetAttribute:
    """Attribute definition for a discovery target."""
    name: str
    type: TargetAttributeType = TargetAttributeType.STRING
    description: Optional[str] = None


@dataclass
class DiscoveryTarget:
    """A discovered target for chaos attacks.

    Usage:
        target = DiscoveryTarget(
            id="container-abc123",
            target_type="container",
            label="payment-service",
            attributes={
                "container.id": "abc123",
                "container.name": "payment-service",
                "container.image": "payment:v1.2.3",
                "host.name": "node-1",
            },
        )
    """
    id: str
    target_type: str
    label: str
    attributes: Dict[str, Any]
    discovered_at: datetime = field(default_factory=datetime.now)

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Steadybit API format."""
        return {
            "id": self.id,
            "targetType": self.target_type,
            "label": self.label,
            "attributes": self.attributes,
        }


@dataclass
class AttackDefinition:
    """Definition of an attack type.

    Usage:
        attack = AttackDefinition(
            id="consciousness-custom-attack",
            name="Custom Attack",
            description="A custom chaos attack",
            category="custom",
            target_type="container",
            parameters=[
                {"name": "intensity", "type": "integer", "default": 50},
            ],
        )
    """
    id: str
    name: str
    description: str
    category: str
    target_type: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)
    version: str = "1.0.0"
    icon: str = "attack"

    def to_api_format(self) -> Dict[str, Any]:
        """Convert to Steadybit API format."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "targetType": self.target_type,
            "parameters": self.parameters,
            "version": self.version,
            "icon": self.icon,
        }


class SteadybitExtension(ABC):
    """Base class for Steadybit extensions.

    Extensions provide:
    - Target discovery
    - Attack definitions
    - Attack execution

    Usage:
        class ConsciousnessExtension(SteadybitExtension):
            def get_target_types(self):
                return [TargetTypeDefinition(...)]

            async def discover_targets(self, target_type):
                return [DiscoveryTarget(...)]

            def get_attack_definitions(self):
                return [AttackDefinition(...)]

            async def prepare_attack(self, attack_id, target, config):
                return {"state": "prepared"}

            async def start_attack(self, state):
                return {"state": "running"}

            async def stop_attack(self, state):
                pass
    """

    @property
    @abstractmethod
    def extension_id(self) -> str:
        """Unique extension ID."""
        pass

    @property
    @abstractmethod
    def extension_name(self) -> str:
        """Human-readable name."""
        pass

    @property
    def extension_version(self) -> str:
        """Extension version."""
        return "1.0.0"

    @abstractmethod
    def get_target_types(self) -> List[Dict[str, Any]]:
        """Get supported target types.

        Returns:
            List of target type definitions
        """
        pass

    @abstractmethod
    async def discover_targets(
        self,
        target_type: str,
    ) -> List[DiscoveryTarget]:
        """Discover targets of the given type.

        Args:
            target_type: Type of targets to discover

        Returns:
            List of discovered targets
        """
        pass

    @abstractmethod
    def get_attack_definitions(self) -> List[AttackDefinition]:
        """Get attack definitions.

        Returns:
            List of attack definitions
        """
        pass

    @abstractmethod
    async def prepare_attack(
        self,
        attack_id: str,
        target: DiscoveryTarget,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare an attack for execution.

        Args:
            attack_id: Attack type ID
            target: Target to attack
            config: Attack configuration

        Returns:
            State for execution
        """
        pass

    @abstractmethod
    async def start_attack(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Start the attack.

        Args:
            state: State from prepare

        Returns:
            Running state
        """
        pass

    @abstractmethod
    async def stop_attack(self, state: Dict[str, Any]):
        """Stop the attack.

        Args:
            state: State from start
        """
        pass

    def to_manifest(self) -> Dict[str, Any]:
        """Generate extension manifest.

        Returns:
            Extension manifest
        """
        return {
            "id": self.extension_id,
            "name": self.extension_name,
            "version": self.extension_version,
            "targetTypes": self.get_target_types(),
            "attacks": [a.to_api_format() for a in self.get_attack_definitions()],
        }


class ExtensionServer:
    """HTTP server for Steadybit extension protocol.

    Usage:
        extension = MyExtension()
        server = ExtensionServer(extension, port=8085)
        await server.start()
    """

    def __init__(
        self,
        extension: SteadybitExtension,
        host: str = "0.0.0.0",
        port: int = 8085,
    ):
        self.extension = extension
        self.host = host
        self.port = port
        self._app = None
        self._attack_states: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        """Start the extension server."""
        try:
            from aiohttp import web
        except ImportError:
            logger.error("aiohttp required for extension server")
            return

        self._app = web.Application()
        self._setup_routes()

        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(
            f"Steadybit extension server started at http://{self.host}:{self.port}"
        )

    def _setup_routes(self):
        """Set up HTTP routes."""
        from aiohttp import web

        self._app.router.add_get("/", self._handle_manifest)
        self._app.router.add_get("/health", self._handle_health)
        self._app.router.add_get("/targetTypes", self._handle_target_types)
        self._app.router.add_post("/discovery", self._handle_discovery)
        self._app.router.add_get("/attacks", self._handle_attacks)
        self._app.router.add_post("/attack/prepare", self._handle_prepare)
        self._app.router.add_post("/attack/start", self._handle_start)
        self._app.router.add_post("/attack/stop", self._handle_stop)

    async def _handle_manifest(self, request):
        """Handle manifest request."""
        from aiohttp import web
        return web.json_response(self.extension.to_manifest())

    async def _handle_health(self, request):
        """Handle health check."""
        from aiohttp import web
        return web.json_response({"status": "healthy"})

    async def _handle_target_types(self, request):
        """Handle target types request."""
        from aiohttp import web
        return web.json_response({
            "targetTypes": self.extension.get_target_types()
        })

    async def _handle_discovery(self, request):
        """Handle discovery request."""
        from aiohttp import web

        try:
            data = await request.json()
            target_type = data.get("targetType", "")

            targets = await self.extension.discover_targets(target_type)

            return web.json_response({
                "targets": [t.to_api_format() for t in targets]
            })
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_attacks(self, request):
        """Handle attacks list request."""
        from aiohttp import web
        return web.json_response({
            "attacks": [
                a.to_api_format()
                for a in self.extension.get_attack_definitions()
            ]
        })

    async def _handle_prepare(self, request):
        """Handle attack prepare request."""
        from aiohttp import web

        try:
            data = await request.json()
            attack_id = data.get("attackId")
            target_data = data.get("target", {})
            config = data.get("config", {})

            target = DiscoveryTarget(
                id=target_data.get("id", "unknown"),
                target_type=target_data.get("targetType", "unknown"),
                label=target_data.get("label", "unknown"),
                attributes=target_data.get("attributes", {}),
            )

            state = await self.extension.prepare_attack(attack_id, target, config)

            # Store state with execution ID
            execution_id = f"exec-{len(self._attack_states)}"
            self._attack_states[execution_id] = state

            return web.json_response({
                "executionId": execution_id,
                "state": state,
            })
        except Exception as e:
            logger.error(f"Prepare error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_start(self, request):
        """Handle attack start request."""
        from aiohttp import web

        try:
            data = await request.json()
            execution_id = data.get("executionId")

            if execution_id not in self._attack_states:
                return web.json_response(
                    {"error": "Unknown execution ID"},
                    status=404,
                )

            state = self._attack_states[execution_id]
            new_state = await self.extension.start_attack(state)
            self._attack_states[execution_id] = new_state

            return web.json_response({
                "executionId": execution_id,
                "state": new_state,
            })
        except Exception as e:
            logger.error(f"Start error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def _handle_stop(self, request):
        """Handle attack stop request."""
        from aiohttp import web

        try:
            data = await request.json()
            execution_id = data.get("executionId")

            if execution_id not in self._attack_states:
                return web.json_response(
                    {"error": "Unknown execution ID"},
                    status=404,
                )

            state = self._attack_states[execution_id]
            await self.extension.stop_attack(state)

            del self._attack_states[execution_id]

            return web.json_response({"status": "stopped"})
        except Exception as e:
            logger.error(f"Stop error: {e}")
            return web.json_response({"error": str(e)}, status=500)


class ConsciousnessExtension(SteadybitExtension):
    """Consciousness Nexus Steadybit extension.

    Provides chaos engineering capabilities for Consciousness services.
    """

    @property
    def extension_id(self) -> str:
        return "consciousness-nexus"

    @property
    def extension_name(self) -> str:
        return "Consciousness Nexus"

    @property
    def extension_version(self) -> str:
        return "1.0.0"

    def __init__(self):
        self._service_registry: Dict[str, Dict[str, Any]] = {}

    def register_service(
        self,
        service_id: str,
        name: str,
        attributes: Dict[str, Any],
    ):
        """Register a service for discovery.

        Args:
            service_id: Unique service ID
            name: Service name
            attributes: Service attributes
        """
        self._service_registry[service_id] = {
            "id": service_id,
            "name": name,
            "attributes": attributes,
        }

    def get_target_types(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "consciousness-service",
                "name": "Consciousness Service",
                "category": "application",
                "attributes": [
                    {"name": "service.name", "type": "string"},
                    {"name": "service.version", "type": "string"},
                    {"name": "service.environment", "type": "string"},
                    {"name": "service.mesh.zone", "type": "string"},
                ],
            },
            {
                "id": "consciousness-processor",
                "name": "Consciousness Processor",
                "category": "application",
                "attributes": [
                    {"name": "processor.type", "type": "string"},
                    {"name": "processor.capabilities", "type": "label"},
                ],
            },
        ]

    async def discover_targets(
        self,
        target_type: str,
    ) -> List[DiscoveryTarget]:
        targets = []

        for service_id, service in self._service_registry.items():
            if target_type == "consciousness-service":
                targets.append(DiscoveryTarget(
                    id=service_id,
                    target_type=target_type,
                    label=service["name"],
                    attributes=service["attributes"],
                ))

        return targets

    def get_attack_definitions(self) -> List[AttackDefinition]:
        return [
            AttackDefinition(
                id="consciousness-delay-processing",
                name="Delay Processing",
                description="Add delay to consciousness processing",
                category="application",
                target_type="consciousness-processor",
                parameters=[
                    {"name": "delay_ms", "type": "integer", "default": 100},
                    {"name": "processor_type", "type": "string", "default": "*"},
                ],
            ),
            AttackDefinition(
                id="consciousness-fail-mesh-node",
                name="Fail Mesh Node",
                description="Simulate mesh node failure",
                category="network",
                target_type="consciousness-service",
                parameters=[
                    {"name": "failure_mode", "type": "select", "options": ["crash", "hang", "slow"]},
                    {"name": "recovery_seconds", "type": "integer", "default": 30},
                ],
            ),
            AttackDefinition(
                id="consciousness-corrupt-state",
                name="Corrupt State",
                description="Inject state corruption for resilience testing",
                category="state",
                target_type="consciousness-service",
                parameters=[
                    {"name": "corruption_type", "type": "select", "options": ["random", "clear", "duplicate"]},
                    {"name": "percentage", "type": "number", "default": 10},
                ],
            ),
        ]

    async def prepare_attack(
        self,
        attack_id: str,
        target: DiscoveryTarget,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "attack_id": attack_id,
            "target": target.to_api_format(),
            "config": config,
            "prepared_at": datetime.now().isoformat(),
        }

    async def start_attack(self, state: Dict[str, Any]) -> Dict[str, Any]:
        attack_id = state.get("attack_id")
        target = state.get("target", {})

        logger.info(f"Starting attack {attack_id} on {target.get('label')}")

        return {
            **state,
            "active": True,
            "started_at": datetime.now().isoformat(),
        }

    async def stop_attack(self, state: Dict[str, Any]):
        attack_id = state.get("attack_id")
        target = state.get("target", {})

        logger.info(f"Stopping attack {attack_id} on {target.get('label')}")
