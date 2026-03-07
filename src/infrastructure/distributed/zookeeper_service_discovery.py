"""Simplified ZooKeeper discovery implementation for tests."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from .service_mesh import (
    InMemoryServiceDiscovery,
    ServiceDiscovery,
    ServiceDiscoveryRequest,
    ServiceInstance,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ZooKeeperConfig:
    hosts: str = "localhost:2181"
    base_path: str = "/services"
    session_timeout: int = 30_000
    connection_timeout: int = 10_000
    auth_scheme: Optional[str] = None
    auth_data: Optional[str] = None
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    ephemeral_nodes: bool = True


def _run_sync(coro: asyncio.Future) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class ZooKeeperServiceDiscovery(ServiceDiscovery):
    """In-memory ZooKeeper mock."""

    def __init__(self, config: Optional[ZooKeeperConfig] = None) -> None:
        self.config = config or ZooKeeperConfig()
        self._registry = InMemoryServiceDiscovery()
        self._watchers: Dict[str, Callable[[List[ServiceInstance]], None]] = {}
        self._registered_services: Dict[str, ServiceInstance] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._health_check_interval = max(self.config.health_check_interval / 1000.0, 0.05)
        self._connected: bool = False
        self._services: Dict[str, ServiceInstance] = {}

    def is_connected(self) -> bool:
        """Check if the service discovery is connected."""
        return self._connected

    def connect(self) -> bool:
        """Connect to ZooKeeper."""
        try:
            self._connected = self._connect_zookeeper()
            return self._connected
        except Exception:
            return False

    def disconnect(self) -> bool:
        """Disconnect from ZooKeeper."""
        try:
            self._disconnect_zookeeper()
            self._connected = False
            return True
        except Exception:
            return False

    def _connect_zookeeper(self) -> bool:
        """Internal method to connect to ZooKeeper."""
        # Simplified implementation - in real implementation would connect to actual ZooKeeper
        logger.info("ZooKeeper 服务发现连接成功")
        return True

    def _disconnect_zookeeper(self) -> None:
        """Internal method to disconnect from ZooKeeper."""
        # Simplified implementation
        pass

    async def register(self, instance: ServiceInstance) -> bool:
        if instance.status is None:
            instance.status = ServiceStatus.HEALTHY
        await self._registry.register(instance)
        self._registered_services[instance.id] = instance
        await self._start_health_check(instance)
        await self._notify(instance.service_name or instance.name)
        return True

    async def deregister(self, instance_id: str) -> bool:
        removed = await self._registry.deregister(instance_id)
        if removed:
            task = self._health_check_tasks.pop(instance_id, None)
            if task:
                task.cancel()
            self._registered_services.pop(instance_id, None)
            await self._notify_all()
        return removed

    async def heartbeat(self, instance_id: str) -> bool:
        return await self._registry.heartbeat(instance_id)

    async def discover(self, request: ServiceDiscoveryRequest) -> List[ServiceInstance]:
        instances = await self._registry.discover(request)
        if request.tags:
            instances = [inst for inst in instances if self._matches_tags(inst, request.tags or [])]
        if request.only_healthy:
            instances = [inst for inst in instances if inst.is_healthy()]
        return instances

    async def _notify(self, service_name: str) -> None:
        if service_name in self._watchers:
            instances = await self.discover(ServiceDiscoveryRequest(service_name))
            try:
                self._watchers[service_name](instances)
            except Exception as exc:  # pragma: no cover
                logger.debug("Watcher callback error: %s", exc)

    async def _notify_all(self) -> None:
        for service_name in list(self._watchers.keys()):
            await self._notify(service_name)

    def _matches_tags(self, instance: ServiceInstance, tags: List[str]) -> bool:
        return set(tags).issubset(set(instance.metadata.get("tags", [])))

    def _serialize_instance(self, instance: ServiceInstance) -> str:
        payload = {
            "id": instance.id,
            "name": instance.name,
            "host": instance.host,
            "port": instance.port,
            "status": instance.status.value,
            "metadata": instance.metadata,
            "weight": instance.weight,
            "last_health_check": instance.last_health_check,
        }
        return json.dumps(payload)

    def _deserialize_instance(self, data: Any) -> Optional[ServiceInstance]:
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        if not isinstance(data, dict) or "id" not in data:
            return None
        return ServiceInstance(
            id=data.get("id"),
            name=data.get("name", ""),
            host=data.get("host", "localhost"),
            port=int(data.get("port", 0)),
            status=ServiceStatus(data.get("status", ServiceStatus.HEALTHY.value)),
            metadata=data.get("metadata", {}),
            weight=data.get("weight", 1),
            last_health_check=data.get("last_health_check"),
        )

    async def _start_health_check(self, instance: ServiceInstance) -> None:
        if instance.id in self._health_check_tasks:
            return

        async def _health_loop() -> None:
            while instance.id in self._registered_services:
                await self._perform_health_check(instance)
                await asyncio.sleep(self._health_check_interval)

        self._health_check_tasks[instance.id] = asyncio.create_task(_health_loop())

    async def _perform_health_check(self, instance: ServiceInstance) -> None:
        instance.last_health_check = time.time()

    def watch_service(self, service_name: str, callback: Callable[[List[ServiceInstance]], None]) -> None:
        self._watchers[service_name] = callback

    def unwatch_service(self, service_name: str) -> None:
        self._watchers.pop(service_name, None)

    def register_service(
        self, name: str, host: str, port: int, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        instance = ServiceInstance(
            name=name,
            service_name=name,
            host=host,
            port=port,
            metadata=metadata or {},
            status=ServiceStatus.HEALTHY,
        )
        return bool(_run_sync(self.register(instance)))

    def discover_service(self, name: str) -> List[ServiceInstance]:
        request = ServiceDiscoveryRequest(service_name=name)
        return list(_run_sync(self.discover(request)))

    def deregister_service(self, instance_id: str) -> bool:
        return bool(_run_sync(self.deregister(instance_id)))


__all__ = ["ZooKeeperConfig", "ZooKeeperServiceDiscovery"]

