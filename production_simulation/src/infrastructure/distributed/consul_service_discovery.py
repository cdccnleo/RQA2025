"""Simplified Consul service discovery used by unit tests."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .service_mesh import (
    InMemoryServiceDiscovery,
    ServiceDiscovery,
    ServiceDiscoveryRequest,
    ServiceInstance,
    ServiceStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ConsulConfig:
    host: str = "localhost"
    port: int = 8500
    scheme: str = "http"
    token: Optional[str] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    health_check_interval: int = 30
    deregister_critical_service_after: str = "30s"


def _run_sync(coro: asyncio.Future) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class ConsulServiceDiscovery(ServiceDiscovery):
    """In-memory Consul mock compatible with tests."""

    def __init__(self, config: Optional[ConsulConfig] = None) -> None:
        self.config = config or ConsulConfig()
        self._registry = InMemoryServiceDiscovery()
        self._request_log: List[Dict[str, Any]] = []
        self._registered_services: Dict[str, ServiceInstance] = {}

    @staticmethod
    def _in_async_context() -> bool:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        return loop.is_running()

    def register(
        self,
        instance: ServiceInstance | str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool | asyncio.Future:
        if isinstance(instance, ServiceInstance):
            coro = self._register_async(instance)
        else:
            service_instance = ServiceInstance(
                id=f"{instance}-{host}:{port}",
                name=str(instance),
                host=host or "localhost",
                port=int(port or 0),
                status=ServiceStatus.HEALTHY,
                metadata=metadata or {},
            )
            coro = self._register_async(service_instance)

        if self._in_async_context():
            return coro
        return _run_sync(coro)

    async def _register_async(self, instance: ServiceInstance) -> bool:
        instance.status = instance.status or ServiceStatus.HEALTHY
        await self._registry.register(instance)
        self._registered_services[instance.id] = instance
        return True

    def deregister(self, instance_id: str) -> bool | asyncio.Future:
        coro = self._deregister_async(instance_id)
        if self._in_async_context():
            return coro
        return _run_sync(coro)

    async def _deregister_async(self, instance_id: str) -> bool:
        removed = await self._registry.deregister(instance_id)
        if removed:
            self._registered_services.pop(instance_id, None)
        return removed

    def heartbeat(self, instance_id: str) -> bool | asyncio.Future:
        coro = self._heartbeat_async(instance_id)
        if self._in_async_context():
            return coro
        return _run_sync(coro)

    async def _heartbeat_async(self, instance_id: str) -> bool:
        return await self._registry.heartbeat(instance_id)

    def discover(
        self, request: ServiceDiscoveryRequest | str
    ) -> List[ServiceInstance] | asyncio.Future:
        if isinstance(request, str):
            request = ServiceDiscoveryRequest(service_name=request)

        coro = self._discover_async(request)
        if self._in_async_context():
            return coro
        return _run_sync(coro)

    async def _discover_async(self, request: ServiceDiscoveryRequest) -> List[ServiceInstance]:
        path = f"/catalog/service/{request.service_name}"
        raw = await self._make_request("GET", path)
        if not raw:
            return []

        instances: List[ServiceInstance] = []
        for payload in raw:
            service = payload.get("Service", {})
            instance = ServiceInstance(
                id=service.get("ID"),
                name=service.get("Service", request.service_name),
                host=service.get("Address", "localhost"),
                port=int(service.get("Port", 0)),
                status=ServiceStatus.HEALTHY,
                metadata=service.get("Meta", {}),
                weight=service.get("Weights", {}).get("Passing", 1),
            )
            instances.append(instance)

        if request.tags:
            instances = [
                inst
                for inst in instances
                if set(request.tags or []).issubset(set(inst.metadata.get("tags", [])))
            ]
        if request.only_healthy:
            instances = [inst for inst in instances if inst.is_healthy()]
        return instances

    async def close(self) -> None:
        self._request_log.clear()

    async def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        await asyncio.sleep(0)
        self._request_log.append({"method": method, "path": path, "data": data})

        if method == "GET" and path.startswith("/catalog/service/"):
            service_name = path.split("/")[-1]
            stored = await self._registry.discover(ServiceDiscoveryRequest(service_name))
            return [
                {
                    "Service": {
                        "ID": inst.id,
                        "Service": inst.service_name,
                        "Address": inst.host,
                        "Port": inst.port,
                        "Meta": inst.metadata,
                        "Weights": {"Passing": inst.weight},
                    }
                }
                for inst in stored
            ]

        if method == "PUT" and path.startswith("/agent/service/register") and data:
            instance = ServiceInstance(
                id=data.get("ID"),
                name=data.get("Name", ""),
                host=data.get("Address", "localhost"),
                port=int(data.get("Port", 0)),
                status=ServiceStatus.HEALTHY,
                metadata=data.get("Meta", {}),
                weight=data.get("Weights", {}).get("Passing", 1),
            )
            await self._registry.register(instance)
            return [{"Service": data}]

        if method == "PUT" and path.startswith("/agent/service/deregister/"):
            instance_id = path.split("/")[-1]
            await self._registry.deregister(instance_id)
            return [{"success": True}]

        return []

    def register_service(
        self, service_name: str, host: str, port: int, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        result = self.register(
            service_name,
            host=host,
            port=port,
            metadata=metadata,
        )
        if isinstance(result, bool):
            return result
        return bool(_run_sync(result))

    def discover_service(self, service_name: str) -> List[ServiceInstance]:
        request = ServiceDiscoveryRequest(service_name=service_name)
        result = self.discover(request)
        if isinstance(result, list):
            return result
        return list(_run_sync(result))

    def deregister_service(self, instance_id: str) -> bool:
        result = self.deregister(instance_id)
        if isinstance(result, bool):
            return result
        return bool(_run_sync(result))


__all__ = ["ConsulConfig", "ConsulServiceDiscovery"]

