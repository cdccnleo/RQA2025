"""Lightweight service mesh utilities used for distributed module tests."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"


@dataclass
class ServiceInstance:
    id: Optional[str] = None
    name: str = ""
    host: str = "localhost"
    port: int = 0
    status: ServiceStatus = ServiceStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    weight: int = 1
    service_name: Optional[str] = None
    last_health_check: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.service_name:
            self.service_name = self.name
        if not self.name:
            self.name = self.service_name or ""
        if self.id is None:
            self.id = f"{self.name}-{self.host}:{self.port}"

    def is_healthy(self) -> bool:
        return self.status == ServiceStatus.HEALTHY

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"


@dataclass
class ServiceDiscoveryRequest:
    service_name: str
    tags: Optional[List[str]] = None
    only_healthy: bool = True


@dataclass
class ServiceCallRequest:
    service_name: str
    method: str
    path: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    body: Any = None
    timeout: float = 5.0


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    timeout_duration: float = 30.0
    recovery_timeout: float = 10.0
    success_threshold: int = 1


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(self, service_name: str, config: CircuitBreakerConfig) -> None:
        self.service_name = service_name
        self.config = config
        self._state = "closed"
        self._failures = 0
        self._last_failure = 0.0
        self._successes = 0

    @property
    def state(self) -> str:
        return self._state

    def call(self, func: Callable[[], Any]) -> Any:
        if self._state == "open":
            if time.time() - self._last_failure < self.config.timeout_duration:
                raise CircuitBreakerOpenError(f"Circuit breaker open for {self.service_name}")
            self._state = "half_open"
            self._successes = 0

        try:
            result = func()
            self._record_success()
            return result
        except Exception:
            self._record_failure()
            raise

    def _record_success(self) -> None:
        if self._state == "half_open":
            self._successes += 1
            if self._successes >= self.config.success_threshold:
                self._state = "closed"
                self._failures = 0
        else:
            self._failures = 0

    def _record_failure(self) -> None:
        self._failures += 1
        self._last_failure = time.time()
        if self._state == "half_open" or self._failures >= self.config.failure_threshold:
            self._state = "open"
        logger.debug("Circuit breaker failure for %s (count=%s)", self.service_name, self._failures)


class ServiceDiscovery:
    async def register(self, instance: ServiceInstance) -> bool:
        raise NotImplementedError

    async def deregister(self, instance_id: str) -> bool:
        raise NotImplementedError

    async def heartbeat(self, instance_id: str) -> bool:
        raise NotImplementedError

    async def discover(self, request: ServiceDiscoveryRequest) -> List[ServiceInstance]:
        raise NotImplementedError


class InMemoryServiceDiscovery(ServiceDiscovery):
    def __init__(self) -> None:
        self._services: Dict[str, Dict[str, ServiceInstance]] = {}

    async def register(self, instance: ServiceInstance) -> bool:
        await asyncio.sleep(0)
        bucket = self._services.setdefault(instance.service_name or instance.name, {})
        bucket[instance.id] = instance
        return True

    async def deregister(self, instance_id: str) -> bool:
        await asyncio.sleep(0)
        removed = False
        for bucket in self._services.values():
            if instance_id in bucket:
                del bucket[instance_id]
                removed = True
        return removed

    async def heartbeat(self, instance_id: str) -> bool:
        await asyncio.sleep(0)
        for bucket in self._services.values():
            inst = bucket.get(instance_id)
            if inst:
                inst.last_health_check = time.time()
                return True
        return False

    async def discover(self, request: ServiceDiscoveryRequest) -> List[ServiceInstance]:
        await asyncio.sleep(0)
        bucket = list(self._services.get(request.service_name, {}).values())
        if request.tags:
            bucket = [
                inst
                for inst in bucket
                if set(request.tags or []).issubset(set(inst.metadata.get("tags", [])))
            ]
        if request.only_healthy:
            bucket = [inst for inst in bucket if inst.is_healthy()]
        return list(bucket)


class LoadBalancer:
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN) -> None:
        self.strategy = strategy
        self._indices: Dict[str, int] = {}

    def select_instance(self, instances: List[ServiceInstance]) -> Optional[ServiceInstance]:
        if not instances:
            return None
        if self.strategy == LoadBalanceStrategy.RANDOM:
            return random.choice(instances)
        service_name = instances[0].service_name or instances[0].name
        index = self._indices.get(service_name, 0)
        chosen = instances[index % len(instances)]
        self._indices[service_name] = (index + 1) % len(instances)
        return chosen


def _run_sync(coro: asyncio.Future) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


class ServiceMeshIntegration:
    def __init__(
        self,
        service_discovery: Optional[ServiceDiscovery] = None,
        load_balancer: Optional[LoadBalancer] = None,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    ) -> None:
        self.service_discovery = service_discovery or InMemoryServiceDiscovery()
        self.load_balancer = load_balancer or LoadBalancer()
        self.circuit_breaker = CircuitBreaker(
            "service-mesh", circuit_breaker_config or CircuitBreakerConfig()
        )

    def register_service(self, instance: ServiceInstance) -> bool:
        return bool(_run_sync(self.service_discovery.register(instance)))

    def discover_service(self, service_name: str) -> List[ServiceInstance]:
        request = ServiceDiscoveryRequest(service_name=service_name)
        return list(_run_sync(self.service_discovery.discover(request)))

    def call_service(self, request: ServiceCallRequest) -> Optional[Dict[str, Any]]:
        instances = self.discover_service(request.service_name)
        if not instances:
            return None
        target = self.load_balancer.select_instance(instances)
        if not target:
            return None

        def _invoke() -> Dict[str, Any]:
            return {
                "service": target.service_name,
                "host": target.host,
                "port": target.port,
                "path": request.path,
                "method": request.method,
            }

        try:
            return self.circuit_breaker.call(_invoke)
        except CircuitBreakerOpenError:
            logger.warning("Circuit breaker open for %s", request.service_name)
            return None


class ServiceMesh:
    def __init__(self) -> None:
        self._services: Dict[str, List[Dict[str, Any]]] = {}

    def register_service(self, name: str, info: Dict[str, Any]) -> bool:
        bucket = self._services.setdefault(name, [])
        bucket.append(dict(info))
        return True

    def discover_service(self, name: str) -> List[Dict[str, Any]]:
        return list(self._services.get(name, []))

    def unregister_service(self, name: str) -> bool:
        return self._services.pop(name, None) is not None

    def health_check(self, name: str) -> bool:
        return name in self._services

    def load_balance(self, name: str) -> Optional[Dict[str, Any]]:
        services = self._services.get(name)
        if not services:
            return None
        return services[0]


__all__ = [
    "ServiceStatus",
    "LoadBalanceStrategy",
    "ServiceInstance",
    "ServiceDiscoveryRequest",
    "ServiceCallRequest",
    "CircuitBreakerConfig",
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "ServiceDiscovery",
    "InMemoryServiceDiscovery",
    "LoadBalancer",
    "ServiceMeshIntegration",
    "ServiceMesh",
]

