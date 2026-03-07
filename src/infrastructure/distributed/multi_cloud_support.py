"""Minimal multi-cloud support primitives for unit tests."""

from __future__ import annotations

import asyncio
import logging
import sys
import types
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    import boto3  # type: ignore
except ImportError:  # pragma: no cover - fallback stub for tests
    boto3 = types.ModuleType("boto3")

    def _missing_client(*args: Any, **kwargs: Any):
        raise ImportError("boto3 not installed")

    boto3.client = _missing_client  # type: ignore[attr-defined]
    sys.modules.setdefault("boto3", boto3)


logger = logging.getLogger(__name__)


class CloudProvider(Enum):
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    ALICLOUD = "alicloud"
    HUAWEI = "huawei"
    TENCENT = "tencent"
    ON_PREMISE = "on_premise"


@dataclass
class CloudConfig:
    provider: CloudProvider
    region: str = "global"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    session_token: Optional[str] = None
    endpoint_url: Optional[str] = None
    credentials: Dict[str, Any] = field(default_factory=dict)
    custom_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CloudServiceInstance:
    instance_id: str
    service_name: str
    cloud_provider: CloudProvider
    region: str
    availability_zone: str
    private_ip: str
    public_ip: Optional[str] = None
    port: int = 80
    health_status: str = "unknown"
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0


class CloudAdapter:
    """Abstract adapter stub; calls operate on in-memory registry."""

    def __init__(self, config: CloudConfig) -> None:
        self.config = config
        self._registry: Dict[str, CloudServiceInstance] = {}

    async def discover_services(self, service_name: str) -> List[CloudServiceInstance]:
        await asyncio.sleep(0)
        return [inst for inst in self._registry.values() if inst.service_name == service_name]

    async def register_service(self, instance: CloudServiceInstance) -> bool:
        await asyncio.sleep(0)
        self._registry[instance.instance_id] = instance
        return True

    async def deregister_service(self, instance_id: str) -> bool:
        await asyncio.sleep(0)
        return self._registry.pop(instance_id, None) is not None

    async def get_service_health(self, instance_id: str) -> str:
        await asyncio.sleep(0)
        instance = self._registry.get(instance_id)
        return instance.health_status if instance else "unknown"

    async def update_service_metadata(self, instance_id: str, metadata: Dict[str, Any]) -> bool:
        await asyncio.sleep(0)
        instance = self._registry.get(instance_id)
        if not instance:
            return False
        instance.metadata.update(metadata)
        return True


def _ensure_boto3_stub() -> Any:
    try:
        import boto3  # type: ignore
        return boto3
    except ImportError:
        module = types.ModuleType("boto3")

        def _client(*args, **kwargs):
            raise ImportError("boto3 not installed")

        module.client = _client  # type: ignore[attr-defined]
        sys.modules.setdefault("boto3", module)
        return module


class AWSAdapter(CloudAdapter):
    """Memory backed adapter; still initialises boto3 client when available."""

    def __init__(self, config: CloudConfig) -> None:
        super().__init__(config)
        self._client = None
        boto3 = _ensure_boto3_stub()
        try:
            self._client = boto3.client(
                "servicediscovery",
                region_name=config.region,
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                aws_session_token=config.session_token,
                endpoint_url=config.endpoint_url,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("AWS client initialisation skipped: %s", exc)
            self._client = None


class AzureAdapter(CloudAdapter):
    pass


class GCPAdapter(CloudAdapter):
    pass


class MultiCloudManager:
    """High level manager aggregating different cloud adapters."""

    def __init__(self, configs: Optional[List[CloudConfig]] = None) -> None:
        self.adapters: Dict[CloudProvider, CloudAdapter] = {}
        self.deployments: Dict[tuple[CloudProvider, str], str] = {}

        if configs:
            for config in configs:
                adapter = self.create_adapter_for_provider(config)
                if adapter:
                    self.add_cloud_adapter(adapter)

    def add_cloud_adapter(self, adapter: CloudAdapter) -> None:
        self.adapters[adapter.config.provider] = adapter

    async def discover_services(
        self,
        service_name: str,
        providers: Optional[List[CloudProvider]] = None,
    ) -> List[CloudServiceInstance]:
        providers = providers or list(self.adapters.keys())
        tasks = [
            self.adapters[provider].discover_services(service_name)
            for provider in providers
            if provider in self.adapters
        ]
        if not tasks:
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        instances: List[CloudServiceInstance] = []
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Discovery exception: %s", result)
                continue
            instances.extend(result)
        return instances

    async def register_service_instance(self, instance: CloudServiceInstance) -> bool:
        adapter = self.adapters.get(instance.cloud_provider)
        if not adapter:
            return False
        return await adapter.register_service(instance)

    async def get_service_health(self, instance: CloudServiceInstance) -> str:
        adapter = self.adapters.get(instance.cloud_provider)
        if not adapter:
            return "unknown"
        return await adapter.get_service_health(instance.instance_id)

    async def sync_service_metadata(
        self,
        instance: CloudServiceInstance,
        metadata: Dict[str, Any],
    ) -> bool:
        adapter = self.adapters.get(instance.cloud_provider)
        if not adapter:
            return False
        return await adapter.update_service_metadata(instance.instance_id, metadata)

    def get_cloud_stats(self) -> Dict[str, Any]:
        return {
            "configured_providers": [provider.value for provider in self.adapters],
            "total_providers": len(self.adapters),
            "providers": {
                provider.value: {"region": adapter.config.region, "status": "configured"}
                for provider, adapter in self.adapters.items()
            },
        }

    async def health_check_all_providers(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for provider, adapter in self.adapters.items():
            instances = await adapter.discover_services("health-check")
            results[provider.value] = {
                "status": "healthy",
                "message": f"{len(instances)} instances registered",
            }
        return results

    def deploy_service(self, service_name: str, provider: CloudProvider) -> str:
        self.deployments[(provider, service_name)] = "deployed"
        return "deployed"

    def get_service_status(self, service_name: str, provider: CloudProvider) -> str:
        return self.deployments.get((provider, service_name), "unknown")

    def create_adapter_for_provider(self, config: CloudConfig) -> Optional[CloudAdapter]:
        if config.provider == CloudProvider.AWS:
            return AWSAdapter(config)
        if config.provider == CloudProvider.AZURE:
            return AzureAdapter(config)
        if config.provider == CloudProvider.GCP:
            return GCPAdapter(config)
        logger.debug("No adapter for provider %s", config.provider)
        return None


class MultiCloudSupport:
    """非常轻量的多云管理门面，满足补充测试需求。"""

    def __init__(self) -> None:
        self._providers: Dict[str, Dict[str, Any]] = {}
        self._deployments: List[Dict[str, Any]] = []

    def register_provider(self, name: str | Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> bool:
        if isinstance(name, dict):
            provider_name = name.get("name")
            if not provider_name:
                return False
            self._providers[provider_name] = dict(name)
            return True
        if not name:
            return False
        self._providers[name] = dict(config or {})
        return True

    def get_provider(self, name: str) -> Optional[Dict[str, Any]]:
        return self._providers.get(name)

    def list_providers(self) -> List[Dict[str, Any]]:
        return [dict(value, name=key) for key, value in self._providers.items()]

    def deploy(self, provider: str | Dict[str, Any], payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if isinstance(provider, dict):
            info = dict(provider)
        else:
            info = {"provider": provider}
            info.update(payload or {})
        self._deployments.append(info)
        return info

    def migrate(self, service: str, source: str, target: str) -> bool:
        self._deployments.append({"service": service, "from": source, "to": target})
        return True

    def remove_provider(self, name: str) -> bool:
        return self._providers.pop(name, None) is not None

    def health_check(self, name: str) -> bool:
        return name in self._providers


def create_multi_cloud_manager(configs: List[CloudConfig]) -> MultiCloudManager:
    return MultiCloudManager(configs)


__all__ = [
    "CloudProvider",
    "CloudConfig",
    "CloudServiceInstance",
    "CloudAdapter",
    "AWSAdapter",
    "AzureAdapter",
    "GCPAdapter",
    "MultiCloudManager",
    "MultiCloudSupport",
    "create_multi_cloud_manager",
]

