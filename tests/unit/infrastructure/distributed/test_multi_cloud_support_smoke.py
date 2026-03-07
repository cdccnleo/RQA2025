import pytest

from src.infrastructure.distributed.multi_cloud_support import (
    CloudConfig,
    CloudProvider,
    CloudServiceInstance,
    MultiCloudManager,
    MultiCloudSupport,
)


@pytest.mark.asyncio
async def test_multi_cloud_manager_basic():
    manager = MultiCloudManager()

    aws_adapter_config = CloudConfig(provider=CloudProvider.AWS, region="us-east-1")
    aws_adapter = manager.create_adapter_for_provider(aws_adapter_config)
    assert aws_adapter is not None
    manager.add_cloud_adapter(aws_adapter)

    instance = CloudServiceInstance(
        instance_id="i-001",
        service_name="orders",
        cloud_provider=CloudProvider.AWS,
        region="us-east-1",
        availability_zone="us-east-1a",
        private_ip="10.0.0.10",
    )

    assert await manager.register_service_instance(instance)
    discover_results = await manager.discover_services("orders")
    assert discover_results and discover_results[0].instance_id == "i-001"

    assert await manager.sync_service_metadata(instance, {"version": "1.0"})
    health = await manager.get_service_health(instance)
    assert isinstance(health, str)

    assert manager.deploy_service("orders", CloudProvider.AWS) == "deployed"
    assert manager.get_service_status("orders", CloudProvider.AWS) == "deployed"


def test_multi_cloud_support_facade():
    support = MultiCloudSupport()

    assert support.register_provider("aws", {"region": "us-east-1"})
    assert support.health_check("aws")

    deployment = support.deploy("aws", {"service": "orders"})
    assert deployment["provider"] == "aws"

    migration = support.migrate("orders", "aws", "gcp")
    assert migration

    providers = support.list_providers()
    assert any(p["name"] == "aws" for p in providers)

    assert support.remove_provider("aws")

