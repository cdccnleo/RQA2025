"""
分布式模块 - 支持分布式训练和协调
"""

from .discovery.service_discovery import ServiceRegistry, ServiceDiscoveryClient
try:
    from .discovery.service_discovery import ServiceDiscovery
except ImportError:
    ServiceDiscovery = None

from .coordinator.coordinator import (
    DistributedCoordinator,
    get_distributed_coordinator
)

__all__ = [
    'ServiceRegistry',
    'ServiceDiscoveryClient',
    'ServiceDiscovery',
    'DistributedCoordinator',
    'get_distributed_coordinator'
]

