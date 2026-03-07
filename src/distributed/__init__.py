"""
分布式协调器层 - 支持分布式训练和协调

符合架构设计:
- 分布式协调器架构设计 (docs\architecture\distributed_coordinator_architecture_design.md)

子模块:
- coordinator: 集群协调器
- registry: 服务注册表
- discovery: 服务发现
- consistency: 一致性管理
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

from .registry import (
    UnifiedWorkerRegistry,
    WorkerType,
    WorkerStatus,
    WorkerNode,
    get_unified_worker_registry
)

__all__ = [
    'ServiceRegistry',
    'ServiceDiscoveryClient',
    'ServiceDiscovery',
    'DistributedCoordinator',
    'get_distributed_coordinator',
    'UnifiedWorkerRegistry',
    'WorkerType',
    'WorkerStatus',
    'WorkerNode',
    'get_unified_worker_registry'
]

