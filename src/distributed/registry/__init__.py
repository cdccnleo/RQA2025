"""
分布式协调器层 - 注册表模块

提供分布式系统中工作节点和服务的注册管理功能。

符合架构设计:
- 分布式协调器架构设计 (docs\architecture\distributed_coordinator_architecture_design.md)

模块组件:
- unified_worker_registry: 统一工作节点注册表

作者: RQA2025 Team
日期: 2026-02-15
"""

from .unified_worker_registry import (
    UnifiedWorkerRegistry,
    WorkerType,
    WorkerStatus,
    WorkerNode,
    get_unified_worker_registry
)

__all__ = [
    'UnifiedWorkerRegistry',
    'WorkerType',
    'WorkerStatus',
    'WorkerNode',
    'get_unified_worker_registry'
]
