"""
cluster_manager.py

集群管理器模块

提供分布式集群的核心管理功能：
- 节点注册和注销
- 集群状态管理
- 健康监控
- 负载均衡协调
- 故障恢复

符合架构设计:
- 分布式协调器架构设计 (docs\architecture\distributed_coordinator_architecture_design.md)
- 集群管理器组件

作者: RQA2025 Team
日期: 2026-02-15
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

# 导入统一工作节点注册表
from src.distributed.registry import (
    get_unified_worker_registry,
    WorkerType,
    WorkerStatus,
    WorkerNode
)

# 配置日志
logger = logging.getLogger(__name__)


@dataclass
class ClusterStats:
    """集群统计信息"""
    total_nodes: int = 0
    active_nodes: int = 0
    offline_nodes: int = 0
    busy_nodes: int = 0
    idle_nodes: int = 0
    by_type: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)


class ClusterManager:
    """
    集群管理器
    
    管理分布式集群的核心功能，整合统一工作节点注册表。
    
    Attributes:
        _registry: 统一工作节点注册表
        _lock: 线程锁
        _running: 是否运行中
        
    Example:
        >>> manager = ClusterManager()
        >>> 
        >>> # 注册节点
        >>> manager.register_node(
        ...     node_id="worker_1",
        ...     node_type=WorkerType.FEATURE_WORKER,
        ...     capabilities={"cpu": 4}
        ... )
        >>> 
        >>> # 获取集群状态
        >>> stats = manager.get_cluster_stats()
    """
    
    def __init__(self):
        """初始化集群管理器"""
        self._registry = get_unified_worker_registry()
        self._lock = threading.RLock()
        self._running = False
        
        logger.info("ClusterManager 初始化完成")
    
    def register_node(
        self,
        node_id: str,
        node_type: WorkerType,
        capabilities: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        注册集群节点
        
        Args:
            node_id: 节点ID
            node_type: 节点类型
            capabilities: 节点能力
            metadata: 元数据
            
        Returns:
            是否注册成功
        """
        return self._registry.register_worker(
            worker_id=node_id,
            worker_type=node_type,
            capabilities=capabilities,
            metadata=metadata
        )
    
    def unregister_node(self, node_id: str) -> bool:
        """
        注销集群节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否注销成功
        """
        return self._registry.unregister_worker(node_id)
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """
        更新节点心跳
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否更新成功
        """
        return self._registry.update_heartbeat(node_id)
    
    def update_node_status(self, node_id: str, status: WorkerStatus) -> bool:
        """
        更新节点状态
        
        Args:
            node_id: 节点ID
            status: 新状态
            
        Returns:
            是否更新成功
        """
        return self._registry.update_status(node_id, status)
    
    def get_node(self, node_id: str) -> Optional[WorkerNode]:
        """
        获取节点信息
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点信息
        """
        return self._registry.get_worker(node_id)
    
    def get_nodes_by_type(self, node_type: WorkerType) -> List[WorkerNode]:
        """
        获取指定类型的节点
        
        Args:
            node_type: 节点类型
            
        Returns:
            节点列表
        """
        return self._registry.get_workers_by_type(node_type)
    
    def get_available_nodes(self, node_type: Optional[WorkerType] = None) -> List[str]:
        """
        获取可用节点ID列表
        
        Args:
            node_type: 可选，指定节点类型
            
        Returns:
            可用节点ID列表
        """
        return self._registry.get_available_workers(node_type)
    
    def get_cluster_stats(self) -> ClusterStats:
        """
        获取集群统计信息
        
        Returns:
            集群统计信息
        """
        registry_stats = self._registry.get_stats()
        
        stats = ClusterStats()
        stats.total_nodes = registry_stats.get("total_workers", 0)
        stats.active_nodes = registry_stats.get("active_workers", 0)
        stats.offline_nodes = registry_stats.get("offline_workers", 0)
        
        # 计算忙碌和空闲节点
        by_type_detailed = registry_stats.get("by_type_detailed", {})
        for type_name, type_stats in by_type_detailed.items():
            stats.idle_nodes += type_stats.get("idle", 0)
            stats.busy_nodes += type_stats.get("busy", 0)
        
        stats.by_type = by_type_detailed
        stats.last_updated = datetime.now()
        
        return stats
    
    def check_cluster_health(self, timeout_seconds: int = 30) -> Dict[str, Any]:
        """
        检查集群健康状态
        
        Args:
            timeout_seconds: 超时时间（秒）
            
        Returns:
            健康状态报告
        """
        unhealthy_nodes = self._registry.check_health(timeout_seconds)
        stats = self.get_cluster_stats()
        
        health_status = {
            "status": "healthy",
            "total_nodes": stats.total_nodes,
            "active_nodes": stats.active_nodes,
            "offline_nodes": stats.offline_nodes,
            "unhealthy_nodes": unhealthy_nodes,
            "unhealthy_count": len(unhealthy_nodes),
            "last_checked": datetime.now().isoformat()
        }
        
        # 判断整体健康状态
        if stats.total_nodes == 0:
            health_status["status"] = "warning"
            health_status["reason"] = "集群中没有节点"
        elif stats.active_nodes == 0:
            health_status["status"] = "critical"
            health_status["reason"] = "集群中没有活跃节点"
        elif len(unhealthy_nodes) > stats.total_nodes * 0.5:
            health_status["status"] = "warning"
            health_status["reason"] = "超过50%的节点不健康"
        
        return health_status
    
    def cleanup_offline_nodes(self, max_offline_minutes: int = 10) -> int:
        """
        清理离线节点
        
        Args:
            max_offline_minutes: 最大离线时间（分钟）
            
        Returns:
            清理的节点数量
        """
        return self._registry.cleanup_offline_workers(max_offline_minutes)
    
    def get_all_nodes(self) -> List[WorkerNode]:
        """
        获取所有节点
        
        Returns:
            节点列表
        """
        return self._registry.get_all_workers()


# 全局集群管理器实例
_global_cluster_manager: Optional[ClusterManager] = None


def get_cluster_manager() -> ClusterManager:
    """
    获取全局集群管理器实例
    
    Returns:
        集群管理器实例
    """
    global _global_cluster_manager
    
    if _global_cluster_manager is None:
        _global_cluster_manager = ClusterManager()
    
    return _global_cluster_manager
