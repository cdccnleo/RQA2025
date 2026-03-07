import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 集群管理器

实现集群管理功能：
- 集群信息管理
- 节点注册和注销
- 集群状态监控
- 集群配置管理
"""

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


logger = logging.getLogger(__name__)


class ClusterStatus(Enum):

    """集群状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


@dataclass
class ClusterInfo:

    """集群信息数据类"""
    cluster_id: str
    name: str
    status: ClusterStatus
    version: str
    created_at: datetime
    node_count: int
    active_nodes: int
    total_cpu: float
    total_memory: float
    metadata: Dict[str, Any]


class ClusterManager:

    """集群管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化集群管理器

        Args:
            config: 配置参数
        """
        self.config = config or {}

        # 集群信息
        self.cluster_info = ClusterInfo(
            cluster_id=str(uuid.uuid4()),
            name=self.config.get('cluster_name', 'RQA2025-Cluster'),
            status=ClusterStatus.ACTIVE,
            version='1.0.0',
            created_at=datetime.now(),
            node_count=0,
            active_nodes=0,
            total_cpu=0.0,
            total_memory=0.0,
            metadata={}
        )

        # 节点管理
        self.nodes: Dict[str, Dict[str, Any]] = {}

        logger.info(f"ClusterManager initialized for cluster {self.cluster_info.cluster_id}")

    def get_cluster_info(self) -> Dict[str, Any]:
        """获取集群信息"""
        return {
            'cluster_id': self.cluster_info.cluster_id,
            'name': self.cluster_info.name,
            'status': self.cluster_info.status.value,
            'version': self.cluster_info.version,
            'created_at': self.cluster_info.created_at.isoformat(),
            'node_count': self.cluster_info.node_count,
            'active_nodes': self.cluster_info.active_nodes,
            'total_cpu': self.cluster_info.total_cpu,
            'total_memory': self.cluster_info.total_memory,
            'metadata': self.cluster_info.metadata
        }

    def register_node(self, node_id: str, node_info: Dict[str, Any]):
        """注册节点"""
        self.nodes[node_id] = node_info
        self.cluster_info.node_count += 1

        # 更新集群资源统计
        if 'cpu_usage' in node_info:
            self.cluster_info.total_cpu += node_info.get('cpu_usage', 0.0)
        if 'memory_usage' in node_info:
            self.cluster_info.total_memory += node_info.get('memory_usage', 0.0)

        logger.info(f"Registered node {node_id} to cluster {self.cluster_info.cluster_id}")

    def unregister_node(self, node_id: str):
        """注销节点"""
        if node_id in self.nodes:
            node_info = self.nodes[node_id]

            # 更新集群资源统计
            if 'cpu_usage' in node_info:
                self.cluster_info.total_cpu -= node_info.get('cpu_usage', 0.0)
            if 'memory_usage' in node_info:
                self.cluster_info.total_memory -= node_info.get('memory_usage', 0.0)

            del self.nodes[node_id]
            self.cluster_info.node_count -= 1

            logger.info(f"Unregistered node {node_id} from cluster {self.cluster_info.cluster_id}")

    def update_cluster_status(self, status: ClusterStatus):
        """更新集群状态"""
        self.cluster_info.status = status
        logger.info(f"Updated cluster status to {status.value}")

    def get_node_list(self) -> List[Dict[str, Any]]:
        """获取节点列表"""
        return list(self.nodes.values())

    def get_cluster_stats(self) -> Dict[str, Any]:
        """获取集群统计信息"""
        return {
            'total_nodes': self.cluster_info.node_count,
            'active_nodes': self.cluster_info.active_nodes,
            'total_cpu': self.cluster_info.total_cpu,
            'total_memory': self.cluster_info.total_memory,
            'average_cpu': self.cluster_info.total_cpu / max(1, self.cluster_info.node_count),
            'average_memory': self.cluster_info.total_memory / max(1, self.cluster_info.node_count)
        }

    @property
    def status(self) -> str:
        """获取集群状态"""
        return self.cluster_info.status.value

    def add_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """添加节点（测试兼容性方法）"""
        self.register_node(node_id, node_info)
        return True

    def remove_node(self, node_id: str) -> bool:
        """移除节点（测试兼容性方法）"""
        if node_id in self.nodes:
            self.unregister_node(node_id)
            return True
        return False

    def get_status(self) -> Dict[str, Any]:
        """获取集群状态（测试兼容性方法）"""
        return {
            "total_nodes": self.cluster_info.node_count,
            "active_nodes": self.cluster_info.active_nodes,
            "status": self.cluster_info.status.value
        }
