"""
集群管理器

负责节点注册、注销和集群状态管理。

从coordinator_core.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import threading
from typing import Dict, List, Any, Optional
from datetime import datetime

from .models import NodeInfo, NodeStatus, ClusterStats

logger = logging.getLogger(__name__)


class ClusterManager:
    """
    集群管理器
    
    负责:
    1. 节点注册和注销
    2. 节点状态管理
    3. 集群状态同步
    4. 集群统计信息
    """
    
    def __init__(self):
        self.nodes: Dict[str, NodeInfo] = {}
        self.stats = ClusterStats()
        self._lock = threading.RLock()
        
        logger.info("集群管理器初始化完成")
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """注册节点"""
        with self._lock:
            try:
                self.nodes[node_info.node_id] = node_info
                self._update_cluster_stats()
                
                logger.info(f"节点 {node_info.node_id} ({node_info.hostname}) 已注册")
                return True
                
            except Exception as e:
                logger.error(f"节点注册失败: {e}")
                return False
    
    def unregister_node(self, node_id: str) -> bool:
        """注销节点"""
        with self._lock:
            try:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    self._update_cluster_stats()
                    
                    logger.info(f"节点 {node_id} 已注销")
                    return True
                    
                return False
                
            except Exception as e:
                logger.error(f"节点注销失败: {e}")
                return False
    
    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """获取节点信息"""
        with self._lock:
            return self.nodes.get(node_id)
    
    def get_all_nodes(self) -> Dict[str, NodeInfo]:
        """获取所有节点"""
        with self._lock:
            return self.nodes.copy()
    
    def get_available_nodes(self) -> Dict[str, NodeInfo]:
        """获取所有可用节点"""
        with self._lock:
            return {
                node_id: node
                for node_id, node in self.nodes.items()
                if node.status == NodeStatus.ONLINE
            }
    
    def update_node_status(self, node_id: str, status: NodeStatus) -> bool:
        """更新节点状态"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].status = status
                self.nodes[node_id].last_heartbeat = datetime.now()
                self._update_cluster_stats()
                return True
            return False
    
    def update_node_load(self, node_id: str, load_factor: float) -> bool:
        """更新节点负载"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].load_factor = load_factor
                return True
            return False
    
    def update_node_heartbeat(self, node_id: str) -> bool:
        """更新节点心跳"""
        with self._lock:
            if node_id in self.nodes:
                self.nodes[node_id].last_heartbeat = datetime.now()
                return True
            return False
    
    def check_node_health(self, node_id: str) -> bool:
        """检查节点健康状态"""
        with self._lock:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            heartbeat_age = (datetime.now() - node.last_heartbeat).total_seconds()
            
            # 超过30秒没有心跳认为节点不健康
            if heartbeat_age > 30:
                node.status = NodeStatus.OFFLINE
                return False
            
            return node.status == NodeStatus.ONLINE
    
    def get_cluster_stats(self) -> ClusterStats:
        """获取集群统计信息"""
        with self._lock:
            self._update_cluster_stats()
            return self.stats
    
    def _update_cluster_stats(self):
        """更新集群统计信息"""
        self.stats.total_nodes = len(self.nodes)
        self.stats.online_nodes = sum(
            1 for node in self.nodes.values()
            if node.status == NodeStatus.ONLINE
        )
        
        if self.nodes:
            self.stats.avg_load_factor = sum(
                node.load_factor for node in self.nodes.values()
            ) / len(self.nodes)
            
            self.stats.total_cpu_cores = sum(
                node.cpu_cores for node in self.nodes.values()
            )
            
            self.stats.total_memory_gb = sum(
                node.memory_gb for node in self.nodes.values()
            )
            
            self.stats.total_gpu_devices = sum(
                len(node.gpu_devices) for node in self.nodes.values()
            )
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        with self._lock:
            self._update_cluster_stats()
            
            return {
                'total_nodes': self.stats.total_nodes,
                'online_nodes': self.stats.online_nodes,
                'avg_load_factor': self.stats.avg_load_factor,
                'total_cpu_cores': self.stats.total_cpu_cores,
                'total_memory_gb': self.stats.total_memory_gb,
                'total_gpu_devices': self.stats.total_gpu_devices,
                'nodes': {
                    node_id: {
                        'hostname': node.hostname,
                        'ip_address': node.ip_address,
                        'status': node.status.value,
                        'cpu_cores': node.cpu_cores,
                        'memory_gb': node.memory_gb,
                        'gpu_devices': node.gpu_devices,
                        'load_factor': node.load_factor,
                        'active_tasks': len(node.active_tasks),
                        'last_heartbeat': node.last_heartbeat.isoformat()
                    }
                    for node_id, node in self.nodes.items()
                }
            }


__all__ = ['ClusterManager']

