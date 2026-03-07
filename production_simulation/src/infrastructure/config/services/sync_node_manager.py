
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional
import time
"""
同步节点管理器
管理分布式配置同步的节点
"""


class SyncStatus(Enum):

    """同步状态枚举"""
    IDLE = "idle"
    SYNCING = "syncing"
    SUCCESS = "success"
    FAILED = "failed"
    OFFLINE = "offline"


@dataclass
class SyncNode:

    """同步节点"""
    node_id: str
    address: str
    port: int
    status: SyncStatus = SyncStatus.IDLE
    version: str = "1.0"
    last_sync_time: Optional[float] = None
    created_at: float = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = time.time()


class SyncNodeManager:

    """同步节点管理器"""

    def __init__(self):

        self._nodes: Dict[str, SyncNode] = {}

    def register_node(self, node_id: str, address: str, port: int) -> bool:
        """注册同步节点"""
        if node_id in self._nodes:
            return False

        self._nodes[node_id] = SyncNode(
            node_id=node_id,
            address=address,
            port=port
        )
        return True

    def unregister_node(self, node_id: str) -> bool:
        """注销同步节点"""
        if node_id not in self._nodes:
            return False

        del self._nodes[node_id]
        return True

    def get_node(self, node_id: str) -> Optional[SyncNode]:
        """获取节点信息"""
        return self._nodes.get(node_id)

    def update_node_status(self, node_id: str, status: SyncStatus) -> bool:
        """更新节点状态"""
        if node_id not in self._nodes:
            return False

        self._nodes[node_id].status = status
        return True

    def update_node_sync_time(self, node_id: str) -> bool:
        """更新节点同步时间"""
        if node_id not in self._nodes:
            return False

        self._nodes[node_id].last_sync_time = time.time()
        return True

    def list_nodes(self) -> List[Dict[str, Any]]:
        """列出所有节点"""
        return [
            {
                "node_id": node.node_id,
                "address": node.address,
                "port": node.port,
                "status": node.status.value,
                "version": node.version,
                "last_sync_time": node.last_sync_time,
                "created_at": node.created_at
            }
            for node in self._nodes.values()
        ]

    def get_active_nodes(self) -> List[str]:
        """获取活跃节点列表"""
        return [
            node_id for node_id, node in self._nodes.items()
            if node.status in [SyncStatus.IDLE, SyncStatus.SUCCESS]
        ]

    def get_node_count(self) -> int:
        """获取节点数量"""
        return len(self._nodes)

    def get_node_status_summary(self) -> Dict[str, int]:
        """获取节点状态统计"""
        summary = {}
        for status in SyncStatus:
            summary[status.value] = 0

        for node in self._nodes.values():
            summary[node.status.value] += 1

        return summary




