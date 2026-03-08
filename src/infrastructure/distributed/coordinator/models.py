"""
分布式协调器数据模型

包含枚举类型和数据类定义。

从coordinator.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class NodeStatus(Enum):
    """节点状态"""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    hostname: str
    ip_address: str
    status: NodeStatus = NodeStatus.ONLINE
    cpu_cores: int = 4
    memory_gb: float = 8.0
    gpu_devices: List[int] = field(default_factory=list)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    active_tasks: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    load_factor: float = 0.0  # 0 - 1, 负载因子


@dataclass
class DistributedTask:
    """分布式任务"""
    task_id: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    created_time: datetime = field(default_factory=datetime.now)
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    timeout_seconds: int = 3600
    retry_count: int = 0
    max_retries: int = 3
    dependencies: Set[str] = field(default_factory=set)
    data: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error_message: Optional[str] = None


@dataclass
class ClusterStats:
    """集群统计信息"""
    total_nodes: int = 0
    online_nodes: int = 0
    total_tasks: int = 0
    running_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_load_factor: float = 0.0
    total_cpu_cores: int = 0
    total_memory_gb: float = 0.0
    total_gpu_devices: int = 0


__all__ = [
    'NodeStatus',
    'TaskStatus',
    'TaskPriority',
    'NodeInfo',
    'DistributedTask',
    'ClusterStats'
]

