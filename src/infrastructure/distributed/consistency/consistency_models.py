"""
一致性数据模型

缓存一致性相关的数据类和枚举。

从cache_consistency.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
import time
import json
from typing import Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConsistencyLevel(Enum):
    """一致性级别"""
    EVENTUAL = "eventual"          # 最终一致性
    STRONG = "strong"             # 强一致性
    CAUSAL = "causal"             # 因果一致性
    SESSION = "session"           # 会话一致性


class NodeStatus(Enum):
    """节点状态"""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    DOWN = "down"
    RECOVERING = "recovering"


class OperationType(Enum):
    """操作类型"""
    SET = "set"
    GET = "get"
    DELETE = "delete"
    EXPIRE = "expire"
    CLEAR = "clear"


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    version: int
    timestamp: float
    ttl: Optional[float] = None
    node_id: str = ""

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() > self.timestamp + self.ttl

    def serialize(self) -> str:
        """序列化"""
        return json.dumps({
            'key': self.key,
            'value': self.value,
            'version': self.version,
            'timestamp': self.timestamp,
            'ttl': self.ttl,
            'node_id': self.node_id
        })

    @classmethod
    def deserialize(cls, data: str) -> 'CacheEntry':
        """反序列化"""
        obj = json.loads(data)
        return cls(**obj)


@dataclass
class LogEntry:
    """日志条目"""
    term: int
    index: int
    operation: OperationType
    key: str
    value: Any = None
    ttl: Optional[float] = None
    timestamp: float = 0.0
    node_id: str = ""

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class NodeInfo:
    """节点信息"""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.FOLLOWER
    last_heartbeat: float = 0.0
    term: int = 0
    voted_for: Optional[str] = None

    def __post_init__(self):
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


__all__ = [
    'ConsistencyLevel',
    'NodeStatus',
    'OperationType',
    'CacheEntry',
    'LogEntry',
    'NodeInfo'
]

