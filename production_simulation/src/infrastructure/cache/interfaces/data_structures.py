"""
data_structures 模块

提供 data_structures 相关功能和接口。
"""

import sys

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
"""
基础设施层 - 数据结构定义

定义缓存系统的数据结构和枚举。
"""


class CacheEvictionStrategy(Enum):
    """缓存淘汰策略枚举"""
    LRU = "lru"      # 最近最少使用
    LFU = "lfu"      # 最少频率使用
    LRU_K = "lru_k"  # LRU - K算法
    ADAPTIVE = "adaptive"  # 自适应缓存
    PRIORITY = "priority"  # 优先级
    COST_AWARE = "cost_aware"  # 成本感知
    FIFO = "fifo"    # 先进先出
    RANDOM = "random" # 随机淘汰
    TTL = "ttl"      # 基于时间过期
    SIZE = "size"    # 基于大小


class AccessPattern(Enum):
    """
    访问模式枚举

    定义缓存访问的不同模式，用于策略选择和性能优化。
    """
    FREQUENT = "frequent"      # 频繁访问 - 高频访问的数据
    MODERATE = "moderate"      # 中等访问 - 正常频率访问
    RARE = "rare"             # 偶尔访问 - 低频访问的数据
    SEQUENTIAL = "sequential"  # 顺序访问 - 按顺序访问的数据
    RANDOM = "random"         # 随机访问 - 随机访问模式
    BURSTY = "bursty"         # 突发访问 - 突发性高负载访问


class ConsistencyLevel(Enum):
    """
    一致性级别枚举

    定义分布式缓存的一致性保证级别。
    """
    STRONG = "strong"         # 强一致性 - 保证所有操作立即可见
    EVENTUAL = "eventual"     # 最终一致性 - 允许短暂不一致
    SESSION = "session"        # 会话一致性 - 单客户端保证
    MONOTONIC_READ = "monotonic_read"    # 单调读一致性
    READ_YOUR_WRITES = "read_your_writes"  # 读己之写一致性
    WEAK = "weak"             # 弱一致性 - 最弱保证


@dataclass
class CacheEntry:
    """
    缓存条目 - 统一数据结构

    表示缓存中的一个数据项，包含键、值、元数据等信息。
    支持过期时间、访问统计、序列化等功能。
    """
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒）
    size_bytes: int = 0        # 数据大小（字节）
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None  # 兼容旧参数

    def __post_init__(self):
        """初始化后自动计算大小"""
        if self.size_bytes == 0:
            self.size_bytes = self._calculate_size()
        if self.timestamp is not None:
            self.created_at = self.timestamp
            # 同步最后访问时间
            if 'last_accessed' in self.metadata:
                self.last_accessed = self.metadata['last_accessed']
            else:
                self.last_accessed = self.timestamp

    def _calculate_size(self) -> int:
        """计算值的大小（字节）"""
        try:
            # 对于字符串，返回其长度
            if isinstance(self.value, str):
                return len(self.value.encode('utf-8'))
            # 对于数字类型，使用固定大小
            elif isinstance(self.value, (int, float)):
                return 8  # 64位数字
            # 对于列表/元组，递归计算大小
            elif isinstance(self.value, (list, tuple)):
                return sum(self._calculate_object_size(item) for item in self.value)
            # 对于字典，递归计算大小
            elif isinstance(self.value, dict):
                size = 0
                for k, v in self.value.items():
                    size += self._calculate_object_size(k) + self._calculate_object_size(v)
                return size
            # 对于其他类型，尝试使用sys.getsizeof
            else:
                return sys.getsizeof(self.value)
        except Exception:
            # 如果计算失败，返回默认值
            return 0

    def _calculate_object_size(self, obj: Any) -> int:
        """递归计算对象大小"""
        try:
            if isinstance(obj, str):
                return len(obj.encode('utf-8'))
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, (list, tuple)):
                return sum(self._calculate_object_size(item) for item in obj)
            elif isinstance(obj, dict):
                size = 0
                for k, v in obj.items():
                    size += self._calculate_object_size(k) + self._calculate_object_size(v)
                return size
            else:
                return sys.getsizeof(obj)
        except Exception:
            return 0

    @property
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        # TTL=0 表示立即过期
        if self.ttl == 0:
            return True
        return (datetime.now() - self.created_at).total_seconds() > self.ttl

    def touch(self) -> None:
        """更新最后访问时间"""
        self.last_accessed = datetime.now()
        self.access_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'key': self.key,
            'value': self.value,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'ttl': self.ttl,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata.copy()
        }


@dataclass
class CacheStats:
    """
    缓存统计信息 - 统一数据结构

    收集和维护缓存系统的各项统计指标。
    支持命中率计算、性能指标监控、容量统计等。
    """
    hits: int = 0              # 命中次数
    misses: int = 0            # 缺失次数
    total_requests: int = 0    # 总请求数
    evictions: int = 0         # 驱逐次数
    total_size_bytes: int = 0  # 总数据大小（字节）
    size: Optional[int] = None  # 兼容字段
    memory_usage: float = 0.0  # 内存使用率（0-1）
    entry_count: int = 0       # 条目数量
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.size is not None:
            try:
                self.total_size_bytes = int(self.size)
            except Exception:
                pass
        self.size = self.total_size_bytes

    @property
    def total_size(self) -> int:
        return self.total_size_bytes

    @property
    def hit_rate(self) -> float:
        """计算命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """计算缺失率"""
        if self.total_requests == 0:
            return 0.0
        return self.misses / self.total_requests

    @property
    def eviction_rate(self) -> float:
        """计算驱逐率"""
        if self.total_requests == 0:
            return 0.0
        return self.evictions / self.total_requests

    def update(self, hit: bool = False, eviction: bool = False,
               size_delta: int = 0, entry_delta: int = 0) -> None:
        """
        更新统计信息

        Args:
            hit: 是否命中
            eviction: 是否发生驱逐
            size_delta: 数据大小变化
            entry_delta: 条目数量变化
        """
        self.total_requests += 1
        if hit:
            self.hits += 1
        else:
            self.misses += 1

        if eviction:
            self.evictions += 1

        self.total_size_bytes += size_delta
        self.entry_count += entry_delta

    def merge(self, other: 'CacheStats') -> None:
        """
        合并另一个CacheStats对象的数据

        Args:
            other: 另一个CacheStats对象
        """
        self.hits += other.hits
        self.misses += other.misses
        self.total_requests += other.total_requests
        self.evictions += other.evictions
        self.total_size_bytes += other.total_size_bytes
        self.memory_usage = max(self.memory_usage, other.memory_usage)
        self.entry_count += other.entry_count

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 统计信息的字典表示
        """
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'eviction_rate': self.eviction_rate,
            'evictions': self.evictions,
            'total_size_bytes': self.total_size_bytes,
            'memory_usage': self.memory_usage,
            'entry_count': self.entry_count,
            'created_at': self.created_at.isoformat()
        }

    def reset(self) -> None:
        """重置所有统计信息"""
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.evictions = 0
        self.total_size_bytes = 0
        self.memory_usage = 0.0
        self.entry_count = 0
        self.created_at = datetime.now()


@dataclass
class PerformanceMetrics:
    """
    性能指标 - 统一数据结构

    收集和维护缓存系统的各项性能指标，用于监控、诊断和优化。
    包含响应时间、命中率、吞吐量、资源使用等关键指标。
    """

    # 时间信息
    timestamp: datetime

    # 核心性能指标
    hit_rate: float          # 命中率 (0.0-1.0)
    response_time: float     # 响应时间 (ms)
    throughput: int          # 吞吐量 (QPS)

    # 资源使用指标
    memory_usage: float      # 内存使用量 (MB)

    # 缓存特定指标
    eviction_rate: float     # 驱逐率
    cache_size: int          # 缓存大小 (条目数)
    miss_penalty: float      # 缓存未命中惩罚时间 (ms)

    @classmethod
    def create_current(cls, hit_rate: float = 0.0, response_time: float = 0.0,
                       throughput: int = 0, memory_usage: float = 0.0,
                       eviction_rate: float = 0.0, cache_size: int = 0,
                       miss_penalty: float = 0.0) -> 'PerformanceMetrics':
        """
        创建当前时间的性能指标实例

        Args:
            hit_rate: 命中率
            response_time: 响应时间
            throughput: 吞吐量
            memory_usage: 内存使用量
            eviction_rate: 驱逐率
            cache_size: 缓存大小
            miss_penalty: 缓存未命中惩罚时间

        Returns:
            PerformanceMetrics: 新的性能指标实例
        """
        return cls(
            timestamp=datetime.now(),
            hit_rate=hit_rate,
            response_time=response_time,
            throughput=throughput,
            memory_usage=memory_usage,
            eviction_rate=eviction_rate,
            cache_size=cache_size,
            miss_penalty=miss_penalty
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 性能指标的字典表示
        """
        return {
            'timestamp': self.timestamp.isoformat(),
            'hit_rate': self.hit_rate,
            'response_time': self.response_time,
            'throughput': self.throughput,
            'memory_usage': self.memory_usage,
            'eviction_rate': self.eviction_rate,
            'cache_size': self.cache_size,
            'miss_penalty': self.miss_penalty
        }

    def is_performance_degraded(self, threshold_hit_rate: float = 0.8,
                                threshold_response_time: float = 100.0) -> bool:
        """
        检查性能是否下降

        Args:
            threshold_hit_rate: 命中率阈值
            threshold_response_time: 响应时间阈值 (ms)

        Returns:
            bool: 性能是否下降
        """
        return self.hit_rate < threshold_hit_rate or self.response_time > threshold_response_time
