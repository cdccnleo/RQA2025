"""
异步数据处理模型

包含异步处理器的数据模型和配置类。

从async_data_processor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AsyncProcessorEventType:
    """异步处理器事件类型"""
    TASK_STARTED = "async_task_started"
    TASK_COMPLETED = "async_task_completed"
    TASK_FAILED = "async_task_failed"
    BATCH_STARTED = "async_batch_started"
    BATCH_COMPLETED = "async_batch_completed"
    RESOURCE_WARNING = "async_resource_warning"
    PERFORMANCE_DEGRADED = "async_performance_degraded"


@dataclass
class AsyncConfig:
    """异步处理配置"""
    max_concurrent_requests: int = 5   # 最大并发请求数
    request_timeout: float = 30.0      # 请求超时时间(秒)
    max_workers: int = 4               # 线程池最大工作线程数
    batch_size: int = 100              # 批量处理大小
    retry_count: int = 3               # 重试次数
    retry_delay: float = 1.0           # 重试延迟(秒)
    enable_caching: bool = True        # 启用缓存
    cache_ttl: int = 300               # 缓存TTL(秒)
    enable_health_check: bool = True   # 启用健康检查
    health_check_interval: int = 60    # 健康检查间隔(秒)


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    active_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    retry_count: int = 0
    timeout_count: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    def update_with_request(self, success: bool, processing_time: float, from_cache: bool = False):
        """更新统计信息"""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_processing_time += processing_time
        self.avg_response_time = self.total_processing_time / max(self.total_requests, 1)
        self.max_response_time = max(self.max_response_time, processing_time)
        self.min_response_time = min(self.min_response_time, processing_time)
        
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        self.last_updated = datetime.now()

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def get_cache_hit_rate(self) -> float:
        """获取缓存命中率"""
        total_cache_attempts = self.cache_hits + self.cache_misses
        if total_cache_attempts == 0:
            return 0.0
        return self.cache_hits / total_cache_attempts

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': self.get_success_rate(),
            'avg_response_time': self.avg_response_time,
            'max_response_time': self.max_response_time,
            'min_response_time': self.min_response_time if self.min_response_time != float('inf') else 0.0,
            'active_requests': self.active_requests,
            'cache_hit_rate': self.get_cache_hit_rate(),
            'retry_count': self.retry_count,
            'timeout_count': self.timeout_count,
            'last_updated': self.last_updated.isoformat()
        }


__all__ = [
    'TaskPriority',
    'AsyncProcessorEventType',
    'AsyncConfig',
    'ProcessingStats'
]

