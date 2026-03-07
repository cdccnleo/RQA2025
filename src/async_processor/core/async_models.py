"""
异步处理数据模型

异步处理器的配置、统计等数据类。

从async_data_processor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

from dataclasses import dataclass, field
from datetime import datetime


class TaskPriority:
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class AsyncProcessorEventType:
    """异步处理器事件类型"""
    TASK_STARTED = "async_task_started"
    TASK_COMPLETED = "async_task_completed"
    TASK_FAILED = "async_task_failed"
    BATCH_STARTED = "async_batch_started"
    BATCH_COMPLETED = "async_batch_completed"
    PROCESSOR_OVERLOADED = "async_processor_overloaded"
    PROCESSOR_RECOVERED = "async_processor_recovered"
    CONFIG_UPDATED = "async_config_updated"
    HEALTH_CHECK_FAILED = "async_health_check_failed"


@dataclass
class AsyncConfig:
    """异步处理配置"""
    max_concurrent_requests: int = 5   # 最大并发请求数
    request_timeout: float = 30.0      # 请求超时时间(秒)
    max_workers: int = 4               # 线程池最大工作线程数
    enable_process_pool: bool = False  # 是否启用进程池
    max_processes: int = 2             # 进程池最大进程数
    batch_size: int = 100              # 批量处理大小
    retry_count: int = 3               # 重试次数
    retry_delay: float = 1.0           # 重试延迟(秒)


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    max_response_time: float = 0.0
    min_response_time: float = float('inf')
    throughput_per_second: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)

    def update_stats(self, response_time: float, success: bool):
        """更新统计信息"""
        self.total_requests += 1
        if success:
            self.completed_requests += 1
        else:
            self.failed_requests += 1

        # 更新响应时间统计
        self.avg_response_time = (
            self.avg_response_time * (self.total_requests - 1) + response_time
        ) / self.total_requests
        self.max_response_time = max(self.max_response_time, response_time)
        self.min_response_time = min(self.min_response_time, response_time)

        # 更新吞吐量
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if elapsed > 0:
            self.throughput_per_second = self.completed_requests / elapsed

        self.last_update = datetime.now()


__all__ = [
    'TaskPriority',
    'AsyncProcessorEventType',
    'AsyncConfig',
    'ProcessingStats'
]

