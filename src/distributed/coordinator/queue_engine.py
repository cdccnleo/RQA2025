"""
队列引擎模块

负责任务队列的管理，包括优先级队列、延迟队列、重试队列等。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import logging
import time
import heapq
from typing import Dict, List, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import threading

logger = logging.getLogger(__name__)


class QueueType(Enum):
    """队列类型"""
    PRIORITY = "priority"       # 优先级队列
    DELAYED = "delayed"         # 延迟队列
    RETRY = "retry"             # 重试队列
    DEAD_LETTER = "dead_letter" # 死信队列


class QueuePriority(Enum):
    """队列优先级"""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1
    BACKGROUND = 0


@dataclass(order=True)
class QueuedTask:
    """队列中的任务"""
    priority: int
    submit_time: float = field(compare=False)
    task_id: str = field(compare=False)
    task_data: Dict[str, Any] = field(compare=False, default_factory=dict)
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    delay_until: Optional[float] = field(compare=False, default=None)
    queue_type: QueueType = field(compare=False, default=QueueType.PRIORITY)
    
    def __post_init__(self):
        # 确保priority是整数
        if not isinstance(self.priority, int):
            self.priority = int(self.priority)


@dataclass
class QueueStats:
    """队列统计"""
    total_enqueued: int = 0
    total_dequeued: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_retried: int = 0
    total_dead_lettered: int = 0
    current_size: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0


@dataclass
class QueueConfig:
    """队列配置"""
    max_queue_size: int = 10000
    max_retry_count: int = 3
    retry_delay_seconds: float = 5.0
    retry_backoff_multiplier: float = 2.0
    max_retry_delay_seconds: float = 300.0  # 最大5分钟
    dead_letter_after_max_retries: bool = True
    enable_priority_aging: bool = True
    priority_aging_interval_seconds: int = 300


class QueueEngine:
    """
    队列引擎
    
    提供以下功能：
    1. 优先级队列管理
    2. 延迟队列支持
    3. 重试队列管理
    4. 死信队列处理
    5. 队列统计和监控
    
    Attributes:
        config: 队列配置
        priority_queue: 优先级队列（使用堆实现）
        delayed_queue: 延迟队列
        retry_queue: 重试队列
        dead_letter_queue: 死信队列
    """
    
    def __init__(self, config: Optional[QueueConfig] = None):
        self.config = config or QueueConfig()
        
        # 队列
        self._priority_queue: List[QueuedTask] = []  # 堆
        self._delayed_queue: List[QueuedTask] = []   # 堆（按delay_until排序）
        self._retry_queue: deque = deque()
        self._dead_letter_queue: deque = deque()
        
        # 任务索引（快速查找）
        self._task_index: Dict[str, QueuedTask] = {}
        self._task_submit_times: Dict[str, float] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        # 统计
        self._stats = QueueStats()
        
        # 回调函数
        self._enqueue_callbacks: List[Callable[[QueuedTask], None]] = []
        self._dequeue_callbacks: List[Callable[[QueuedTask], None]] = []
        self._dead_letter_callbacks: List[Callable[[QueuedTask], None]] = []
        
        # 运行状态
        self._running = False
        self._maintenance_task = None
        
        logger.info("QueueEngine initialized")
    
    async def initialize(self) -> bool:
        """
        初始化队列引擎
        
        Returns:
            bool: 初始化是否成功
        """
        self._running = True
        
        # 启动维护任务
        import asyncio
        self._maintenance_task = asyncio.create_task(self._maintenance_loop())
        
        logger.info("QueueEngine initialized successfully")
        return True
    
    async def shutdown(self) -> bool:
        """
        关闭队列引擎
        
        Returns:
            bool: 关闭是否成功
        """
        self._running = False
        
        # 停止维护任务
        if self._maintenance_task:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
        
        logger.info("QueueEngine shutdown successfully")
        return True
    
    def enqueue_task(
        self,
        task_id: str,
        task_data: Dict[str, Any],
        priority: QueuePriority = QueuePriority.NORMAL,
        delay_seconds: Optional[float] = None
    ) -> bool:
        """
        将任务加入队列
        
        Args:
            task_id: 任务ID
            task_data: 任务数据
            priority: 优先级
            delay_seconds: 延迟时间（秒）
            
        Returns:
            bool: 是否成功加入队列
        """
        with self._lock:
            # 检查队列大小
            total_size = len(self._priority_queue) + len(self._delayed_queue) + len(self._retry_queue)
            if total_size >= self.config.max_queue_size:
                logger.error(f"Queue full, cannot enqueue task {task_id}")
                return False
            
            # 检查任务是否已存在
            if task_id in self._task_index:
                logger.warning(f"Task {task_id} already in queue, updating")
                self._remove_task(task_id)
            
            # 创建队列任务
            current_time = time.time()
            
            if delay_seconds and delay_seconds > 0:
                # 延迟任务
                queued_task = QueuedTask(
                    priority=priority.value,
                    submit_time=current_time,
                    task_id=task_id,
                    task_data=task_data,
                    delay_until=current_time + delay_seconds,
                    queue_type=QueueType.DELAYED
                )
                heapq.heappush(self._delayed_queue, queued_task)
            else:
                # 普通优先级任务
                queued_task = QueuedTask(
                    priority=priority.value,
                    submit_time=current_time,
                    task_id=task_id,
                    task_data=task_data,
                    queue_type=QueueType.PRIORITY
                )
                heapq.heappush(self._priority_queue, queued_task)
            
            # 索引任务
            self._task_index[task_id] = queued_task
            self._task_submit_times[task_id] = current_time
            
            # 更新统计
            self._stats.total_enqueued += 1
            self._stats.current_size = total_size + 1
            
            # 触发回调
            for callback in self._enqueue_callbacks:
                try:
                    callback(queued_task)
                except Exception as e:
                    logger.error(f"Enqueue callback error: {e}")
            
            logger.debug(f"Task {task_id} enqueued with priority {priority.name}")
            return True
    
    def dequeue_task(self, timeout: Optional[float] = None) -> Optional[QueuedTask]:
        """
        从队列中取出任务
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Optional[QueuedTask]: 队列任务，如果队列为空则返回None
        """
        with self._lock:
            # 首先检查重试队列
            if self._retry_queue:
                queued_task = self._retry_queue.popleft()
                if self._is_ready(queued_task):
                    self._complete_dequeue(queued_task)
                    return queued_task
                else:
                    # 还没准备好，放回队列
                    self._retry_queue.appendleft(queued_task)
            
            # 然后检查优先级队列
            if self._priority_queue:
                queued_task = heapq.heappop(self._priority_queue)
                self._complete_dequeue(queued_task)
                return queued_task
            
            return None
    
    def _complete_dequeue(self, queued_task: QueuedTask):
        """完成出队操作"""
        # 从索引中移除
        if queued_task.task_id in self._task_index:
            del self._task_index[queued_task.task_id]
        
        # 更新统计
        self._stats.total_dequeued += 1
        self._stats.current_size = len(self._priority_queue) + len(self._delayed_queue) + len(self._retry_queue)
        
        # 计算等待时间
        wait_time = (time.time() - queued_task.submit_time) * 1000  # 毫秒
        self._update_wait_time_stats(wait_time)
        
        # 触发回调
        for callback in self._dequeue_callbacks:
            try:
                callback(queued_task)
            except Exception as e:
                logger.error(f"Dequeue callback error: {e}")
        
        logger.debug(f"Task {queued_task.task_id} dequeued")
    
    def _is_ready(self, queued_task: QueuedTask) -> bool:
        """检查任务是否准备好执行"""
        if queued_task.delay_until is None:
            return True
        return time.time() >= queued_task.delay_until
    
    def retry_task(self, task_id: str, task_data: Optional[Dict] = None) -> bool:
        """
        将任务加入重试队列
        
        Args:
            task_id: 任务ID
            task_data: 任务数据（可选，更新数据）
            
        Returns:
            bool: 是否成功加入重试队列
        """
        with self._lock:
            # 查找原任务
            original_task = self._task_index.get(task_id)
            
            if original_task:
                retry_count = original_task.retry_count + 1
                
                # 检查是否超过最大重试次数
                if retry_count > self.config.max_retry_count:
                    logger.warning(f"Task {task_id} exceeded max retries, moving to dead letter queue")
                    self._move_to_dead_letter(original_task)
                    return False
                
                # 计算重试延迟（指数退避）
                delay = min(
                    self.config.retry_delay_seconds * (self.config.retry_backoff_multiplier ** (retry_count - 1)),
                    self.config.max_retry_delay_seconds
                )
                
                # 创建重试任务
                retry_task = QueuedTask(
                    priority=original_task.priority,
                    submit_time=original_task.submit_time,
                    task_id=task_id,
                    task_data=task_data or original_task.task_data,
                    retry_count=retry_count,
                    max_retries=self.config.max_retry_count,
                    delay_until=time.time() + delay,
                    queue_type=QueueType.RETRY
                )
                
                # 加入重试队列
                self._retry_queue.append(retry_task)
                self._task_index[task_id] = retry_task
                
                # 更新统计
                self._stats.total_retried += 1
                
                logger.info(f"Task {task_id} scheduled for retry {retry_count}/{self.config.max_retry_count} in {delay:.1f}s")
                return True
            else:
                logger.warning(f"Task {task_id} not found for retry")
                return False
    
    def complete_task(self, task_id: str, success: bool = True) -> bool:
        """
        标记任务完成
        
        Args:
            task_id: 任务ID
            success: 是否成功
            
        Returns:
            bool: 是否成功标记
        """
        with self._lock:
            if task_id in self._task_index:
                del self._task_index[task_id]
            
            if success:
                self._stats.total_completed += 1
                logger.debug(f"Task {task_id} completed successfully")
            else:
                self._stats.total_failed += 1
                logger.debug(f"Task {task_id} failed")
            
            return True
    
    def _move_to_dead_letter(self, queued_task: QueuedTask):
        """将任务移到死信队列"""
        self._dead_letter_queue.append(queued_task)
        self._stats.total_dead_lettered += 1
        
        # 从索引中移除
        if queued_task.task_id in self._task_index:
            del self._task_index[queued_task.task_id]
        
        # 触发回调
        for callback in self._dead_letter_callbacks:
            try:
                callback(queued_task)
            except Exception as e:
                logger.error(f"Dead letter callback error: {e}")
        
        logger.warning(f"Task {queued_task.task_id} moved to dead letter queue")
    
    def _remove_task(self, task_id: str) -> bool:
        """从队列中移除任务"""
        # 从优先级队列中移除
        self._priority_queue = [t for t in self._priority_queue if t.task_id != task_id]
        heapq.heapify(self._priority_queue)
        
        # 从延迟队列中移除
        self._delayed_queue = [t for t in self._delayed_queue if t.task_id != task_id]
        heapq.heapify(self._delayed_queue)
        
        # 从重试队列中移除
        self._retry_queue = deque([t for t in self._retry_queue if t.task_id != task_id])
        
        # 从索引中移除
        if task_id in self._task_index:
            del self._task_index[task_id]
        
        return True
    
    async def _maintenance_loop(self):
        """维护循环"""
        import asyncio
        
        while self._running:
            try:
                # 处理延迟队列中的就绪任务
                self._process_delayed_tasks()
                
                # 处理重试队列中的就绪任务
                self._process_retry_tasks()
                
                # 优先级老化
                if self.config.enable_priority_aging:
                    self._apply_priority_aging()
                
                await asyncio.sleep(1)  # 每秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Maintenance loop error: {e}")
                await asyncio.sleep(5)
    
    def _process_delayed_tasks(self):
        """处理延迟队列中的就绪任务"""
        with self._lock:
            current_time = time.time()
            ready_tasks = []
            
            # 找出所有就绪的延迟任务
            while self._delayed_queue and self._delayed_queue[0].delay_until <= current_time:
                task = heapq.heappop(self._delayed_queue)
                ready_tasks.append(task)
            
            # 将就绪任务移到优先级队列
            for task in ready_tasks:
                task.queue_type = QueueType.PRIORITY
                task.delay_until = None
                heapq.heappush(self._priority_queue, task)
                logger.debug(f"Delayed task {task.task_id} moved to priority queue")
    
    def _process_retry_tasks(self):
        """处理重试队列中的就绪任务"""
        with self._lock:
            current_time = time.time()
            ready_tasks = []
            remaining_tasks = deque()
            
            for task in self._retry_queue:
                if task.delay_until <= current_time:
                    ready_tasks.append(task)
                else:
                    remaining_tasks.append(task)
            
            self._retry_queue = remaining_tasks
            
            # 将就绪任务移到优先级队列
            for task in ready_tasks:
                task.queue_type = QueueType.PRIORITY
                task.delay_until = None
                heapq.heappush(self._priority_queue, task)
                logger.debug(f"Retry task {task.task_id} moved to priority queue")
    
    def _apply_priority_aging(self):
        """应用优先级老化"""
        # 这里可以实现优先级老化逻辑
        # 例如：长时间等待的任务提升优先级
        pass
    
    def _update_wait_time_stats(self, wait_time_ms: float):
        """更新等待时间统计"""
        # 使用指数移动平均
        alpha = 0.1
        self._stats.avg_wait_time_ms = (
            alpha * wait_time_ms + (1 - alpha) * self._stats.avg_wait_time_ms
        )
        self._stats.max_wait_time_ms = max(self._stats.max_wait_time_ms, wait_time_ms)
    
    def get_task_position(self, task_id: str) -> Optional[int]:
        """
        获取任务在队列中的位置
        
        Args:
            task_id: 任务ID
            
        Returns:
            Optional[int]: 位置（从1开始），如果不在队列中则返回None
        """
        with self._lock:
            # 在优先级队列中查找
            for i, task in enumerate(sorted(self._priority_queue)):
                if task.task_id == task_id:
                    return i + 1
            
            # 在延迟队列中查找
            for i, task in enumerate(sorted(self._delayed_queue)):
                if task.task_id == task_id:
                    return i + 1
            
            return None
    
    def get_queue_stats(self) -> QueueStats:
        """
        获取队列统计
        
        Returns:
            QueueStats: 队列统计
        """
        with self._lock:
            self._stats.current_size = len(self._priority_queue) + len(self._delayed_queue) + len(self._retry_queue)
            return self._stats
    
    def get_queue_sizes(self) -> Dict[str, int]:
        """
        获取各队列大小
        
        Returns:
            Dict[str, int]: 队列大小字典
        """
        with self._lock:
            return {
                'priority': len(self._priority_queue),
                'delayed': len(self._delayed_queue),
                'retry': len(self._retry_queue),
                'dead_letter': len(self._dead_letter_queue),
                'total': len(self._priority_queue) + len(self._delayed_queue) + len(self._retry_queue)
            }
    
    def register_enqueue_callback(self, callback: Callable[[QueuedTask], None]):
        """注册入队回调"""
        self._enqueue_callbacks.append(callback)
    
    def register_dequeue_callback(self, callback: Callable[[QueuedTask], None]):
        """注册出队回调"""
        self._dequeue_callbacks.append(callback)
    
    def register_dead_letter_callback(self, callback: Callable[[QueuedTask], None]):
        """注册死信回调"""
        self._dead_letter_callbacks.append(callback)


# 便捷函数
def create_default_queue_engine() -> QueueEngine:
    """创建默认队列引擎"""
    return QueueEngine()


def create_high_throughput_queue_engine() -> QueueEngine:
    """创建高吞吐量队列引擎"""
    config = QueueConfig(
        max_queue_size=50000,
        max_retry_count=5,
        retry_delay_seconds=1.0,
        retry_backoff_multiplier=1.5
    )
    return QueueEngine(config)


def create_reliable_queue_engine() -> QueueEngine:
    """创建高可靠性队列引擎"""
    config = QueueConfig(
        max_queue_size=10000,
        max_retry_count=10,
        retry_delay_seconds=10.0,
        retry_backoff_multiplier=2.0,
        dead_letter_after_max_retries=True
    )
    return QueueEngine(config)


__all__ = [
    'QueueEngine',
    'QueueConfig',
    'QueueStats',
    'QueuedTask',
    'QueueType',
    'QueuePriority',
    'create_default_queue_engine',
    'create_high_throughput_queue_engine',
    'create_reliable_queue_engine'
]