"""
批量任务处理器

支持将多个小任务合并为批量任务执行，提高吞吐量
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable, Coroutine
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """批量处理策略"""
    SIZE_BASED = "size"      # 基于任务数量
    TIME_BASED = "time"      # 基于时间窗口
    HYBRID = "hybrid"        # 混合策略


@dataclass
class BatchConfig:
    """批量处理配置"""
    strategy: BatchStrategy = BatchStrategy.HYBRID
    max_batch_size: int = 100           # 最大批量大小
    max_wait_time_ms: int = 1000        # 最大等待时间（毫秒）
    min_batch_size: int = 10            # 最小批量大小（用于时间策略）
    enable_compression: bool = True     # 启用数据压缩


@dataclass
class BatchTask:
    """批量任务项"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int
    submitted_at: datetime
    timeout_seconds: Optional[int] = None


class BatchProcessor:
    """
    批量任务处理器

    将多个小任务合并为批量任务执行，优势：
    - 减少网络往返次数
    - 提高数据库批量操作效率
    - 降低系统调用开销

    使用场景：
    - 数据采集批量写入
    - 特征批量计算
    - 订单批量处理
    """

    def __init__(
        self,
        config: Optional[BatchConfig] = None,
        batch_handler: Optional[Callable[[List[BatchTask]], Coroutine]] = None
    ):
        """
        初始化批量处理器

        Args:
            config: 批量处理配置
            batch_handler: 批量处理函数
        """
        self._config = config or BatchConfig()
        self._batch_handler = batch_handler

        # 按任务类型分组的待处理队列
        self._pending_batches: Dict[str, List[BatchTask]] = {}
        self._batch_locks: Dict[str, asyncio.Lock] = {}
        self._batch_timers: Dict[str, Optional[asyncio.Task]] = {}

        # 统计信息
        self._stats = {
            'total_batches_processed': 0,
            'total_tasks_processed': 0,
            'avg_batch_size': 0,
            'avg_processing_time_ms': 0
        }

        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动批量处理器"""
        if self._running:
            return

        self._running = True
        self._main_loop_task = asyncio.create_task(self._main_loop())
        logger.info("✅ 批量任务处理器已启动")

    async def stop(self):
        """停止批量处理器"""
        if not self._running:
            return

        self._running = False

        # 取消主循环
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass

        # 取消所有定时器
        for timer in self._batch_timers.values():
            if timer:
                timer.cancel()

        # 处理剩余批次
        await self._flush_all_batches()

        logger.info("✅ 批量任务处理器已停止")

    async def submit(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        priority: int = 5,
        timeout_seconds: Optional[int] = None
    ) -> bool:
        """
        提交任务到批量处理器

        Args:
            task_id: 任务ID
            task_type: 任务类型（用于分组）
            payload: 任务数据
            priority: 优先级
            timeout_seconds: 超时时间

        Returns:
            bool: 是否成功提交
        """
        if not self._running:
            logger.warning("批量处理器未运行")
            return False

        # 创建批量任务
        batch_task = BatchTask(
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            priority=priority,
            submitted_at=datetime.now(),
            timeout_seconds=timeout_seconds
        )

        # 获取或创建锁
        if task_type not in self._batch_locks:
            self._batch_locks[task_type] = asyncio.Lock()

        async with self._batch_locks[task_type]:
            # 初始化批次
            if task_type not in self._pending_batches:
                self._pending_batches[task_type] = []

            # 添加任务到批次
            self._pending_batches[task_type].append(batch_task)

            # 检查是否需要立即处理
            should_flush = self._should_flush_batch(task_type)

            if should_flush:
                # 取消定时器（如果存在）
                if task_type in self._batch_timers and self._batch_timers[task_type]:
                    self._batch_timers[task_type].cancel()
                    self._batch_timers[task_type] = None

                # 立即处理批次
                asyncio.create_task(self._process_batch(task_type))
            else:
                # 设置定时器（如果是第一个任务或定时器已取消）
                if task_type not in self._batch_timers or not self._batch_timers[task_type]:
                    self._batch_timers[task_type] = asyncio.create_task(
                        self._batch_timer(task_type)
                    )

        return True

    def _should_flush_batch(self, task_type: str) -> bool:
        """
        检查是否应该立即处理批次

        Args:
            task_type: 任务类型

        Returns:
            bool: 是否应该立即处理
        """
        batch = self._pending_batches.get(task_type, [])

        if not batch:
            return False

        if self._config.strategy == BatchStrategy.SIZE_BASED:
            # 基于大小：达到最大批量大小
            return len(batch) >= self._config.max_batch_size

        elif self._config.strategy == BatchStrategy.TIME_BASED:
            # 基于时间：检查最早任务是否超时
            if batch:
                oldest_task = min(batch, key=lambda t: t.submitted_at)
                elapsed_ms = (datetime.now() - oldest_task.submitted_at).total_seconds() * 1000
                return elapsed_ms >= self._config.max_wait_time_ms
            return False

        else:  # HYBRID
            # 混合策略：达到最大批量或超时
            if len(batch) >= self._config.max_batch_size:
                return True

            if batch:
                oldest_task = min(batch, key=lambda t: t.submitted_at)
                elapsed_ms = (datetime.now() - oldest_task.submitted_at).total_seconds() * 1000
                # 达到最小批量且超时，或未达最小批量但严重超时（2倍）
                if len(batch) >= self._config.min_batch_size:
                    return elapsed_ms >= self._config.max_wait_time_ms
                else:
                    return elapsed_ms >= self._config.max_wait_time_ms * 2

            return False

    async def _batch_timer(self, task_type: str):
        """
        批次定时器

        在指定时间后触发批次处理

        Args:
            task_type: 任务类型
        """
        try:
            await asyncio.sleep(self._config.max_wait_time_ms / 1000)

            async with self._batch_locks.get(task_type, asyncio.Lock()):
                if task_type in self._pending_batches and self._pending_batches[task_type]:
                    await self._process_batch(task_type)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"批次定时器错误: {e}")
        finally:
            if task_type in self._batch_timers:
                self._batch_timers[task_type] = None

    async def _process_batch(self, task_type: str):
        """
        处理批次

        Args:
            task_type: 任务类型
        """
        async with self._batch_locks.get(task_type, asyncio.Lock()):
            batch = self._pending_batches.get(task_type, [])
            if not batch:
                return

            # 提取批次并清空
            tasks_to_process = batch.copy()
            self._pending_batches[task_type] = []

        # 处理批次
        start_time = time.time()

        try:
            if self._batch_handler:
                await self._batch_handler(tasks_to_process)
            else:
                # 默认处理：逐个处理任务
                for task in tasks_to_process:
                    logger.debug(f"处理任务: {task.task_id}")

            # 更新统计
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_stats(len(tasks_to_process), processing_time_ms)

            logger.debug(f"批量处理完成: {task_type}, 任务数: {len(tasks_to_process)}")

        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            # 这里可以添加失败重试逻辑

    async def _flush_all_batches(self):
        """处理所有待处理批次"""
        for task_type in list(self._pending_batches.keys()):
            async with self._batch_locks.get(task_type, asyncio.Lock()):
                if task_type in self._pending_batches and self._pending_batches[task_type]:
                    await self._process_batch(task_type)

    async def _main_loop(self):
        """主循环 - 定期检查和强制处理超时的批次"""
        while self._running:
            try:
                await asyncio.sleep(5)  # 每5秒检查一次

                # 检查所有批次
                for task_type in list(self._pending_batches.keys()):
                    if self._should_flush_batch(task_type):
                        async with self._batch_locks.get(task_type, asyncio.Lock()):
                            # 取消定时器
                            if task_type in self._batch_timers and self._batch_timers[task_type]:
                                self._batch_timers[task_type].cancel()
                                self._batch_timers[task_type] = None

                            if self._pending_batches.get(task_type):
                                asyncio.create_task(self._process_batch(task_type))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"主循环错误: {e}")

    def _update_stats(self, batch_size: int, processing_time_ms: float):
        """
        更新统计信息

        Args:
            batch_size: 批次大小
            processing_time_ms: 处理时间（毫秒）
        """
        self._stats['total_batches_processed'] += 1
        self._stats['total_tasks_processed'] += batch_size

        # 更新平均值
        n = self._stats['total_batches_processed']
        self._stats['avg_batch_size'] = (
            (self._stats['avg_batch_size'] * (n - 1) + batch_size) / n
        )
        self._stats['avg_processing_time_ms'] = (
            (self._stats['avg_processing_time_ms'] * (n - 1) + processing_time_ms) / n
        )

    def get_pending_count(self, task_type: Optional[str] = None) -> int:
        """
        获取待处理任务数量

        Args:
            task_type: 任务类型，None则返回所有类型

        Returns:
            int: 待处理任务数
        """
        if task_type:
            return len(self._pending_batches.get(task_type, []))
        return sum(len(batch) for batch in self._pending_batches.values())

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            **self._stats,
            'pending_tasks': self.get_pending_count(),
            'batch_groups': len(self._pending_batches),
            'config': {
                'strategy': self._config.strategy.value,
                'max_batch_size': self._config.max_batch_size,
                'max_wait_time_ms': self._config.max_wait_time_ms
            }
        }

    def get_batch_status(self) -> Dict[str, Any]:
        """
        获取批次状态

        Returns:
            Dict[str, Any]: 各类型批次的当前状态
        """
        status = {}
        for task_type, batch in self._pending_batches.items():
            if batch:
                oldest = min(batch, key=lambda t: t.submitted_at)
                newest = max(batch, key=lambda t: t.submitted_at)
                status[task_type] = {
                    'count': len(batch),
                    'oldest_age_ms': (datetime.now() - oldest.submitted_at).total_seconds() * 1000,
                    'newest_age_ms': (datetime.now() - newest.submitted_at).total_seconds() * 1000
                }
        return status
