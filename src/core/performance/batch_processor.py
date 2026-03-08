"""
自适应批处理系统

功能:
- 动态批大小调整（基于吞吐量）
- 时间窗口和大小窗口批处理策略
- 背压处理机制
- 与异步数据库和HTTP客户端集成

性能目标:
- 批处理吞吐量提升 300%
- 延迟控制在 100ms 以内
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import (
    Callable, Generic, List, Optional, TypeVar,
    Any, Dict, AsyncIterator, Union
)
from enum import Enum
from collections import deque
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchStrategy(Enum):
    """批处理策略枚举"""
    TIME_WINDOW = "time_window"      # 时间窗口策略
    SIZE_WINDOW = "size_window"      # 大小窗口策略
    ADAPTIVE = "adaptive"            # 自适应策略


@dataclass
class BatchConfig:
    """批处理配置"""
    # 基础配置
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    max_batch_size: int = 1000
    min_batch_size: int = 10
    max_wait_time: float = 0.1  # 秒

    # 自适应配置
    target_latency_ms: float = 100.0
    latency_threshold_high: float = 150.0
    latency_threshold_low: float = 50.0
    adjustment_factor: float = 0.2

    # 背压配置
    max_inflight_batches: int = 5
    max_queue_size: int = 10000

    # 性能监控
    enable_metrics: bool = True
    metrics_window_size: int = 100


@dataclass
class BatchMetrics:
    """批处理指标"""
    total_batches: int = 0
    total_items: int = 0
    total_latency_ms: float = 0.0
    avg_batch_size: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_per_sec: float = 0.0
    current_batch_size: int = 0
    queue_size: int = 0
    dropped_items: int = 0
    last_adjustment_time: float = 0.0

    def record_batch(self, size: int, latency_ms: float) -> None:
        """记录批次处理结果"""
        self.total_batches += 1
        self.total_items += size
        self.total_latency_ms += latency_ms
        self.avg_batch_size = self.total_items / self.total_batches
        self.avg_latency_ms = self.total_latency_ms / self.total_batches

    def update_throughput(self, window_seconds: float) -> None:
        """更新吞吐量"""
        if window_seconds > 0:
            self.throughput_per_sec = self.total_items / window_seconds


class AdaptiveBatchSizer:
    """
    自适应批大小调整器

    根据实际处理延迟动态调整批大小，以达到目标延迟
    """

    def __init__(self, config: BatchConfig):
        self.config = config
        self.current_size = config.max_batch_size // 2
        self.latency_history: deque = deque(maxlen=config.metrics_window_size)
        self.adjustment_count = 0

    def record_latency(self, latency_ms: float) -> None:
        """记录处理延迟"""
        self.latency_history.append(latency_ms)

    def adjust_batch_size(self) -> int:
        """
        根据延迟历史调整批大小

        Returns:
            调整后的批大小
        """
        if len(self.latency_history) < 10:
            return self.current_size

        avg_latency = sum(self.latency_history) / len(self.latency_history)
        old_size = self.current_size

        if avg_latency > self.config.latency_threshold_high:
            # 延迟过高，减小批大小
            reduction = int(self.current_size * self.config.adjustment_factor)
            self.current_size = max(
                self.config.min_batch_size,
                self.current_size - reduction
            )
            logger.debug(
                f"批大小减小: {old_size} -> {self.current_size} "
                f"(延迟: {avg_latency:.2f}ms)"
            )

        elif avg_latency < self.config.latency_threshold_low:
            # 延迟过低，增加批大小
            increase = int(self.current_size * self.config.adjustment_factor)
            self.current_size = min(
                self.config.max_batch_size,
                self.current_size + increase
            )
            logger.debug(
                f"批大小增加: {old_size} -> {self.current_size} "
                f"(延迟: {avg_latency:.2f}ms)"
            )

        self.adjustment_count += 1
        return self.current_size

    def get_current_size(self) -> int:
        """获取当前批大小"""
        return self.current_size


class BatchProcessor(Generic[T, R]):
    """
    自适应批处理器

    支持多种批处理策略，自动调整批大小以优化吞吐量
    """

    def __init__(
        self,
        processor: Callable[[List[T]], asyncio.Future[List[R]]],
        config: Optional[BatchConfig] = None,
        name: str = "BatchProcessor"
    ):
        self.processor = processor
        self.config = config or BatchConfig()
        self.name = name

        # 队列和状态
        self._queue: asyncio.Queue[tuple[T, asyncio.Future[R]]] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._shutdown = False
        self._processing_task: Optional[asyncio.Task] = None

        # 自适应调整器
        self._adaptive_sizer = AdaptiveBatchSizer(self.config)

        # 指标
        self._metrics = BatchMetrics()
        self._start_time = time.time()
        self._latency_history: deque = deque(maxlen=self.config.metrics_window_size)

        # 背压控制
        self._inflight_batches = 0
        self._backpressure_event = asyncio.Event()
        self._backpressure_event.set()

    async def start(self) -> None:
        """启动批处理器"""
        if self._processing_task is None:
            self._processing_task = asyncio.create_task(
                self._processing_loop(),
                name=f"{self.name}_loop"
            )
            logger.info(f"批处理器 '{self.name}' 已启动")

    async def stop(self) -> None:
        """停止批处理器"""
        self._shutdown = True

        if self._processing_task:
            # 等待队列中的请求处理完成
            await self._queue.join()
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
            self._processing_task = None

        logger.info(f"批处理器 '{self.name}' 已停止")

    async def submit(self, item: T) -> R:
        """
        提交单个项目进行处理

        Args:
            item: 要处理的项目

        Returns:
            处理结果

        Raises:
            RuntimeError: 如果处理器已关闭
            asyncio.TimeoutError: 如果背压导致超时
        """
        if self._shutdown:
            raise RuntimeError("批处理器已关闭")

        # 背压检查
        if not self._backpressure_event.is_set():
            try:
                await asyncio.wait_for(
                    self._backpressure_event.wait(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError("背压超时，系统负载过高")

        # 创建结果Future
        future: asyncio.Future[R] = asyncio.get_event_loop().create_future()

        try:
            self._queue.put_nowait((item, future))
            self._metrics.queue_size = self._queue.qsize()
        except asyncio.QueueFull:
            self._metrics.dropped_items += 1
            raise RuntimeError("队列已满，请求被拒绝")

        return await future

    async def submit_many(self, items: List[T]) -> List[R]:
        """
        批量提交项目

        Args:
            items: 要处理的项目列表

        Returns:
            处理结果列表
        """
        futures = [self.submit(item) for item in items]
        return await asyncio.gather(*futures)

    async def _processing_loop(self) -> None:
        """批处理主循环"""
        while not self._shutdown:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except Exception as e:
                logger.exception(f"批处理循环错误: {e}")
                await asyncio.sleep(0.1)

    async def _collect_batch(self) -> List[tuple[T, asyncio.Future[R]]]:
        """
        收集一批待处理项目

        Returns:
            收集到的项目批次
        """
        batch: List[tuple[T, asyncio.Future[R]]] = []

        # 确定批大小
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            batch_size = self._adaptive_sizer.get_current_size()
        else:
            batch_size = self.config.max_batch_size

        # 等待第一个项目
        try:
            item = await asyncio.wait_for(
                self._queue.get(),
                timeout=1.0
            )
            batch.append(item)
        except asyncio.TimeoutError:
            return batch

        # 收集更多项目（时间窗口或大小窗口）
        start_time = time.time()
        timeout = self.config.max_wait_time

        while len(batch) < batch_size:
            elapsed = time.time() - start_time
            remaining = timeout - elapsed

            if remaining <= 0:
                break

            try:
                item = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=remaining
                )
                batch.append(item)
            except asyncio.TimeoutError:
                break

        return batch

    async def _process_batch(
        self,
        batch: List[tuple[T, asyncio.Future[R]]]
    ) -> None:
        """
        处理一批项目

        Args:
            batch: 待处理的批次
        """
        start_time = time.time()
        items = [item for item, _ in batch]
        futures = [future for _, future in batch]

        # 背压控制
        self._inflight_batches += 1
        if self._inflight_batches >= self.config.max_inflight_batches:
            self._backpressure_event.clear()

        try:
            # 执行批处理
            results = await self.processor(items)

            # 设置结果
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # 所有项目标记为失败
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            logger.exception(f"批处理失败: {e}")

        finally:
            # 更新指标
            latency_ms = (time.time() - start_time) * 1000
            self._latency_history.append(latency_ms)
            self._metrics.record_batch(len(batch), latency_ms)
            self._metrics.current_batch_size = len(batch)

            # 自适应调整
            if self.config.strategy == BatchStrategy.ADAPTIVE:
                self._adaptive_sizer.record_latency(latency_ms)
                if len(self._latency_history) >= 10:
                    self._adaptive_sizer.adjust_batch_size()

            # 释放背压
            self._inflight_batches -= 1
            if self._inflight_batches < self.config.max_inflight_batches:
                self._backpressure_event.set()

            # 标记队列任务完成
            for _ in batch:
                self._queue.task_done()

    def get_metrics(self) -> BatchMetrics:
        """获取当前指标"""
        self._metrics.update_throughput(time.time() - self._start_time)
        return self._metrics

    @property
    def is_running(self) -> bool:
        """检查处理器是否正在运行"""
        return self._processing_task is not None and not self._processing_task.done()


class BatchingIterator(Generic[T]):
    """
    异步迭代器批处理包装器

    将异步迭代器的数据分批处理
    """

    def __init__(
        self,
        iterator: AsyncIterator[T],
        batch_size: int = 100,
        max_wait_time: float = 0.1
    ):
        self.iterator = iterator
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time

    def __aiter__(self) -> AsyncIterator[List[T]]:
        return self

    async def __anext__(self) -> List[T]:
        batch: List[T] = []
        start_time = time.time()

        async for item in self.iterator:
            batch.append(item)

            if len(batch) >= self.batch_size:
                return batch

            if time.time() - start_time >= self.max_wait_time:
                return batch

        if batch:
            return batch

        raise StopAsyncIteration


class ParallelBatchProcessor(Generic[T, R]):
    """
    并行批处理器

    支持多个批处理器并行工作，提高吞吐量
    """

    def __init__(
        self,
        processor: Callable[[List[T]], asyncio.Future[List[R]]],
        config: Optional[BatchConfig] = None,
        num_workers: int = 4,
        name: str = "ParallelBatchProcessor"
    ):
        self.config = config or BatchConfig()
        self.num_workers = num_workers
        self.name = name

        # 创建多个批处理器
        self._processors: List[BatchProcessor[T, R]] = [
            BatchProcessor(
                processor=processor,
                config=config,
                name=f"{name}_worker_{i}"
            )
            for i in range(num_workers)
        ]

        # 轮询计数器
        self._round_robin_counter = 0

    async def start(self) -> None:
        """启动所有处理器"""
        await asyncio.gather(*[p.start() for p in self._processors])
        logger.info(f"并行批处理器 '{self.name}' 已启动 ({self.num_workers} 个工作者)")

    async def stop(self) -> None:
        """停止所有处理器"""
        await asyncio.gather(*[p.stop() for p in self._processors])
        logger.info(f"并行批处理器 '{self.name}' 已停止")

    async def submit(self, item: T) -> R:
        """
        提交项目（轮询选择处理器）

        Args:
            item: 要处理的项目

        Returns:
            处理结果
        """
        processor = self._processors[self._round_robin_counter]
        self._round_robin_counter = (self._round_robin_counter + 1) % self.num_workers
        return await processor.submit(item)

    async def submit_many(self, items: List[T]) -> List[R]:
        """
        批量提交项目

        Args:
            items: 要处理的项目列表

        Returns:
            处理结果列表
        """
        # 将项目分配给不同的处理器
        chunks: List[List[T]] = [[] for _ in range(self.num_workers)]
        for i, item in enumerate(items):
            chunks[i % self.num_workers].append(item)

        # 并行提交
        tasks = [
            self._processors[i].submit_many(chunk)
            for i, chunk in enumerate(chunks) if chunk
        ]

        results = await asyncio.gather(*tasks)

        # 合并结果（保持原始顺序）
        merged: List[Optional[R]] = [None] * len(items)
        for i, chunk_results in enumerate(results):
            for j, result in enumerate(chunk_results):
                original_index = j * self.num_workers + i
                merged[original_index] = result

        return [r for r in merged if r is not None]

    def get_metrics(self) -> Dict[str, Any]:
        """获取所有处理器的指标"""
        return {
            f"worker_{i}": p.get_metrics()
            for i, p in enumerate(self._processors)
        }


# 便捷函数

async def batch_process(
    items: List[T],
    processor: Callable[[List[T]], asyncio.Future[List[R]]],
    batch_size: int = 100,
    max_wait_time: float = 0.1
) -> List[R]:
    """
    便捷函数：批量处理项目

    Args:
        items: 要处理的项目列表
        processor: 批处理函数
        batch_size: 批大小
        max_wait_time: 最大等待时间

    Returns:
        处理结果列表
    """
    config = BatchConfig(
        strategy=BatchStrategy.SIZE_WINDOW,
        max_batch_size=batch_size,
        max_wait_time=max_wait_time
    )

    batch_processor = BatchProcessor(processor, config)
    await batch_processor.start()

    try:
        return await batch_processor.submit_many(items)
    finally:
        await batch_processor.stop()


@asynccontextmanager
async def create_batch_processor(
    processor: Callable[[List[T]], asyncio.Future[List[R]]],
    config: Optional[BatchConfig] = None,
    name: str = "BatchProcessor"
):
    """
    批处理器上下文管理器

    Args:
        processor: 批处理函数
        config: 批处理配置
        name: 处理器名称

    Yields:
        BatchProcessor 实例
    """
    bp = BatchProcessor(processor, config, name)
    await bp.start()
    try:
        yield bp
    finally:
        await bp.stop()
