# -*- coding: utf-8 -*-
"""
RQA2025 流处理层基础处理器接口
Stream Processing Layer Base Processor Interface

定义流处理器的基础接口和抽象类。
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable, List
from abc import ABC, abstractmethod
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

# 获取统一基础设施集成层的日志适配器
try:
    from src.core.integration import get_models_adapter
    models_adapter = get_models_adapter()
    from src.infrastructure.logging.core.interfaces import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


@dataclass
class StreamProcessingResult:

    """流处理结果"""
    event_id: str
    processing_status: str
    processed_data: Dict[str, Any]
    processing_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProcessingStatus:

    """处理状态枚举"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class StreamMetrics:

    """流处理指标"""
    processor_id: str
    total_events_processed: int = 0
    total_processing_time_ms: float = 0.0
    successful_events: int = 0
    failed_events: int = 0
    queue_size: int = 0
    last_updated: Optional[datetime] = None

    @property
    def error_rate(self) -> float:
        """错误率"""
        total = self.total_events_processed
        return self.failed_events / total if total > 0 else 0.0

    @property
    def avg_processing_time_ms(self) -> float:
        """平均处理时间"""
        total = self.total_events_processed
        return self.total_processing_time_ms / total if total > 0 else 0.0

    @property
    def throughput_per_second(self) -> float:
        """每秒吞吐量"""
        if not self.last_updated:
            return 0.0

        time_diff = (datetime.now() - self.last_updated).total_seconds()
        return self.total_events_processed / time_diff if time_diff > 0 else 0.0

    def update_metrics(self, processing_time_ms: float, is_successful: bool):
        """更新指标"""
        self.total_events_processed += 1
        self.total_processing_time_ms += processing_time_ms
        self.last_updated = datetime.now()

        if is_successful:
            self.successful_events += 1
        else:
            self.failed_events += 1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'processor_id': self.processor_id,
            'total_events_processed': self.total_events_processed,
            'total_processing_time_ms': self.total_processing_time_ms,
            'successful_events': self.successful_events,
            'failed_events': self.failed_events,
            'queue_size': self.queue_size,
            'error_rate': self.error_rate,
            'avg_processing_time_ms': self.avg_processing_time_ms,
            'throughput_per_second': self.throughput_per_second,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }


class StreamProcessorBase(ABC):

    """
    流处理器基础抽象类
    """

    def __init__(self, processor_id: str, config: Optional[Dict[str, Any]] = None):

        self.processor_id = processor_id
        self.config = config or {}

        # 处理配置
        self.max_concurrent_events = self.config.get('max_concurrent_events', 100)
        self.processing_timeout = self.config.get('processing_timeout', 30.0)
        self.enable_monitoring = self.config.get('enable_monitoring', True)

        # 状态管理
        self.is_running = False
        self.processing_queue = asyncio.Queue(maxsize=self.max_concurrent_events)
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_events)
        self.processing_tasks: List[asyncio.Task] = []

        # 指标收集
        self.metrics = StreamMetrics(processor_id)

        # 事件处理器映射
        self.event_handlers: Dict[str, Callable] = {}

        logger.info(f"流处理器 {processor_id} 已初始化")

    @abstractmethod
    async def process_event(self, event: Any) -> StreamProcessingResult:
        """
        处理单个事件

        Args:
            event: 待处理的流事件

        Returns:
            StreamProcessingResult: 处理结果
        """

    async def start_processing(self):
        """启动事件处理"""
        if self.is_running:
            logger.warning(f"处理器 {self.processor_id} 已在运行中")
            return

        self.is_running = True
        logger.info(f"启动流处理器 {self.processor_id}")

        # 启动处理任务（不等待完成，让它们在后台运行）
        self.processing_tasks = []
        for _ in range(self.max_concurrent_events):
            task = asyncio.create_task(self._processing_loop())
            self.processing_tasks.append(task)

        # 启动监控任务
        if self.enable_monitoring:
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.processing_tasks.append(monitoring_task)

    async def stop_processing(self):
        """停止事件处理"""
        if not self.is_running:
            logger.info(f"处理器 {self.processor_id} 已停止")
            return

        self.is_running = False
        logger.info(f"正在停止流处理器 {self.processor_id}")

        # 取消所有处理任务
        if hasattr(self, 'processing_tasks'):
            for task in self.processing_tasks:
                if not task.done():
                    task.cancel()
            
            # 等待任务取消完成（设置超时避免无限等待）
            if self.processing_tasks:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*self.processing_tasks, return_exceptions=True),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"停止处理器 {self.processor_id} 超时，强制取消任务")
                    for task in self.processing_tasks:
                        if not task.done():
                            task.cancel()

        # 等待队列处理完成（设置超时避免无限等待）
        try:
            await asyncio.wait_for(self.processing_queue.join(), timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(f"等待队列处理完成超时，强制停止")

        # 关闭线程池
        self.executor.shutdown(wait=True)

        logger.info(f"流处理器 {self.processor_id} 已停止")

    async def submit_event(self, event: Any) -> bool:
        """
        提交事件到处理队列

        Args:
            event: 待处理的流事件

        Returns:
            bool: 提交是否成功
        """
        if not self.is_running:
            logger.warning(f"处理器 {self.processor_id} 未运行，无法提交事件")
            return False

        try:
            await asyncio.wait_for(
                self.processing_queue.put(event),
                timeout=5.0
            )
            return True
        except asyncio.TimeoutError:
            logger.error(f"提交事件 {getattr(event, 'event_id', 'unknown')} 超时")
            return False

    def register_event_handler(self, event_type: str, handler: Callable):
        """
        注册事件处理器

        Args:
            event_type: 事件类型
            handler: 处理函数
        """
        self.event_handlers[event_type] = handler
        logger.info(f"为事件类型 {event_type} 注册了处理器")

    async def _processing_loop(self):
        """处理循环"""
        while self.is_running:
            try:
                # 获取待处理事件（设置超时避免无限等待）
                try:
                    event = await asyncio.wait_for(
                        self.processing_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    # 超时后继续循环，检查is_running状态
                    continue

                # 处理事件
                start_time = datetime.now()
                result = await self._safe_process_event(event)
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                # 更新指标
                self.metrics.update_metrics(
                    processing_time, result.processing_status == ProcessingStatus.COMPLETED)
                self.metrics.queue_size = self.processing_queue.qsize()

                # 处理完成回调
                await self._on_processing_complete(result)

                self.processing_queue.task_done()

            except asyncio.CancelledError:
                # 任务被取消，正常退出
                logger.info(f"处理循环 {self.processor_id} 被取消")
                break
            except Exception as e:
                logger.error(f"处理循环异常: {str(e)}")
                # 发生异常时短暂等待，避免快速循环
                await asyncio.sleep(0.1)

    async def _safe_process_event(self, event: Any) -> StreamProcessingResult:
        """安全的事件处理"""
        try:
            # 执行实际处理
            result = await asyncio.wait_for(
                self.process_event(event),
                timeout=self.processing_timeout
            )

            return result

        except asyncio.TimeoutError:
            logger.error("处理事件超时")
            return StreamProcessingResult(
                event_id=getattr(event, 'event_id', 'unknown'),
                processing_status=ProcessingStatus.TIMEOUT,
                processed_data={},
                processing_time_ms=self.processing_timeout * 1000,
                error_message="Processing timeout"
            )

        except Exception as e:
            logger.error(f"处理事件异常: {str(e)}")
            return StreamProcessingResult(
                event_id=getattr(event, 'event_id', 'unknown'),
                processing_status=ProcessingStatus.FAILED,
                processed_data={},
                processing_time_ms=0.0,
                error_message=str(e)
            )

    async def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                await self._collect_system_metrics()

                # 检查健康状态
                health_status = await self._check_health()

                # 记录监控数据
                await self._record_monitoring_data(health_status)

                # 每30秒监控一次，但检查is_running状态
                for _ in range(30):  # 30秒 = 30次 * 1秒
                    if not self.is_running:
                        break
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                # 任务被取消，正常退出
                logger.info(f"监控循环 {self.processor_id} 被取消")
                break
            except Exception as e:
                logger.error(f"监控循环异常: {str(e)}")
                await asyncio.sleep(1)  # 异常时短暂等待

    async def _collect_system_metrics(self):
        """收集系统指标"""
        # 收集队列大小
        self.metrics.queue_size = self.processing_queue.qsize()

        # 可以在这里添加更多系统指标收集

    async def _check_health(self) -> Dict[str, Any]:
        """检查处理器健康状态"""
        return {
            'processor_id': self.processor_id,
            'is_running': self.is_running,
            'queue_size': self.processing_queue.qsize(),
            'error_rate': self.metrics.error_rate,
            'throughput': self.metrics.throughput_per_second,
            'last_updated': datetime.now()
        }

    async def _record_monitoring_data(self, health_status: Dict[str, Any]):
        """记录监控数据"""
        # 这里可以发送监控数据到监控系统
        # 例如：发送到Prometheus、ELK等

    async def _on_processing_complete(self, result: StreamProcessingResult):
        """处理完成回调"""
        # 可以在这里添加后处理逻辑
        # 例如：发送结果到下游系统、记录审计日志等

    def get_processor_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return {
            'processor_id': self.processor_id,
            'is_running': self.is_running,
            'queue_size': self.processing_queue.qsize(),
            'metrics': self.metrics.to_dict(),
            'config': self.config
        }


__all__ = [
    'StreamProcessorBase',
    'StreamProcessingResult',
    'ProcessingStatus',
    'StreamMetrics'
]
