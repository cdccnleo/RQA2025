"""
事件处理组件

负责事件的执行、批处理和错误处理。
"""

from typing import List, Tuple, Optional
import logging
import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass

from ..models import Event, EventHandler
from ..context import HandlerExecutionContext  # 从独立模块导入，避免循环导入

logger = logging.getLogger(__name__)


@dataclass
class EventProcessingResult:
    """事件处理结果"""
    event_id: Optional[str]
    event_type: str
    success: bool
    processing_time: float
    sync_handlers_executed: int = 0
    async_handlers_executed: int = 0
    errors: Optional[List[str]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class EventProcessor:
    """事件处理组件"""

    def __init__(self, subscriber, statistics_manager, lock, retry_manager=None,
                 performance_monitor=None):
        """初始化事件处理组件

        Args:
            subscriber: 事件订阅组件
            statistics_manager: 统计管理器
            lock: 线程锁
            retry_manager: 重试管理器（可选）
            performance_monitor: 性能监控器（可选）
        """
        self.subscriber = subscriber
        self.statistics_manager = statistics_manager
        self._lock = lock
        self._retry_manager = retry_manager
        self._performance_monitor = performance_monitor

    def handle_event(self, event: Event) -> EventProcessingResult:
        """处理单个事件

        Args:
            event: 事件对象

        Returns:
            事件处理结果
        """
        start_time = time.time()
        event_type_str = str(event.event_type)

        result = EventProcessingResult(
            event_id=getattr(event, 'event_id', ''),
            event_type=event_type_str,
            success=False,
            processing_time=0.0
        )

        try:
            # 获取处理器列表
            handlers, async_handlers = self.subscriber.get_handlers(event_type_str)

            # 执行同步处理器
            sync_success, sync_count = self._execute_sync_handlers(event, handlers)

            # 执行异步处理器
            async_success, async_count = self._execute_async_handlers(event, async_handlers)

            # 计算整体成功状态
            result.success = sync_success and async_success
            result.sync_handlers_executed = sync_count
            result.async_handlers_executed = async_count

            # 更新统计和持久化
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            self._update_event_statistics(
                event, result.success, processing_time,
                sync_count, async_count
            )

            # 添加成功处理日志
            logger.info(f"✅ 事件处理完成: {event.event_id} ({event_type_str}), "
                       f"成功: {result.success}, "
                       f"同步处理器: {sync_count}, 异步处理器: {async_count}, "
                       f"处理时间: {processing_time:.3f}s")

        except Exception as e:
            logger.error(f"❌ 处理事件失败: {e}", exc_info=True)
            if hasattr(result, 'errors') and result.errors is not None:
                result.errors.append(str(e))
            else:
                result.errors = [str(e)]
            self._handle_event_error(event, event_type_str)

        return result

    def _execute_sync_handlers(self, event: Event, handlers: List[EventHandler]) -> Tuple[bool, int]:
        """执行同步处理器，返回执行结果和数量"""
        success = True
        executed_count = 0

        for handler_info in handlers:
            try:
                context = HandlerExecutionContext(
                    event=event,
                    handler_info=handler_info,
                    start_time=time.time()
                )
                self._sync_handler_wrapper_with_context(context)
                executed_count += 1
            except Exception as e:
                logger.error(f"同步处理器异常: {e}")
                success = False
                self._handle_sync_handler_error(event, handler_info, e)

        return success, executed_count

    def _sync_handler_wrapper_with_context(self, context: HandlerExecutionContext) -> None:
        """使用上下文包装的同步处理器执行"""
        try:
            # 检查是否过期
            if context.is_expired:
                logger.warning(f"处理器执行超时: {context.handler_info}")
                return

            # 执行处理器
            self._sync_handler_wrapper(context.handler_info, context.event)

        except Exception as e:
            logger.error(f"同步处理器执行失败: {e}")

    def _sync_handler_wrapper(self, handler_info: EventHandler, event: Event) -> None:
        """同步处理器包装器"""
        try:
            # 执行处理器
            handler_info.handler(event)

            # 记录性能（如果启用）
            if self._performance_monitor:
                # 这里可以记录处理器执行时间等指标
                pass

        except Exception as e:
            logger.error(f"同步处理器执行异常: {handler_info.handler.__name__}, 错误: {e}")
            raise

    def _execute_async_handlers(self, event: Event, async_handlers: List[EventHandler]) -> Tuple[bool, int]:
        """执行异步处理器，返回执行结果和数量"""
        success = True
        executed_count = 0

        for handler_info in async_handlers:
            try:
                context = HandlerExecutionContext(
                    event=event,
                    handler_info=handler_info,
                    start_time=time.time()
                )
                asyncio.create_task(self._async_handler_wrapper_with_context(context))
                executed_count += 1
            except Exception as e:
                logger.error(f"异步处理器异常: {e}")
                success = False
                self._handle_async_handler_error(event, handler_info, e)

        return success, executed_count

    def _async_handler_wrapper_with_context(self, context: HandlerExecutionContext) -> None:
        """使用上下文包装的异步处理器执行"""
        try:
            # 检查是否过期
            if context.is_expired:
                logger.warning(f"异步处理器执行超时: {context.handler_info}")
                return

            # 执行处理器
            asyncio.create_task(self._async_handler_wrapper(context.handler_info.handler, context.event))

        except Exception as e:
            logger.error(f"异步处理器执行失败: {e}")

    def _async_handler_wrapper(self, handler: callable, event: Event) -> None:
        """异步处理器包装器"""
        try:
            asyncio.create_task(handler(event))
        except Exception as e:
            logger.error(f"异步处理器执行异常: {handler.__name__}, 错误: {e}")

    def handle_batch(self, events: List[Event]) -> List[EventProcessingResult]:
        """处理事件批次

        Args:
            events: 事件列表

        Returns:
            处理结果列表
        """
        if not events:
            return []

        # 按事件类型分组
        event_groups = defaultdict(list)
        for event in events:
            event_type_str = str(event.event_type)
            event_groups[event_type_str].append(event)

        results = []
        # 批量处理每种类型的事件
        for event_type, event_batch in event_groups.items():
            try:
                batch_results = self._handle_event_batch(event_type, event_batch)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"批量处理事件失败: {event_type}, 错误: {e}")

        return results

    def _handle_event_batch(self, event_type: str, events: List[Event]) -> List[EventProcessingResult]:
        """处理同类型事件批次"""
        handlers, async_handlers = self.subscriber.get_handlers(event_type)

        results = []
        # 批量执行同步处理器
        for handler_info in handlers:
            try:
                for event in events:
                    result = self.handle_event(event)
                    results.append(result)
            except Exception as e:
                logger.error(f"批量同步处理器异常: {e}")

        # 批量执行异步处理器
        for handler_info in async_handlers:
            try:
                for event in events:
                    asyncio.create_task(self._async_handler_wrapper(handler_info.handler, event))
            except Exception as e:
                logger.error(f"批量异步处理器异常: {e}")

        return results

    def _handle_sync_handler_error(self, event: Event, handler_info: EventHandler, error: Exception):
        """处理同步处理器错误"""
        logger.error(f"同步处理器错误: {handler_info.handler.__name__}, 事件: {event.event_id}, 错误: {error}")

    def _handle_async_handler_error(self, event: Event, handler_info: EventHandler, error: Exception):
        """处理异步处理器错误"""
        logger.error(f"异步处理器错误: {handler_info.handler.__name__}, 事件: {event.event_id}, 错误: {error}")

    def _handle_event_error(self, event: Event, event_type_str: str):
        """处理事件错误"""
        logger.error(f"处理事件失败: {event.event_id} ({event_type_str})")

    def _update_event_statistics(self, event: Event, success: bool, processing_time: float,
                                sync_count: int, async_count: int):
        """更新事件统计"""
        error_count = 0 if success else 1
        self.statistics_manager.update_statistics(
            str(event.event_type),
            handler_count=sync_count + async_count,
            error_count=error_count
        )

