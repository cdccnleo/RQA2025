"""
事件监控组件

负责事件的监控、统计和健康检查。
"""

from typing import Dict, Any, Optional, Union
import logging
import time

from ..types import EventType
from ...foundation.base import ComponentHealth

logger = logging.getLogger(__name__)


class EventMonitor:
    """事件监控组件"""

    def __init__(self, statistics_manager, event_queue, max_queue_size: int,
                 worker_threads: list, start_time: float):
        """初始化事件监控组件

        Args:
            statistics_manager: 统计管理器
            event_queue: 事件队列
            max_queue_size: 最大队列大小
            worker_threads: 工作线程列表
            start_time: 启动时间
        """
        self.statistics_manager = statistics_manager
        self._event_queue = event_queue
        self.max_queue_size = max_queue_size
        self._worker_threads = worker_threads
        self._start_time = start_time

    def check_health(self, status) -> ComponentHealth:
        """检查事件总线健康状态

        Args:
            status: 组件状态

        Returns:
            健康状态
        """
        try:
            # 检查基本状态
            if status in ['ERROR', 'UNHEALTHY']:
                return ComponentHealth.UNHEALTHY

            # 检查队列状态
            queue_size = self._event_queue.qsize()
            if queue_size > self.max_queue_size * 0.9:  # 队列使用率超过90%
                return ComponentHealth.UNHEALTHY

            # 检查工作线程状态
            active_threads = sum(1 for t in self._worker_threads if t.is_alive())
            if active_threads < len(self._worker_threads):
                return ComponentHealth.UNHEALTHY

            return ComponentHealth.HEALTHY

        except Exception as e:
            logger.error(f"EventBus健康检查失败: {str(e)}")
            return ComponentHealth.UNHEALTHY

    def get_statistics(self, event_counter: int, processed_counter: int,
                      handlers: Dict, async_handlers: Dict,
                      filter_manager, routing_manager,
                      performance_monitor=None) -> Dict[str, Any]:
        """获取EventBus统计信息

        Args:
            event_counter: 事件计数器
            processed_counter: 已处理事件计数
            handlers: 同步处理器字典
            async_handlers: 异步处理器字典
            filter_manager: 过滤器管理器
            routing_manager: 路由管理器
            performance_monitor: 性能监控器（可选）

        Returns:
            统计信息字典
        """
        try:
            # 合并管理器统计和核心统计
            stats = self.statistics_manager.get_statistics()
            stats.update({
                "total_events_published": event_counter,
                "total_events_processed": processed_counter,
                "active_handlers": (
                    sum(len(h) for h in handlers.values()) +
                    sum(len(h) for h in async_handlers.values())
                ),
                "queue_size": self._event_queue.qsize() if hasattr(self._event_queue, 'qsize') else 0,
                "worker_threads": len(self._worker_threads),
                "event_filters": len(filter_manager._filters),
                "event_transformers": len(getattr(filter_manager, '_transformers', [])),
                "event_routes": len(routing_manager._routes),
                "dead_letter_queue_size": len(routing_manager._dead_letter_queue),
                "uptime": time.time() - self._start_time,
            })

            # 添加性能监控信息（如果启用）
            if performance_monitor:
                try:
                    perf_stats = {
                        "avg_processing_time": getattr(performance_monitor, 'avg_processing_time', 0),
                        "max_processing_time": getattr(performance_monitor, 'max_processing_time', 0),
                        "total_processed_events": getattr(performance_monitor, 'total_processed', 0),
                    }
                    stats.update(perf_stats)
                except Exception:
                    pass

            return stats

        except Exception as e:
            return {"error": f"获取统计信息失败: {str(e)}"}

    def get_subscriber_count(self, event_type: Union[EventType, str],
                            handlers: Dict, async_handlers: Dict) -> int:
        """获取订阅者数量

        Args:
            event_type: 事件类型
            handlers: 同步处理器字典
            async_handlers: 异步处理器字典

        Returns:
            订阅者数量
        """
        # 直接从处理器字典计算当前订阅者数量，而不是依赖统计数据
        # 使用与EventSubscriber相同的字符串转换逻辑
        if hasattr(event_type, 'value'):
            event_type_str = event_type.value
        else:
            event_type_str = str(event_type)

        sync_count = len(handlers.get(event_type_str, []))
        async_count = len(async_handlers.get(event_type_str, []))
        return sync_count + async_count

    def get_event_statistics(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        return self.statistics_manager.get_event_statistics()

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return self.statistics_manager.get_performance_stats()

    def get_recent_events(self, minutes: int = 5) -> Dict[str, int]:
        """获取最近的事件

        Args:
            minutes: 最近多少分钟

        Returns:
            事件类型和数量的字典
        """
        return self.statistics_manager.get_recent_events(minutes)

    def health_check(self) -> bool:
        """健康检查（简化版）"""
        try:
            # 检查队列是否可用
            if not hasattr(self._event_queue, 'qsize'):
                return False

            # 检查队列是否已满
            queue_size = self._event_queue.qsize()
            if queue_size >= self.max_queue_size:
                return False

            # 检查工作线程是否正常运行
            active_threads = sum(1 for t in self._worker_threads if t.is_alive())
            if active_threads == 0:
                return False

            return True

        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return False

