"""
异步事件处理器

异步处理器的事件驱动架构实现。

从async_data_processor.py中提取以改善代码组织。

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import asyncio
import logging
from typing import Dict, Any, Callable
from datetime import datetime

from .async_models import AsyncProcessorEventType

logger = logging.getLogger(__name__)


# 导入事件总线
try:
    from src.core.event_bus.event_bus import Event, EventType, EventPriority
    event_bus_available = True
except ImportError:
    event_bus_available = False
    logger.warning("Event bus not available, using fallback event handling")


# 导入数据集成
try:
    from src.core.integration import record_data_metric
except ImportError:
    record_data_metric = lambda *args, **kwargs: None


class AsyncEventHandler:
    """
    异步事件处理器
    
    负责:
    1. 事件注册和分发
    2. 事件队列管理
    3. 异步事件处理
    """
    
    def __init__(self):
        self.event_handlers: Dict[str, Callable] = {}
        self.event_queue = asyncio.Queue()
        self.event_processor_task = None
        self.processing_task = None  # 添加processing_task属性

        # 注册默认事件处理器
        self._register_default_handlers()

        logger.info("异步事件处理器初始化完成")
    
    def _register_default_handlers(self):
        """注册默认事件处理器"""
        self.event_handlers = {
            'task_started': self._handle_task_started,
            'task_completed': self._handle_task_completed,  # 使用字符串键名匹配测试期望
            'task_failed': self._handle_task_failed,
            'batch_started': self._handle_batch_started,
            'batch_completed': self._handle_batch_completed,
            'processor_overloaded': self._handle_processor_overloaded,
            'config_updated': self._handle_config_updated,
        }
    
    async def _process_events(self):
        """异步处理事件队列"""
        while True:
            try:
                event = await self.event_queue.get()
                await self._dispatch_event(event)
                self.event_queue.task_done()
            except Exception as e:
                logger.error(f"事件处理异常: {e}")
    
    async def _dispatch_event(self, event):
        """分发事件到对应处理器"""
        event_type = event.get('type')
        handler = self.event_handlers.get(event_type)
        if handler:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"事件处理器异常 ({event_type}): {e}")
        else:
            logger.debug(f"未找到事件处理器: {event_type}")
    
    async def publish_event(self, event_type: str, data: Dict[str, Any], priority: str = "normal"):
        """发布异步事件"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'source': 'AsyncDataProcessor'
        }

        # 添加到异步队列
        try:
            await self.event_queue.put(event)
        except Exception as e:
            logger.error(f"发布事件失败: {e}")

        # 同时发布到统一事件总线
        if event_bus_available:
            try:
                event_obj = Event(
                    type=EventType.CUSTOM,
                    data={
                        'async_event_type': event_type,
                        'async_data': data,
                        'source': 'AsyncDataProcessor'
                    },
                    priority=EventPriority.NORMAL if priority == "normal" else EventPriority.HIGH
                )
                # 假设有全局事件总线实例
                # global_event_bus.publish(event_obj)
            except Exception as e:
                logger.debug(f"事件总线发布失败: {e}")
    
    # 事件处理器方法
    async def _handle_task_started(self, event):
        """处理任务开始事件"""
        task_id = event['data'].get('task_id')
        logger.info(f"任务开始处理: {task_id}")
        record_data_metric("async_task_started", 1, "count", {"task_id": task_id})
    
    async def _handle_task_completed(self, event):
        """处理任务完成事件"""
        task_id = event['data'].get('task_id')
        duration = event['data'].get('duration', 0)
        logger.info(f"任务完成: {task_id}, 耗时: {duration:.2f}s")
        record_data_metric("async_task_completed", 1, "count", {"task_id": task_id, "duration": duration})
    
    async def _handle_task_failed(self, event):
        """处理任务失败事件"""
        task_id = event['data'].get('task_id')
        error = event['data'].get('error', 'Unknown error')
        logger.warning(f"任务失败: {task_id}, 错误: {error}")
        record_data_metric("async_task_failed", 1, "count", {"task_id": task_id, "error": str(error)})
    
    async def _handle_batch_started(self, event):
        """处理批量任务开始事件"""
        batch_id = event['data'].get('batch_id')
        task_count = event['data'].get('task_count', 0)
        logger.info(f"批量任务开始: {batch_id}, 任务数: {task_count}")
        record_data_metric("async_batch_started", task_count, "count", {"batch_id": batch_id})
    
    async def _handle_batch_completed(self, event):
        """处理批量任务完成事件"""
        batch_id = event['data'].get('batch_id')
        total_duration = event['data'].get('total_duration', 0)
        success_count = event['data'].get('success_count', 0)
        logger.info(f"批量任务完成: {batch_id}, 总耗时: {total_duration:.2f}s, 成功: {success_count}")
        record_data_metric("async_batch_completed", success_count, "count", {"batch_id": batch_id})
    
    async def _handle_processor_overloaded(self, event):
        """处理处理器过载事件"""
        load_factor = event['data'].get('load_factor', 0)
        logger.warning(f"异步处理器过载，负载因子: {load_factor}")
        record_data_metric("async_processor_overloaded", load_factor, "gauge")
    
    async def _handle_config_updated(self, event):
        """处理配置更新事件"""
        config_changes = event['data'].get('changes', {})
        logger.info(f"异步处理器配置更新: {config_changes}")


__all__ = ['AsyncEventHandler']

