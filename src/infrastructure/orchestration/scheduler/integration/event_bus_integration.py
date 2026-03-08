"""
调度器与事件总线集成模块

实现事件驱动的任务调度，支持：
- 事件触发任务执行
- 任务状态变更事件发布
- 任务完成事件通知
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass

from ..base import Task, TaskStatus, JobType

try:
    from src.core.event_bus.core import EventBus
    from src.core.event_bus.models import Event, EventPriority
    from src.core.event_bus.types import EventType
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False
    EventBus = None
    Event = None
    EventPriority = None
    EventType = None

logger = logging.getLogger(__name__)


# 调度器事件类型
class SchedulerEventType:
    """调度器事件类型定义"""
    TASK_CREATED = "scheduler.task.created"
    TASK_STARTED = "scheduler.task.started"
    TASK_COMPLETED = "scheduler.task.completed"
    TASK_FAILED = "scheduler.task.failed"
    TASK_CANCELLED = "scheduler.task.cancelled"
    TASK_TIMEOUT = "scheduler.task.timeout"
    TASK_RETRYING = "scheduler.task.retrying"

    JOB_TRIGGERED = "scheduler.job.triggered"
    JOB_COMPLETED = "scheduler.job.completed"
    JOB_FAILED = "scheduler.job.failed"

    WORKER_STARTED = "scheduler.worker.started"
    WORKER_STOPPED = "scheduler.worker.stopped"
    WORKER_ERROR = "scheduler.worker.error"

    SCHEDULER_STARTED = "scheduler.started"
    SCHEDULER_STOPPED = "scheduler.stopped"
    SCHEDULER_ERROR = "scheduler.error"


@dataclass
class TaskEventData:
    """任务事件数据"""
    task_id: str
    task_type: str
    status: str
    timestamp: str
    payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "error": self.error,
            "execution_time": self.execution_time
        }


class EventBusIntegration:
    """事件总线集成器"""

    def __init__(self, event_bus: Optional[Any] = None):
        """
        初始化事件总线集成器

        Args:
            event_bus: 事件总线实例，None则尝试获取全局实例
        """
        self._event_bus = event_bus
        self._initialized = False
        self._event_subscriptions: List[str] = []

        if not EVENT_BUS_AVAILABLE:
            logger.warning("事件总线模块不可用，事件集成功能将被禁用")
            return

        # 如果没有提供事件总线，尝试获取全局实例
        if self._event_bus is None:
            try:
                self._event_bus = EventBus.get_instance()
            except Exception as e:
                logger.warning(f"无法获取事件总线实例: {e}")
                return

        self._initialized = True
        logger.info("事件总线集成器初始化成功")

    def is_available(self) -> bool:
        """检查事件总线是否可用"""
        return self._initialized and self._event_bus is not None

    # ========== 事件发布方法 ==========

    async def publish_task_created(self, task: Task):
        """发布任务创建事件"""
        if not self.is_available():
            return

        try:
            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="created",
                timestamp=datetime.now().isoformat(),
                payload=task.payload
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_CREATED,
                data=event_data.to_dict(),
                priority=EventPriority.NORMAL if EventPriority else 2
            )
        except Exception as e:
            logger.error(f"发布任务创建事件失败: {e}")

    async def publish_task_started(self, task: Task):
        """发布任务开始事件"""
        if not self.is_available():
            return

        try:
            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="started",
                timestamp=datetime.now().isoformat(),
                payload=task.payload
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_STARTED,
                data=event_data.to_dict(),
                priority=EventPriority.HIGH if EventPriority else 1
            )
        except Exception as e:
            logger.error(f"发布任务开始事件失败: {e}")

    async def publish_task_completed(self, task: Task, result: Any = None):
        """发布任务完成事件"""
        if not self.is_available():
            return

        try:
            execution_time = None
            if task.started_at and task.completed_at:
                execution_time = (task.completed_at - task.started_at).total_seconds()

            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="completed",
                timestamp=datetime.now().isoformat(),
                payload=task.payload,
                execution_time=execution_time
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_COMPLETED,
                data=event_data.to_dict(),
                priority=EventPriority.NORMAL if EventPriority else 2
            )
        except Exception as e:
            logger.error(f"发布任务完成事件失败: {e}")

    async def publish_task_failed(self, task: Task, error: str):
        """发布任务失败事件"""
        if not self.is_available():
            return

        try:
            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="failed",
                timestamp=datetime.now().isoformat(),
                payload=task.payload,
                error=error
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_FAILED,
                data=event_data.to_dict(),
                priority=EventPriority.HIGH if EventPriority else 1
            )
        except Exception as e:
            logger.error(f"发布任务失败事件失败: {e}")

    async def publish_task_timeout(self, task: Task):
        """发布任务超时事件"""
        if not self.is_available():
            return

        try:
            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="timeout",
                timestamp=datetime.now().isoformat(),
                payload=task.payload,
                error="Task execution timeout"
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_TIMEOUT,
                data=event_data.to_dict(),
                priority=EventPriority.HIGH if EventPriority else 1
            )
        except Exception as e:
            logger.error(f"发布任务超时事件失败: {e}")

    async def publish_task_retrying(self, task: Task, retry_count: int):
        """发布任务重试事件"""
        if not self.is_available():
            return

        try:
            event_data = TaskEventData(
                task_id=task.id,
                task_type=task.type,
                status="retrying",
                timestamp=datetime.now().isoformat(),
                payload={**task.payload, "retry_count": retry_count}
            )

            await self._publish_event(
                event_type=SchedulerEventType.TASK_RETRYING,
                data=event_data.to_dict(),
                priority=EventPriority.NORMAL if EventPriority else 2
            )
        except Exception as e:
            logger.error(f"发布任务重试事件失败: {e}")

    async def publish_scheduler_started(self, config: Dict[str, Any]):
        """发布调度器启动事件"""
        if not self.is_available():
            return

        try:
            await self._publish_event(
                event_type=SchedulerEventType.SCHEDULER_STARTED,
                data={
                    "timestamp": datetime.now().isoformat(),
                    "config": config
                },
                priority=EventPriority.HIGH if EventPriority else 1
            )
        except Exception as e:
            logger.error(f"发布调度器启动事件失败: {e}")

    async def publish_scheduler_stopped(self, stats: Dict[str, Any]):
        """发布调度器停止事件"""
        if not self.is_available():
            return

        try:
            await self._publish_event(
                event_type=SchedulerEventType.SCHEDULER_STOPPED,
                data={
                    "timestamp": datetime.now().isoformat(),
                    "statistics": stats
                },
                priority=EventPriority.HIGH if EventPriority else 1
            )
        except Exception as e:
            logger.error(f"发布调度器停止事件失败: {e}")

    async def _publish_event(self, event_type: str, data: Dict[str, Any], priority: int = 2):
        """
        发布事件到事件总线

        Args:
            event_type: 事件类型
            data: 事件数据
            priority: 事件优先级
        """
        if not self.is_available():
            return

        try:
            # 创建事件
            if Event:
                event = Event(
                    event_type=event_type,
                    data=data,
                    priority=priority,
                    timestamp=datetime.now()
                )
                await self._event_bus.publish(event)
            else:
                # 直接调用发布方法
                await self._event_bus.publish(event_type, data)

            logger.debug(f"事件已发布: {event_type}")
        except Exception as e:
            logger.error(f"发布事件失败 [{event_type}]: {e}")

    # ========== 事件订阅方法 ==========

    def subscribe_to_task_events(self, handler: Callable):
        """
        订阅任务相关事件

        Args:
            handler: 事件处理函数
        """
        if not self.is_available():
            return

        try:
            # 订阅所有任务事件
            task_events = [
                SchedulerEventType.TASK_CREATED,
                SchedulerEventType.TASK_STARTED,
                SchedulerEventType.TASK_COMPLETED,
                SchedulerEventType.TASK_FAILED,
                SchedulerEventType.TASK_CANCELLED,
                SchedulerEventType.TASK_TIMEOUT
            ]

            for event_type in task_events:
                subscription_id = self._event_bus.subscribe(event_type, handler)
                self._event_subscriptions.append(subscription_id)

            logger.info(f"已订阅任务事件，共 {len(task_events)} 种")
        except Exception as e:
            logger.error(f"订阅任务事件失败: {e}")

    def subscribe_to_external_events(self, event_type: str, handler: Callable):
        """
        订阅外部事件（用于事件驱动任务触发）

        Args:
            event_type: 外部事件类型
            handler: 事件处理函数
        """
        if not self.is_available():
            return

        try:
            subscription_id = self._event_bus.subscribe(event_type, handler)
            self._event_subscriptions.append(subscription_id)
            logger.info(f"已订阅外部事件: {event_type}")
        except Exception as e:
            logger.error(f"订阅外部事件失败 [{event_type}]: {e}")

    def unsubscribe_all(self):
        """取消所有事件订阅"""
        if not self.is_available():
            return

        try:
            for subscription_id in self._event_subscriptions:
                self._event_bus.unsubscribe(subscription_id)

            self._event_subscriptions.clear()
            logger.info("已取消所有事件订阅")
        except Exception as e:
            logger.error(f"取消事件订阅失败: {e}")


class EventDrivenTaskTrigger:
    """事件驱动任务触发器"""

    def __init__(self, scheduler: Any, event_bus_integration: EventBusIntegration):
        """
        初始化事件驱动任务触发器

        Args:
            scheduler: 调度器实例
            event_bus_integration: 事件总线集成器
        """
        self._scheduler = scheduler
        self._event_bus = event_bus_integration
        self._event_task_mappings: Dict[str, List[Dict[str, Any]]] = {}

    def register_event_trigger(
        self,
        event_type: str,
        task_type: str,
        payload_template: Optional[Dict[str, Any]] = None,
        priority: int = 5,
        timeout_seconds: Optional[int] = None
    ):
        """
        注册事件触发器

        当指定类型的事件发生时，自动创建并执行任务

        Args:
            event_type: 监听的事件类型
            task_type: 要执行的任务类型
            payload_template: 任务数据模板（支持从事件数据提取字段）
            priority: 任务优先级
            timeout_seconds: 任务超时时间
        """
        if event_type not in self._event_task_mappings:
            self._event_task_mappings[event_type] = []

        self._event_task_mappings[event_type].append({
            "task_type": task_type,
            "payload_template": payload_template or {},
            "priority": priority,
            "timeout_seconds": timeout_seconds
        })

        # 订阅事件
        self._event_bus.subscribe_to_external_events(
            event_type,
            self._handle_event
        )

        logger.info(f"已注册事件触发器: {event_type} -> {task_type}")

    async def _handle_event(self, event: Any):
        """
        处理事件并触发任务

        Args:
            event: 事件对象
        """
        try:
            # 获取事件类型
            event_type = event.event_type if hasattr(event, 'event_type') else str(event)

            # 查找对应的任务配置
            task_configs = self._event_task_mappings.get(event_type, [])

            for config in task_configs:
                # 构建任务payload
                payload = self._build_payload(
                    config["payload_template"],
                    event
                )

                # 提交任务
                task_id = await self._scheduler.submit_task(
                    task_type=config["task_type"],
                    payload=payload,
                    priority=config["priority"],
                    timeout_seconds=config["timeout_seconds"]
                )

                logger.info(f"事件 {event_type} 触发任务 {config['task_type']} (ID: {task_id})")

        except Exception as e:
            logger.error(f"处理事件触发任务失败: {e}")

    def _build_payload(
        self,
        template: Dict[str, Any],
        event: Any
    ) -> Dict[str, Any]:
        """
        构建任务payload

        支持从事件数据中提取字段，使用 ${event.field} 语法

        Args:
            template: 模板
            event: 事件对象

        Returns:
            Dict[str, Any]: 构建后的payload
        """
        payload = {}

        # 获取事件数据
        event_data = {}
        if hasattr(event, 'data'):
            event_data = event.data
        elif hasattr(event, 'to_dict'):
            event_data = event.to_dict()
        elif isinstance(event, dict):
            event_data = event

        # 处理模板
        for key, value in template.items():
            if isinstance(value, str) and value.startswith("${event.") and value.endswith("}"):
                # 提取字段路径
                field_path = value[8:-1]  # 移除 ${event. 和 }
                field_value = self._get_nested_value(event_data, field_path)
                payload[key] = field_value
            else:
                payload[key] = value

        # 添加事件元数据
        payload["_event_triggered"] = True
        payload["_event_timestamp"] = datetime.now().isoformat()

        return payload

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """
        获取嵌套字典值

        Args:
            data: 数据字典
            path: 字段路径，如 "user.name"

        Returns:
            Any: 字段值
        """
        keys = path.split(".")
        value = data

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value


# 全局集成实例
_event_bus_integration_instance: Optional[EventBusIntegration] = None


def get_event_bus_integration(event_bus: Optional[Any] = None) -> EventBusIntegration:
    """
    获取事件总线集成实例（单例）

    Args:
        event_bus: 事件总线实例

    Returns:
        EventBusIntegration: 事件总线集成实例
    """
    global _event_bus_integration_instance

    if _event_bus_integration_instance is None:
        _event_bus_integration_instance = EventBusIntegration(event_bus)

    return _event_bus_integration_instance


def reset_event_bus_integration():
    """重置事件总线集成实例（用于测试）"""
    global _event_bus_integration_instance
    _event_bus_integration_instance = None
