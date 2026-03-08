"""
Saga Choreography Module

提供协作式Saga的实现，基于事件驱动处理分布式事务。
"""

import logging
from typing import Any, Dict, List

from ..events.events import DomainEvent, EventHandler, CompensationHandler, HandlerResult


logger = logging.getLogger(__name__)


class ChoreographySaga:
    """
    协作式Saga
    
    基于事件驱动的分布式事务实现，服务间通过事件协作完成事务。
    
    功能：
    1. 注册事件处理器
    2. 监听和处理事件
    3. 触发补偿事件
    4. 维护本地状态
    """
    
    def __init__(self, event_bus=None):
        """
        初始化协作式Saga
        
        Args:
            event_bus: 事件总线
        """
        self.event_bus = event_bus
        self.event_handlers: Dict[str, List[EventHandler]] = {}
        self.compensation_handlers: Dict[str, CompensationHandler] = {}
        
    def register_handler(self, event_type: str, handler: EventHandler) -> None:
        """
        注册事件处理器
        
        Args:
            event_type: 事件类型
            handler: 事件处理器
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
        logger.info(f"Registered handler for event type: {event_type}")
        
    def register_compensation(self, action_type: str, handler: CompensationHandler) -> None:
        """
        注册补偿处理器
        
        Args:
            action_type: 动作类型
            handler: 补偿处理器
        """
        self.compensation_handlers[action_type] = handler
        logger.info(f"Registered compensation handler for: {action_type}")
        
    async def handle_event(self, event: DomainEvent) -> None:
        """
        处理事件
        
        Args:
            event: 领域事件
        """
        handlers = self.event_handlers.get(event.type, [])
        
        if not handlers:
            logger.warning(f"No handlers registered for event type: {event.type}")
            return
            
        for handler in handlers:
            try:
                result = await handler.process(event)
                
                if result.success:
                    # 发布成功事件
                    if result.next_event and self.event_bus:
                        await self.event_bus.publish(result.next_event)
                else:
                    # 发布补偿事件
                    await self._publish_compensation_event(event)
                    
            except Exception as e:
                logger.error(f"Error handling event {event.type}: {e}")
                # 发布补偿事件
                await self._publish_compensation_event(event)
                
    async def _publish_compensation_event(self, failed_event: DomainEvent) -> None:
        """
        发布补偿事件
        
        Args:
            failed_event: 失败的事件
        """
        if not self.event_bus:
            return
            
        try:
            from ..events.events import CompensationEvent
            
            compensation_event = CompensationEvent(
                saga_id=failed_event.saga_id,
                failed_event_type=failed_event.type,
                failed_event_data=failed_event.data
            )
            
            await self.event_bus.publish(compensation_event)
            logger.info(f"Published compensation event for saga {failed_event.saga_id}")
            
        except Exception as e:
            logger.error(f"Failed to publish compensation event: {e}")
