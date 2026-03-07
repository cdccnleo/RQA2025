"""
事件总线组件模块（别名模块）
提供向后兼容的导入路径

实际实现在 components/ 目录中
"""

try:
    from .components import (
        EventPublisher,
        EventSubscriber,
        EventProcessor,
        EventMonitor
    )
    from .core import EventBus
except ImportError:
    # 提供基础实现
    class EventPublisher:
        pass
    
    class EventSubscriber:
        pass
    
    class EventProcessor:
        pass
    
    class EventMonitor:
        pass
    
    class EventBus:
        pass

__all__ = ['EventPublisher', 'EventSubscriber', 'EventProcessor', 'EventMonitor', 'EventBus']

