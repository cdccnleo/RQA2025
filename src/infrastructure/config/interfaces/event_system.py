from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional

class IConfigEventSystem(ABC):
    """配置事件系统接口"""

    @abstractmethod
    def publish(self, event_type: str, payload: Dict) -> None:
        """发布事件
        Args:
            event_type: 事件类型标识符
            payload: 事件负载数据
        Raises:
            EventDeliveryError: 当事件投递失败时抛出
        """
        pass

    @abstractmethod
    def subscribe(self,
                 event_type: str,
                 handler: Callable[[Dict], None]) -> str:
        """订阅事件
        Args:
            event_type: 要订阅的事件类型
            handler: 事件处理函数
        Returns:
            订阅ID(用于取消订阅)
        """
        pass

    @abstractmethod
    def get_dead_letters(self) -> List[Dict]:
        """获取死信队列
        Returns:
            包含失败事件信息的字典列表
        """
        pass

    @abstractmethod
    def clear_dead_letters(self) -> None:
        """清空死信队列"""
        pass


class IEventSubscriber(ABC):
    """事件订阅者接口"""

    @abstractmethod
    def handle_event(self, event: Dict) -> bool:
        """处理接收到的事件
        Args:
            event: 包含事件数据和元信息的字典
        Returns:
            bool: 处理是否成功
        Raises:
            EventProcessingError: 当处理失败时抛出
        """
        pass
