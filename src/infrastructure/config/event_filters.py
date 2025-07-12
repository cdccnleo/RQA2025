from abc import ABC, abstractmethod
from typing import Dict, Any
from threading import Lock

class IEventFilter(ABC):
    """事件过滤器接口"""
    @abstractmethod
    def filter(self, event: Dict[str, Any]) -> bool:
        """过滤事件，返回True表示允许通过"""
        pass

class SensitiveDataFilter(IEventFilter):
    """敏感数据过滤器"""
    def __init__(self, sensitive_keys: list = None):
        self.sensitive_keys = sensitive_keys or ['password', 'secret', 'token']
        self._lock = Lock()

    def filter(self, event: Dict) -> bool:
        """过滤事件中的敏感数据"""
        # 处理配置变更事件
        if event.get('type') == 'config_updated':
            key = event.get('key', '')
            new_value = event.get('new_value')

            # 过滤API密钥
            if 'api_key' in key.lower() and isinstance(new_value, str):
                # 保留前7个字符（包括下划线）
                prefix = new_value[:7]
                event['new_value'] = prefix + '****'

            # 过滤密码字段
            if 'password' in key.lower() and new_value is not None:
                event['new_value'] = '***'

        return True

class EventTypeFilter(IEventFilter):
    """事件类型过滤器"""
    def __init__(self, allowed_types: list):
        self.allowed_types = set(allowed_types)
        self._lock = Lock()

    def filter(self, event: Dict[str, Any]) -> bool:
        """只允许特定类型的事件通过"""
        with self._lock:
            return event.get('type') in self.allowed_types
