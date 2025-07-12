"""配置事件过滤器实现"""
from typing import Set, Optional, Pattern
import re
from dataclasses import dataclass
from .config_event import ConfigEvent

@dataclass
class EventFilter:
    """基础事件过滤器"""
    event_types: Set[str]  # 要过滤的事件类型集合
    key_pattern: Optional[str] = None  # 键匹配模式(正则表达式)

    def filter(self, event: ConfigEvent) -> bool:
        """过滤事件
        Args:
            event: 配置事件对象
        Returns:
            bool: 是否通过过滤
        """
        if event.event_type not in self.event_types:
            return False

        if self.key_pattern is not None:
            return bool(re.fullmatch(self.key_pattern, event.key))

        return True


class EnvironmentFilter(EventFilter):
    """环境过滤器 - 按环境过滤配置事件"""
    def __init__(self, env: str):
        """初始化环境过滤器
        Args:
            env: 要过滤的环境名称(如'prod')
        """
        super().__init__(
            event_types={"config_updated", "config_loaded"},
            key_pattern=rf"{env}\..*"
        )


class SensitiveDataFilter(EventFilter):
    """敏感数据过滤器 - 过滤包含敏感信息的配置变更"""
    def __init__(self):
        """初始化敏感数据过滤器"""
        super().__init__(
            event_types={"config_updated"},
            key_pattern=".*(password|secret|token).*"
        )


class EventTypeFilter:
    """事件类型过滤器 - 按事件类型白名单过滤"""
    def __init__(self, allowed_types: Set[str]):
        self.allowed_types = allowed_types
    
    def filter(self, event: ConfigEvent) -> bool:
        """过滤事件"""
        return event.event_type in self.allowed_types


class CompositeFilter:
    """复合过滤器 - 组合多个过滤器"""
    def __init__(self, filters: list):
        self.filters = filters
    
    def filter(self, event: ConfigEvent) -> bool:
        """应用所有过滤器"""
        return all(f.filter(event) for f in self.filters)
