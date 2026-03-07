"""
事件过滤器模块
提供配置事件过滤功能
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import re
from enum import Enum


class FilterType(Enum):

    """过滤器类型"""
    INCLUDE = "include"
    EXCLUDE = "exclude"


class IEventFilterComponent(ABC):

    """EventFilter组件接口"""

    @abstractmethod
    def should_process(self, event: Dict[str, Any]) -> bool:
        """
        判断是否应该处理事件

        Args:
            event: 事件数据

        Returns:
            是否应该处理
        """

    @abstractmethod
    def get_filter_info(self) -> Dict[str, Any]:
        """
        获取过滤器信息

        Returns:
            过滤器信息
        """


# 类型别名
IEventFilter = IEventFilterComponent


class EventTypeFilter(IEventFilter):

    """事件类型过滤器"""

    def __init__(self, event_types: List[str], filter_type: FilterType = FilterType.INCLUDE):

        self.event_types = set(event_types)
        self.filter_type = filter_type

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        event_type = event.get("type", "")

        if self.filter_type == FilterType.INCLUDE:
            return event_type in self.event_types
        else:
            return event_type not in self.event_types

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器信息"""
        return {
            "type": "EventTypeFilter",
            "event_types": list(self.event_types),
            "filter_type": self.filter_type.value
        }


class SensitiveDataFilter(IEventFilter):

    """敏感数据过滤器"""

    def __init__(self, sensitive_keys: List[str],


                 replacement: str = "***",
                 filter_type: FilterType = FilterType.EXCLUDE):
        self.sensitive_keys = set(sensitive_keys)
        self.replacement = replacement
        self.filter_type = filter_type

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        # 检查是否包含敏感数据
        has_sensitive_data = self._contains_sensitive_data(event)

        if self.filter_type == FilterType.EXCLUDE:
            return not has_sensitive_data
        else:
            return has_sensitive_data

    def _contains_sensitive_data(self, data: Any, path: str = "") -> bool:
        """检查是否包含敏感数据"""
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if key in self.sensitive_keys:
                    return True
                if self._contains_sensitive_data(value, current_path):
                    return True
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                if self._contains_sensitive_data(item, current_path):
                    return True
        return False

    def sanitize_data(self, data: Any) -> Any:
        """清理敏感数据"""
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in self.sensitive_keys:
                    result[key] = self.replacement
                else:
                    result[key] = self.sanitize_data(value)
            return result
        elif isinstance(data, list):
            return [self.sanitize_data(item) for item in data]
        else:
            return data

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器信息"""
        return {
            "type": "SensitiveDataFilter",
            "sensitive_keys": list(self.sensitive_keys),
            "replacement": self.replacement,
            "filter_type": self.filter_type.value
        }


class PatternFilter(IEventFilter):

    """模式过滤器"""

    def __init__(self, pattern: str, field: str = "message",


                 filter_type: FilterType = FilterType.INCLUDE):
        self.pattern = re.compile(pattern)
        self.field = field
        self.filter_type = filter_type

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        field_value = event.get(self.field, "")

        if isinstance(field_value, str):
            matches = bool(self.pattern.search(field_value))
        else:
            matches = False

        if self.filter_type == FilterType.INCLUDE:
            return matches
        else:
            return not matches

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器信息"""
        return {
            "type": "PatternFilter",
            "pattern": self.pattern.pattern,
            "field": self.field,
            "filter_type": self.filter_type.value
        }


class TimeRangeFilter(IEventFilter):

    """时间范围过滤器"""

    def __init__(self, start_time: Optional[str] = None,


                 end_time: Optional[str] = None,
                 filter_type: FilterType = FilterType.INCLUDE):
        self.start_time = start_time
        self.end_time = end_time
        self.filter_type = filter_type

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        event_time = event.get("timestamp", "")

        if not event_time:
            return True

        # 简单的时间比较（假设时间格式为ISO格式）
        try:
            from datetime import datetime
            event_dt = datetime.fromisoformat(event_time.replace('Z', '+00:00'))

            if self.start_time:
                start_dt = datetime.fromisoformat(self.start_time.replace('Z', '+00:00'))
                if event_dt < start_dt:
                    return False

            if self.end_time:
                end_dt = datetime.fromisoformat(self.end_time.replace('Z', '+00:00'))
                if event_dt > end_dt:
                    return False

            return True
        except Exception:
            return True

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器信息"""
        return {
            "type": "TimeRangeFilter",
            "start_time": self.start_time,
            "end_time": self.end_time,
            "filter_type": self.filter_type.value
        }


class CompositeFilter(IEventFilter):

    """复合过滤器"""

    def __init__(self, filters: List[IEventFilter],


                 operator: str = "AND"):
        self.filters = filters
        self.operator = operator.upper()

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        if not self.filters:
            return True

        results = [filter_obj.should_process(event) for filter_obj in self.filters]

        if self.operator == "AND":
            return all(results)
        elif self.operator == "OR":
            return any(results)
        else:
            return True

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器信息"""
        return {
            "type": "CompositeFilter",
            "operator": self.operator,
            "filters": [filter_obj.get_filter_info() for filter_obj in self.filters]
        }


class EventFilterChain:

    """事件过滤器链"""

    def __init__(self):

        self.filters: List[IEventFilter] = []

    def add_filter(self, filter_obj: IEventFilter) -> None:
        """添加过滤器"""
        self.filters.append(filter_obj)

    def remove_filter(self, filter_obj: IEventFilter) -> bool:
        """移除过滤器"""
        if filter_obj in self.filters:
            self.filters.remove(filter_obj)
            return True
        return False

    def clear_filters(self) -> None:
        """清空所有过滤器"""
        self.filters.clear()

    def should_process(self, event: Dict[str, Any]) -> bool:
        """判断是否应该处理事件"""
        for filter_obj in self.filters:
            if not filter_obj.should_process(event):
                return False
        return True

    def process_event(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        处理事件

        Args:
            event: 原始事件

        Returns:
            处理后的事件，如果不应该处理则返回None
        """
        if not self.should_process(event):
            return None

        # 应用敏感数据过滤器
        processed_event = event.copy()
        for filter_obj in self.filters:
            if isinstance(filter_obj, SensitiveDataFilter):
                processed_event = filter_obj.sanitize_data(processed_event)

        return processed_event

    def get_filter_info(self) -> Dict[str, Any]:
        """获取过滤器链信息"""
        return {
            "filter_count": len(self.filters),
            "filters": [filter_obj.get_filter_info() for filter_obj in self.filters]
        }


class EventFilterFactory:

    """事件过滤器工厂"""

    @staticmethod
    def create_type_filter(event_types: List[str],


                           filter_type: FilterType = FilterType.INCLUDE) -> EventTypeFilter:
        """创建事件类型过滤器"""
        return EventTypeFilter(event_types, filter_type)

    @staticmethod
    def create_sensitive_filter(sensitive_keys: List[str],


                                replacement: str = "***",
                                filter_type: FilterType = FilterType.EXCLUDE) -> SensitiveDataFilter:
        """创建敏感数据过滤器"""
        return SensitiveDataFilter(sensitive_keys, replacement, filter_type)

    @staticmethod
    def create_pattern_filter(pattern: str,


                              field: str = "message",
                              filter_type: FilterType = FilterType.INCLUDE) -> PatternFilter:
        """创建模式过滤器"""
        return PatternFilter(pattern, field, filter_type)

    @staticmethod
    def create_time_filter(start_time: Optional[str] = None,


                           end_time: Optional[str] = None,
                           filter_type: FilterType = FilterType.INCLUDE) -> TimeRangeFilter:
        """创建时间范围过滤器"""
        return TimeRangeFilter(start_time, end_time, filter_type)

    @staticmethod
    def create_composite_filter(filters: List[IEventFilter],


                                operator: str = "AND") -> CompositeFilter:
        """创建复合过滤器"""
        return CompositeFilter(filters, operator)

    @staticmethod
    def create_default_filter_chain() -> EventFilterChain:
        """创建默认过滤器链"""
        chain = EventFilterChain()

        # 添加敏感数据过滤器
        sensitive_filter = EventFilterFactory.create_sensitive_filter([
            "password", "token", "secret", "key", "credential"])

        chain.add_filter(sensitive_filter)

        return chain
