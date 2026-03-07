from enum import Enum, IntEnum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any

class AlertLevel(Enum):
    INFO = ("info", 0)
    WARNING = ("warning", 1)
    ERROR = ("error", 2)
    CRITICAL = ("critical", 3)
    
    def __init__(self, string_value, int_value):
        self._string_value = string_value
        self._int_value = int_value
    
    @property
    def value(self):
        """返回字符串值，以匹配test_enum_values测试"""
        return self._string_value
    
    @property
    def int_value(self):
        """返回整数值，用于排序比较"""
        return self._int_value
    
    def __lt__(self, other):
        """支持枚举实例比较"""
        if isinstance(other, AlertLevel):
            return self._int_value < other._int_value
        return NotImplemented
    
    def __le__(self, other):
        """支持枚举实例比较"""
        if isinstance(other, AlertLevel):
            return self._int_value <= other._int_value
        return NotImplemented
    
    def __gt__(self, other):
        """支持枚举实例比较"""
        if isinstance(other, AlertLevel):
            return self._int_value > other._int_value
        return NotImplemented
    
    def __ge__(self, other):
        """支持枚举实例比较"""
        if isinstance(other, AlertLevel):
            return self._int_value >= other._int_value
        return NotImplemented

@dataclass(frozen=False)
class AlertData:
    level: AlertLevel
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
