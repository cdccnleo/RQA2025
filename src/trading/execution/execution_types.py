"""执行引擎类型定义"""

from enum import Enum


class ExecutionMode(Enum):
    """执行模式枚举"""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"

    @property
    def value(self):
        """获取枚举值"""
        return self._value_


__all__ = ['ExecutionMode', 'ExecutionStatus']

