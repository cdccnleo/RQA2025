
from enum import Enum
"""
健康状态定义

统一管理健康状态相关的枚举和常量
"""


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"
