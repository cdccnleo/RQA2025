"""
健康检查数据模型

包含健康检查相关的枚举和数据类定义。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List


class HealthCheckType(Enum):
    """健康检查类型"""
    BASIC = "basic"
    DETAILED = "detailed"
    FULL = "full"


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AsyncHealthCheckConfig:
    """异步健康检查配置"""
    enabled: bool = True
    check_interval: int = 60
    timeout: int = 30
    retry_count: int = 3
    parallel_checks: int = 5

