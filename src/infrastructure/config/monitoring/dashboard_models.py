
# ==================== 枚举定义 ====================

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple
import datetime as dt
"""
监控面板数据模型

定义监控系统的数据结构和枚举
"""


class MetricType(Enum):
    """指标类型枚举"""
    COUNTER = "counter"          # 计数器
    GAUGE = "gauge"              # 仪表
    HISTOGRAM = "histogram"      # 直方图
    SUMMARY = "summary"          # 摘要


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"                # 信息
    WARNING = "warning"          # 警告
    ERROR = "error"              # 错误
    CRITICAL = "critical"        # 严重


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"            # 活跃
    RESOLVED = "resolved"        # 已解决
    ACKNOWLEDGED = "acknowledged"  # 已确认

# ==================== 数据类定义 ====================


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: Union[int, float]
    type: MetricType
    timestamp: dt.datetime
    labels: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Metric:
    """指标定义"""
    name: str
    type: MetricType
    description: str
    unit: Optional[str] = None
    labels: Optional[List[str]] = None
    help_text: Optional[str] = None


@dataclass
class Alert:
    """告警定义"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    timestamp: dt.datetime
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None
    value: Optional[Union[int, float]] = None
    threshold: Optional[Union[int, float]] = None


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    collection_interval: int = 15  # 秒
    retention_days: int = 30
    alerting_enabled: bool = True
    metrics_endpoint: Optional[str] = None
    alerting_endpoint: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """性能指标 (兼容原有接口)"""
    timestamp: dt.datetime
    operation_type: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SystemResources:
    """系统资源使用情况"""
    timestamp: dt.datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: Dict[str, float]
    network_io: Dict[str, int]
    load_average: Optional[Tuple[float, ...]] = None


@dataclass
class ConfigOperationStats:
    """配置操作统计"""
    operation: str
    count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_execution: Optional[dt.datetime] = None

    def add_metric(self, duration: float, success: bool):
        """添加指标数据"""
        self.count += 1
        self.total_time += duration
        self.avg_time = self.total_time / self.count
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.last_execution = dt.datetime.now()

        if success:
            self.success_count += 1
        else:
            self.error_count += 1

    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.count == 0:
            return 0.0
        return self.success_count / self.count

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'operation': self.operation,
            'count': self.count,
            'total_time': self.total_time,
            'avg_time': self.avg_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0.0,
            'max_time': self.max_time,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.get_success_rate(),
            'last_execution': self.last_execution.isoformat() if self.last_execution else None
        }




