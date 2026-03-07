"""
基础设施层参数对象定义

使用参数对象模式简化长参数列表，提高代码可读性和可维护性

作者: RQA2025团队
创建时间: 2025-10-23
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


# ==================== 健康检查参数对象 ====================

@dataclass
class HealthCheckParams:
    """健康检查参数对象"""
    
    service_name: str
    timeout: int = 30
    retry_count: int = 3
    check_dependencies: bool = True
    include_details: bool = True
    check_timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.check_timestamp is None:
            self.check_timestamp = datetime.now()


@dataclass
class ServiceHealthReportParams:
    """服务健康报告参数对象"""
    
    include_metrics: bool = True
    include_history: bool = False
    history_limit: int = 10
    detailed_report: bool = False
    services_filter: Optional[List[str]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.services_filter is None:
            self.services_filter = []


@dataclass
class HealthCheckResultParams:
    """健康检查结果参数对象"""
    
    service_name: str
    healthy: bool
    status: str
    timestamp: Optional[datetime] = None
    version: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    issues: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    response_time: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.version is None:
            self.version = "1.0.0"
        if self.details is None:
            self.details = {}
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []


# ==================== 配置验证参数对象 ====================

@dataclass
class ConfigValidationParams:
    """配置验证参数对象"""
    
    value: Any
    expected_type: Optional[type] = None
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    allowed_values: Optional[List[Any]] = None
    regex_pattern: Optional[str] = None
    custom_validator: Optional[callable] = None
    
    def validate(self) -> bool:
        """执行验证"""
        # 类型检查
        if self.expected_type and not isinstance(self.value, self.expected_type):
            return False
        
        # 范围检查
        if isinstance(self.value, (int, float)):
            if self.min_value is not None and self.value < self.min_value:
                return False
            if self.max_value is not None and self.value > self.max_value:
                return False
        
        # 允许值检查
        if self.allowed_values and self.value not in self.allowed_values:
            return False
        
        # 自定义验证器
        if self.custom_validator:
            return self.custom_validator(self.value)
        
        return True


# ==================== 服务初始化参数对象 ====================

@dataclass
class ServiceInitializationParams:
    """服务初始化参数对象"""
    
    config: Optional[Dict[str, Any]] = None
    enable_monitoring: bool = True
    enable_caching: bool = True
    enable_logging: bool = True
    auto_retry: bool = True
    max_retries: int = 3
    timeout: int = 30
    startup_delay: int = 0
    dependencies: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.dependencies is None:
            self.dependencies = []


# ==================== 监控参数对象 ====================

@dataclass
class MonitoringParams:
    """监控参数对象"""
    
    metric_name: str
    metric_value: Any
    metric_type: str = "gauge"  # gauge, counter, histogram, summary
    tags: Optional[Dict[str, str]] = None
    timestamp: Optional[datetime] = None
    unit: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class AlertParams:
    """告警参数对象"""
    
    alert_level: str  # info, warning, error, critical
    alert_message: str
    alert_source: str
    timestamp: Optional[datetime] = None
    context: Optional[Dict[str, Any]] = None
    notify_channels: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context is None:
            self.context = {}
        if self.notify_channels is None:
            self.notify_channels = ["log"]


# ==================== 资源管理参数对象 ====================

@dataclass
class ResourceAllocationParams:
    """资源分配参数对象"""
    
    resource_type: str
    resource_amount: float
    requester_id: str
    priority: int = 0
    timeout: int = 30
    allow_overcommit: bool = False
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ResourceUsageParams:
    """资源使用参数对象"""
    
    resource_type: str
    current_usage: float
    total_capacity: float
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95
    unit: str = "%"
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def usage_percentage(self) -> float:
        """计算使用百分比"""
        if self.total_capacity == 0:
            return 0.0
        return (self.current_usage / self.total_capacity) * 100
    
    @property
    def is_warning_level(self) -> bool:
        """是否达到警告级别"""
        return self.usage_percentage >= (self.warning_threshold * 100)
    
    @property
    def is_critical_level(self) -> bool:
        """是否达到危险级别"""
        return self.usage_percentage >= (self.critical_threshold * 100)


# ==================== 缓存参数对象 ====================

@dataclass
class CacheOperationParams:
    """缓存操作参数对象"""
    
    operation: str  # get, set, delete, clear
    key: Optional[str] = None
    value: Optional[Any] = None
    ttl: Optional[int] = None
    namespace: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.options is None:
            self.options = {}


# ==================== 日志参数对象 ====================

@dataclass
class LogRecordParams:
    """日志记录参数对象"""
    
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: str
    logger_name: Optional[str] = None
    exc_info: Optional[Exception] = None
    extra: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.extra is None:
            self.extra = {}


# ==================== 导出列表 ====================

__all__ = [
    # 健康检查
    'HealthCheckParams',
    'ServiceHealthReportParams',
    'HealthCheckResultParams',
    
    # 配置验证
    'ConfigValidationParams',
    
    # 服务初始化
    'ServiceInitializationParams',
    
    # 监控和告警
    'MonitoringParams',
    'AlertParams',
    
    # 资源管理
    'ResourceAllocationParams',
    'ResourceUsageParams',
    
    # 缓存操作
    'CacheOperationParams',
    
    # 日志记录
    'LogRecordParams',
]

