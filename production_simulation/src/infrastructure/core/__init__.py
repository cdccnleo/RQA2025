"""
基础设施层核心组件

提供基础设施层的核心服务接口、异常处理、常量定义等

模块结构:
- infrastructure_service_provider.py - 基础设施服务提供者
- health_check_interface.py - 健康检查接口
- exceptions.py - 异常定义
- constants.py - 常量定义
- parameter_objects.py - 参数对象（新增）
- mock_services.py - Mock服务基类（新增）
- component_registry.py - 组件注册

作者: RQA2025团队
创建时间: 2025-10-23
"""

# ==================== 核心服务 ====================

from .infrastructure_service_provider import (
    InfrastructureServiceProvider,
    InfrastructureServiceStatus,
    get_infrastructure_service_provider
)

# ==================== 健康检查 ====================

from .health_check_interface import (
    HealthCheckResult,
    HealthCheckInterface,
    InfrastructureHealthChecker,
    get_infrastructure_health_checker,
    register_infrastructure_service,
    check_infrastructure_health
)

# ==================== 异常处理 ====================

from .exceptions import (
    InfrastructureException,
    ConfigurationError,
    CacheError,
    LoggingError,
    MonitoringError,
    ResourceError,
    NetworkError,
    DatabaseError,
    FileSystemError,
    SecurityError,
    HealthCheckError,
    VersionError,
    handle_infrastructure_exception,
    validate_config_value,
    validate_resource_limits,
    validate_file_path,
    check_health_status
)

# ==================== 常量定义 ====================

from .constants import (
    # 常量类
    CacheConstants,
    ConfigConstants,
    MonitoringConstants,
    ResourceConstants,
    NetworkConstants,
    SecurityConstants,
    DatabaseConstants,
    FileSystemConstants,
    TimeConstants,
    CommonConstants,
    LoggingConstants,
    HealthConstants,
    ResourceLimits,
    PerformanceBenchmarks,
    ErrorConstants,
    NotificationConstants,
    
    # 快捷常量
    DEFAULT_TIMEOUT,
    DEFAULT_CACHE_SIZE,
    DEFAULT_POOL_SIZE,
    DEFAULT_QUEUE_SIZE,
    MAX_RETRY_ATTEMPTS,
    RETRY_BACKOFF_FACTOR,
    DEFAULT_LOG_RETENTION,
    HEALTH_CHECK_TIMEOUT,
    CONFIG_CACHE_TTL,
    CPU_WARNING_THRESHOLD,
    API_RESPONSE_ACCEPTABLE
)

# ==================== 参数对象 (新增) ====================

from .parameter_objects import (
    # 健康检查参数
    HealthCheckParams,
    ServiceHealthReportParams,
    HealthCheckResultParams,
    
    # 配置验证参数
    ConfigValidationParams,
    
    # 服务初始化参数
    ServiceInitializationParams,
    
    # 监控和告警参数
    MonitoringParams,
    AlertParams,
    
    # 资源管理参数
    ResourceAllocationParams,
    ResourceUsageParams,
    
    # 缓存操作参数
    CacheOperationParams,
    
    # 日志记录参数
    LogRecordParams
)

# ==================== Mock服务基类 (新增) ====================

from .mock_services import (
    BaseMockService,
    SimpleMockDict,
    SimpleMockLogger,
    SimpleMockMonitor
)

# ==================== 导出列表 ====================

__all__ = [
    # 核心服务
    'InfrastructureServiceProvider',
    'InfrastructureServiceStatus',
    'get_infrastructure_service_provider',
    
    # 健康检查
    'HealthCheckResult',
    'HealthCheckInterface',
    'InfrastructureHealthChecker',
    'get_infrastructure_health_checker',
    'register_infrastructure_service',
    'check_infrastructure_health',
    
    # 异常处理
    'InfrastructureException',
    'ConfigurationError',
    'CacheError',
    'LoggingError',
    'MonitoringError',
    'ResourceError',
    'NetworkError',
    'DatabaseError',
    'FileSystemError',
    'SecurityError',
    'HealthCheckError',
    'VersionError',
    'handle_infrastructure_exception',
    'validate_config_value',
    'validate_resource_limits',
    'validate_file_path',
    'check_health_status',
    
    # 常量类
    'CacheConstants',
    'ConfigConstants',
    'MonitoringConstants',
    'ResourceConstants',
    'NetworkConstants',
    'SecurityConstants',
    'DatabaseConstants',
    'FileSystemConstants',
    'TimeConstants',
    'CommonConstants',
    'LoggingConstants',
    'HealthConstants',
    'ResourceLimits',
    'PerformanceBenchmarks',
    'ErrorConstants',
    'NotificationConstants',
    
    # 快捷常量
    'DEFAULT_TIMEOUT',
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_POOL_SIZE',
    'DEFAULT_QUEUE_SIZE',
    'MAX_RETRY_ATTEMPTS',
    'RETRY_BACKOFF_FACTOR',
    'DEFAULT_LOG_RETENTION',
    'HEALTH_CHECK_TIMEOUT',
    'CONFIG_CACHE_TTL',
    'CPU_WARNING_THRESHOLD',
    'API_RESPONSE_ACCEPTABLE',
    
    # 参数对象 (新增)
    'HealthCheckParams',
    'ServiceHealthReportParams',
    'HealthCheckResultParams',
    'ConfigValidationParams',
    'ServiceInitializationParams',
    'MonitoringParams',
    'AlertParams',
    'ResourceAllocationParams',
    'ResourceUsageParams',
    'CacheOperationParams',
    'LogRecordParams',
    
    # Mock服务 (新增)
    'BaseMockService',
    'SimpleMockDict',
    'SimpleMockLogger',
    'SimpleMockMonitor',
]

