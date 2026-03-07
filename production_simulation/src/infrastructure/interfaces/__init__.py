
from .standard_interfaces import *
from .infrastructure_services import *

"""
RQA2025 Infrastructure Interfaces

基础设施层提供纯粹的技术服务接口，不依赖任何业务层。
所有上层业务模块都应该依赖这些基础设施服务接口。

接口分类：
1. standard_interfaces.py - 标准基础设施接口（兼容性保留）
2. infrastructure_services.py - 基础设施服务接口（推荐使用）
"""

__all__ = [
    # 标准基础设施接口（兼容性保留）
    'DataRequest',
    'DataResponse',
    'IServiceProvider',
    'ICacheProvider',
    'ILogger',
    'IConfigProvider',
    'IHealthCheck',
    'Event',
    'IEventBus',
    'IMonitor',

    # 基础设施服务接口（推荐使用）
    'IConfigManager',
    'ICacheService',
    'IMultiLevelCache',
    'ILogManager',
    'IMonitor',
    'ISecurityManager',
    'IHealthChecker',
    'IResourceManager',
    'IEventBus',
    'IServiceContainer',
    'IInfrastructureServiceProvider',

    # 数据结构
    'CacheEntry',
    'LogEntry',
    'MetricData',
    'UserCredentials',
    'SecurityToken',
    'HealthCheckResult',
    'ResourceQuota',
    'Event',
    'InfrastructureServiceStatus',
    'LogLevel'
]
