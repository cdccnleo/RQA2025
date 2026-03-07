"""
Core Service Container模块别名

提供向后兼容的导入路径
从container.service_container导入
"""

from .container.service_container import (
    DependencyContainer,
    Lifecycle,
    ServiceHealth
)

# 别名
ServiceContainer = DependencyContainer
ServiceConfig = dict  # 配置使用字典
ServiceInstance = object  # 服务实例
ServiceStatus = ServiceHealth  # 状态别名

try:
    # 尝试导入更多组件
    from .container.service_container import LoadBalancer, LoadBalancingStrategy
except ImportError:
    from enum import Enum
    
    class LoadBalancingStrategy(Enum):
        """负载均衡策略"""
        ROUND_ROBIN = "round_robin"
        RANDOM = "random"
        LEAST_CONNECTIONS = "least_connections"
    
    class LoadBalancer:
        """负载均衡器"""
        def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN):
            self.strategy = strategy

__all__ = [
    'ServiceContainer',
    'DependencyContainer',
    'ServiceConfig',
    'ServiceInstance',
    'ServiceStatus',
    'ServiceHealth',
    'Lifecycle',
    'LoadBalancer',
    'LoadBalancingStrategy'
]

