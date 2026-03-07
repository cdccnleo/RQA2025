"""
核心服务集成管理器 - 入口点

这是service_integration_manager的简化入口点，重新导出所有组件。

拆分后的模块结构：
- integration_models.py: 数据模型
- connection_pool.py: 连接池管理
- service_registry.py: 服务注册表
- cache_manager.py: 缓存管理
- integration_monitor.py: 性能监控
- service_executor.py: 服务执行器
- integration_manager_core.py: 核心管理器
"""

import logging
from typing import Dict, Any

# 导入所有组件
from .integration_models import ServiceCall, ServiceEndpoint
from .connection_pool import ConnectionPool, ConnectionPoolManager
from .service_registry import ServiceRegistry
from .cache_manager import CacheManager
from .integration_monitor import PerformanceMonitor
from .service_executor import ServiceExecutor
from .integration_manager_core import ServiceIntegrationManagerRefactored

logger = logging.getLogger(__name__)

# 为向后兼容，提供原有的类名
ServiceIntegrationManager = ServiceIntegrationManagerRefactored

# 单例模式
_service_integration_manager_instance = None


def get_service_integration_manager() -> ServiceIntegrationManager:
    """获取服务集成管理器单例"""
    global _service_integration_manager_instance
    if _service_integration_manager_instance is None:
        _service_integration_manager_instance = ServiceIntegrationManager()
    return _service_integration_manager_instance


def init_service_integration(enable_high_performance: bool = True):
    """初始化服务集成"""
    global _service_integration_manager_instance
    _service_integration_manager_instance = ServiceIntegrationManager(
        max_workers=50 if enable_high_performance else 20,
        enable_caching=True
    )

    if enable_high_performance:
        _service_integration_manager_instance.optimize_for_high_load()

    logger.info("服务集成管理器已初始化")


# 重新导出所有组件
__all__ = [
    # 数据模型
    'ServiceCall',
    'ServiceEndpoint',
    
    # 组件
    'ConnectionPool',
    'ConnectionPoolManager',
    'ServiceRegistry',
    'CacheManager',
    'PerformanceMonitor',
    'ServiceExecutor',
    
    # 管理器
    'ServiceIntegrationManager',
    'ServiceIntegrationManagerRefactored',
    
    # 工具函数
    'get_service_integration_manager',
    'init_service_integration',
]


if __name__ == "__main__":
    # 使用示例
    init_service_integration(enable_high_performance=True)

    manager = get_service_integration_manager()

    # 注册服务
    endpoint = ServiceEndpoint(
        service_name="data_service",
        endpoint_url="http://localhost:8001/api"
    )
    manager.register_service("data_service", endpoint)

    # 调用服务
    call = ServiceCall(
        service_name="data_service",
        method_name="get_market_data",
        parameters={"symbol": "AAPL"}
    )
    result = manager.call_service(call)
    print(f"调用结果: {result}")

    # 获取性能统计
    stats = manager.get_performance_stats()
    print(f"性能统计: {stats}")
