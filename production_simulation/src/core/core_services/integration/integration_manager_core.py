"""
服务集成管理器核心

重构后的服务集成管理器，使用组合模式。
"""

import logging
from typing import Dict, Any

from .integration_models import ServiceCall, ServiceEndpoint
from .service_registry import ServiceRegistry
from .connection_pool import ConnectionPoolManager
from .cache_manager import CacheManager
from .integration_monitor import PerformanceMonitor
from .service_executor import ServiceExecutor

logger = logging.getLogger(__name__)


class ServiceIntegrationManagerRefactored:
    """重构后的服务集成管理器 - 组合模式：使用专门的组件"""

    def __init__(self, max_workers: int = 20, enable_caching: bool = True):
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        self.logger = logging.getLogger(__name__)

        # 初始化专门的组件
        self.registry = ServiceRegistry()
        self.pool_manager = ConnectionPoolManager()
        self.cache_manager = CacheManager() if enable_caching else None
        self.monitor = PerformanceMonitor()
        self.executor = ServiceExecutor(
            self.registry,
            self.pool_manager,
            self.cache_manager,
            self.monitor
        )

    def register_service(self, service_name: str, endpoint: ServiceEndpoint) -> None:
        """注册服务 - 代理到注册表"""
        self.registry.register_service(service_name, endpoint)

    def unregister_service(self, service_name: str) -> bool:
        """注销服务 - 代理到注册表"""
        return self.registry.unregister_service(service_name)

    def call_service(self, call: ServiceCall) -> Dict[str, Any]:
        """调用服务 - 代理到执行器"""
        return self.executor.call_service(call)

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'call_stats': self.monitor.get_stats(),
            'service_count': self.registry.get_service_count(),
            'connection_pools': self.pool_manager.get_pool_stats(),
            'cache_size': self.cache_manager.get_stats()['cache_size'] if self.cache_manager else 0,
            'executor_active_threads': self.max_workers,
            'optimization_enabled': self.enable_caching
        }

    def optimize_for_high_load(self):
        """高负载优化配置"""
        self.max_workers = 50
        if self.cache_manager:
            self.cache_manager._max_cache_size = 50000
            self.cache_manager._cache_ttl = 180

        self.logger.info("已启用高负载优化配置")

    def shutdown(self):
        """关闭管理器"""
        self.pool_manager.close_all_pools()
        if self.cache_manager:
            self.cache_manager.clear()
        self.logger.info("服务集成管理器已关闭")

