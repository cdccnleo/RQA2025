"""
服务执行器

执行实际的服务调用逻辑。
"""

import logging
import time
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from .integration_models import ServiceCall, ServiceEndpoint
from .service_registry import ServiceRegistry
from .connection_pool import ConnectionPoolManager
from .cache_manager import CacheManager
from .integration_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class ServiceExecutor:
    """服务执行器 - 职责：执行实际的服务调用"""

    def __init__(self, registry: ServiceRegistry,
                 pool_manager: ConnectionPoolManager,
                 cache_manager: Optional[CacheManager] = None,
                 monitor: Optional[PerformanceMonitor] = None):
        self.registry = registry
        self.pool_manager = pool_manager
        self.cache_manager = cache_manager
        self.monitor = monitor
        self._executor = ThreadPoolExecutor(max_workers=20)

    def call_service(self, call: ServiceCall) -> Dict[str, Any]:
        """调用服务"""
        start_time = time.time()

        try:
            # 检查缓存
            if self.cache_manager:
                cache_key = f"{call.service_name}:{call.method_name}:{str(call.parameters)}"
                cached_result = self.cache_manager.get(cache_key)
                if cached_result:
                    if self.monitor:
                        self.monitor.record_call(time.time() - start_time, True)
                    return cached_result

            # 获取服务端点
            endpoint = self.registry.get_service(call.service_name)
            if not endpoint:
                raise ValueError(f"服务 '{call.service_name}' 未注册")

            # 获取连接池
            pool = self.pool_manager.get_connection_pool(
                call.service_name,
                endpoint.connection_pool_size
            )

            # 执行服务调用（这里是模拟实现）
            result = self._execute_service_call(call, endpoint)

            # 缓存结果
            if self.cache_manager:
                self.cache_manager.set(cache_key, result)

            # 记录性能
            if self.monitor:
                self.monitor.record_call(time.time() - start_time, True)

            return result

        except Exception as e:
            if self.monitor:
                self.monitor.record_call(time.time() - start_time, False)
            logger.error(f"服务调用失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': time.time()
            }

    def _execute_service_call(self, call: ServiceCall,
                             endpoint: ServiceEndpoint) -> Dict[str, Any]:
        """执行实际的服务调用"""
        # 这里是模拟实现，实际应该调用真实的服务
        logger.info(f"调用服务: {call.service_name}.{call.method_name}")

        # 模拟服务调用延迟
        time.sleep(0.1)

        return {
            'success': True,
            'service': call.service_name,
            'method': call.method_name,
            'result': f"模拟结果 for {call.parameters}",
            'timestamp': time.time()
        }

