#!/usr/bin/env python3
"""
RQA2025 健康层基础设施适配器

基于适配器模式设计，为健康管理系统提供统一的基础设施服务访问接口，
消除代码重复，实现集中化管理和异步处理能力。
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from ...business_adapters import IBusinessAdapter, BusinessLayerType
from src.infrastructure.logging.core.unified_logger import get_unified_logger
# 可选导入基础设施服务
try:
    from src.infrastructure.cache.core.unified_cache import UnifiedCacheManager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from src.infrastructure.health.core.enhanced_health_checker import EnhancedHealthChecker
    HEALTH_CHECKER_AVAILABLE = True
except ImportError:
    HEALTH_CHECKER_AVAILABLE = False

try:
    from src.infrastructure.monitoring.core.unified_monitoring import UnifiedMonitoring
    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

try:
    from src.infrastructure.config.core.unified_config import UnifiedConfigManager
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


logger = get_unified_logger(__name__)


class HealthCheckType(Enum):
    """健康检查类型枚举"""
    BASIC = "basic"           # 基础健康检查
    ADVANCED = "advanced"     # 高级健康检查
    PERFORMANCE = "performance"  # 性能健康检查
    SECURITY = "security"     # 安全健康检查
    COMPLIANCE = "compliance"  # 合规健康检查


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    service_name: str
    status: HealthStatus
    response_time: float
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class AsyncHealthCheckConfig:
    """异步健康检查配置"""
    timeout: float = 30.0
    retry_count: int = 3
    retry_delay: float = 1.0
    concurrent_limit: int = 10
    enable_caching: bool = True
    cache_ttl: int = 300


class IHealthAdapter(ABC):
    """健康适配器接口"""

    @abstractmethod
    async def perform_health_check(self, service_name: str,
                                   check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
        """执行健康检查"""

    @abstractmethod
    async def perform_batch_health_check(self, service_names: List[str],
                                         check_type: HealthCheckType = HealthCheckType.BASIC) -> Dict[str, HealthCheckResult]:
        """批量执行健康检查"""

    @abstractmethod
    async def get_health_status_stream(self, service_names: List[str],
                                       interval: float = 60.0) -> AsyncGenerator[Dict[str, HealthCheckResult], None]:
        """获取健康状态流"""

    @abstractmethod
    async def register_health_check(self, service_name: str,
                                    check_function: Callable,
                                    config: Optional[Dict[str, Any]] = None) -> bool:
        """注册健康检查函数"""

    @abstractmethod
    async def unregister_health_check(self, service_name: str) -> bool:
        """注销健康检查函数"""


class HealthLayerAdapter(IBusinessAdapter, IHealthAdapter):
    """
    健康层适配器实现

    提供统一的基础设施服务访问接口，支持异步健康检查，
    实现集中化管理和性能优化。
    """

    def __init__(self):
        self._layer_type = BusinessLayerType.HEALTH
        self._config = AsyncHealthCheckConfig()
        self._health_checks: Dict[str, Callable] = {}
        self._check_configs: Dict[str, Dict[str, Any]] = {}
        self._cache_manager: Optional[UnifiedCacheManager] = None
        self._monitoring: Optional[UnifiedMonitoring] = None
        self._config_manager: Optional[UnifiedConfigManager] = None
        self._health_checker: Optional[EnhancedHealthChecker] = None
        self._executor = ThreadPoolExecutor(max_workers=self._config.concurrent_limit)
        self._logger = get_unified_logger(f"{__name__}.HealthLayerAdapter")

        # 初始化基础设施服务
        self._initialize_infrastructure()

    @property
    def layer_type(self) -> BusinessLayerType:
        """获取业务层类型"""
        return self._layer_type

    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        return {
            'cache_manager': self._cache_manager,
            'monitoring': self._monitoring,
            'config_manager': self._config_manager,
            'health_checker': self._health_checker,
            'executor': self._executor,
            'logger': self._logger
        }

    def get_health_adapter(self) -> 'HealthLayerAdapter':
        """获取健康适配器实例"""
        return self

    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""
        # 健康层适配器直接返回自身作为服务桥接器
        if service_name == "health_adapter":
            return self
        return None

    def health_check(self) -> Dict[str, Any]:
        """同步健康检查"""
        try:
            # 创建新的事件循环来运行异步方法
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.health_status_async())
            loop.close()
            return result
        except Exception as e:
            self._logger.error(f"同步健康检查失败: {e}")
            return {
                "status": "critical",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _initialize_infrastructure(self):
        """初始化基础设施服务"""
        try:
            # 可选初始化基础设施服务
            if CACHE_AVAILABLE:
                self._cache_manager = UnifiedCacheManager()
                self._logger.info("缓存管理器初始化成功")
            else:
                self._cache_manager = None
                self._logger.warning("缓存管理器不可用，使用内存缓存")

            if MONITORING_AVAILABLE:
                self._monitoring = UnifiedMonitoring()
                self._logger.info("监控服务初始化成功")
            else:
                self._monitoring = None
                self._logger.warning("监控服务不可用")

            if CONFIG_AVAILABLE:
                self._config_manager = UnifiedConfigManager()
                # 加载配置
                health_config = self._config_manager.get_config('health_layer_adapter', {})
                if health_config:
                    self._config = AsyncHealthCheckConfig(**health_config)
                self._logger.info("配置管理器初始化成功")
            else:
                self._config_manager = None
                self._logger.warning("配置管理器不可用，使用默认配置")

            if HEALTH_CHECKER_AVAILABLE:
                self._health_checker = EnhancedHealthChecker()
                self._logger.info("健康检查器初始化成功")
            else:
                self._health_checker = None
                self._logger.warning("增强健康检查器不可用")

            self._logger.info("健康层适配器基础设施服务初始化完成")

        except Exception as e:
            self._logger.error(f"基础设施服务初始化失败: {e}")
            # 不抛出异常，允许部分服务不可用时继续运行
            self._cache_manager = None
            self._monitoring = None
            self._config_manager = None
            self._health_checker = None

    async def perform_health_check(self, service_name: str,
                                   check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
        """执行单个健康检查"""
        start_time = time.time()

        try:
            # 检查缓存
            if self._config.enable_caching:
                cache_key = f"health_check:{service_name}:{check_type.value}"
                cached_result = await self._get_cached_result(cache_key)
                if cached_result:
                    return cached_result

            # 获取健康检查函数
            check_function = self._health_checks.get(service_name)
            if not check_function:
                return HealthCheckResult(
                    service_name=service_name,
                    status=HealthStatus.UNKNOWN,
                    response_time=time.time() - start_time,
                    timestamp=datetime.now(),
                    error_message=f"未注册的健康检查服务: {service_name}"
                )

            # 执行健康检查
            config = self._check_configs.get(service_name, {})
            result = await self._execute_health_check(service_name, check_function, check_type, config)

            # 缓存结果
            if self._config.enable_caching:
                await self._cache_result(cache_key, result)

            # 记录监控指标
            await self._record_health_metrics(service_name, result)

            return result

        except Exception as e:
            self._logger.error(f"健康检查执行失败 {service_name}: {e}")
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    async def perform_batch_health_check(self, service_names: List[str],
                                         check_type: HealthCheckType = HealthCheckType.BASIC) -> Dict[str, HealthCheckResult]:
        """批量执行健康检查"""
        if not service_names:
            return {}

        # 创建并发任务
        tasks = [
            self.perform_health_check(service_name, check_type)
            for service_name in service_names
        ]

        # 限制并发数量
        semaphore = asyncio.Semaphore(self._config.concurrent_limit)

        async def limited_check(service_name: str):
            async with semaphore:
                return await self.perform_health_check(service_name, check_type)

        # 执行并发检查
        results = await asyncio.gather(*[limited_check(name) for name in service_names])

        return {result.service_name: result for result in results}

    async def get_health_status_stream(self, service_names: List[str],
                                       interval: float = 60.0) -> AsyncGenerator[Dict[str, HealthCheckResult], None]:
        """获取健康状态流"""
        while True:
            try:
                results = await self.perform_batch_health_check(service_names)
                yield results
                await asyncio.sleep(interval)
            except Exception as e:
                self._logger.error(f"健康状态流生成失败: {e}")
                await asyncio.sleep(interval)

    async def register_health_check(self, service_name: str,
                                    check_function: Callable,
                                    config: Optional[Dict[str, Any]] = None) -> bool:
        """注册健康检查函数"""
        try:
            self._health_checks[service_name] = check_function
            if config:
                self._check_configs[service_name] = config

            self._logger.info(f"健康检查函数注册成功: {service_name}")
            return True

        except Exception as e:
            self._logger.error(f"健康检查函数注册失败 {service_name}: {e}")
            return False

    async def unregister_health_check(self, service_name: str) -> bool:
        """注销健康检查函数"""
        try:
            if service_name in self._health_checks:
                del self._health_checks[service_name]
                if service_name in self._check_configs:
                    del self._check_configs[service_name]

                self._logger.info(f"健康检查函数注销成功: {service_name}")
                return True
            else:
                self._logger.warning(f"尝试注销未注册的健康检查函数: {service_name}")
                return False

        except Exception as e:
            self._logger.error(f"健康检查函数注销失败 {service_name}: {e}")
            return False

    async def _execute_health_check(self, service_name: str, check_function: Callable,
                                    check_type: HealthCheckType, config: Dict[str, Any]) -> HealthCheckResult:
        """执行健康检查"""
        start_time = time.time()

        try:
            # 根据检查类型设置超时时间
            timeout = config.get('timeout', self._config.timeout)

            # 执行健康检查（支持同步和异步函数）
            if asyncio.iscoroutinefunction(check_function):
                result = await asyncio.wait_for(
                    check_function(check_type, config),
                    timeout=timeout
                )
            else:
                # 对于同步函数，使用线程池执行
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(self._executor, check_function, check_type, config),
                    timeout=timeout
                )

            # 解析结果
            if isinstance(result, dict):
                status = HealthStatus(result.get('status', 'unknown'))
                details = result.get('details', {})
                error_message = result.get('error_message')
            elif isinstance(result, HealthStatus):
                status = result
                details = {}
                error_message = None
            else:
                status = HealthStatus.HEALTHY if result else HealthStatus.CRITICAL
                details = {}
                error_message = None

            return HealthCheckResult(
                service_name=service_name,
                status=status,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                details=details,
                error_message=error_message
            )

        except asyncio.TimeoutError:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                error_message=f"健康检查超时 ({timeout}s)"
            )

        except Exception as e:
            return HealthCheckResult(
                service_name=service_name,
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                timestamp=datetime.now(),
                error_message=str(e)
            )

    async def _get_cached_result(self, cache_key: str) -> Optional[HealthCheckResult]:
        """获取缓存的健康检查结果"""
        if not self._cache_manager:
            return None

        try:
            cached_data = await self._cache_manager.get_async(cache_key)
            if cached_data:
                # 检查缓存是否过期
                cached_time = cached_data.get('timestamp')
                if cached_time and (datetime.now() - datetime.fromisoformat(cached_time)).seconds < self._config.cache_ttl:
                    return HealthCheckResult(**cached_data)
        except Exception as e:
            self._logger.warning(f"获取缓存结果失败 {cache_key}: {e}")

        return None

    async def _cache_result(self, cache_key: str, result: HealthCheckResult):
        """缓存健康检查结果"""
        if not self._cache_manager:
            return

        try:
            cache_data = {
                'service_name': result.service_name,
                'status': result.status.value,
                'response_time': result.response_time,
                'timestamp': result.timestamp.isoformat(),
                'details': result.details,
                'error_message': result.error_message
            }
            await self._cache_manager.set_async(cache_key, cache_data, ttl=self._config.cache_ttl)
        except Exception as e:
            self._logger.warning(f"缓存结果失败 {cache_key}: {e}")

    async def _record_health_metrics(self, service_name: str, result: HealthCheckResult):
        """记录健康检查指标"""
        if not self._monitoring:
            return

        try:
            metrics = {
                'service_name': service_name,
                'status': result.status.value,
                'response_time': result.response_time,
                'timestamp': result.timestamp.isoformat(),
                'has_error': result.error_message is not None
            }

            await self._monitoring.record_metrics_async('health_check', metrics)
        except Exception as e:
            self._logger.warning(f"记录健康指标失败 {service_name}: {e}")

    async def get_health_status_summary(self, service_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """获取健康状态汇总"""
        if service_names is None:
            service_names = list(self._health_checks.keys())

        results = await self.perform_batch_health_check(service_names)

        summary = {
            'total_services': len(service_names),
            'healthy_count': 0,
            'warning_count': 0,
            'critical_count': 0,
            'unknown_count': 0,
            'average_response_time': 0.0,
            'timestamp': datetime.now().isoformat(),
            'details': {}
        }

        total_response_time = 0.0

        for service_name, result in results.items():
            summary['details'][service_name] = {
                'status': result.status.value,
                'response_time': result.response_time,
                'error_message': result.error_message
            }

            total_response_time += result.response_time

            if result.status == HealthStatus.HEALTHY:
                summary['healthy_count'] += 1
            elif result.status == HealthStatus.WARNING:
                summary['warning_count'] += 1
            elif result.status == HealthStatus.CRITICAL:
                summary['critical_count'] += 1
            else:
                summary['unknown_count'] += 1

        if results:
            summary['average_response_time'] = total_response_time / len(results)

        return summary

    def cleanup(self):
        """清理资源"""
        if self._executor:
            self._executor.shutdown(wait=True)
        self._logger.info("健康层适配器资源清理完成")


# 全局健康层适配器实例
_health_adapter_instance: Optional[HealthLayerAdapter] = None
_health_adapter_lock = asyncio.Lock()


async def get_health_layer_adapter() -> HealthLayerAdapter:
    """获取健康层适配器实例（单例模式）"""
    global _health_adapter_instance

    if _health_adapter_instance is None:
        async with _health_adapter_lock:
            if _health_adapter_instance is None:
                _health_adapter_instance = HealthLayerAdapter()

    return _health_adapter_instance


# 便捷函数
async def perform_health_check_async(service_name: str,
                                     check_type: HealthCheckType = HealthCheckType.BASIC) -> HealthCheckResult:
    """便捷的异步健康检查函数"""
    adapter = await get_health_layer_adapter()
    return await adapter.perform_health_check(service_name, check_type)


async def perform_batch_health_check_async(service_names: List[str],
                                           check_type: HealthCheckType = HealthCheckType.BASIC) -> Dict[str, HealthCheckResult]:
    """便捷的批量异步健康检查函数"""
    adapter = await get_health_layer_adapter()
    return await adapter.perform_batch_health_check(service_names, check_type)


async def register_health_check_async(service_name: str,
                                      check_function: Callable,
                                      config: Optional[Dict[str, Any]] = None) -> bool:
    """便捷的健康检查注册函数"""
    adapter = await get_health_layer_adapter()
    return await adapter.register_health_check(service_name, check_function, config)
