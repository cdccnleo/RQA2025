#!/usr/bin/env python3
"""
RQA2025 统一适配器架构

整合所有业务层适配器的通用功能，提供统一的适配器基类、
工厂模式、服务生命周期管理和性能监控能力。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable, Union
import logging
import time
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .interfaces import (
    IBusinessAdapter, IAdapterComponent, IServiceBridge, IFallbackService,
    BusinessLayerType, ICoreComponent, ComponentLifecycle
)
# 避免循环导入，后面再导入BaseBusinessAdapter
# from .business_adapters import BaseBusinessAdapter
from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class ServiceConfig:

    """服务配置"""
    name: str
    primary_factory: Callable[[], Any]
    fallback_factory: Optional[Callable[[], Any]] = None
    health_check_interval: int = 30
    enable_monitoring: bool = True
    enable_caching: bool = True


@dataclass
class AdapterMetrics:

    """适配器指标"""
    service_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    fallback_count: int = 0
    recovery_count: int = 0
    error_count: int = 0
    last_health_check: Optional[datetime] = None
    average_response_time: float = 0.0


@dataclass
class ServiceStatus:

    """服务状态"""
    name: str
    status: str  # 'primary', 'fallback', 'unavailable'
    last_check: datetime
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    health_score: float = 1.0


# =============================================================================
# 统一适配器基类
# =============================================================================


class UnifiedBusinessAdapter(IBusinessAdapter, ICoreComponent):

    """统一业务层适配器基类

    提供所有业务层适配器的通用功能：
    - 服务生命周期管理
    - 性能监控和指标收集
    - 自动降级和恢复
    - 健康检查和状态管理
    - 缓存和配置管理
    """

    def __init__(self, layer_type: BusinessLayerType):
        """初始化统一业务层适配器

        Args:
            layer_type: 业务层类型
        """
        # 设置基础属性
        self._layer_type = layer_type

        # 初始化各组件
        self._init_lifecycle_management()
        self._init_resource_management()
        self._init_service_management()
        self._init_health_check_executor(layer_type)
        self._init_service_configs()
        self._start_cleanup_task()
        self._add_default_lifecycle_listeners()

    def _init_lifecycle_management(self):
        """初始化生命周期管理"""
        self.lifecycle_status = ComponentLifecycle.CREATED
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        self.uptime_seconds = 0.0

        # 生命周期事件监听器
        self.lifecycle_listeners: List[Callable] = []
        self.shutdown_timeout = 30  # 关闭超时时间(秒)

    def _init_resource_management(self):
        """初始化资源管理"""
        self.managed_resources: List[Any] = []
        self.resource_cleanup_handlers: List[Callable] = []

    def _init_service_management(self):
        """初始化服务管理"""
        self._services: Dict[str, Any] = {}
        self._last_recovery_attempt: Dict[str, datetime] = {}
        self.service_configs: Dict[str, ServiceConfig] = {}
        self.service_status: Dict[str, ServiceStatus] = {}
        self.metrics = AdapterMetrics()

    def _init_health_check_executor(self, layer_type: BusinessLayerType):
        """初始化健康检查执行器"""
        self.health_check_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"{layer_type.value}_health"
        )

    def _init_service_configs(self):
        """初始化服务配置 - 默认实现，子类可重写"""
        # 默认的基础服务配置
        self.service_configs.update({
            'cache_manager': ServiceConfig(
                name='cache_manager',
                primary_factory=self._create_cache_manager,
                fallback_factory=self._create_fallback_cache
            ),
            'config_manager': ServiceConfig(
                name='config_manager',
                primary_factory=self._create_config_manager,
                fallback_factory=self._create_fallback_config
            ),
            'logger': ServiceConfig(
                name='logger',
                primary_factory=self._create_logger,
                fallback_factory=self._create_fallback_logger
            ),
            'monitoring': ServiceConfig(
                name='monitoring',
                primary_factory=self._create_monitoring,
                fallback_factory=self._create_fallback_monitoring
            ),
            'health_checker': ServiceConfig(
                name='health_checker',
                primary_factory=self._create_health_checker,
                fallback_factory=self._create_fallback_health_checker
            )
        })

    # =========================================================================
    # 服务工厂方法 - 由子类实现
    # =========================================================================

    def _create_cache_manager(self):
        """创建缓存管理器"""
        from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
        return UnifiedCacheManager()

    def _create_config_manager(self):
        """创建配置管理器"""
        from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
        return UnifiedConfigManager()

    def _create_logger(self):
        """创建日志器"""
        return get_unified_logger(f"{self.layer_type.value}_layer")

    def _create_monitoring(self):
        """创建监控系统"""
        from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring
        return UnifiedMonitoring()

    def _create_health_checker(self):
        """创建健康检查器 - 使用延迟导入避免循环依赖"""
        try:
            # 延迟导入，避免循环依赖
            from infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
            return EnhancedHealthChecker()
        except ImportError as e:
            # 如果导入失败，创建一个基本的健康检查器
            self.logger.warning(f"无法导入增强健康检查器，使用基本实现: {e}")
            return self._create_basic_health_checker()

    def _create_basic_health_checker(self):
        """创建基本的健康检查器实现"""
        class BasicHealthChecker:
            def health_check(self):
                return {
                    'service': 'basic_health_checker',
                    'healthy': True,
                    'status': 'healthy',
                    'message': '基础健康检查器运行正常'
                }

            @property
            def service_name(self):
                return 'basic_health_checker'

            @property
            def service_version(self):
                return '1.0.0'

        return BasicHealthChecker()

    # =========================================================================
    # 降级服务工厂方法
    # =========================================================================

    def _create_fallback_cache(self):
        """创建降级缓存管理器"""
        try:
            from .fallback_services import get_fallback_cache_manager
            return get_fallback_cache_manager()
        except ImportError:
            return self._create_basic_fallback_cache()

    def _create_fallback_config(self):
        """创建降级配置管理器"""
        try:
            from .fallback_services import get_fallback_config_manager
            return get_fallback_config_manager()
        except ImportError:
            return self._create_basic_fallback_config()

    def _create_fallback_logger(self):
        """创建降级日志器"""
        try:
            from .fallback_services import get_fallback_logger
            return get_fallback_logger()
        except ImportError:
            return self._create_basic_fallback_logger()

    def _create_fallback_monitoring(self):
        """创建降级监控系统"""
        try:
            from .fallback_services import get_fallback_monitoring
            return get_fallback_monitoring()
        except ImportError:
            return self._create_basic_fallback_monitoring()

    def _create_fallback_health_checker(self):
        """创建降级健康检查器"""
        try:
            from .fallback_services import get_fallback_health_checker
            return get_fallback_health_checker()
        except ImportError:
            return self._create_basic_fallback_health_checker()

    # =========================================================================
    # 基础降级服务实现
    # =========================================================================

    def _create_basic_fallback_cache(self):
        """创建基础降级缓存"""

        class BasicFallbackCache:

            def get(self, key, default=None): return default

            def set(self, key, value, ttl=3600): return True

            def delete(self, key): return True

        return BasicFallbackCache()

    def _create_basic_fallback_config(self):
        """创建基础降级配置"""

        class BasicFallbackConfig:

            def get(self, key, default=None): return default

            def set(self, key, value): return True

        return BasicFallbackConfig()

    def _create_basic_fallback_logger(self):
        """创建基础降级日志器"""
        import logging

        class BasicFallbackLogger:

            def info(self, msg): logging.info(f"[降级] {msg}")

            def warning(self, msg): logging.warning(f"[降级] {msg}")

            def error(self, msg): logging.error(f"[降级] {msg}")

            def debug(self, msg): logging.debug(f"[降级] {msg}")

        return BasicFallbackLogger()

    def _create_basic_fallback_monitoring(self):
        """创建基础降级监控"""

        class BasicFallbackMonitoring:

            def record_metric(self, name, value, tags=None): pass

            def record_event(self, name, data=None): pass

        return BasicFallbackMonitoring()

    def _create_basic_fallback_health_checker(self):
        """创建基础降级健康检查器"""

        class BasicFallbackHealthChecker:

            def check_health(self): return {'status': 'degraded', 'message': '降级模式'}

        return BasicFallbackHealthChecker()

    # =========================================================================
    # 生命周期管理方法
    # =========================================================================

    def add_lifecycle_listener(self, status: ComponentLifecycle, listener: Callable):
        """添加生命周期监听器"""
        if not hasattr(listener, '__call__'):
            raise ValueError("监听器必须是可调用的")

        self.lifecycle_listeners.append((status, listener))
        logger.debug(f"添加生命周期监听器: {status.value}")

    def remove_lifecycle_listener(self, status: ComponentLifecycle, listener: Callable):
        """移除生命周期监听器"""
        self.lifecycle_listeners = [
            (s, l) for s, l in self.lifecycle_listeners
            if not (s == status and l == listener)
        ]
        logger.debug(f"移除生命周期监听器: {status.value}")

    def _notify_lifecycle_listeners(self, status: ComponentLifecycle):
        """通知生命周期监听器"""
        for listener_status, listener in self.lifecycle_listeners:
            if listener_status == status:
                try:
                    listener()
                except Exception as e:
                    logger.error(f"生命周期监听器执行失败: {status.value}, 错误: {e}")

    def change_lifecycle_status(self, new_status: ComponentLifecycle):
        """改变生命周期状态"""
        old_status = self.lifecycle_status
        self.lifecycle_status = new_status

        # 通知监听器
        self._notify_lifecycle_listeners(new_status)

        logger.info(f"生命周期状态变更: {old_status.value} -> {new_status.value}")

    def start(self) -> bool:
        """启动适配器"""
        if self.lifecycle_status in [ComponentLifecycle.RUNNING, ComponentLifecycle.STARTING]:
            logger.warning(f"{self.layer_type.value}层适配器已在运行中")
            return True

        try:
            self.change_lifecycle_status(ComponentLifecycle.STARTING)

            # 执行启动逻辑
            if self.initialize():
                self.change_lifecycle_status(ComponentLifecycle.RUNNING)
                return True
            else:
                self.change_lifecycle_status(ComponentLifecycle.ERROR)
                return False

        except Exception as e:
            logger.error(f"{self.layer_type.value}层适配器启动失败: {e}")
            self.change_lifecycle_status(ComponentLifecycle.ERROR)
            return False

    def stop(self, timeout: Optional[int] = None) -> bool:
        """停止适配器"""
        if self.lifecycle_status in [ComponentLifecycle.STOPPED, ComponentLifecycle.STOPPING]:
            logger.warning(f"{self.layer_type.value}层适配器已停止")
            return True

        timeout = timeout or self.shutdown_timeout

        try:
            self.change_lifecycle_status(ComponentLifecycle.STOPPING)

            # 执行清理逻辑
            self._cleanup_resources(timeout)

            self.change_lifecycle_status(ComponentLifecycle.STOPPED)
            return True

        except Exception as e:
            logger.error(f"{self.layer_type.value}层适配器停止失败: {e}")
            self.change_lifecycle_status(ComponentLifecycle.ERROR)
            return False

    def restart(self) -> bool:
        """重启适配器"""
        logger.info(f"重启{self.layer_type.value}层适配器")

        if not self.stop():
            logger.error(f"停止适配器失败，无法重启")
            return False

        # 等待一段时间确保完全停止
        import time
        time.sleep(1)

        return self.start()

    def _cleanup_resources(self, timeout: int):
        """清理资源"""
        logger.info(f"开始清理{self.layer_type.value}层适配器资源")

        # 停止健康检查线程
        if hasattr(self, 'health_check_executor'):
            self.health_check_executor.shutdown(wait=True, timeout=timeout)

        # 执行自定义清理处理器
        for cleanup_handler in self.resource_cleanup_handlers:
            try:
                cleanup_handler()
            except Exception as e:
                logger.error(f"资源清理处理器执行失败: {e}")

        # 清理托管资源
        for resource in self.managed_resources:
            try:
                if hasattr(resource, 'close'):
                    resource.close()
                elif hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'stop'):
                    resource.stop()
            except Exception as e:
                logger.error(f"资源清理失败: {e}")

        self.managed_resources.clear()
        logger.info(f"{self.layer_type.value}层适配器资源清理完成")

    def add_managed_resource(self, resource: Any):
        """添加托管资源"""
        self.managed_resources.append(resource)
        logger.debug(f"添加托管资源: {type(resource).__name__}")

    def add_resource_cleanup_handler(self, handler: Callable):
        """添加资源清理处理器"""
        if not hasattr(handler, '__call__'):
            raise ValueError("清理处理器必须是可调用的")

        self.resource_cleanup_handlers.append(handler)
        logger.debug("添加资源清理处理器")

    def get_lifecycle_info(self) -> Dict[str, Any]:
        """获取生命周期信息"""
        return {
            'layer_type': self.layer_type.value,
            'current_status': self.lifecycle_status.value,
            'created_at': self.created_at.isoformat(),
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stop_time': self.stop_time.isoformat() if self.stop_time else None,
            'uptime_seconds': self.uptime_seconds,
            'last_activity': self.last_activity.isoformat(),
            'managed_resources_count': len(self.managed_resources),
            'lifecycle_listeners_count': len(self.lifecycle_listeners)
        }

    def _register_adapter_classes(self):
        """注册适配器类 - 默认空实现，子类可重写"""

    def _start_cleanup_task(self):
        """启动清理任务 - 默认空实现，子类可重写"""

    def _add_default_lifecycle_listeners(self):
        """添加默认的生命周期监听器 - 默认空实现，子类可重写"""

    # =========================================================================
    # 核心接口实现
    # =========================================================================

    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        self._update_activity()
        self.metrics.service_calls += 1

        services = {}
        for service_name in self.service_configs.keys():
            services[service_name] = self._get_service(service_name)

        return services

    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""
        self._update_activity()
        return self._get_service(service_name)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        self._update_activity()
        self.metrics.last_health_check = datetime.now()

        health_status = {
            'layer_type': self.layer_type.value,
            'lifecycle_status': self.lifecycle_status.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'services': {},
            'overall_status': 'healthy',
            'health_score': 0.0,
            'metrics': self._get_metrics_dict()
        }

        total_score = 0.0
        service_count = 0

        for service_name, status in self.service_status.items():
            health_status['services'][service_name] = {
                'status': status.status,
                'health_score': status.health_score,
                'last_check': status.last_check.isoformat(),
                'consecutive_failures': status.consecutive_failures
            }
            total_score += status.health_score
            service_count += 1

            if status.status == 'unavailable' or status.health_score < 0.5:
                health_status['overall_status'] = 'degraded'

        if service_count > 0:
            health_status['health_score'] = total_score / service_count

        return health_status

    @property
    def layer_type(self) -> BusinessLayerType:
        """获取业务层类型"""
        return self._layer_type

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            'component_type': 'UnifiedBusinessAdapter',
            'layer_type': self.layer_type.value,
            'lifecycle_status': self.lifecycle_status.value,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'uptime_seconds': self.uptime_seconds,
            'managed_resources_count': len(self.managed_resources),
            'lifecycle_listeners_count': len(self.lifecycle_listeners)
        }

    def validate_config(self) -> bool:
        """验证配置"""
        # 基础配置验证
        try:
            # 检查服务配置是否有效
            if not hasattr(self, 'service_configs'):
                return False

            for service_name, config in self.service_configs.items():
                if not hasattr(config, 'primary_factory'):
                    return False

            return True
        except Exception as e:
            logger.warning(f"配置验证失败: {e}")
            return False

    def initialize(self) -> bool:
        """初始化适配器"""
        try:
            self.lifecycle_status = ComponentLifecycle.INITIALIZING
            logger.info(f"初始化{self.layer_type.value}层适配器")

            success_count = 0
            for service_name, config in self.service_configs.items():
                if self._init_service_with_degradation(service_name, config):
                    success_count += 1

            # 初始化层特定服务
            self._init_layer_specific_services()

            self.lifecycle_status = ComponentLifecycle.INITIALIZED
            logger.info(
                f"{self.layer_type.value}层适配器初始化完成，{success_count}/{len(self.service_configs)}个服务成功")
            return True

        except Exception as e:
            self.lifecycle_status = ComponentLifecycle.ERROR
            logger.error(f"{self.layer_type.value}层适配器初始化失败: {e}")
            return False

    # =========================================================================
    # 辅助方法
    # =========================================================================

    def _init_service_with_degradation(self, service_name: str, config: ServiceConfig) -> bool:
        """带降级的服务初始化"""
        try:
            # 尝试初始化主服务
            service = config.primary_factory()
            self._services[service_name] = service
            self.service_status[service_name] = ServiceStatus(
                name=service_name,
                status='primary',
                last_check=datetime.now(),
                health_score=1.0
            )
            logger.debug(f"{service_name} 主服务初始化成功")
            return True

        except Exception as e:
            logger.warning(f"{service_name} 主服务初始化失败: {e}")

            # 尝试降级服务
            if config.fallback_factory:
                try:
                    fallback_service = config.fallback_factory()
                    self._services[service_name] = fallback_service
                    self.service_status[service_name] = ServiceStatus(
                        name=service_name,
                        status='fallback',
                        last_check=datetime.now(),
                        health_score=0.7,
                        last_error=str(e)
                    )
                    self.metrics.fallback_count += 1
                    logger.info(f"{service_name} 降级服务初始化成功")
                    return True

                except Exception as fallback_e:
                    logger.error(f"{service_name} 降级服务初始化失败: {fallback_e}")

            # 完全不可用
            self.service_status[service_name] = ServiceStatus(
                name=service_name,
                status='unavailable',
                last_check=datetime.now(),
                health_score=0.0,
                last_error=str(e)
            )
            return False

    def _get_service(self, service_name: str) -> Optional[Any]:
        """获取服务实例，支持自动恢复"""
        service = self._services.get(service_name)

        # 检查服务是否需要恢复
        if service_name in self.service_status:
            status = self.service_status[service_name]
            if status.status in ['fallback', 'unavailable']:
                if self._should_attempt_recovery(service_name, status):
                    if self._attempt_service_recovery(service_name):
                        service = self._services.get(service_name)

        return service

    def _should_attempt_recovery(self, service_name: str, status: ServiceStatus) -> bool:
        """判断是否应该尝试恢复服务"""
        if service_name not in self._last_recovery_attempt:
            return True

        # 简单的恢复策略：失败次数少于3次，且距离上次尝试超过5分钟
        time_since_last_attempt = (
            datetime.now() - self._last_recovery_attempt[service_name]).total_seconds()
        return status.consecutive_failures < 3 and time_since_last_attempt > 300

    def _attempt_service_recovery(self, service_name: str) -> bool:
        """尝试恢复服务"""
        self._last_recovery_attempt[service_name] = datetime.now()

        if service_name not in self.service_configs:
            return False

        config = self.service_configs[service_name]

        try:
            # 尝试重新初始化主服务
            service = config.primary_factory()
            self._services[service_name] = service
            self.service_status[service_name] = ServiceStatus(
                name=service_name,
                status='primary',
                last_check=datetime.now(),
                health_score=1.0
            )
            self.metrics.recovery_count += 1
            logger.info(f"{service_name} 服务恢复成功")
            return True

        except Exception as e:
            logger.warning(f"{service_name} 服务恢复失败: {e}")
            status = self.service_status[service_name]
            status.consecutive_failures += 1
            status.last_error = str(e)
            return False

    def _update_activity(self):
        """更新活动时间"""
        self.last_activity = datetime.now()
        if self.lifecycle_status == ComponentLifecycle.INITIALIZED:
            self.lifecycle_status = ComponentLifecycle.RUNNING

    def _get_metrics_dict(self) -> Dict[str, Any]:
        """获取指标字典"""
        return {
            'service_calls': self.metrics.service_calls,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'fallback_count': self.metrics.fallback_count,
            'recovery_count': self.metrics.recovery_count,
            'error_count': self.metrics.error_count,
            'average_response_time': self.metrics.average_response_time
        }


# =============================================================================
# 统一适配器工厂
# =============================================================================


class UnifiedAdapterFactory:

    """统一适配器工厂"""

    def __init__(self):

        self._adapters: Dict[BusinessLayerType, UnifiedBusinessAdapter] = {}
        self._adapter_classes: Dict[BusinessLayerType, Type[UnifiedBusinessAdapter]] = {}
        self._adapter_configs: Dict[BusinessLayerType, Dict[str, Any]] = {}

        # 监控和指标
        self._factory_metrics = {
            'created_adapters': 0,
            'active_adapters': 0,
            'failed_creations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def register_adapter_class(self, layer_type: BusinessLayerType,


                               adapter_class: Type[UnifiedBusinessAdapter],
                               config: Optional[Dict[str, Any]] = None):
        """注册适配器类"""
        self._adapter_classes[layer_type] = adapter_class
        if config:
            self._adapter_configs[layer_type] = config
        logger.info(f"注册适配器类: {layer_type.value} -> {adapter_class.__name__}")

    def get_adapter(self, layer_type: BusinessLayerType) -> Optional[UnifiedBusinessAdapter]:
        """获取适配器实例"""
        if layer_type not in self._adapters:
            if layer_type in self._adapter_classes and self._adapter_classes[layer_type] is not None:
                try:
                    adapter_class = self._adapter_classes[layer_type]
                    config = self._adapter_configs.get(layer_type, {})

                    # 创建实例
                    if config:
                        adapter = adapter_class(config=config)
                    else:
                        adapter = adapter_class()

                    # 初始化适配器
                    if hasattr(adapter, 'initialize') and not adapter.initialize():
                        logger.error(f"适配器初始化失败: {layer_type.value}")
                        return None

                    self._adapters[layer_type] = adapter
                    self._factory_metrics['created_adapters'] += 1
                    logger.info(f"成功创建适配器实例: {layer_type.value}")
                    return adapter
                except Exception as e:
                    logger.error(f"创建适配器实例失败: {layer_type.value}, 错误: {e}")
                    self._factory_metrics['failed_creations'] += 1
                    return None
            else:
                logger.warning(f"未注册的适配器类型: {layer_type.value}")
                return None

        self._factory_metrics['cache_hits'] += 1
        return self._adapters[layer_type]

    def get_all_adapters(self) -> Dict[BusinessLayerType, UnifiedBusinessAdapter]:
        """获取所有适配器"""
        for layer_type in BusinessLayerType:
            if layer_type not in self._adapters:
                self.get_adapter(layer_type)

        return self._adapters.copy()

    def health_check_all(self) -> Dict[str, Any]:
        """检查所有适配器的健康状态"""
        overall_health = {
            'timestamp': datetime.now().isoformat(),
            'factory_metrics': self._factory_metrics.copy(),
            'adapters': {},
            'overall_status': 'healthy',
            'active_adapters': len(self._adapters),
            'registered_classes': len([cls for cls in self._adapter_classes.values() if cls is not None])
        }

        degraded_count = 0
        unhealthy_count = 0

        for layer_type, adapter in self._adapters.items():
            try:
                adapter_health = adapter.health_check()
                overall_health['adapters'][layer_type.value] = adapter_health

                status = adapter_health.get('overall_status', 'unknown')
                if status == 'unhealthy':
                    unhealthy_count += 1
                    overall_health['overall_status'] = 'unhealthy'
                elif status == 'degraded':
                    degraded_count += 1
                    if overall_health['overall_status'] == 'healthy':
                        overall_health['overall_status'] = 'degraded'

            except Exception as e:
                logger.error(f"适配器健康检查失败: {layer_type.value}, 错误: {e}")
                overall_health['adapters'][layer_type.value] = {
                    'status': 'error',
                    'error': str(e),
                    'overall_status': 'unhealthy'
                }
                unhealthy_count += 1
                overall_health['overall_status'] = 'unhealthy'

        overall_health.update({
            'summary': {
                'healthy_adapters': len(self._adapters) - degraded_count - unhealthy_count,
                'degraded_adapters': degraded_count,
                'unhealthy_adapters': unhealthy_count,
                'total_instances': len(self._adapters)
            }
        })

        return overall_health

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'factory_metrics': self._factory_metrics.copy(),
            'adapter_metrics': {},
            'instance_counts': {
                layer_type.value: 1 if layer_type in self._adapters else 0
                for layer_type in BusinessLayerType
            },
            'cache_efficiency': {
                'total_requests': self._factory_metrics['cache_hits'] + self._factory_metrics['cache_misses'],
                'hit_rate': (self._factory_metrics['cache_hits']
                             / max(self._factory_metrics['cache_hits'] + self._factory_metrics['cache_misses'], 1))
            }
        }


# =============================================================================
# 全局适配器工厂实例
# =============================================================================

_unified_adapter_factory = UnifiedAdapterFactory()


def get_unified_adapter_factory() -> UnifiedAdapterFactory:
    """获取统一适配器工厂实例"""
    return _unified_adapter_factory


def register_adapter_class(layer_type: BusinessLayerType,


                           adapter_class: Type[UnifiedBusinessAdapter]):
    """注册适配器类"""
    _unified_adapter_factory.register_adapter_class(layer_type, adapter_class)


def get_adapter(layer_type: BusinessLayerType) -> Optional[UnifiedBusinessAdapter]:
    """获取适配器实例"""
    return _unified_adapter_factory.get_adapter(layer_type)


def get_all_adapters() -> Dict[BusinessLayerType, UnifiedBusinessAdapter]:
    """获取所有适配器"""
    return _unified_adapter_factory.get_all_adapters()


def health_check_all_adapters() -> Dict[str, Any]:
    """检查所有适配器的健康状态"""
    return _unified_adapter_factory.health_check_all()


def get_adapter_performance_report() -> Dict[str, Any]:
    """获取适配器性能报告"""
    return _unified_adapter_factory.get_performance_report()


# =============================================================================
# 便捷函数
# =============================================================================


def create_service_config(name: str, primary_factory: Callable,


                          fallback_factory: Optional[Callable] = None, **kwargs) -> ServiceConfig:
    """创建服务配置的便捷函数"""
    return ServiceConfig(
        name=name,
        primary_factory=primary_factory,
        fallback_factory=fallback_factory,
        **kwargs
    )


# =============================================================================
# 生命周期管理扩展
# =============================================================================


def add_lifecycle_listener(adapter: UnifiedBusinessAdapter,


                           status: ComponentLifecycle,
                           listener: Callable):
    """为适配器添加生命周期监听器"""
    adapter.add_lifecycle_listener(status, listener)


def remove_lifecycle_listener(adapter: UnifiedBusinessAdapter,


                              status: ComponentLifecycle,
                              listener: Callable):
    """从适配器移除生命周期监听器"""
    adapter.remove_lifecycle_listener(status, listener)


def start_adapter(adapter: UnifiedBusinessAdapter) -> bool:
    """启动适配器"""
    return adapter.start()


def stop_adapter(adapter: UnifiedBusinessAdapter, timeout: Optional[int] = None) -> bool:
    """停止适配器"""
    return adapter.stop(timeout)


def restart_adapter(adapter: UnifiedBusinessAdapter) -> bool:
    """重启适配器"""
    return adapter.restart()


def get_adapter_lifecycle_info(adapter: UnifiedBusinessAdapter) -> Dict[str, Any]:
    """获取适配器生命周期信息"""
    return adapter.get_lifecycle_info()


# =============================================================================
# 健康层适配器注册
# =============================================================================

def register_health_adapter():
    """注册健康层适配器"""
    from .health_adapter import HealthLayerAdapter
    from .business_adapters import BusinessLayerType

    try:
        register_adapter_class(BusinessLayerType.HEALTH, HealthLayerAdapter)
        logger.info("健康层适配器注册成功")
        return True
    except Exception as e:
        logger.error(f"健康层适配器注册失败: {e}")
        return False


# 初始化时自动注册健康适配器
try:
    register_health_adapter()
except Exception as e:
    logger.warning(f"健康适配器自动注册失败: {e}")


__all__ = [
    # 核心类
    'UnifiedBusinessAdapter',
    'UnifiedAdapterFactory',

    # 数据类
    'ServiceConfig',
    'AdapterMetrics',
    'ServiceStatus',

    # 全局函数
    'get_unified_adapter_factory',
    'register_adapter_class',
    'get_adapter',
    'get_all_adapters',
    'health_check_all_adapters',
    'get_adapter_performance_report',

    # 便捷函数
    'create_service_config',

    # 生命周期管理
    'add_lifecycle_listener',
    'remove_lifecycle_listener',
    'start_adapter',
    'stop_adapter',
    'restart_adapter',
    'get_adapter_lifecycle_info'
]


# =============================================================================
# 生命周期管理扩展
# =============================================================================


def add_lifecycle_listener(adapter: UnifiedBusinessAdapter,


                           status: ComponentLifecycle,
                           listener: Callable):
    """为适配器添加生命周期监听器"""
    adapter.add_lifecycle_listener(status, listener)


def remove_lifecycle_listener(adapter: UnifiedBusinessAdapter,


                              status: ComponentLifecycle,
                              listener: Callable):
    """从适配器移除生命周期监听器"""
    adapter.remove_lifecycle_listener(status, listener)


def start_adapter(adapter: UnifiedBusinessAdapter) -> bool:
    """启动适配器"""
    return adapter.start()


def stop_adapter(adapter: UnifiedBusinessAdapter, timeout: Optional[int] = None) -> bool:
    """停止适配器"""
    return adapter.stop(timeout)


def restart_adapter(adapter: UnifiedBusinessAdapter) -> bool:
    """重启适配器"""
    return adapter.restart()


def get_adapter_lifecycle_info(adapter: UnifiedBusinessAdapter) -> Dict[str, Any]:
    """获取适配器生命周期信息"""
    return adapter.get_lifecycle_info()


# =============================================================================
# 健康层适配器注册
# =============================================================================

def register_health_adapter():
    """注册健康层适配器"""
    from .health_adapter import HealthLayerAdapter
    from .business_adapters import BusinessLayerType

    try:
        register_adapter_class(BusinessLayerType.HEALTH, HealthLayerAdapter)
        logger.info("健康层适配器注册成功")
        return True
    except Exception as e:
        logger.error(f"健康层适配器注册失败: {e}")
        return False


# 初始化时自动注册健康适配器
try:
    register_health_adapter()
except Exception as e:
    logger.warning(f"健康适配器自动注册失败: {e}")
