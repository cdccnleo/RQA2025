#!/usr/bin/env python3
"""
RQA2025 统一业务层基础设施适配器

基于适配器模式设计，为不同业务层提供统一的基础设施服务访问接口，
消除代码重复，实现集中化管理。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type
import logging
import threading
import time
from enum import Enum
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class BusinessLayerType(Enum):

    """业务层类型枚举"""
    DATA = "data"
    FEATURES = "features"
    TRADING = "trading"
    RISK = "risk"
    MODELS = "models"
    ENGINE = "engine"
    HEALTH = "health"


class IBusinessAdapter(ABC):

    """业务层适配器接口"""

    @property
    @abstractmethod
    def layer_type(self) -> BusinessLayerType:
        """获取业务层类型"""

    @abstractmethod
    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""

    @abstractmethod
    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""


class BaseBusinessAdapter(IBusinessAdapter):

    """基础业务层适配器"""

    def __init__(self, layer_type: BusinessLayerType):

        self._layer_type = layer_type
        self._service_bridges: Dict[str, Any] = {}
        self._infrastructure_services: Dict[str, Any] = {}
        self._initialized = False

        # 初始化基础设施服务映射
        self._init_infrastructure_services()

    @property
    def layer_type(self) -> BusinessLayerType:

        return self._layer_type

    def _init_infrastructure_services(self):
        """初始化基础设施服务映射（使用统一服务注册表）"""
        try:
            from src.infrastructure.core import get_service_registry
            from src.infrastructure.logging.core.unified_logger import get_unified_logger
            
            registry = get_service_registry()
            
            # 确保服务已注册
            self._ensure_services_registered(registry)
            
            # 从服务注册表获取服务
            self._infrastructure_services = {
                'config_manager': registry.get_service('config_manager'),
                'cache_manager': registry.get_service('cache_manager'),
                'logger': get_unified_logger(f"{self._layer_type.value}_layer"),
                'monitoring': registry.get_service('monitoring'),
                'health_checker': registry.get_service('health_checker')
            }

            # 只在第一次初始化时记录日志，避免重复日志
            if not hasattr(self.__class__, '_initialized_layers'):
                self.__class__._initialized_layers = set()
            
            if self._layer_type.value not in self.__class__._initialized_layers:
                logger.info(f"{self._layer_type.value}层基础设施服务初始化完成")
                self.__class__._initialized_layers.add(self._layer_type.value)
            else:
                logger.debug(f"{self._layer_type.value}层基础设施服务已初始化（使用服务注册表单例）")

        except ImportError as e:
            logger.warning(f"{self._layer_type.value}层基础设施服务部分导入失败: {e}")
            # 使用降级模式
            self._init_fallback_services()
    
    def _ensure_services_registered(self, registry):
        """
        确保基础设施服务已注册到服务注册表
        
        Args:
            registry: 服务注册表实例
        """
        try:
            # 注册配置管理器
            if not registry.is_service_registered('config_manager'):
                from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
                registry.register_singleton('config_manager', service_class=UnifiedConfigManager)
            
            # 注册缓存管理器
            if not registry.is_service_registered('cache_manager'):
                from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
                registry.register_singleton('cache_manager', service_class=UnifiedCacheManager)
            
            # 注册监控服务
            if not registry.is_service_registered('monitoring'):
                from src.infrastructure.monitoring import ContinuousMonitoringSystem
                registry.register_singleton('monitoring', service_class=ContinuousMonitoringSystem)
            
            # 注册健康检查器
            if not registry.is_service_registered('health_checker'):
                from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
                registry.register_singleton('health_checker', service_class=EnhancedHealthChecker)
                
        except ImportError as e:
            logger.debug(f"部分基础设施服务导入失败（非关键）: {e}")

    def _init_fallback_services(self):
        """初始化降级服务"""
        try:
            # 延迟导入，避免循环依赖
            import importlib
            fallback_module = importlib.import_module('src.core.integration.fallback_services')
            get_fallback_service = fallback_module.get_fallback_service

            self._infrastructure_services = {
                'config_manager': get_fallback_service('config_manager'),
                'cache_manager': get_fallback_service('cache_manager'),
                'logger': get_fallback_service('logger'),
                'monitoring': get_fallback_service('monitoring'),
                'health_checker': get_fallback_service('health_checker')
            }

            logger.info(f"{self._layer_type.value}层降级服务初始化完成")
        except Exception as e:
            logger.error(f"降级服务导入失败: {e}")
            # 使用最基本的降级实现
            self._infrastructure_services = self._create_basic_fallback_services()

    def _create_basic_fallback_services(self) -> Dict[str, Any]:
        """创建最基本的降级服务"""
        import logging

        class BasicLogger:

            def info(self, msg): logging.info(msg)

            def warning(self, msg): logging.warning(msg)

            def error(self, msg): logging.error(msg)

            def debug(self, msg): logging.debug(msg)

        return {
            'config_manager': None,
            'cache_manager': None,
            'logger': BasicLogger(),
            'monitoring': None,
            'health_checker': None
        }

    def get_infrastructure_services(self) -> Dict[str, Any]:
        """获取基础设施服务"""
        return self._infrastructure_services.copy()

    def get_service_bridge(self, service_name: str) -> Optional[Any]:
        """获取服务桥接器"""
        return self._service_bridges.get(service_name)

    def get_config_manager(self) -> Any:
        """获取配置管理器"""
        return self._infrastructure_services.get('config_manager')

    def get_cache_manager(self) -> Any:
        """获取缓存管理器"""
        return self._infrastructure_services.get('cache_manager')

    def get_logger(self) -> Any:
        """获取日志器"""
        return self._infrastructure_services.get('logger')

    def get_monitoring(self) -> Any:
        """获取监控器"""
        return self._infrastructure_services.get('monitoring')

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'layer_type': self._layer_type.value,
            'infrastructure_services': {},
            'service_bridges': {},
            'overall_status': 'healthy'
        }

        # 检查基础设施服务
        for service_name, service in self._infrastructure_services.items():
            try:
                if hasattr(service, 'health_check'):
                    service_health = service.health_check()
                    health_status['infrastructure_services'][service_name] = service_health
                else:
                    health_status['infrastructure_services'][service_name] = {'status': 'unknown'}
            except Exception as e:
                health_status['infrastructure_services'][service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'

        # 检查服务桥接器
        for bridge_name, bridge in self._service_bridges.items():
            try:
                if hasattr(bridge, 'health_check'):
                    bridge_health = bridge.health_check()
                    health_status['service_bridges'][bridge_name] = bridge_health
                else:
                    health_status['service_bridges'][bridge_name] = {'status': 'unknown'}
            except Exception as e:
                health_status['service_bridges'][bridge_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }
                health_status['overall_status'] = 'degraded'

        return health_status


class UnifiedBusinessAdapterFactory:

    """统一业务层适配器工厂 - 增强版

    提供高性能、动态配置、监控集成的适配器工厂服务
    支持并发访问、自动故障转移、性能监控等高级功能
    """

    def __init__(self):

        self._adapters: Dict[BusinessLayerType, IBusinessAdapter] = {}
        self._adapter_classes: Dict[BusinessLayerType, Type[IBusinessAdapter]] = {}
        self._adapter_configs: Dict[BusinessLayerType, Dict[str, Any]] = {}
        self._adapter_metrics: Dict[BusinessLayerType, Dict[str, Any]] = {}
        self._adapter_instances: Dict[BusinessLayerType, List[IBusinessAdapter]] = {}

        # 性能和并发控制
        self._lock = threading.Lock()
        self._max_instances_per_type = 5  # 每个类型的最大实例数
        self._instance_cleanup_interval = 300  # 实例清理间隔(秒)

        # 监控和指标
        self._factory_metrics = {
            'created_adapters': 0,
            'active_adapters': 0,
            'failed_creations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # 注册适配器类
        self._register_adapter_classes()
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动实例清理任务"""

        def cleanup_worker():

            while True:
                try:
                    time.sleep(self._instance_cleanup_interval)
                    self._cleanup_idle_instances()
                except Exception as e:
                    logger.error(f"实例清理任务异常: {e}")

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def _cleanup_idle_instances(self):
        """清理空闲实例"""
        with self._lock:
            current_time = datetime.now()
            for layer_type, instances in self._adapter_instances.items():
                active_instances = []
                for instance in instances:
                    # 检查实例是否仍然活跃（这里可以根据实际需求实现更复杂的逻辑）
                    if hasattr(instance, '_last_activity'):
                        last_activity = getattr(instance, '_last_activity', current_time)
                        if current_time - last_activity < timedelta(hours=1):
                            active_instances.append(instance)
                        else:
                            logger.info(f"清理空闲实例: {layer_type.value}")
                    else:
                        active_instances.append(instance)

                self._adapter_instances[layer_type] = active_instances

    def _register_adapter_classes(self):
        """注册适配器类"""
        # 使用延迟导入避免循环导入问题
        try:
            from .data_adapter import DataLayerAdapter
            data_adapter = DataLayerAdapter
        except ImportError:
            data_adapter = None

        try:
            from .features_adapter import FeaturesLayerAdapter
            features_adapter = FeaturesLayerAdapter
        except ImportError:
            features_adapter = None

        try:
            from .trading_adapter import TradingLayerAdapter
            trading_adapter = TradingLayerAdapter
        except ImportError:
            trading_adapter = None

        try:
            from .risk_adapter import RiskLayerAdapter
            risk_adapter = RiskLayerAdapter
        except ImportError:
            risk_adapter = None

        try:
            from .models_adapter import ModelsLayerAdapter
            models_adapter = ModelsLayerAdapter
        except ImportError:
            models_adapter = None

        self._adapter_classes = {
            BusinessLayerType.DATA: data_adapter,
            BusinessLayerType.FEATURES: features_adapter,
            BusinessLayerType.TRADING: trading_adapter,
            BusinessLayerType.RISK: risk_adapter,
            BusinessLayerType.MODELS: models_adapter
        }

    def register_adapter_class(self, layer_type: BusinessLayerType,


                               adapter_class: Type[IBusinessAdapter],
                               config: Optional[Dict[str, Any]] = None):
        """动态注册适配器类"""
        with self._lock:
            self._adapter_classes[layer_type] = adapter_class
            if config:
                self._adapter_configs[layer_type] = config
            logger.info(f"动态注册适配器类: {layer_type.value} -> {adapter_class.__name__}")

    def configure_adapter(self, layer_type: BusinessLayerType, config: Dict[str, Any]):
        """配置适配器"""
        with self._lock:
            self._adapter_configs[layer_type] = config
            logger.info(f"更新适配器配置: {layer_type.value}")

    def get_adapter_config(self, layer_type: BusinessLayerType) -> Optional[Dict[str, Any]]:
        """获取适配器配置"""
        return self._adapter_configs.get(layer_type)

    def get_adapter(self, layer_type: BusinessLayerType, use_load_balancing: bool = True) -> IBusinessAdapter:
        """获取业务层适配器 - 支持负载均衡"""
        with self._lock:
            # 检查缓存
            if layer_type in self._adapters and not use_load_balancing:
                self._factory_metrics['cache_hits'] += 1
                return self._adapters[layer_type]

            self._factory_metrics['cache_misses'] += 1

            # 创建适配器实例
            adapter = self._create_adapter_instance(layer_type)
            if adapter:
                self._factory_metrics['created_adapters'] += 1
                if not use_load_balancing:
                    self._adapters[layer_type] = adapter
            else:
                self._factory_metrics['failed_creations'] += 1
                # 返回基础适配器作为降级方案
                adapter = BaseBusinessAdapter(layer_type)
                logger.warning(f"使用基础适配器作为降级方案: {layer_type.value}")

            return adapter

    def _create_adapter_instance(self, layer_type: BusinessLayerType) -> Optional[IBusinessAdapter]:
        """创建适配器实例"""
        try:
            if layer_type in self._adapter_classes and self._adapter_classes[layer_type] is not None:
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

                # 记录实例
                if layer_type not in self._adapter_instances:
                    self._adapter_instances[layer_type] = []
                self._adapter_instances[layer_type].append(adapter)

                # 更新指标
                self._adapter_metrics[layer_type] = self._adapter_metrics.get(layer_type, {
                    'instances': 0,
                    'created_at': datetime.now().isoformat()
                })
                self._adapter_metrics[layer_type]['instances'] += 1

                logger.info(f"成功创建适配器实例: {layer_type.value}")
                return adapter
            else:
                logger.warning(f"未注册的适配器类型: {layer_type.value}")
                return None

        except Exception as e:
            logger.error(f"创建适配器实例失败: {layer_type.value}, 错误: {e}")
            return None

    def get_all_adapters(self) -> Dict[BusinessLayerType, IBusinessAdapter]:
        """获取所有适配器"""
        for layer_type in BusinessLayerType:
            if layer_type not in self._adapters:
                self.get_adapter(layer_type)

        return self._adapters.copy()

    def health_check_all(self) -> Dict[str, Any]:
        """检查所有适配器的健康状态 - 增强版"""
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
                'total_instances': sum(len(instances) for instances in self._adapter_instances.values())
            }
        })

        return overall_health

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'factory_metrics': self._factory_metrics.copy(),
            'adapter_metrics': self._adapter_metrics.copy(),
            'instance_counts': {
                layer_type.value: len(instances)
                for layer_type, instances in self._adapter_instances.items()
            },
            'cache_efficiency': {
                'total_requests': self._factory_metrics['cache_hits'] + self._factory_metrics['cache_misses'],
                'hit_rate': (self._factory_metrics['cache_hits']
                             / max(self._factory_metrics['cache_hits'] + self._factory_metrics['cache_misses'], 1))
            }
        }

    def reload_adapter(self, layer_type: BusinessLayerType) -> bool:
        """重新加载适配器"""
        with self._lock:
            try:
                if layer_type in self._adapters:
                    del self._adapters[layer_type]

                if layer_type in self._adapter_instances:
                    # 清理旧实例
                    for instance in self._adapter_instances[layer_type]:
                        if hasattr(instance, 'shutdown'):
                            instance.shutdown()
                    del self._adapter_instances[layer_type]

                # 重新创建
                adapter = self.get_adapter(layer_type, use_load_balancing=False)
                logger.info(f"成功重新加载适配器: {layer_type.value}")
                return adapter is not None

            except Exception as e:
                logger.error(f"重新加载适配器失败: {layer_type.value}, 错误: {e}")
                return False


# 全局适配器工厂实例
_adapter_factory = UnifiedBusinessAdapterFactory()


def get_business_adapter(layer_type: BusinessLayerType) -> IBusinessAdapter:
    """获取业务层适配器"""
    return _adapter_factory.get_adapter(layer_type)


def get_all_business_adapters() -> Dict[BusinessLayerType, IBusinessAdapter]:
    """获取所有业务层适配器"""
    return _adapter_factory.get_all_adapters()


def health_check_business_adapters() -> Dict[str, Any]:
    """检查所有业务层适配器的健康状态"""
    return _adapter_factory.health_check_all()


# 便捷函数

def get_data_adapter() -> IBusinessAdapter:
    """获取数据层适配器"""
    return get_business_adapter(BusinessLayerType.DATA)


def get_features_adapter() -> IBusinessAdapter:
    """获取特征层适配器"""
    return get_business_adapter(BusinessLayerType.FEATURES)


def get_trading_adapter() -> IBusinessAdapter:
    """获取交易层适配器"""
    return get_business_adapter(BusinessLayerType.TRADING)


def get_risk_adapter() -> IBusinessAdapter:
    """获取风控层适配器"""
    return get_business_adapter(BusinessLayerType.RISK)


def get_models_adapter() -> IBusinessAdapter:
    """获取模型层适配器"""
    return get_business_adapter(BusinessLayerType.MODELS)


# 新增的高级功能便捷函数

def get_adapter_performance_report() -> Dict[str, Any]:
    """获取适配器性能报告"""
    return _adapter_factory.get_performance_report()


def reload_business_adapter(layer_type: BusinessLayerType) -> bool:
    """重新加载业务层适配器"""
    return _adapter_factory.reload_adapter(layer_type)


def configure_business_adapter(layer_type: BusinessLayerType, config: Dict[str, Any]):
    """配置业务层适配器"""
    _adapter_factory.configure_adapter(layer_type, config)


def register_business_adapter_class(layer_type: BusinessLayerType,


                                    adapter_class: Type[IBusinessAdapter],
                                    config: Optional[Dict[str, Any]] = None):
    """注册业务层适配器类"""
    _adapter_factory.register_adapter_class(layer_type, adapter_class, config)


def get_health_adapter() -> IBusinessAdapter:
    """获取健康层适配器"""
    return get_business_adapter(BusinessLayerType.HEALTH)


# 导入统一适配器工厂函数 - 暂时注释掉，避免循环导入
# from .adapters import get_unified_adapter_factory, register_adapter_class
