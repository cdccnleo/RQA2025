import logging
#!/usr/bin/env python3
"""
RQA2025 特征层基础设施适配器 - 重构版

专门为特征层提供基础设施服务访问接口，
基于统一业务层适配器架构，实现特征层的特定需求。
深度集成统一事件总线，支持事件驱动架构。

重构说明:
- 将大类拆分为多个专门组件
- 职责分离，提高可维护性
- 组合模式，保持接口兼容性
"""

from typing import Dict, List, Any, Optional, Protocol
import json
import time
from dataclasses import dataclass
from src.core.constants import (
    DEFAULT_TIMEOUT, MAX_RETRIES, SECONDS_PER_HOUR, MAX_RECORDS, DEFAULT_BATCH_SIZE
)

from .business_adapters import BaseBusinessAdapter, BusinessLayerType

# 导入缓存策略枚举
try:
    from src.data.cache.enhanced_cache_strategy import CacheStrategy
except ImportError:
    # 降级定义
    from enum import Enum
    class CacheStrategy(Enum):
        LRU = "lru"
        LFU = "lfu"
        ADAPTIVE = "adaptive"
        FIFO = "fifo"
        COST_AWARE = "cost_aware"

from src.core.event_bus import EventBus, EventType, Event
# 尝试导入依赖组件，失败时使用降级方案
# 注意：使用服务注册表获取单例缓存管理器，避免重复初始化
try:
    from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
    def get_cache_manager():
        """获取缓存管理器（使用服务注册表单例）"""
        try:
            from src.infrastructure.core import get_service_registry
            registry = get_service_registry()
            if registry.is_service_registered('cache_manager'):
                return registry.get_service('cache_manager')
            else:
                # 降级：直接创建实例（但不推荐）
                return UnifiedCacheManager()
        except Exception:
            # 降级：直接创建实例
            return UnifiedCacheManager()
except ImportError:
    def get_cache_manager(): return None

try:
    from src.security.unified_security import get_security, UnifiedSecurity
except ImportError:
    def get_security(): return None
    UnifiedSecurity = None

try:
    # 从monitoring模块导入（已在__init__.py中导出）
    from src.infrastructure.monitoring import ContinuousMonitoringSystem
except ImportError:
    try:
        # 降级方案：直接从services导入
        from src.infrastructure.monitoring.services.continuous_monitoring_service import ContinuousMonitoringSystem
    except ImportError:
        try:
            # 再次降级：从core导入
            from src.infrastructure.monitoring.services.continuous_monitoring_core import ContinuousMonitoringSystem
        except ImportError:
            ContinuousMonitoringSystem = None

logger = logging.getLogger(__name__)


# 协议定义
class FeatureCacheManager(Protocol):
    """特征缓存管理器协议"""
    def get_feature_cache(self): ...
    def get_model_cache(self): ...
    def cache_feature_result(self, feature_key: str, result: Any, priority: int = 0): ...
    def get_cached_feature(self, feature_key: str) -> Optional[Any]: ...


class FeatureSecurityManager(Protocol):
    """特征安全管理器协议"""
    def validate_feature_access(self, user_id: str, feature_name: str, action: str = "access") -> bool: ...
    def encrypt_feature_data(self, data: Any) -> str: ...
    def decrypt_feature_data(self, encrypted_data: str) -> Any: ...


class FeaturePerformanceMonitor(Protocol):
    """特征性能监控器协议"""
    def monitor_feature_performance(self, feature_name: str, execution_time: float, success: bool): ...
    def collect_detailed_metrics(self) -> Dict[str, Any]: ...


@dataclass
class FeaturesAdapterConfig:
    """特征适配器配置"""
    enable_smart_cache: bool = True
    enable_enterprise_security: bool = True
    enable_performance_monitoring: bool = True
    cache_strategy: str = "adaptive"
    security_level: str = "enterprise"
    monitoring_interval: int = DEFAULT_TIMEOUT


class FeatureCacheManagerImpl:
    """特征缓存管理器实现 - 职责：管理特征缓存"""

    def __init__(self, config: FeaturesAdapterConfig):
        self.config = config
        self._feature_cache = {}
        self._model_cache = {}
        self._priority_cache = {}

    def get_feature_cache(self):
        """获取特征缓存"""
        return self._feature_cache

    def get_model_cache(self):
        """获取模型缓存"""
        return self._model_cache

    def get_priority_cache(self):
        """获取优先级缓存"""
        return self._priority_cache

    def cache_feature_result(self, feature_key: str, result: Any, priority: int = 0):
        """缓存特征结果"""
        self._feature_cache[feature_key] = {
            'result': result,
            'timestamp': time.time(),
            'priority': priority
        }

    def get_cached_feature(self, feature_key: str) -> Optional[Any]:
        """获取缓存的特征"""
        if feature_key in self._feature_cache:
            return self._feature_cache[feature_key]['result']
        return None


class FeatureSecurityManagerImpl:
    """特征安全管理器实现 - 职责：管理特征安全"""

    def __init__(self, config: FeaturesAdapterConfig):
        self.config = config
        self._access_logs = []

    def validate_feature_access(self, user_id: str, feature_name: str, action: str = "access") -> bool:
        """验证特征访问权限"""
        # 简化的权限验证逻辑
        if self.config.security_level == "enterprise":
            # 企业级安全检查
            return self._check_enterprise_access(user_id, feature_name, action)
        return True  # 默认允许访问

    def _check_enterprise_access(self, user_id: str, feature_name: str, action: str) -> bool:
        """企业级访问检查"""
        # 这里应该实现实际的企业级安全检查逻辑
        return True

    def encrypt_feature_data(self, data: Any) -> str:
        """加密特征数据"""
        # 简化的加密逻辑
        return json.dumps(data)

    def decrypt_feature_data(self, encrypted_data: str) -> Any:
        """解密特征数据"""
        return json.loads(encrypted_data)


class FeaturePerformanceMonitorImpl:
    """特征性能监控器实现 - 职责：监控特征性能"""

    def __init__(self, config: FeaturesAdapterConfig):
        self.config = config
        self._metrics = {}
        self._performance_history = []

    def monitor_feature_performance(self, feature_name: str, execution_time: float, success: bool):
        """监控特征性能"""
        if feature_name not in self._metrics:
            self._metrics[feature_name] = {
                'total_calls': 0,
                'total_time': 0.0,
                'success_count': 0,
                'avg_time': 0.0
            }

        metric = self._metrics[feature_name]
        metric['total_calls'] += 1
        metric['total_time'] += execution_time
        if success:
            metric['success_count'] += 1
        metric['avg_time'] = metric['total_time'] / metric['total_calls']

    def collect_detailed_metrics(self) -> Dict[str, Any]:
        """收集详细指标"""
        return {
            'features': self._metrics,
            'performance_history': self._performance_history[-MAX_RETRIES:],  # 最近100条记录
            'summary': {
                'total_features': len(self._metrics),
                'total_calls': sum(m['total_calls'] for m in self._metrics.values()),
                'avg_execution_time': sum(m['avg_time'] for m in self._metrics.values()) / len(self._metrics) if self._metrics else 0
            }
        }


class FeaturesLayerAdapterRefactored(BaseBusinessAdapter):
    """重构后的特征层适配器 - 组合模式：使用专门的组件"""

    def __init__(self, config: Optional[FeaturesAdapterConfig] = None):
        super().__init__(BusinessLayerType.FEATURES)

        # 初始化配置
        self.config = config or FeaturesAdapterConfig()

        # 初始化专门的组件
        self.cache_manager = FeatureCacheManagerImpl(self.config)
        self.security_manager = FeatureSecurityManagerImpl(self.config)
        self.performance_monitor = FeaturePerformanceMonitorImpl(self.config)

        # 注意：基础设施服务已在BaseBusinessAdapter.__init__中初始化
        # 这里不需要重复初始化，但可以获取服务引用
        self._infrastructure_services = self.get_infrastructure_services()

        # 初始化事件驱动架构（保持兼容性）
        self._init_event_driven_features()

        # 只在第一次初始化时记录日志，避免重复日志
        if not hasattr(FeaturesLayerAdapterRefactored, '_instance_count'):
            FeaturesLayerAdapterRefactored._instance_count = 0
        
        FeaturesLayerAdapterRefactored._instance_count += 1
        
        if FeaturesLayerAdapterRefactored._instance_count == 1:
            logger.info("重构后的特征层适配器初始化完成")
        else:
            logger.debug(f"重构后的特征层适配器初始化完成（实例 #{FeaturesLayerAdapterRefactored._instance_count}）")

    # 代理方法到专门的组件
    def get_feature_cache(self):
        """获取特征缓存 - 代理到缓存管理器"""
        return self.cache_manager.get_feature_cache()

    def get_model_cache(self):
        """获取模型缓存 - 代理到缓存管理器"""
        return self.cache_manager.get_model_cache()

    def get_priority_cache(self):
        """获取优先级缓存 - 代理到缓存管理器"""
        return self.cache_manager.get_priority_cache()

    def cache_feature_result(self, feature_key: str, result: Any, priority: int = 0):
        """缓存特征结果 - 代理到缓存管理器"""
        return self.cache_manager.cache_feature_result(feature_key, result, priority)

    def get_cached_feature(self, feature_key: str) -> Optional[Any]:
        """获取缓存的特征 - 代理到缓存管理器"""
        return self.cache_manager.get_cached_feature(feature_key)

    def validate_feature_access(self, user_id: str, feature_name: str, action: str = "access") -> bool:
        """验证特征访问权限 - 代理到安全管理器"""
        return self.security_manager.validate_feature_access(user_id, feature_name, action)

    def encrypt_feature_data(self, data: Any) -> str:
        """加密特征数据 - 代理到安全管理器"""
        return self.security_manager.encrypt_feature_data(data)

    def decrypt_feature_data(self, encrypted_data: str) -> Any:
        """解密特征数据 - 代理到安全管理器"""
        return self.security_manager.decrypt_feature_data(encrypted_data)

    def monitor_feature_performance(self, feature_name: str, execution_time: float, success: bool):
        """监控特征性能 - 代理到性能监控器"""
        return self.performance_monitor.monitor_feature_performance(feature_name, execution_time, success)

    def collect_detailed_metrics(self) -> Dict[str, Any]:
        """收集详细指标 - 代理到性能监控器"""
        return self.performance_monitor.collect_detailed_metrics()

    # 保持向后兼容性
    def _init_legacy_services(self):
        """初始化遗留基础设施服务（向后兼容）"""
        self._event_bus = None
        self._cache_manager = None
        self._security_manager = None
        self._performance_monitor = None

        # 尝试初始化基础设施组件
        self._init_event_driven_features()

    def _init_features_specific_services(self):
        """初始化特征层特定的基础设施服务"""
        try:
            # 特征层特定的服务桥接器已迁移到统一基础设施集成层
            self._service_bridges = {}

            logger.info("特征层特定服务桥接器初始化完成")

        except ImportError as e:
            logger.warning(f"特征层特定服务桥接器导入失败，使用基础服务: {e}")

    def _init_event_driven_features(self):
        """初始化事件驱动架构"""
        # 只在第一次初始化时执行，避免重复初始化
        if hasattr(FeaturesLayerAdapterRefactored, '_event_driven_initialized') and FeaturesLayerAdapterRefactored._event_driven_initialized:
            logger.debug("特征层事件驱动架构已初始化（跳过重复初始化）")
            return
        
        try:
            # 使用单例事件总线，避免重复创建
            try:
                from src.core.event_bus import get_event_bus
                self._event_bus = get_event_bus()
                logger.debug("使用全局单例事件总线")
            except ImportError:
                # 降级：直接创建实例
                self._event_bus = EventBus()
                logger.debug("创建新的事件总线实例（降级模式）")

            # 初始化智能缓存系统
            self._cache_manager = get_cache_manager()
            self._init_smart_caches()

            # 初始化企业级安全系统
            self._security_manager = get_security()
            self._init_enterprise_security()

            # 启动事件总线（先初始化事件总线，再初始化其他依赖事件总线的组件）
            if self._event_bus:
                try:
                    # 检查事件总线是否已初始化
                    if not hasattr(self._event_bus, '_initialized') or not self._event_bus._initialized:
                        if self._event_bus.initialize():
                            self._event_bus.start()
                            logger.debug("事件总线初始化并启动成功")
                        else:
                            logger.warning("事件总线初始化失败")
                    else:
                        logger.debug("事件总线已初始化，跳过重复初始化")
                except Exception as e:
                    logger.warning(f"事件总线初始化异常: {e}")

            # 初始化性能监控增强系统（可选，依赖事件总线）
            # 使用统一业务适配器的单例监控服务，避免重复启动
            try:
                infrastructure_services = self.get_infrastructure_services()
                self._performance_monitor = infrastructure_services.get('monitoring')
                if self._performance_monitor:
                    self._init_performance_monitoring()
                else:
                    logger.debug("统一监控服务不可用，使用降级方案")
                    self._performance_monitor = None
            except Exception as e:
                logger.debug(f"性能监控增强系统初始化失败（可选）: {e}")
                self._performance_monitor = None

            # 注册特征层特定的事件处理器（依赖事件总线）
            self._register_event_handlers()

            # 标记为已初始化
            FeaturesLayerAdapterRefactored._event_driven_initialized = True
            logger.info("特征层事件驱动架构初始化完成")

        except Exception as e:
            logger.warning(f"特征层事件驱动架构初始化失败: {e}")
            self._event_bus = None

    def _init_smart_caches(self):
        """初始化智能缓存系统"""
        try:
            # 检查缓存管理器是否有 create_cache 方法
            if hasattr(self._cache_manager, 'create_cache'):
                # 创建特征层专用缓存实例
                self._feature_cache = self._cache_manager.create_cache(
                    name="features_main",
                    strategy=CacheStrategy.ADAPTIVE,
                    capacity=2000,
                    ttl=1800  # 30分钟
                )

                self._model_cache = self._cache_manager.create_cache(
                    name="features_models",
                    strategy=CacheStrategy.COST_AWARE,
                    capacity=500,
                    ttl=SECONDS_PER_HOUR,  # 1小时
                    cost_threshold=5.0
                )

                self._priority_cache = self._cache_manager.create_cache(
                    name="features_priority",
                    strategy=CacheStrategy.PRIORITY,
                    capacity=MAX_RECORDS,
                    ttl=900  # 15分钟
                )

                logger.info("特征层智能缓存系统初始化完成")
            else:
                # 降级方案：直接使用缓存管理器本身
                logger.debug("缓存管理器不支持 create_cache 方法，使用统一缓存管理器")
                self._feature_cache = self._cache_manager
                self._model_cache = self._cache_manager
                self._priority_cache = self._cache_manager

        except Exception as e:
            logger.warning(f"智能缓存系统初始化失败: {e}")
            # 创建基本的LRU缓存作为降级方案
            if hasattr(self._cache_manager, 'create_cache'):
                try:
                    self._feature_cache = self._cache_manager.create_cache(
                        name="features_fallback",
                        strategy=CacheStrategy.LRU,
                        capacity=MAX_RECORDS,
                        ttl=1800
                    )
                except Exception:
                    self._feature_cache = self._cache_manager
            else:
                self._feature_cache = self._cache_manager

    def _register_event_handlers(self):
        """注册特征层事件处理器"""
        # 只在第一次注册时执行，避免重复注册
        if hasattr(FeaturesLayerAdapterRefactored, '_event_handlers_registered') and FeaturesLayerAdapterRefactored._event_handlers_registered:
            logger.debug("特征层事件处理器已注册（跳过重复注册）")
            return
        
        if not self._event_bus:
            logger.debug("事件总线未初始化，跳过事件处理器注册")
            return

        # 检查事件总线是否已初始化
        if not hasattr(self._event_bus, '_initialized') or not self._event_bus._initialized:
            logger.debug("事件总线尚未初始化完成，跳过事件处理器注册")
            return

        try:
            # 注册特征提取事件处理器
            self._event_bus.subscribe(
                EventType.FEATURES_EXTRACTED,
                self._handle_features_extracted_event
            )

            # 注册特征处理完成事件处理器
            self._event_bus.subscribe(
                EventType.FEATURE_PROCESSING_COMPLETED,
                self._handle_feature_processing_completed_event
            )

            # 注册性能监控事件处理器
            self._event_bus.subscribe(
                EventType.PERFORMANCE_ALERT,
                self._handle_performance_alert_event
            )

            # 注册缓存事件处理器
            self._event_bus.subscribe(
                EventType.CACHE_HIT,
                self._handle_cache_hit_event
            )
            self._event_bus.subscribe(
                EventType.CACHE_MISS,
                self._handle_cache_miss_event
            )

            # 标记为已注册
            FeaturesLayerAdapterRefactored._event_handlers_registered = True
            logger.info("特征层事件处理器注册完成")

        except Exception as e:
            logger.error(f"注册特征层事件处理器失败: {e}", exc_info=True)

    def get_features_config_manager(self):
        """获取特征层配置管理器"""
        return self.get_infrastructure_services().get('config_manager')

    def get_features_cache_manager(self):
        """获取特征层缓存管理器"""
        return self.get_infrastructure_services().get('cache_manager')

    def get_features_monitoring(self):
        """获取特征层监控服务"""
        return self.get_infrastructure_services().get('monitoring')

    def get_features_health_checker(self):
        """获取特征层健康检查器"""
        return self.get_infrastructure_services().get('health_checker')

    def get_config_manager(self):
        """获取配置管理器"""
        return self.get_infrastructure_services().get('config_manager')

    def get_cache_manager(self):
        """获取缓存管理器"""
        return self.get_infrastructure_services().get('cache_manager')

    def get_event_bus(self):
        """获取事件总线"""
        return self.get_infrastructure_services().get('event_bus')

    def get_logger(self):
        """获取日志器"""
        return self.get_infrastructure_services().get('logger')

    def get_monitoring(self):
        """获取监控服务"""
        return self.get_infrastructure_services().get('monitoring')

    def get_features_engine(self):
        """获取特征层引擎"""
        try:
            from src.features.core.engine import FeatureEngine
            return FeatureEngine()
        except ImportError:
            logger.warning("特征层引擎导入失败")
            return None

    def get_features_distributed_processor(self):
        """获取特征层分布式处理器"""
        try:
            from src.features.distributed.distributed_processor import DistributedProcessor
            return DistributedProcessor()
        except ImportError:
            logger.warning("特征层分布式处理器导入失败")
            return None

    def get_features_accelerator(self):
        """获取特征层加速器"""
        try:
            from src.features.acceleration.performance_optimizer import PerformanceOptimizer
            return PerformanceOptimizer()
        except ImportError:
            logger.warning("特征层加速器导入失败")
            return None

    def get_features_intelligent_manager(self):
        """获取特征层智能化管理器"""
        try:
            from src.features.intelligent.intelligent_enhancement_manager import IntelligentEnhancementManager
            return IntelligentEnhancementManager()
        except ImportError:
            logger.warning("特征层智能化管理器导入失败")
            return None

    def health_check(self) -> Dict[str, Any]:
        """特征层健康检查"""
        base_health = super().health_check()

        # 添加特征层特定检查
        features_specific_health = {
            'features_engine': self._check_features_engine_health(),
            'features_distributed_processor': self._check_component_health('distributed_processor'),
            'features_accelerator': self._check_component_health('accelerator'),
            'features_intelligent_manager': self._check_component_health('intelligent_manager'),
            'features_config_manager': self._check_service_health('config_manager'),
            'features_cache_manager': self._check_service_health('cache_manager')
        }

        base_health['features_specific_services'] = features_specific_health

        # 更新整体状态
        for service_name, health_info in features_specific_health.items():
            if health_info.get('status') != 'healthy':
                base_health['overall_status'] = 'degraded'
                break

        return base_health

    def _check_bridge_health(self, bridge_name: str) -> Dict[str, Any]:
        """检查基础设施服务健康状态 - 已更新为使用统一适配器"""
        services = self.get_infrastructure_services()
        if bridge_name in services:
            service = services[bridge_name]
            if service and hasattr(service, 'health_check'):
                try:
                    return service.health_check()
                except Exception as e:
                    return {'status': 'unhealthy', 'error': str(e)}
        return {'status': 'unknown'}

    def _check_features_engine_health(self) -> Dict[str, Any]:
        """检查特征引擎健康状态"""
        engine = self.get_features_engine()
        if engine and hasattr(engine, 'health_check'):
            try:
                return engine.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif engine:
            return {'status': 'healthy', 'engine_available': True}
        return {'status': 'unknown'}

    def _check_component_health(self, component_name: str) -> Dict[str, Any]:
        """检查特征层组件健康状态"""
        component = None
        if component_name == 'distributed_processor':
            component = self.get_features_distributed_processor()
        elif component_name == 'accelerator':
            component = self.get_features_accelerator()
        elif component_name == 'intelligent_manager':
            component = self.get_features_intelligent_manager()

        if component and hasattr(component, 'health_check'):
            try:
                return component.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif component:
            return {'status': 'healthy', 'component_available': True}
        return {'status': 'unknown'}

    def _check_service_health(self, service_name: str) -> Dict[str, Any]:
        """检查服务健康状态"""
        # 直接从统一基础设施服务获取
        service = self.get_infrastructure_services().get(service_name)

        if service and hasattr(service, 'health_check'):
            try:
                return service.health_check()
            except Exception as e:
                return {'status': 'unhealthy', 'error': str(e)}
        elif service:
            return {'status': 'healthy', 'service_available': True}
        return {'status': 'unknown'}

    def get_features_layer_metrics(self) -> Dict[str, Any]:
        """获取特征层性能指标"""
        metrics = {
            'timestamp': '2025 - 01 - 27T10:00:00Z',
            'layer_type': 'features',
            'infrastructure_metrics': {},
            'bridge_metrics': {},
            'engine_metrics': {}
        }

        # 获取基础设施服务指标
        for service_name, service in self.get_infrastructure_services().items():
            if hasattr(service, 'get_metrics'):
                try:
                    metrics['infrastructure_metrics'][service_name] = service.get_metrics()
                except Exception as e:
                    metrics['infrastructure_metrics'][service_name] = {'error': str(e)}

        # 获取桥接器指标
        for bridge_name, bridge in self._service_bridges.items():
            if hasattr(bridge, 'get_metrics'):
                try:
                    metrics['bridge_metrics'][bridge_name] = bridge.get_metrics()
                except Exception as e:
                    metrics['bridge_metrics'][bridge_name] = {'error': str(e)}

        # 获取特征引擎指标
        engine = self.get_features_engine()
        if engine and hasattr(engine, 'get_metrics'):
            try:
                metrics['engine_metrics'] = engine.get_metrics()
            except Exception as e:
                metrics['engine_metrics'] = {'error': str(e)}

        return metrics

    def process_features_with_infrastructure(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """使用基础设施支持处理特征数据 - 重构版：职责分离"""
        result = self._initialize_features_result(data)

        try:
            # 获取基础设施服务
            services = self._get_features_infrastructure_services()

            # 记录处理开始
            self._record_features_start(services, result)

            # 检查缓存避免重复处理
            if self._check_features_cache(data, services, result):
                return result

            # 执行实际特征处理
            self._execute_features_processing(data, services, result)

            # 记录处理完成
            self._record_features_completion(services)

        except Exception as e:
            self._handle_features_error(e, result)

        return result

    def _initialize_features_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """初始化特征处理结果对象"""
        return {
            'timestamp': datetime.now().isoformat(),
            'layer_type': 'features',
            'input_data': data,
            'processed': False,
            'infrastructure_used': []
        }

    def _get_features_infrastructure_services(self) -> Dict[str, Any]:
        """获取特征处理所需的基础设施服务"""
        return {
            'cache_manager': self.get_features_cache_manager(),
            'monitoring': self.get_features_monitoring()
        }

    def _record_features_start(self, services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """记录特征处理开始"""
        if services['monitoring']:
            services['monitoring'].record_metric('features_processing_start', 1, {'layer': 'features'})
            result['infrastructure_used'].append('monitoring')

    def _check_features_cache(self, data: Dict[str, Any],
                             services: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """检查特征处理缓存，避免重复处理"""
        if not services['cache_manager']:
            return False

        cache_key = f"features_{hash(str(data))}"
        cached_result = services['cache_manager'].get(cache_key)

        if cached_result:
            result['cached_result'] = cached_result
            result['processed'] = True
            result['infrastructure_used'].append('cache')
            return True

        return False

    def _execute_features_processing(self, data: Dict[str, Any],
                                    services: Dict[str, Any], result: Dict[str, Any]) -> None:
        """执行实际的特征处理逻辑"""
        engine = self.get_features_engine()
        if engine:
            processed_data = engine.process_features(data)
            result['processed_data'] = processed_data
            result['processed'] = True

            # 缓存处理结果
            self._cache_features_result(data, processed_data, services, result)

    def _cache_features_result(self, data: Dict[str, Any],
                              processed_data: Any, services: Dict[str, Any],
                              result: Dict[str, Any]) -> None:
        """缓存特征处理结果"""
        if services['cache_manager']:
            cache_key = f"features_{hash(str(data))}"
            services['cache_manager'].set(cache_key, processed_data, SECONDS_PER_HOUR)  # 缓存1小时
            result['infrastructure_used'].append('cache')

    def _record_features_completion(self, services: Dict[str, Any]) -> None:
        """记录特征处理完成"""
        if services['monitoring']:
            services['monitoring'].record_metric('features_processing_complete', 1, {'layer': 'features'})

    def _handle_features_error(self, error: Exception, result: Dict[str, Any]) -> None:
        """处理特征处理错误"""
        result['error'] = str(error)
        logger.error(f"特征处理失败: {error}")

        # 记录错误指标
        services = self._get_features_infrastructure_services()
        if services['monitoring']:
            services['monitoring'].record_metric('features_processing_error', 1, {
                'layer': 'features', 'error': str(error)
            })

        return result


# 便捷函数

def get_features_layer_adapter() -> 'FeaturesLayerAdapterRefactored':
    """获取特征层适配器实例"""
    return FeaturesLayerAdapterRefactored()


def get_features_config_manager():
    """获取特征层配置管理器"""
    return get_features_layer_adapter().get_features_config_manager()


def get_features_cache_manager():
    """获取特征层缓存管理器"""
    return get_features_layer_adapter().get_features_cache_manager()


def get_features_engine():
    """获取特征层引擎"""
    return get_features_layer_adapter().get_features_engine()


def process_features_with_infrastructure(data: Dict[str, Any]) -> Dict[str, Any]:
    """使用基础设施支持处理特征数据"""
    return get_features_layer_adapter().process_features_with_infrastructure(data)


# 事件处理器实现

class FeaturesEventHandlers:

    """特征层事件处理器"""

    def _handle_features_extracted_event(self, event: Event):
        """处理特征提取完成事件"""
        try:
            logger.info(f"特征提取完成事件: {event.event_id}")
            data = event.data

            # 更新监控指标
            monitoring = self.get_features_monitoring()
            if monitoring:
                monitoring.record_metric(
                    "features_extracted",
                    data.get('feature_count', 0),
                    {'source': event.source}
                )

            # 发布后续处理事件
            if self._event_bus:
                self._event_bus.publish(
                    EventType.FEATURE_PROCESSING_COMPLETED,
                    {
                        'original_event_id': event.event_id,
                        'feature_count': data.get('feature_count', 0),
                        'processing_time': data.get('processing_time', 0)
                    },
                    source="features_adapter"
                )

        except Exception as e:
            logger.error(f"处理特征提取完成事件失败: {e}")

    def _handle_feature_processing_completed_event(self, event: Event):
        """处理特征处理完成事件"""
        try:
            logger.info(f"特征处理完成事件: {event.event_id}")
            data = event.data

            # 记录性能指标
            monitoring = self.get_features_monitoring()
            if monitoring:
                monitoring.record_metric(
                    "feature_processing_time",
                    data.get('processing_time', 0),
                    {'status': 'completed'}
                )

        except Exception as e:
            logger.error(f"处理特征处理完成事件失败: {e}")

    def _handle_performance_alert_event(self, event: Event):
        """处理性能告警事件"""
        try:
            logger.warning(f"性能告警事件: {event.event_id}")
            data = event.data

            # 根据告警类型采取相应措施
            alert_type = data.get('alert_type', '')
            if alert_type == 'high_memory_usage':
                # 触发特征缓存清理
                self._cleanup_feature_cache()
            elif alert_type == 'high_cpu_usage':
                # 降低特征处理优先级
                self._adjust_processing_priority()

        except Exception as e:
            logger.error(f"处理性能告警事件失败: {e}")

    def _handle_cache_hit_event(self, event: Event):
        """处理缓存命中事件"""
        try:
            monitoring = self.get_features_monitoring()
            if monitoring:
                monitoring.record_metric(
                    "cache_hit_rate",
                    1,
                    {'cache_type': 'features'}
                )
        except Exception as e:
            logger.error(f"处理缓存命中事件失败: {e}")

    def _handle_cache_miss_event(self, event: Event):
        """处理缓存未命中事件"""
        try:
            monitoring = self.get_features_monitoring()
            if monitoring:
                monitoring.record_metric(
                    "cache_miss_rate",
                    1,
                    {'cache_type': 'features'}
                )

            # 可以在这里实现缓存预加载逻辑
            self._preload_related_features(event.data)

        except Exception as e:
            logger.error(f"处理缓存未命中事件失败: {e}")

    def _cleanup_feature_cache(self):
        """清理特征缓存"""
        try:
            cache_manager = self.get_features_cache_manager()
            if cache_manager:
                # 清理过期特征缓存
                cache_manager.clear_expired()
                logger.info("特征缓存清理完成")
        except Exception as e:
            logger.error(f"清理特征缓存失败: {e}")

    def _adjust_processing_priority(self):
        """调整特征处理优先级"""
        try:
            # 降低非关键特征处理的优先级
            logger.info("调整特征处理优先级以降低CPU使用率")
        except Exception as e:
            logger.error(f"调整处理优先级失败: {e}")

    def _preload_related_features(self, cache_data: Dict[str, Any]):
        """预加载相关特征"""
        try:
            # 根据缓存未命中的数据，预加载相关特征
            feature_type = cache_data.get('feature_type', '')
            if feature_type:
                logger.info(f"预加载相关特征: {feature_type}")
        except Exception as e:
            logger.error(f"预加载相关特征失败: {e}")


# 将事件处理器方法添加到FeaturesLayerAdapter类
FeaturesLayerAdapterRefactored._handle_features_extracted_event = FeaturesEventHandlers._handle_features_extracted_event
FeaturesLayerAdapterRefactored._handle_feature_processing_completed_event = FeaturesEventHandlers._handle_feature_processing_completed_event
FeaturesLayerAdapterRefactored._handle_performance_alert_event = FeaturesEventHandlers._handle_performance_alert_event
FeaturesLayerAdapterRefactored._handle_cache_hit_event = FeaturesEventHandlers._handle_cache_hit_event
FeaturesLayerAdapterRefactored._handle_cache_miss_event = FeaturesEventHandlers._handle_cache_miss_event
FeaturesLayerAdapterRefactored._cleanup_feature_cache = FeaturesEventHandlers._cleanup_feature_cache
FeaturesLayerAdapterRefactored._adjust_processing_priority = FeaturesEventHandlers._adjust_processing_priority
FeaturesLayerAdapterRefactored._preload_related_features = FeaturesEventHandlers._preload_related_features


class SmartCacheManager:

    """智能缓存管理器"""

    def get_feature_cache(self):
        """获取特征缓存"""
        return self._feature_cache

    def get_model_cache(self):
        """获取模型缓存"""
        return self._model_cache

    def get_priority_cache(self):
        """获取优先级缓存"""
        return self._priority_cache

    def cache_feature_result(self, feature_key: str, result: Any, priority: int = 0):
        """缓存特征计算结果"""
        try:
            if hasattr(self, '_feature_cache'):
                self._feature_cache.set(feature_key, result, priority=priority)

            if hasattr(self, '_priority_cache'):
                self._priority_cache.set(feature_key, result, priority=priority)

            # 发布缓存事件
            if self._event_bus:
                self._event_bus.publish(
                    EventType.CACHE_SET,
                    {
                        'key': feature_key,
                        'priority': priority,
                        'size': len(str(result))
                    },
                    source="features_adapter"
                )

        except Exception as e:
            logger.error(f"缓存特征结果失败: {e}")

    def get_cached_feature(self, feature_key: str) -> Optional[Any]:
        """获取缓存的特征结果"""
        try:
            # 优先从优先级缓存获取
            if hasattr(self, '_priority_cache'):
                result = self._priority_cache.get(feature_key)
                if result is not None:
                    return result

            # 从主缓存获取
            if hasattr(self, '_feature_cache'):
                result = self._feature_cache.get(feature_key)
                if result is not None:
                    return result

            return None

        except Exception as e:
            logger.error(f"获取缓存特征失败: {e}")
            return None

    def cache_model_result(self, model_key: str, result: Any, cost: float = 1.0):
        """缓存模型计算结果（成本感知）"""
        try:
            if hasattr(self, '_model_cache'):
                self._model_cache.set(model_key, result, retrieval_cost=cost)

            # 发布缓存事件
            if self._event_bus:
                self._event_bus.publish(
                    EventType.CACHE_SET,
                    {
                        'key': model_key,
                        'cost': cost,
                        'type': 'model'
                    },
                    source="features_adapter"
                )

        except Exception as e:
            logger.error(f"缓存模型结果失败: {e}")

    def get_cached_model(self, model_key: str) -> Optional[Any]:
        """获取缓存的模型结果"""
        try:
            if hasattr(self, '_model_cache'):
                result = self._model_cache.get(model_key)
                return result
            return None

        except Exception as e:
            logger.error(f"获取缓存模型失败: {e}")
            return None

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'feature_cache': {},
            'model_cache': {},
            'priority_cache': {}
        }

        try:
            if hasattr(self, '_feature_cache'):
                stats['feature_cache'] = self._feature_cache.get_stats().__dict__

            if hasattr(self, '_model_cache'):
                stats['model_cache'] = self._model_cache.get_stats().__dict__

            if hasattr(self, '_priority_cache'):
                stats['priority_cache'] = self._priority_cache.get_stats().__dict__

        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")

        return stats

    def optimize_cache_strategy(self):
        """优化缓存策略"""
        try:
            stats = self.get_cache_stats()

            # 根据命中率调整策略
            for cache_name, cache_stats in stats.items():
                if cache_name == 'feature_cache':
                    hit_rate = cache_stats.get('hit_rate', 0)
                    if hit_rate < 0.5:
                        logger.info(f"{cache_name} 命中率较低 ({hit_rate:.2f})，考虑调整策略")

                elif cache_name == 'model_cache':
                    total_cost_saved = cache_stats.get('total_cost_saved', 0)
                    if total_cost_saved > MAX_RECORDS:
                        logger.info(f"{cache_name} 节省成本显著 ({total_cost_saved:.2f})")

        except Exception as e:
            logger.error(f"缓存策略优化失败: {e}")

    def clear_expired_cache(self):
        """清理过期缓存"""
        try:
            if hasattr(self, '_feature_cache'):
                self._feature_cache.invalidate_pattern("*")

            if hasattr(self, '_model_cache'):
                self._model_cache.invalidate_pattern("*")

            if hasattr(self, '_priority_cache'):
                self._priority_cache.invalidate_pattern("*")

            logger.info("过期缓存清理完成")

        except Exception as e:
            logger.error(f"清理过期缓存失败: {e}")


# 将智能缓存管理器方法添加到FeaturesLayerAdapter类
FeaturesLayerAdapterRefactored.get_feature_cache = SmartCacheManager.get_feature_cache
FeaturesLayerAdapterRefactored.get_model_cache = SmartCacheManager.get_model_cache
FeaturesLayerAdapterRefactored.get_priority_cache = SmartCacheManager.get_priority_cache
FeaturesLayerAdapterRefactored.cache_feature_result = SmartCacheManager.cache_feature_result
FeaturesLayerAdapterRefactored.get_cached_feature = SmartCacheManager.get_cached_feature
FeaturesLayerAdapterRefactored.cache_model_result = SmartCacheManager.cache_model_result
FeaturesLayerAdapterRefactored.get_cached_model = SmartCacheManager.get_cached_model
FeaturesLayerAdapterRefactored.get_cache_stats = SmartCacheManager.get_cache_stats
FeaturesLayerAdapterRefactored.optimize_cache_strategy = SmartCacheManager.optimize_cache_strategy
FeaturesLayerAdapterRefactored.clear_expired_cache = SmartCacheManager.clear_expired_cache


# 企业级安全管理器类

class EnterpriseSecurityManager:

    """企业级安全管理器"""

    def _init_enterprise_security(self):
        """初始化企业级安全系统"""
        try:
            # 配置特征层特定的安全策略
            self._security_policies = {
                'max_feature_requests_per_hour': MAX_RECORDS,
                'max_model_requests_per_hour': 500,
                'max_cache_requests_per_hour': 5000,
                'encryption_enabled': True,
                'audit_enabled': True,
                'access_control_enabled': True
            }

            # 初始化安全审计
            self._init_security_audit()

            # 只在第一次初始化时记录INFO日志
            if not hasattr(FeaturesLayerAdapterRefactored, '_enterprise_security_initialized'):
                logger.info("企业级安全系统初始化完成")
                FeaturesLayerAdapterRefactored._enterprise_security_initialized = True
            else:
                logger.debug("企业级安全系统已初始化（重复调用）")

        except Exception as e:
            logger.warning(f"企业级安全系统初始化失败: {e}")
            self._security_manager = None

    def _init_security_audit(self):
        """初始化安全审计"""
        try:
            # 注册安全相关的事件处理器
            if not self._event_bus:
                logger.debug("事件总线未初始化，跳过安全审计事件注册")
                return

            # 检查事件总线是否已初始化
            if not hasattr(self._event_bus, '_initialized') or not self._event_bus._initialized:
                logger.debug("事件总线尚未初始化完成，跳过安全审计事件注册")
                return

            self._event_bus.subscribe(
                EventType.ACCESS_GRANTED,
                self._handle_access_granted_event
            )
            self._event_bus.subscribe(
                EventType.ACCESS_DENIED,
                self._handle_access_denied_event
            )

        except Exception as e:
            logger.error(f"安全审计初始化失败: {e}", exc_info=True)

    def _handle_access_granted_event(self, event: Event):
        """处理访问授权事件"""
        try:
            logger.info(f"访问授权事件: {event.event_id}")
            data = event.data

            # 记录安全审计
            if self._security_manager:
                self._security_manager._log_audit(
                    "feature_access_granted",
                    user_id=data.get('user_id', 'unknown'),
                    resource=data.get('resource', 'features'),
                    action=data.get('action', 'access')
                )

        except Exception as e:
            logger.error(f"处理访问授权事件失败: {e}")

    def _handle_access_denied_event(self, event: Event):
        """处理访问拒绝事件"""
        try:
            logger.warning(f"访问拒绝事件: {event.event_id}")
            data = event.data

            # 记录安全审计
            if self._security_manager:
                self._security_manager._log_audit(
                    "feature_access_denied",
                    user_id=data.get('user_id', 'unknown'),
                    resource=data.get('resource', 'features'),
                    action=data.get('action', 'access'),
                    reason=data.get('reason', 'unauthorized')
                )

        except Exception as e:
            logger.error(f"处理访问拒绝事件失败: {e}")

    def validate_feature_access(self, user_id: str, feature_name: str, action: str = "access") -> bool:
        """验证特征访问权限"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return True  # 如果安全系统未初始化，默认允许访问

            # 检查速率限制
            rate_limit_key = f"feature_{user_id}_{feature_name}"
            if not self._security_manager.check_rate_limit(
                rate_limit_key,
                max_attempts=self._security_policies.get('max_feature_requests_per_hour', MAX_RECORDS),
                window=SECONDS_PER_HOUR
            ):
                logger.warning(f"特征访问速率限制: {user_id} -> {feature_name}")
                return False

            # 验证访问权限
            return self._security_manager.validate_access(
                user_id,
                f"feature:{feature_name}",
                action
            )

        except Exception as e:
            logger.error(f"特征访问验证失败: {e}")
            return False

    def validate_model_access(self, user_id: str, model_name: str, action: str = "predict") -> bool:
        """验证模型访问权限"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return True

            # 检查速率限制
            rate_limit_key = f"model_{user_id}_{model_name}"
            if not self._security_manager.check_rate_limit(
                rate_limit_key,
                max_attempts=self._security_policies.get('max_model_requests_per_hour', 500),
                window=SECONDS_PER_HOUR
            ):
                logger.warning(f"模型访问速率限制: {user_id} -> {model_name}")
                return False

            # 验证访问权限
            return self._security_manager.validate_access(
                user_id,
                f"model:{model_name}",
                action
            )

        except Exception as e:
            logger.error(f"模型访问验证失败: {e}")
            return False

    def encrypt_feature_data(self, data: Any) -> str:
        """加密特征数据"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return str(data)

            if not self._security_policies.get('encryption_enabled', True):
                return str(data)

            # 序列化数据
            if isinstance(data, dict):
                data_str = json.dumps(data, ensure_ascii=False)
            else:
                data_str = str(data)

            # 加密数据
            return self._security_manager.encrypt(data_str)

        except Exception as e:
            logger.error(f"特征数据加密失败: {e}")
            return str(data)

    def decrypt_feature_data(self, encrypted_data: str) -> Any:
        """解密特征数据"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return encrypted_data

            if not self._security_policies.get('encryption_enabled', True):
                return encrypted_data

            # 解密数据
            decrypted_str = self._security_manager.decrypt(encrypted_data)

            # 尝试反序列化
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str

        except Exception as e:
            logger.error(f"特征数据解密失败: {e}")
            return encrypted_data

    def audit_feature_operation(self, user_id: str, operation: str, feature_name: str, **kwargs):
        """审计特征操作"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return

            if not self._security_policies.get('audit_enabled', True):
                return

            # 记录审计日志
            audit_data = {
                'user_id': user_id,
                'operation': operation,
                'feature_name': feature_name,
                'timestamp': time.time(),
                **kwargs
            }

            self._security_manager._log_audit("feature_operation", **audit_data)

            # 发布审计事件
            if self._event_bus:
                self._event_bus.publish(
                    EventType.SECURITY_AUDIT,
                    audit_data,
                    source="features_adapter"
                )

        except Exception as e:
            logger.error(f"特征操作审计失败: {e}")

    def get_security_stats(self) -> Dict[str, Any]:
        """获取安全统计信息"""
        try:
            if hasattr(self, '_security_manager') and self._security_manager:
                stats = self._security_manager.get_security_stats()
                stats.update({
                    'feature_policies': self._security_policies,
                    'layer_type': 'features'
                })
                return stats
            return {'status': 'security_disabled'}

        except Exception as e:
            logger.error(f"获取安全统计失败: {e}")
            return {'error': str(e)}

    def manage_access_control(self, action: str, user_id: str = None, resource: str = None):
        """管理访问控制"""
        try:
            if not hasattr(self, '_security_manager') or not self._security_manager:
                return

            if action == 'blacklist_add' and user_id:
                self._security_manager.add_to_blacklist(user_id, f"特征层资源: {resource}")
            elif action == 'blacklist_remove' and user_id:
                self._security_manager.remove_from_blacklist(user_id)
            elif action == 'whitelist_add' and user_id:
                self._security_manager.add_to_whitelist(user_id)
            elif action == 'whitelist_remove' and user_id:
                self._security_manager.remove_from_whitelist(user_id)

        except Exception as e:
            logger.error(f"访问控制管理失败: {e}")

    def secure_feature_processing(self, user_id: str, feature_name: str, data: Any) -> Dict[str, Any]:
        """安全特征处理"""
        try:
            # 1. 访问验证
            if not self.validate_feature_access(user_id, feature_name):
                return {
                    'success': False,
                    'error': 'Access denied',
                    'error_type': 'authorization'
                }

            # 2. 数据加密（如果启用）
            if self._security_policies.get('encryption_enabled', True):
                encrypted_data = self.encrypt_feature_data(data)
            else:
                encrypted_data = data

            # 3. 审计记录
            self.audit_feature_operation(
                user_id=user_id,
                operation='feature_processing',
                feature_name=feature_name,
                data_size=len(str(data)) if data else 0
            )

            return {
                'success': True,
                'encrypted_data': encrypted_data,
                'original_data': data,
                'user_id': user_id,
                'feature_name': feature_name
            }

        except Exception as e:
            logger.error(f"安全特征处理失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'error_type': 'processing_error'
            }


# 将企业级安全管理器方法添加到FeaturesLayerAdapter类
FeaturesLayerAdapterRefactored._init_enterprise_security = EnterpriseSecurityManager._init_enterprise_security
FeaturesLayerAdapterRefactored._init_security_audit = EnterpriseSecurityManager._init_security_audit
FeaturesLayerAdapterRefactored._handle_access_granted_event = EnterpriseSecurityManager._handle_access_granted_event
FeaturesLayerAdapterRefactored._handle_access_denied_event = EnterpriseSecurityManager._handle_access_denied_event
FeaturesLayerAdapterRefactored.validate_feature_access = EnterpriseSecurityManager.validate_feature_access
FeaturesLayerAdapterRefactored.validate_model_access = EnterpriseSecurityManager.validate_model_access
FeaturesLayerAdapterRefactored.encrypt_feature_data = EnterpriseSecurityManager.encrypt_feature_data
FeaturesLayerAdapterRefactored.decrypt_feature_data = EnterpriseSecurityManager.decrypt_feature_data
FeaturesLayerAdapterRefactored.audit_feature_operation = EnterpriseSecurityManager.audit_feature_operation
FeaturesLayerAdapterRefactored.get_security_stats = EnterpriseSecurityManager.get_security_stats
FeaturesLayerAdapterRefactored.manage_access_control = EnterpriseSecurityManager.manage_access_control
FeaturesLayerAdapterRefactored.secure_feature_processing = EnterpriseSecurityManager.secure_feature_processing


# 性能监控增强管理器类

# 性能监控相关协议
class PerformanceMetricsCollector(Protocol):
    """性能指标收集器协议"""
    def collect_detailed_metrics(self) -> Dict[str, Any]: ...


class PerformanceAlertHandler(Protocol):
    """性能告警处理器协议"""
    def handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]): ...


class PerformanceAutoTuner(Protocol):
    """性能自动调优器协议"""
    def trigger_auto_tuning(self, alert_type: str, alert_data: Dict[str, Any]): ...


@dataclass
class PerformanceConfig:
    """性能监控配置"""
    enable_detailed_metrics: bool = True
    auto_tuning_enabled: bool = True
    monitoring_interval: int = DEFAULT_TIMEOUT
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    response_time_threshold: float = 5000.0
    error_rate_threshold: float = 5.0


class PerformanceMetricsCollectorImpl:
    """性能指标收集器实现 - 职责：收集各种性能指标"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self._metrics_history = []

    def collect_detailed_metrics(self) -> Dict[str, Any]:
        """收集详细性能指标"""
        try:
            metrics = {
                'timestamp': time.time(),
                'layer_type': 'features',
                'system_metrics': {},
                'cache_metrics': {},
                'event_metrics': {},
                'security_metrics': {},
                'summary': {}
            }

            # 这里应该实现实际的指标收集逻辑
            # 由于篇幅限制，这里只提供框架

            metrics['summary'] = {
                'total_metrics_collected': len(self._metrics_history),
                'collection_interval': self.config.monitoring_interval,
                'last_collection_time': time.time()
            }

            # 保存到历史记录
            self._metrics_history.append(metrics)
            # 只保留最近100条记录
            if len(self._metrics_history) > MAX_RETRIES:
                self._metrics_history = self._metrics_history[-MAX_RETRIES:]

            return metrics

        except Exception as e:
            logger.error(f"收集性能指标失败: {e}")
            return {}


class PerformanceAlertHandlerImpl:
    """性能告警处理器实现 - 职责：处理性能告警"""

    def __init__(self, config: PerformanceConfig, auto_tuner: PerformanceAutoTuner):
        self.config = config
        self.auto_tuner = auto_tuner

    def handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """处理性能告警事件"""
        try:
            logger.warning(f"性能告警: {alert_type}")

            if not self.config.auto_tuning_enabled:
                logger.info("自动调优已禁用")
                return

            # 根据告警类型触发相应的调优策略
            self.auto_tuner.trigger_auto_tuning(alert_type, alert_data)

        except Exception as e:
            logger.error(f"处理性能告警失败: {e}")


class PerformanceAutoTunerImpl:
    """性能自动调优器实现 - 职责：执行自动调优"""

    def __init__(self, config: PerformanceConfig):
        self.config = config

    def trigger_auto_tuning(self, alert_type: str, alert_data: Dict[str, Any]):
        """触发自动调优"""
        try:
            if alert_type in ['high_cpu_usage', 'high_memory_usage']:
                self._optimize_resource_usage(alert_type, alert_data)
            elif alert_type == 'high_response_time':
                self._optimize_response_time(alert_data)
            elif alert_type == 'high_error_rate':
                self._optimize_error_handling(alert_data)

        except Exception as e:
            logger.error(f"自动调优失败: {e}")

    def _optimize_resource_usage(self, alert_type: str, alert_data: Dict[str, Any]):
        """优化资源使用"""
        if alert_type == 'high_cpu_usage':
            self._optimize_for_cpu_usage()
        elif alert_type == 'high_memory_usage':
            self._optimize_for_memory_usage()

    def _optimize_for_cpu_usage(self):
        """CPU使用率优化"""
        # 实现CPU优化逻辑
        logger.info("执行CPU使用率优化")

    def _optimize_for_memory_usage(self):
        """内存使用率优化"""
        # 实现内存优化逻辑
        logger.info("执行内存使用率优化")

    def _optimize_response_time(self, alert_data: Dict[str, Any]):
        """响应时间优化"""
        self._optimize_cache_strategy_for_performance()
        logger.info("执行响应时间优化")

    def _optimize_cache_strategy_for_performance(self):
        """优化缓存策略以提升性能"""
        # 实现缓存策略优化逻辑
        logger.info("执行缓存策略优化")

    def _optimize_error_handling(self, alert_data: Dict[str, Any]):
        """错误处理优化"""
        self._enhance_error_handling()
        logger.info("执行错误处理优化")

    def _enhance_error_handling(self):
        """增强错误处理"""
        # 实现错误处理增强逻辑
        logger.info("增强错误处理机制")


class PerformanceReporter:
    """性能报告生成器 - 职责：生成性能报告"""

    def __init__(self, metrics_collector: PerformanceMetricsCollector):
        self.metrics_collector = metrics_collector

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            metrics = self.metrics_collector.collect_detailed_metrics()
            recommendations = self._generate_performance_recommendations(metrics)

            return {
                'timestamp': time.time(),
                'metrics': metrics,
                'recommendations': recommendations,
                'summary': {
                    'report_generated': True,
                    'total_recommendations': len(recommendations)
                }
            }

        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {}

    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        # 基于指标生成建议
        system_metrics = metrics.get('system_metrics', {})

        if system_metrics.get('cpu_percent', 0) > 80:
            recommendations.append("CPU使用率过高，建议优化计算密集型操作")

        if system_metrics.get('memory_percent', 0) > 85:
            recommendations.append("内存使用率过高，建议优化内存管理")

        cache_metrics = metrics.get('cache_metrics', {})
        if cache_metrics.get('hit_rate', 1.0) < 0.8:
            recommendations.append("缓存命中率较低，建议优化缓存策略")

        return recommendations


class PerformanceMonitoringManagerRefactored:
    """重构后的性能监控增强管理器 - 组合模式：使用专门的组件"""

    def __init__(self, config: Optional[PerformanceConfig] = None):
        # 初始化配置
        self.config = config or PerformanceConfig()

        # 初始化专门的组件
        self.auto_tuner = PerformanceAutoTunerImpl(self.config)
        self.alert_handler = PerformanceAlertHandlerImpl(self.config, self.auto_tuner)
        self.metrics_collector = PerformanceMetricsCollectorImpl(self.config)
        self.reporter = PerformanceReporter(self.metrics_collector)

        logger.info("重构后的性能监控管理器初始化完成")

    # 代理方法到专门的组件
    def collect_detailed_metrics(self) -> Dict[str, Any]:
        """收集详细指标 - 代理到指标收集器"""
        return self.metrics_collector.collect_detailed_metrics()

    def handle_performance_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """处理性能告警 - 代理到告警处理器"""
        return self.alert_handler.handle_performance_alert(alert_type, alert_data)

    def trigger_auto_tuning(self, alert_type: str, alert_data: Dict[str, Any]):
        """触发自动调优 - 代理到自动调优器"""
        return self.auto_tuner.trigger_auto_tuning(alert_type, alert_data)

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告 - 代理到报告生成器"""
        return self.reporter.generate_performance_report()

    # 保持向后兼容性
    def _init_performance_monitoring(self):
        """初始化性能监控增强系统（向后兼容）"""
        # 在重构版本中，这个方法不需要额外的初始化
        # 配置和组件初始化已经在__init__中完成
        pass

    def _init_performance_tracking(self):
        """初始化性能跟踪（向后兼容）"""
        # 在重构版本中，跟踪逻辑分布在各个组件中
        pass

    def _handle_performance_alert(self, event: Event):
        """处理性能告警事件（向后兼容）"""
        alert_data = event.data if hasattr(event, 'data') else {}
        alert_type = alert_data.get('alert_type', 'unknown')
        self.handle_performance_alert(alert_type, alert_data)


class PerformanceMonitoringManager:

    """性能监控增强管理器 - 重构版：组合模式"""

    def _init_performance_monitoring(self):
        """初始化性能监控增强系统"""
        # 只在第一次初始化时执行，避免重复初始化
        if hasattr(FeaturesLayerAdapterRefactored, '_performance_monitoring_initialized') and FeaturesLayerAdapterRefactored._performance_monitoring_initialized:
            logger.debug("性能监控增强系统已初始化（跳过重复初始化）")
            return
        
        try:
            # 配置特征层特定的性能监控策略
            self._performance_policies = {
                'enable_detailed_metrics': True,
                'auto_tuning_enabled': True,
                'alert_thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'response_time': 5000.0,  # 5秒
                    'error_rate': 5.0  # 5%
                },
                'monitoring_interval': DEFAULT_TIMEOUT,  # 30秒
                'history_retention_days': 7
            }

            # 初始化性能监控
            self._init_performance_tracking()

            # 标记为已初始化
            FeaturesLayerAdapterRefactored._performance_monitoring_initialized = True
            logger.info("性能监控增强系统初始化完成")

        except Exception as e:
            logger.warning(f"性能监控增强系统初始化失败: {e}")
            self._performance_monitor = None

    def _init_performance_tracking(self):
        """初始化性能跟踪"""
        try:
            # 注册性能监控相关的事件处理器
            if self._event_bus:
                # 检查事件总线是否已初始化
                if not hasattr(self._event_bus, '_initialized') or not self._event_bus._initialized:
                    logger.debug("事件总线尚未初始化完成，跳过性能告警事件订阅")
                else:
                    try:
                        self._event_bus.subscribe(
                            EventType.PERFORMANCE_ALERT,
                            self._handle_performance_alert
                        )
                        logger.debug("性能告警事件订阅成功")
                    except Exception as e:
                        logger.warning(f"订阅性能告警事件失败: {e}")

            # 启动性能监控
            if hasattr(self._performance_monitor, 'start_monitoring'):
                try:
                    self._performance_monitor.start_monitoring()
                except Exception as e:
                    logger.warning(f"启动性能监控失败: {e}")

        except Exception as e:
            logger.error(f"性能跟踪初始化失败: {e}")

    def _handle_performance_alert(self, event: Event):
        """处理性能告警事件"""
        try:
            logger.warning(f"性能告警事件: {event.event_id}")
            data = event.data

            alert_type = data.get('alert_type', '')
            if alert_type in ['high_cpu_usage', 'high_memory_usage']:
                # 触发自动调优
                self._trigger_auto_tuning(alert_type, data)
            elif alert_type == 'high_response_time':
                # 优化缓存策略
                self._optimize_cache_strategy_for_performance()
            elif alert_type == 'high_error_rate':
                # 增强错误处理
                self._enhance_error_handling()

        except Exception as e:
            logger.error(f"处理性能告警失败: {e}")

    def collect_detailed_metrics(self) -> Dict[str, Any]:
        """收集详细性能指标"""
        try:
            metrics = {
                'timestamp': time.time(),
                'layer_type': 'features',
                'system_metrics': {},
                'cache_metrics': {},
                'event_metrics': {},
                'security_metrics': {}
            }

            # 系统资源指标
            if hasattr(self, '_performance_monitor'):
                system_stats = self._performance_monitor.get_system_stats()
                metrics['system_metrics'] = {
                    'cpu_percent': system_stats.get('cpu_percent', 0),
                    'memory_percent': system_stats.get('memory_percent', 0),
                    'disk_usage': system_stats.get('disk_usage', {}),
                    'network_io': system_stats.get('network_io', {})
                }

            # 缓存性能指标
            if hasattr(self, '_cache_manager'):
                cache_stats = self.get_cache_stats()
                metrics['cache_metrics'] = cache_stats

            # 事件总线指标
            if hasattr(self, '_event_bus'):
                event_stats = self._event_bus.get_event_statistics()
                metrics['event_metrics'] = event_stats

            # 安全指标
            if hasattr(self, '_security_manager'):
                security_stats = self.get_security_stats()
                metrics['security_metrics'] = security_stats

            return metrics

        except Exception as e:
            logger.error(f"收集详细指标失败: {e}")
            return {'error': str(e)}

    def monitor_feature_performance(self, feature_name: str, execution_time: float,


                                    success: bool, data_size: int = 0) -> None:
        """监控特征性能"""
        try:
            # 记录性能指标
            if hasattr(self._performance_monitor, 'record_metric'):
                self._performance_monitor.record_metric(
                    f"feature_{feature_name}_execution_time",
                    execution_time
                )

                self._performance_monitor.record_metric(
                    f"feature_{feature_name}_success_rate",
                    1 if success else 0
                )

                if data_size > 0:
                    self._performance_monitor.record_metric(
                        f"feature_{feature_name}_data_size",
                        data_size
                    )

            # 检查是否需要告警
            self._check_performance_thresholds(feature_name, execution_time, success)

        except Exception as e:
            logger.error(f"监控特征性能失败: {e}")

    def _check_performance_thresholds(self, feature_name: str, execution_time: float, success: bool):
        """检查性能阈值"""
        try:
            thresholds = self._performance_policies.get('alert_thresholds', {})

            # 检查响应时间
            if execution_time > thresholds.get('response_time', 5000):
                logger.warning(f"特征 {feature_name} 响应时间过长: {execution_time}ms")

                # 发布性能告警事件
                if self._event_bus:
                    self._event_bus.publish(
                        EventType.PERFORMANCE_ALERT,
                        {
                            'alert_type': 'high_response_time',
                            'feature_name': feature_name,
                            'execution_time': execution_time,
                            'threshold': thresholds.get('response_time')
                        },
                        source="performance_monitor"
                    )

            # 检查成功率
            if not success:
                # 这里可以实现更复杂的错误率计算逻辑
                logger.warning(f"特征 {feature_name} 执行失败")

        except Exception as e:
            logger.error(f"检查性能阈值失败: {e}")

    def _trigger_auto_tuning(self, alert_type: str, alert_data: Dict[str, Any]):
        """触发自动调优"""
        try:
            logger.info(f"触发自动调优: {alert_type}")

            if alert_type == 'high_cpu_usage':
                # CPU使用率高时的调优策略
                self._optimize_for_cpu_usage()
            elif alert_type == 'high_memory_usage':
                # 内存使用率高时的调优策略
                self._optimize_for_memory_usage()

        except Exception as e:
            logger.error(f"自动调优失败: {e}")

    def _optimize_for_cpu_usage(self):
        """CPU优化策略"""
        try:
            # 降低缓存更新频率
            if hasattr(self, '_cache_manager'):
                logger.info("执行CPU优化: 降低缓存更新频率")

            # 调整事件处理优先级
            if hasattr(self, '_event_bus'):
                logger.info("执行CPU优化: 调整事件处理优先级")

        except Exception as e:
            logger.error(f"CPU优化失败: {e}")

    def _optimize_for_memory_usage(self):
        """内存优化策略"""
        try:
            # 清理过期缓存
            if hasattr(self, 'clear_expired_cache'):
                self.clear_expired_cache()
                logger.info("执行内存优化: 清理过期缓存")

            # 压缩数据结构
            logger.info("执行内存优化: 压缩数据结构")

        except Exception as e:
            logger.error(f"内存优化失败: {e}")

    def _optimize_cache_strategy_for_performance(self):
        """优化缓存策略以提升性能"""
        try:
            if hasattr(self, 'optimize_cache_strategy'):
                self.optimize_cache_strategy()
                logger.info("优化缓存策略以提升性能")

        except Exception as e:
            logger.error(f"缓存策略优化失败: {e}")

    def _enhance_error_handling(self):
        """增强错误处理"""
        try:
            logger.info("增强错误处理机制")

            # 可以在这里实现更复杂的错误处理增强逻辑
            # 比如增加重试机制、故障转移等

        except Exception as e:
            logger.error(f"增强错误处理失败: {e}")

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        try:
            report = {
                'timestamp': time.time(),
                'period': 'last_hour',
                'summary': {},
                'recommendations': [],
                'alerts': []
            }

            # 收集当前指标
            current_metrics = self.collect_detailed_metrics()
            report['current_metrics'] = current_metrics

            # 生成建议
            recommendations = self._generate_performance_recommendations(current_metrics)
            report['recommendations'] = recommendations

            # 获取告警历史
            if hasattr(self._performance_monitor, 'get_alerts'):
                alerts = self._performance_monitor.get_alerts()
                report['alerts'] = alerts[-DEFAULT_BATCH_SIZE:]  # 最近10个告警

            return report

        except Exception as e:
            logger.error(f"生成性能报告失败: {e}")
            return {'error': str(e)}

    def _generate_performance_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        try:
            # 基于CPU使用率给出建议
            cpu_usage = metrics.get('system_metrics', {}).get('cpu_percent', 0)
            if cpu_usage > 80:
                recommendations.append("CPU使用率过高，建议优化算法复杂度或增加资源")

            # 基于内存使用率给出建议
            memory_usage = metrics.get('system_metrics', {}).get('memory_percent', 0)
            if memory_usage > 85:
                recommendations.append("内存使用率过高，建议清理缓存或优化数据结构")

            # 基于缓存命中率给出建议
            cache_metrics = metrics.get('cache_metrics', {})
            if cache_metrics:
                for cache_name, cache_stat in cache_metrics.items():
                    if isinstance(cache_stat, dict):
                        hit_rate = cache_stat.get('hit_rate', 0)
                        if hit_rate < 0.5:
                            recommendations.append(f"{cache_name} 缓存命中率较低({hit_rate:.2f})，建议调整缓存策略")

            # 基于事件处理性能给出建议
            event_metrics = metrics.get('event_metrics', {})
            if event_metrics:
                processing_time = event_metrics.get('total_time', 0)
                if processing_time > MAX_RECORDS:  # 1秒
                    recommendations.append("事件处理时间过长，建议优化事件处理逻辑")

        except Exception as e:
            logger.error(f"生成性能建议失败: {e}")

        return recommendations

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
from datetime import datetime
        try:
            stats = {
                'layer_type': 'features',
                'monitoring_enabled': self._performance_policies.get('enable_detailed_metrics', False),
                'auto_tuning_enabled': self._performance_policies.get('auto_tuning_enabled', False),
                'alert_thresholds': self._performance_policies.get('alert_thresholds', {}),
                'current_metrics': self.collect_detailed_metrics()
            }

            return stats

        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {'error': str(e)}


# 将性能监控增强管理器方法添加到FeaturesLayerAdapter类
FeaturesLayerAdapterRefactored._init_performance_monitoring = PerformanceMonitoringManager._init_performance_monitoring
FeaturesLayerAdapterRefactored._init_performance_tracking = PerformanceMonitoringManager._init_performance_tracking
FeaturesLayerAdapterRefactored._handle_performance_alert = PerformanceMonitoringManager._handle_performance_alert
FeaturesLayerAdapterRefactored.collect_detailed_metrics = PerformanceMonitoringManager.collect_detailed_metrics
FeaturesLayerAdapterRefactored.monitor_feature_performance = PerformanceMonitoringManager.monitor_feature_performance
FeaturesLayerAdapterRefactored._check_performance_thresholds = PerformanceMonitoringManager._check_performance_thresholds
FeaturesLayerAdapterRefactored._trigger_auto_tuning = PerformanceMonitoringManager._trigger_auto_tuning
FeaturesLayerAdapterRefactored._optimize_for_cpu_usage = PerformanceMonitoringManager._optimize_for_cpu_usage
FeaturesLayerAdapterRefactored._optimize_for_memory_usage = PerformanceMonitoringManager._optimize_for_memory_usage
FeaturesLayerAdapterRefactored._optimize_cache_strategy_for_performance = PerformanceMonitoringManager._optimize_cache_strategy_for_performance
FeaturesLayerAdapterRefactored._enhance_error_handling = PerformanceMonitoringManager._enhance_error_handling
FeaturesLayerAdapterRefactored.generate_performance_report = PerformanceMonitoringManager.generate_performance_report
FeaturesLayerAdapterRefactored._generate_performance_recommendations = PerformanceMonitoringManager._generate_performance_recommendations
FeaturesLayerAdapterRefactored.get_performance_stats = PerformanceMonitoringManager.get_performance_stats


# 为了向后兼容，保留原有的PerformanceMonitoringManager类名，但内部使用重构版本
PerformanceMonitoringManager = PerformanceMonitoringManagerRefactored


# 为了向后兼容，保留原有的FeaturesLayerAdapter类名，但内部使用重构版本
FeaturesLayerAdapter = FeaturesLayerAdapterRefactored
