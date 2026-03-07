import logging
#!/usr/bin/env python3
"""
RQA2025 数据层基础设施适配器 - 统一架构版本

专门为数据层提供基础设施服务访问接口，
基于统一适配器架构，实现数据层的特定需求。
支持业务流程驱动的数据处理和分析。
"""

from typing import Any
from ...core.business_adapters import BaseBusinessAdapter, BusinessLayerType
from src.services.infrastructure.service_container import ServiceConfig

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = logging.getLogger(__name__)


class DataLayerAdapter(BaseBusinessAdapter):

    """数据层适配器 - 统一架构版本

    基于统一适配器架构的数据层实现，支持：
    1. 数据流处理和缓存管理
    2. 多源数据适配器集成
    3. 实时数据质量监控
    4. 业务流程驱动的数据服务
    """

    def __init__(self):

        super().__init__(BusinessLayerType.DATA)
        self._init_data_layer_services()
        # 数据层特定的初始化将在_init_layer_specific_services中完成

    def _init_data_layer_services(self):
        """初始化数据层特定的服务"""
        self._data_flow_manager = None
        self._cache_integration_manager = None

    def _init_service_configs(self):
        """初始化数据层特定的服务配置"""
        # 首先调用父类的初始化方法
        super()._init_service_configs()

        # 添加数据层特有的event_bus服务
        self.service_configs['event_bus'] = ServiceConfig(
            name='event_bus',
            primary_factory=self._create_event_bus,
            fallback_factory=self._create_fallback_event_bus,
            health_check_interval=60  # 数据层对事件总线要求较高
        )

    def _init_layer_specific_services(self):
        """初始化数据层特定的服务"""
        # 数据层特有的数据流管理器
        try:
            from .data import DataFlowManager, CacheIntegrationManager
            self._data_flow_manager = DataFlowManager()
            self._cache_integration_manager = CacheIntegrationManager()
            logger.info("数据层特定服务初始化完成")
        except Exception as e:
            logger.warning(f"数据层特定服务初始化失败: {e}")
            self._data_flow_manager = None
            self._cache_integration_manager = None

    def _create_cache_manager(self):
        """创建缓存管理器"""
        try:
            from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
            return UnifiedCacheManager()
        except ImportError:
            # 兼容性处理
            from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
            return UnifiedCacheManager()

    def _create_config_manager(self):
        """创建配置管理器"""
        try:
            from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
            return UnifiedConfigManager()
        except ImportError:
            # 兼容性处理 - 使用简化版本
            logger.warning("UnifiedConfigManager不可用，使用简化配置管理器")
            return None

    def _create_logger(self):
        """创建日志器"""
        return get_unified_logger('data_layer')

    def _create_monitoring(self):
        """创建监控服务"""
        from src.infrastructure.monitoring.unified_monitoring import UnifiedMonitoring
        return UnifiedMonitoring()

    def _create_event_bus(self):
        """创建事件总线"""
        try:
            from src.event_bus.event_bus import EventBus
            return EventBus()
        except ImportError:
            # 兼容性处理
            from src.event_bus.bus_components import EventBus
            return EventBus()

    def _create_health_checker(self):
        """创建健康检查器"""
        from src.infrastructure.health.components.enhanced_health_checker import EnhancedHealthChecker
        return EnhancedHealthChecker()

    def _create_fallback_cache(self):
        """创建降级缓存"""
        try:
            from .fallback_services import get_fallback_cache_manager
            return get_fallback_cache_manager()
        except ImportError:
            # 使用基类方法
            return super()._create_fallback_cache()

    def _create_fallback_config(self):
        """创建降级配置"""
        try:
            from .fallback_services import get_fallback_config_manager
            return get_fallback_config_manager()
        except ImportError:
            # 使用基类方法
            return super()._create_fallback_config()

    def _create_fallback_logger(self):
        """创建降级日志器"""
        try:
            from .fallback_services import get_fallback_logger
            return get_fallback_logger()
        except ImportError:
            # 使用基类方法
            return super()._create_fallback_logger()

    def _create_fallback_monitoring(self):
        """创建降级监控"""
        try:
            from .fallback_services import get_fallback_monitoring
            return get_fallback_monitoring()
        except ImportError:
            # 使用基类方法
            return super()._create_fallback_monitoring()

    def _create_fallback_event_bus(self):
        """创建降级事件总线"""
        # 简单的事件总线降级实现

        class FallbackEventBus:

            def publish(self, event, data=None):

                logger.debug(f"Fallback EventBus: {event} - {data}")

            def subscribe(self, event, handler):

                pass
        return FallbackEventBus()

    def _create_fallback_health_checker(self):
        """创建降级健康检查器"""
        try:
            from .fallback_services import get_fallback_health_checker
            return get_fallback_health_checker()
        except ImportError:
            # 使用基类方法
            return super()._create_fallback_health_checker()

    # ===========================================
    # 数据层特定功能扩展
    # ===========================================

    def get_data_flow_manager(self):
        """获取数据流管理器"""
        return self._data_flow_manager

    def get_cache_integration_manager(self):
        """获取缓存集成管理器"""
        return self._cache_integration_manager

    def process_data_stream(self, data_stream: Any) -> Any:
        """处理数据流"""
        if self._data_flow_manager:
            return self._data_flow_manager.process_stream(data_stream)
        else:
            logger.warning("数据流管理器不可用")
            return data_stream

    def integrate_cache(self, data: Any, cache_key: str) -> Any:
        """集成缓存处理"""
        if self._cache_integration_manager:
            return self._cache_integration_manager.integrate_cache(data, cache_key)
        else:
            logger.warning("缓存集成管理器不可用")
            return data
    # ===========================================
    # 兼容性接口 - 向后兼容
    # ===========================================

    def get_data_cache_bridge(self):
        """获取数据缓存桥接器 - 兼容性接口"""
        return self.get_infrastructure_services().get('cache_manager')

    def get_data_config_bridge(self):
        """获取数据配置桥接器 - 兼容性接口"""
        return self.get_infrastructure_services().get('config_manager')

    def get_data_monitoring_bridge(self):
        """获取数据监控桥接器 - 兼容性接口"""
        return self.get_infrastructure_services().get('monitoring')

    def get_data_health_bridge(self):
        """获取数据健康检查桥接器 - 兼容性接口"""
        return self.get_infrastructure_services().get('health_checker')

    def get_monitoring(self):
        """获取监控服务 - 兼容性接口"""
        return self.get_infrastructure_services().get('monitoring')


# 便捷函数

def get_data_layer_adapter() -> DataLayerAdapter:
    """获取数据层适配器实例"""
    from .business_adapters import get_data_adapter
    return get_data_adapter()


def get_data_cache_bridge():
    """获取数据缓存桥接器"""
    return get_data_layer_adapter().get_data_cache_bridge()


def get_data_config_bridge():
    """获取数据配置桥接器"""
    return get_data_layer_adapter().get_data_config_bridge()


def get_data_monitoring_bridge():
    """获取数据监控桥接器"""
    return get_data_layer_adapter().get_data_monitoring_bridge()


def get_data_infrastructure_manager():
    """获取数据层基础设施集成管理器"""
    return get_data_layer_adapter().get_data_infrastructure_manager()
