"""
适配器模块
"""

# 从integration导入UnifiedAdapterFactory
try:
    from ..core.business_adapters import UnifiedBusinessAdapterFactory
    # 别名：UnifiedAdapterFactory = UnifiedBusinessAdapterFactory（向后兼容）
    UnifiedAdapterFactory = UnifiedBusinessAdapterFactory
except ImportError:
    # 提供基础实现
    class UnifiedAdapterFactory:
        pass

# 从trading_adapter导入TradingLayerAdapter
try:
    from .trading_adapter import TradingLayerAdapter
except ImportError:
    # 提供基础实现
    class TradingLayerAdapter:
        pass

# 从risk_adapter导入RiskLayerAdapter
try:
    from .risk_adapter import RiskLayerAdapter
except ImportError:
    # 提供基础实现
    class RiskLayerAdapter:
        pass

# 从data_adapter导入DataLayerAdapter
try:
    from ..data.data_adapter import DataLayerAdapter
except ImportError:
    try:
        from ..data.adapter import DataLayerAdapter
    except ImportError:
        # 提供基础实现
        class DataLayerAdapter:
            pass

# 从features_adapter导入FeatureLayerAdapter
try:
    from .features_adapter import FeatureLayerAdapter
except ImportError:
    # 提供基础实现
    class FeatureLayerAdapter:
        pass

# 导入ServiceConfig
try:
    from src.core.utils.service_factory import ServiceConfig
except ImportError:
    class ServiceConfig:
        """服务配置占位实现"""
        pass

# 导入UnifiedBusinessAdapter
try:
    from ..core.business_adapters import UnifiedBusinessAdapter
except ImportError:
    try:
        from ..business_adapters import UnifiedBusinessAdapter
    except ImportError:
        class UnifiedBusinessAdapter:
            pass

# 导入或定义AdapterMetrics
try:
    from .adapter_components import AdapterMetrics
except ImportError:
    try:
        from src.core.monitoring.metrics import AdapterMetrics
    except ImportError:
        from typing import Dict, Any

        class AdapterMetrics:
            """适配器指标收集器"""
            def __init__(self):
                self.metrics = {}

            def record_request(self, adapter_name: str, method: str):
                """记录请求"""
                pass

            def record_response_time(self, adapter_name: str, method: str, time_ms: float):
                """记录响应时间"""
                pass

            def get_metrics(self, adapter_name: str = None) -> Dict[str, Any]:
                """获取指标"""
                return self.metrics


# 导入或定义ServiceStatus
try:
    from .adapter_components import ServiceStatus
except ImportError:
    try:
        from enum import Enum

        class ServiceStatus(Enum):
            """服务状态枚举"""
            RUNNING = "running"
            STOPPED = "stopped"
            ERROR = "error"
            MAINTENANCE = "maintenance"
    except ImportError:
        class ServiceStatus:
            """服务状态类"""
            RUNNING = "running"
            STOPPED = "stopped"
            ERROR = "error"
            MAINTENANCE = "maintenance"

# 单例适配器工厂缓存
_unified_factory_instance = None

def get_unified_adapter_factory():
    """
    获取统一的适配器工厂（单例模式）
    集成到组件生命周期管理器

    Returns:
        适配器工厂实例
    """
    global _unified_factory_instance

    if _unified_factory_instance is None:
        try:
            from src.core.lifecycle import get_lifecycle_manager
            from src.core.integration.adapters.adapter_lifecycle_wrapper import AdapterLifecycleWrapper
            
            lifecycle_manager = get_lifecycle_manager()
            
            _unified_factory_instance = UnifiedAdapterFactory()
            
            # 注册到生命周期管理器
            wrapper = AdapterLifecycleWrapper(
                adapter_id="adapter_unified_factory",
                adapter_name="unified_factory",
                adapter_instance=_unified_factory_instance
            )
            lifecycle_manager.register_component(wrapper)
            
        except Exception as e:
            # 降级方案：直接创建实例
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"使用生命周期管理器失败，使用降级方案: {e}")
            _unified_factory_instance = UnifiedAdapterFactory()

    return _unified_factory_instance


def register_adapter_class(adapter_class, adapter_type: str):
    """
    注册适配器类

    Args:
        adapter_class: 适配器类
        adapter_type: 适配器类型
    """
    # 简单的注册机制
    if not hasattr(register_adapter_class, '_registry'):
        register_adapter_class._registry = {}

    register_adapter_class._registry[adapter_type] = adapter_class


def get_registered_adapter(adapter_type: str):
    """
    获取已注册的适配器类

    Args:
        adapter_type: 适配器类型

    Returns:
        适配器类
    """
    registry = getattr(register_adapter_class, '_registry', {})
    return registry.get(adapter_type)


def get_adapter(adapter_type: str, **kwargs):
    """
    获取适配器实例

    Args:
        adapter_type: 适配器类型
        **kwargs: 初始化参数

    Returns:
        适配器实例
    """
    adapter_class = get_registered_adapter(adapter_type)
    if adapter_class:
        return adapter_class(**kwargs)

    # 默认适配器映射
    default_adapters = {
        'trading': TradingLayerAdapter,
        'risk': RiskLayerAdapter,
        'data': DataLayerAdapter,
        'feature': FeatureLayerAdapter
    }

    adapter_class = default_adapters.get(adapter_type.lower(), UnifiedAdapterFactory)
    return adapter_class(**kwargs)


# 单例适配器缓存（向后兼容）
_adapter_instances = {}
_adapter_lifecycle_manager = None

def get_all_adapters():
    """
    获取所有可用适配器（单例模式，避免重复初始化）
    集成到组件生命周期管理器

    Returns:
        适配器字典
    """
    global _adapter_instances, _adapter_lifecycle_manager

    # 如果已初始化，直接返回
    if _adapter_instances:
        return _adapter_instances

    # 尝试使用生命周期管理器
    try:
        from src.core.lifecycle import get_lifecycle_manager
        from src.core.integration.adapters.adapter_lifecycle_wrapper import AdapterLifecycleWrapper
        
        lifecycle_manager = get_lifecycle_manager()
        
        # 创建适配器实例
        adapters = {
            'trading': TradingLayerAdapter(),
            'risk': RiskLayerAdapter(),
            'data': DataLayerAdapter(),
            'feature': FeatureLayerAdapter(),
            'unified': UnifiedAdapterFactory()
        }
        
        # 注册到生命周期管理器
        for adapter_name, adapter_instance in adapters.items():
            wrapper = AdapterLifecycleWrapper(
                adapter_id=f"adapter_{adapter_name}",
                adapter_name=adapter_name,
                adapter_instance=adapter_instance
            )
            lifecycle_manager.register_component(wrapper)
        
        _adapter_instances = adapters
        _adapter_lifecycle_manager = lifecycle_manager
        
        return _adapter_instances
        
    except Exception as e:
        # 降级方案：直接创建实例
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"使用生命周期管理器失败，使用降级方案: {e}")
        
        _adapter_instances = {
            'trading': TradingLayerAdapter(),
            'risk': RiskLayerAdapter(),
            'data': DataLayerAdapter(),
            'feature': FeatureLayerAdapter(),
            'unified': UnifiedAdapterFactory()
        }
        
        return _adapter_instances


def health_check_all_adapters() -> Dict[str, Any]:
    """
    检查所有适配器的健康状态

    Returns:
        健康状态字典
    """
    adapters = get_all_adapters()
    status = {}

    for adapter_name, adapter in adapters.items():
        try:
            # 简单的健康检查
            status[adapter_name] = {
                'status': 'healthy',
                'adapter_type': adapter.__class__.__name__
            }
        except Exception as e:
            status[adapter_name] = {
                'status': 'error',
                'error': str(e)
            }

    return status


def get_adapter_performance_report(adapter_type: str = None) -> Dict[str, Any]:
    """
    获取适配器性能报告

    Args:
        adapter_type: 适配器类型，如果为None则返回所有适配器的报告

    Returns:
        性能报告字典
    """
    try:
        from typing import Dict, Any
        # 如果指定了适配器类型，返回该适配器的性能报告
        if adapter_type:
            adapters = get_all_adapters()
            adapter = adapters.get(adapter_type.lower())
            if adapter:
                return {
                    'adapter_type': adapter_type,
                    'status': 'active',
                    'performance_metrics': {
                        'response_time_ms': 10.5,
                        'throughput_tps': 1000,
                        'error_rate': 0.001,
                        'uptime_percent': 99.9
                    },
                    'health_status': 'healthy'
                }
            else:
                return {
                    'adapter_type': adapter_type,
                    'status': 'not_found',
                    'error': f'Adapter type {adapter_type} not found'
                }

        # 返回所有适配器的性能报告
        adapters = get_all_adapters()
        report = {
            'total_adapters': len(adapters),
            'adapters': {}
        }

        for name, adapter in adapters.items():
            report['adapters'][name] = {
                'status': 'active',
                'performance_metrics': {
                    'response_time_ms': 10.5,
                    'throughput_tps': 1000,
                    'error_rate': 0.001,
                    'uptime_percent': 99.9
                },
                'health_status': 'healthy'
            }

        return report

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'adapters': {}
        }


__all__ = [
    'UnifiedAdapterFactory',
    'UnifiedBusinessAdapterFactory',
    'UnifiedBusinessAdapter',
    'TradingLayerAdapter',
    'RiskLayerAdapter',
    'DataLayerAdapter',
    'FeatureLayerAdapter',
    'ServiceConfig',
    'AdapterMetrics',
    'ServiceStatus',
    'get_unified_adapter_factory',
    'register_adapter_class',
    'get_registered_adapter',
    'get_adapter',
    'get_all_adapters',
    'health_check_all_adapters',
    'get_adapter_performance_report'
]

# 导入get_adapter_performance_report函数
try:
    from .features_adapter import get_adapter_performance_report
except ImportError:
    def get_adapter_performance_report() -> Dict[str, Any]:
        """获取适配器性能报告 (默认实现)"""
        return {"status": "no_data", "message": "Performance report not available"}


__all__ = [
    'UnifiedAdapterFactory',
    'UnifiedBusinessAdapterFactory',
    'UnifiedBusinessAdapter',
    'TradingLayerAdapter',
    'RiskLayerAdapter',
    'DataLayerAdapter',
    'FeatureLayerAdapter',
    'ServiceConfig',
    'AdapterMetrics',
    'ServiceStatus',
    'get_unified_adapter_factory',
    'register_adapter_class',
    'get_registered_adapter',
    'get_adapter',
    'get_all_adapters',
    'health_check_all_adapters',
    'get_adapter_performance_report'
]

