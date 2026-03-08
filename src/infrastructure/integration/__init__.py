"""
RQA2025 统一基础设施集成层

此模块提供了系统各层之间的统一集成功能，消除代码重复，实现集中化管理。
包括业务层适配器、基础设施服务直接访问、降级服务等核心功能。
"""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# 延迟导入统一业务层适配器，避免循环依赖


def get_data_adapter():
    """延迟导入获取数据适配器"""
    from .business_adapters import get_data_adapter as _get_data_adapter
    return _get_data_adapter()


def get_features_adapter():
    """延迟导入获取特征适配器"""
    from .business_adapters import get_features_adapter as _get_features_adapter
    return _get_features_adapter()


def get_trading_adapter():
    """延迟导入获取交易适配器"""
    from .business_adapters import get_trading_adapter as _get_trading_adapter
    return _get_trading_adapter()


def get_risk_adapter():
    """延迟导入获取风控适配器"""
    from .business_adapters import get_risk_adapter as _get_risk_adapter
    return _get_risk_adapter()


def get_models_adapter():
    """延迟导入获取模型适配器"""
    from .business_adapters import get_models_adapter as _get_models_adapter
    return _get_models_adapter()


def get_business_adapter(layer_type):
    """延迟导入获取业务适配器"""
    from .business_adapters import get_business_adapter as _get_business_adapter
    return _get_business_adapter(layer_type)


def get_all_business_adapters():
    """延迟导入获取所有业务适配器"""
    from .business_adapters import get_all_business_adapters as _get_all_business_adapters
    return _get_all_business_adapters()


def health_check_business_adapters():
    """延迟导入健康检查业务适配器"""
    from .business_adapters import health_check_business_adapters as _health_check_business_adapters
    return _health_check_business_adapters()

# 导入业务层专用适配器 - 延迟导入避免循环依赖


def _import_data_adapter():
    """延迟导入数据层适配器"""
    try:
        from .data_adapter import (
            DataLayerAdapter,
            get_data_layer_adapter,
            get_data_cache_bridge,
            get_data_config_bridge,
            get_data_monitoring_bridge,
            get_data_infrastructure_manager
        )
        return {
            'DataLayerAdapter': DataLayerAdapter,
            'get_data_layer_adapter': get_data_layer_adapter,
            'get_data_cache_bridge': get_data_cache_bridge,
            'get_data_config_bridge': get_data_config_bridge,
            'get_data_monitoring_bridge': get_data_monitoring_bridge,
            'get_data_infrastructure_manager': get_data_infrastructure_manager
        }
    except ImportError as e:
        logger.warning(f"导入数据层适配器失败: {e}")
        return None


# 创建延迟导入的代理对象
_data_adapter_imports = None


def get_data_layer_adapter():
    """获取数据层适配器（延迟导入）"""
    global _data_adapter_imports
    if _data_adapter_imports is None:
        _data_adapter_imports = _import_data_adapter()
    if _data_adapter_imports and 'get_data_layer_adapter' in _data_adapter_imports:
        return _data_adapter_imports['get_data_layer_adapter']()
    return None


def DataLayerAdapter(*args, **kwargs):
    """数据层适配器类（延迟导入）"""
    global _data_adapter_imports
    if _data_adapter_imports is None:
        _data_adapter_imports = _import_data_adapter()
    if _data_adapter_imports and 'DataLayerAdapter' in _data_adapter_imports:
        return _data_adapter_imports['DataLayerAdapter'](*args, **kwargs)
    return None

# 导入特征层专用适配器 - 延迟导入避免循环依赖


def _import_features_adapter():
    """延迟导入特征层适配器"""
    try:
        from .adapters.features_adapter import (
            FeaturesLayerAdapter,
            get_features_layer_adapter,
            get_features_config_manager,
            get_features_cache_manager,
            get_features_engine,
            process_features_with_infrastructure
        )
        return {
            'FeaturesLayerAdapter': FeaturesLayerAdapter,
            'get_features_layer_adapter': get_features_layer_adapter,
            'get_features_config_manager': get_features_config_manager,
            'get_features_cache_manager': get_features_cache_manager,
            'get_features_engine': get_features_engine,
            'process_features_with_infrastructure': process_features_with_infrastructure
        }
    except ImportError as e:
        logger.warning(f"导入特征层适配器失败: {e}")
        return None


# 创建延迟导入的代理对象
_features_adapter_imports = None


def get_features_layer_adapter():
    """获取特征层适配器（延迟导入）"""
    global _features_adapter_imports
    if _features_adapter_imports is None:
        _features_adapter_imports = _import_features_adapter()
    if _features_adapter_imports and 'get_features_layer_adapter' in _features_adapter_imports:
        return _features_adapter_imports['get_features_layer_adapter']()
    return None


def FeaturesLayerAdapter(*args, **kwargs):
    """特征层适配器类（延迟导入）"""
    global _features_adapter_imports
    if _features_adapter_imports is None:
        _features_adapter_imports = _import_features_adapter()
    if _features_adapter_imports and 'FeaturesLayerAdapter' in _features_adapter_imports:
        return _features_adapter_imports['FeaturesLayerAdapter'](*args, **kwargs)
    return None

# 导入交易层专用适配器 - 延迟导入避免循环依赖


def _import_trading_adapter():
    """延迟导入交易层适配器"""
    try:
        from .trading_adapter import (
            TradingLayerAdapter,
            get_trading_layer_adapter,
            get_trading_engine,
            get_order_manager,
            get_execution_engine,
            execute_trade_with_infrastructure
        )
        return {
            'TradingLayerAdapter': TradingLayerAdapter,
            'get_trading_layer_adapter': get_trading_layer_adapter,
            'get_trading_engine': get_trading_engine,
            'get_order_manager': get_order_manager,
            'get_execution_engine': get_execution_engine,
            'execute_trade_with_infrastructure': execute_trade_with_infrastructure
        }
    except ImportError as e:
        logger.warning(f"导入交易层适配器失败: {e}")
        return None


# 创建延迟导入的代理对象
_trading_adapter_imports = None


def get_trading_layer_adapter():
    """获取交易层适配器（延迟导入）"""
    global _trading_adapter_imports
    if _trading_adapter_imports is None:
        _trading_adapter_imports = _import_trading_adapter()
    if _trading_adapter_imports and 'get_trading_layer_adapter' in _trading_adapter_imports:
        return _trading_adapter_imports['get_trading_layer_adapter']()
    return None


def TradingLayerAdapter(*args, **kwargs):
    """交易层适配器类（延迟导入）"""
    global _trading_adapter_imports
    if _trading_adapter_imports is None:
        _trading_adapter_imports = _import_trading_adapter()
    if _trading_adapter_imports and 'TradingLayerAdapter' in _trading_adapter_imports:
        return _trading_adapter_imports['TradingLayerAdapter'](*args, **kwargs)
    return None

# 导入风控层专用适配器 - 延迟导入避免循环依赖


def _import_risk_adapter():
    """延迟导入风控层适配器"""
    try:
        from .risk_adapter import (
            RiskLayerAdapter,
            get_risk_layer_adapter,
            get_risk_manager,
            get_risk_monitor,
            get_risk_calculator,
            assess_risk_with_infrastructure
        )
        return {
            'RiskLayerAdapter': RiskLayerAdapter,
            'get_risk_layer_adapter': get_risk_layer_adapter,
            'get_risk_manager': get_risk_manager,
            'get_risk_monitor': get_risk_monitor,
            'get_risk_calculator': get_risk_calculator,
            'assess_risk_with_infrastructure': assess_risk_with_infrastructure
        }
    except ImportError as e:
        logger.warning(f"导入风控层适配器失败: {e}")
        return None


# 创建延迟导入的代理对象
_risk_adapter_imports = None


def get_risk_layer_adapter():
    """获取风控层适配器（延迟导入）"""
    global _risk_adapter_imports
    if _risk_adapter_imports is None:
        _risk_adapter_imports = _import_risk_adapter()
    if _risk_adapter_imports and 'get_risk_layer_adapter' in _risk_adapter_imports:
        return _risk_adapter_imports['get_risk_layer_adapter']()
    return None


def RiskLayerAdapter(*args, **kwargs):
    """风控层适配器类（延迟导入）"""
    global _risk_adapter_imports
    if _risk_adapter_imports is None:
        _risk_adapter_imports = _import_risk_adapter()
    if _risk_adapter_imports and 'RiskLayerAdapter' in _risk_adapter_imports:
        return _risk_adapter_imports['RiskLayerAdapter'](*args, **kwargs)
    return None


# 导入降级服务
try:
    from .fallback_services import (
        get_fallback_service,
        get_all_fallback_services,
        health_check_fallback_services,
        get_fallback_config_manager,
        get_fallback_cache_manager,
        get_fallback_logger,
        get_fallback_monitoring,
        get_fallback_health_checker
    )
except ImportError as e:
    logger.warning(f"导入降级服务失败: {e}")

    def get_fallback_service(*args, **kwargs):
        raise RuntimeError("Fallback service not available") from e

# 导入统一接口 (新架构)
try:
    from .interfaces import (
        # 核心接口
        ICoreComponent, ILayerComponent,
        # 适配器接口
        IBusinessAdapter, IAdapterComponent,
        # 服务桥接接口
        IServiceBridge, IFallbackService,
        # 管理器接口
        IComponentManager, IInterfaceManager,
        # 实现类
        LayerInterfaceManager, CoreLayerInterface,
        # 便捷函数
        create_layer_interface_manager, create_core_layer_interface,
        validate_component_interface
    )
    # 类型别名从其他模块导入
    try:
        from .unified_business_adapters import BusinessLayerType
    except ImportError:
        from enum import Enum
        class BusinessLayerType(Enum):
            DATA = "data"
            FEATURES = "features"
            ML = "ml"
            STRATEGY = "strategy"
            TRADING = "trading"
            RISK = "risk"
    
    try:
        from .unified_business_adapters import ComponentLifecycle
    except ImportError:
        from enum import Enum
        class ComponentLifecycle(Enum):
            INITIALIZED = "initialized"
            STARTED = "started"
            STOPPED = "stopped"
except ImportError as e:
    logger.warning(f"导入统一接口失败: {e}")

    # 提供基础实现
    class IServiceComponent:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("ServiceComponent interface unavailable") from e

# 导入统一适配器架构 (新架构)
try:
    from .adapters import (
        # 核心类
        UnifiedBusinessAdapter, UnifiedAdapterFactory,
        # 数据类
        ServiceConfig, AdapterMetrics, ServiceStatus,
        # 全局函数
        get_unified_adapter_factory, register_adapter_class,
        get_adapter, get_all_adapters, health_check_all_adapters,
        get_adapter_performance_report
    )
except ImportError as e:
    logger.warning(f"导入统一适配器架构失败: {e}")

# 导入原有组件 (兼容性保持)
try:
    from .system_integration_manager import SystemIntegrationManager
    from .interface import SystemLayerInterfaceManager  # 重命名后的系统层接口管理器
    from .layer_interface import LayerInterface  # 即将废弃，使用CoreLayerInterface
    from .interface import ICoreComponent  # 兼容性接口
    from src.infrastructure.config.core.unified_manager import UnifiedConfigManager
    from .data import DataFlowManager, CacheIntegrationManager
except ImportError as e:
    logger.warning(f"导入原有集成组件失败: {e}")

    # 提供基础定义确保向后兼容性
    from abc import ABC, abstractmethod
    from typing import Any, Dict, Optional

    class ICoreComponent(ABC):
        """核心组件接口基础定义"""
        @property
        @abstractmethod
        def component_id(self) -> str:
            """组件ID"""

        @abstractmethod
        def initialize(self) -> bool:
            """初始化组件"""

        @abstractmethod
        def shutdown(self) -> bool:
            """关闭组件"""

        @abstractmethod
        def health_check(self) -> Dict[str, Any]:
            """健康检查"""

    class IServiceComponent(ABC):
        """服务组件接口基础定义"""
        @property
        @abstractmethod
        def service_name(self) -> str:
            """服务名称"""

        @abstractmethod
        def start_service(self) -> bool:
            """启动服务"""

        @abstractmethod
        def stop_service(self) -> bool:
            """停止服务"""

        @abstractmethod
        def get_service_status(self) -> Dict[str, Any]:
            """获取服务状态"""

    class ILayerComponent(ABC):
        """层组件接口基础定义"""
        @property
        @abstractmethod
        def layer_name(self) -> str:
            """层名称"""

        @abstractmethod
        def get_layer_info(self) -> Dict[str, Any]:
            """获取层信息"""

        @abstractmethod
        def validate_layer_dependencies(self) -> bool:
            """验证层依赖"""

# 导出统一基础设施集成接口
__all__ = [
    # 统一接口 (新架构)
    # 核心接口
    'ICoreComponent', 'IServiceComponent', 'ILayerComponent',
    # 适配器接口
    'IBusinessAdapter', 'IAdapterComponent',
    # 服务桥接接口
    'IServiceBridge', 'IFallbackService',
    # 管理器接口
    'IComponentManager', 'IInterfaceManager',
    # 实现类
    'LayerInterfaceManager', 'CoreLayerInterface',
    # 便捷函数
    'create_layer_interface_manager', 'create_core_layer_interface',
    'validate_component_interface',
    # 枚举和类型
    'BusinessLayerType', 'ComponentLifecycle',

    # 统一适配器架构 (新架构)
    # 核心类
    'UnifiedBusinessAdapter', 'UnifiedAdapterFactory',
    # 数据类
    'ServiceConfig', 'AdapterMetrics', 'ServiceStatus',
    # 全局函数
    'get_unified_adapter_factory', 'register_adapter_class',
    'get_adapter', 'get_all_adapters', 'health_check_all_adapters',
    'get_adapter_performance_report',
    # 便捷函数
    'create_service_config',

    # 统一业务层适配器
    'UnifiedBusinessAdapterFactory',
    'get_business_adapter',
    'get_all_business_adapters',
    'health_check_business_adapters',
    'get_data_adapter',
    'get_features_adapter',
    'get_trading_adapter',
    'get_risk_adapter',

    # 数据层适配器
    'DataLayerAdapter',
    'get_data_layer_adapter',
    'get_data_cache_bridge',
    'get_data_config_bridge',
    'get_data_monitoring_bridge',
    'get_data_infrastructure_manager',

    # 特征层适配器
    'FeaturesLayerAdapter',
    'get_features_layer_adapter',
    'get_features_config_manager',
    'get_features_cache_manager',
    'get_features_engine',
    'process_features_with_infrastructure',

    # 交易层适配器
    'TradingLayerAdapter',
    'get_trading_layer_adapter',
    'get_trading_engine',
    'get_order_manager',
    'get_execution_engine',
    'execute_trade_with_infrastructure',

    # 风控层适配器
    'RiskLayerAdapter',
    'get_risk_layer_adapter',
    'get_risk_manager',
    'get_risk_monitor',
    'get_risk_calculator',
    'assess_risk_with_infrastructure',

    # 降级服务
    'get_fallback_service',
    'get_all_fallback_services',
    'health_check_fallback_services',
    'get_fallback_config_manager',
    'get_fallback_cache_manager',
    'get_fallback_logger',
    'get_fallback_monitoring',
    'get_fallback_health_checker',

    # 原有组件 (兼容性保持)
    'SystemIntegrationManager',
    'SystemLayerInterfaceManager',  # 重命名后的系统层接口管理器
    'LayerInterface',  # 即将废弃，使用CoreLayerInterface
    'ICoreComponent',  # 兼容性接口
    'UnifiedConfigManager',
    'DataFlowManager',
    'CacheIntegrationManager',

    # 向后兼容函数
    'log_data_operation',
    'record_data_metric',
    'get_data_config',
    'DataSourceType'
]

# 导入数据源类型枚举（用于向后兼容）
try:
    from src.data.interfaces.standard_interfaces import DataSourceType
except ImportError:
    # 定义简化的DataSourceType枚举用于向后兼容
    from enum import Enum

    class DataSourceType(Enum):

        STOCK = "stock"
        FUTURES = "futures"
        OPTIONS = "options"
        INDEX = "index"
        BONDS = "bonds"
        FOREX = "forex"
        CRYPTO = "crypto"
        GENERAL = "general"

# 数据操作日志记录函数（向后兼容）


def log_data_operation(operation: str, data_type, details: dict, level: str = "info") -> None:
    """
    记录数据操作日志（统一接口）

    Args:
        operation: 操作名称
        data_type: 数据类型
        details: 操作详情
        level: 日志级别
    """
    try:
        data_adapter = get_data_adapter()
        logger = data_adapter.get_logger()
        if logger:
            message = f"数据操作: {operation}, 类型: {data_type}, 详情: {details}"
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            elif level == "debug":
                logger.debug(message)
            else:
                logger.info(message)
        else:
            # 降级到标准logging
            import logging
            logging.getLogger(__name__).info(f"[降级日志] {operation}: {details}")
    except Exception as e:
        # 最后的降级方案
        import logging
        logging.getLogger(__name__).warning(f"日志记录失败: {e}")

# 数据指标记录函数（向后兼容）


def record_data_metric(metric_name: str, value, data_type, tags: dict = None) -> None:
    """
    记录数据指标（统一接口）

    Args:
        metric_name: 指标名称
        value: 指标值
        data_type: 数据类型
        tags: 标签
    """
    try:
        data_adapter = get_data_adapter()
        monitoring = data_adapter.get_monitoring()
        if monitoring:
            monitoring.record_metric(metric_name, value, tags or {},
                                     description=f"数据指标: {metric_name}")
        else:
            logger = data_adapter.get_logger()
            if logger:
                logger.info(f"数据指标记录: {metric_name}={value}, 类型: {data_type}, 标签: {tags}")
    except Exception:
        # 降级处理
        import logging
        logging.getLogger(__name__).info(f"[降级指标] {metric_name}={value}")

# 获取数据配置函数（向后兼容）


def get_data_config(key: str, default=None):
    """
    获取数据配置（统一接口）

    Args:
        key: 配置键

        default: 默认值

    Returns:
        配置值
    """
    try:
        data_adapter = get_data_adapter()
        config_manager = data_adapter.get_config_manager()
        if config_manager:
            return config_manager.get(key, default)
        return default
    except Exception:
        return default


# 版本信息
__version__ = "2.0.0"
__description__ = "统一基础设施集成层 - 消除代码重复，实现集中化管理"
