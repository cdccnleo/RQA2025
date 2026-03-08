#!/usr/bin/env python3
"""
RQA2025 统一集成接口规范

整合所有核心集成接口定义，消除重复和分散问题。
提供统一的接口体系，支持业务层适配器、组件管理、服务桥接等功能。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 枚举定义
# =============================================================================


class BusinessLayerType(Enum):

    """业务层类型枚举"""
    DATA = "data"
    FEATURES = "features"
    TRADING = "trading"
    RISK = "risk"
    MODELS = "models"
    ENGINE = "engine"


class ComponentLifecycle(Enum):

    """组件生命周期状态"""
    CREATED = "created"
    INITIALIZING = "initializing"
    INITIALIZED = "initialized"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


# =============================================================================
# 核心组件接口
# =============================================================================


class ICoreComponent(ABC):

    """核心组件统一接口"""

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""


class IServiceComponent(ICoreComponent):

    """服务组件接口"""

    @abstractmethod
    def start(self) -> bool:
        """启动服务"""

    @abstractmethod
    def stop(self) -> bool:
        """停止服务"""

    @abstractmethod
    def restart(self) -> bool:
        """重启服务"""


class ILayerComponent(ICoreComponent):

    """层组件接口"""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """处理数据"""

    @abstractmethod
    def get_layer_type(self) -> BusinessLayerType:
        """获取层类型"""


# =============================================================================
# 适配器接口
# =============================================================================


class IBusinessAdapter(ICoreComponent):

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


class IAdapterComponent(ICoreComponent):

    """适配器组件接口"""

    @abstractmethod
    def adapt(self, source: Any, target_type: Type) -> Any:
        """适配转换"""

    @abstractmethod
    def validate_adaptation(self, source: Any, target: Any) -> bool:
        """验证适配结果"""


# =============================================================================
# 服务桥接接口
# =============================================================================


class IServiceBridge(ABC):

    """服务桥接器接口"""

    @abstractmethod
    def get_service(self, service_name: str) -> Optional[Any]:
        """获取服务实例"""

    @abstractmethod
    def register_service(self, service_name: str, service_instance: Any) -> bool:
        """注册服务实例"""

    @abstractmethod
    def unregister_service(self, service_name: str) -> bool:
        """注销服务实例"""


class IFallbackService(ABC):

    """降级服务接口"""

    @abstractmethod
    def is_available(self) -> bool:
        """检查服务是否可用"""

    @abstractmethod
    def get_fallback_service(self) -> Any:
        """获取降级服务实例"""


# =============================================================================
# 管理器接口
# =============================================================================


class IComponentManager(ABC):

    """组件管理器接口"""

    @abstractmethod
    def register_component(self, component: ICoreComponent) -> bool:
        """注册组件"""

    @abstractmethod
    def unregister_component(self, component_id: str) -> bool:
        """注销组件"""

    @abstractmethod
    def get_component(self, component_id: str) -> Optional[ICoreComponent]:
        """获取组件"""

    @abstractmethod
    def list_components(self) -> List[str]:
        """列出所有组件"""


class IInterfaceManager(ABC):

    """接口管理器接口"""

    @abstractmethod
    def register_interface(self, interface_name: str, interface_obj: Any) -> bool:
        """注册接口"""

    @abstractmethod
    def get_interface(self, interface_name: str) -> Optional[Any]:
        """获取接口"""

    @abstractmethod
    def list_interfaces(self) -> List[str]:
        """列出所有接口"""


# =============================================================================
# 层接口管理器 (原LayerInterface重命名)
# =============================================================================


class LayerInterfaceManager:

    """层接口管理器 - 负责单层的接口标准化"""

    def __init__(self, layer_name: str):

        self.layer_name = layer_name
        self.methods: Dict[str, Callable] = {}
        self.interfaces: Dict[str, Any] = {}
        self.components: Dict[str, ICoreComponent] = {}
        self.logger = logging.getLogger(f"{__name__}.{layer_name}")

    def register_method(self, method_name: str, method_func: Callable) -> None:
        """注册层方法"""
        self.methods[method_name] = method_func
        self.logger.info(f"注册 {self.layer_name} 层方法: {method_name}")

    def register_interface(self, interface_name: str, interface_obj: Any) -> None:
        """注册层接口"""
        self.interfaces[interface_name] = interface_obj
        self.logger.info(f"注册 {self.layer_name} 层接口: {interface_name}")

    def register_component(self, component_name: str, component: ICoreComponent) -> None:
        """注册层组件"""
        self.components[component_name] = component
        self.logger.info(f"注册 {self.layer_name} 层组件: {component_name}")

    def get_method(self, method_name: str) -> Optional[Callable]:
        """获取层方法"""
        return self.methods.get(method_name)

    def get_interface(self, interface_name: str) -> Optional[Any]:
        """获取层接口"""
        return self.interfaces.get(interface_name)

    def get_component(self, component_name: str) -> Optional[ICoreComponent]:
        """获取层组件"""
        return self.components.get(component_name)

    def list_methods(self) -> List[str]:
        """列出所有方法"""
        return list(self.methods.keys())

    def list_interfaces(self) -> List[str]:
        """列出所有接口"""
        return list(self.interfaces.keys())

    def list_components(self) -> List[str]:
        """列出所有组件"""
        return list(self.components.keys())

    def validate_interface(self) -> Dict[str, Any]:
        """验证接口完整性"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }

        # 检查必要的方法是否存在
        required_methods = self._get_required_methods()
        for method in required_methods:
            if method not in self.methods:
                validation_result['valid'] = False
                validation_result['errors'].append(f"缺少必要方法: {method}")

        # 检查必要接口是否存在
        required_interfaces = self._get_required_interfaces()
        for interface_name in required_interfaces:
            if interface_name not in self.interfaces:
                validation_result['valid'] = False
                validation_result['errors'].append(f"缺少必要接口: {interface_name}")

        return validation_result

    def _get_required_methods(self) -> List[str]:
        """获取必需的方法列表"""
        return ['initialize', 'get_status', 'health_check']

    def _get_required_interfaces(self) -> List[str]:
        """获取必需的接口列表"""
        return ['config_manager', 'logger']


# =============================================================================
# 核心层接口 (原LayerInterface重命名)
# =============================================================================


class CoreLayerInterface(ILayerComponent):

    """核心层接口 - 实现组件生命周期管理"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化核心层接口

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.initialized = False
        self.lifecycle_status = ComponentLifecycle.CREATED
        self.created_at = datetime.now()
        self.logger = logging.getLogger(f"{__name__}.CoreLayerInterface")

    def initialize(self) -> bool:
        """初始化组件

        Returns:
            初始化是否成功
        """
        try:
            self.lifecycle_status = ComponentLifecycle.INITIALIZING
            self.initialized = True
            self.lifecycle_status = ComponentLifecycle.INITIALIZED
            self.logger.info("核心层接口初始化成功")
            return True
        except Exception as e:
            self.lifecycle_status = ComponentLifecycle.ERROR
            self.logger.error(f"核心层接口初始化失败: {e}")
            return False

    def process(self, data: Any) -> Any:
        """处理数据

        Args:
            data: 输入数据

        Returns:
            处理后的数据
        """
        if not self.initialized:
            self.logger.warning("组件未初始化，无法处理数据")
            return data

        try:
            self.lifecycle_status = ComponentLifecycle.RUNNING
            self.logger.info(f"处理数据: {type(data)}")
            # 这里可以添加实际的数据处理逻辑
            return data
        except Exception as e:
            self.lifecycle_status = ComponentLifecycle.ERROR
            self.logger.error(f"数据处理失败: {e}")
            return data

    def validate_config(self) -> bool:
        """验证配置

        Returns:
            验证结果
        """
        self.logger.info("验证配置")
        # 这里可以添加配置验证逻辑
        return True

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态

        Returns:
            状态信息字典
        """
        return {
            'component_type': 'CoreLayerInterface',
            'initialized': self.initialized,
            'lifecycle_status': self.lifecycle_status.value,
            'created_at': self.created_at.isoformat(),
            'config_valid': self.validate_config(),
            'last_updated': datetime.now().isoformat()
        }

    def get_layer_type(self) -> BusinessLayerType:
        """获取层类型"""
        return BusinessLayerType.ENGINE  # 默认核心层

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'status': 'healthy' if self.initialized else 'unhealthy',
            'lifecycle_status': self.lifecycle_status.value,
            'timestamp': datetime.now().isoformat(),
            'checks': {
                'initialization': self.initialized,
                'configuration': self.validate_config()
            }
        }


# =============================================================================
# 便捷函数和工具
# =============================================================================


def create_layer_interface_manager(layer_name: str) -> LayerInterfaceManager:
    """创建层接口管理器"""
    return LayerInterfaceManager(layer_name)


def create_core_layer_interface(config: Optional[Dict[str, Any]] = None) -> CoreLayerInterface:
    """创建核心层接口"""
    return CoreLayerInterface(config)


def validate_component_interface(component: Any) -> Dict[str, Any]:
    """验证组件接口实现"""
    validation_result = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'missing_methods': [],
        'missing_properties': []
    }

    # 检查必需的方法
    required_methods = ['initialize', 'get_status', 'validate_config', 'health_check']
    for method in required_methods:
        if not hasattr(component, method) or not callable(getattr(component, method, None)):
            validation_result['valid'] = False
            validation_result['missing_methods'].append(method)

    # 检查可选的方法
    optional_methods = ['process', 'start', 'stop', 'restart']
    for method in optional_methods:
        if not hasattr(component, method) or not callable(getattr(component, method, None)):
            validation_result['warnings'].append(f"缺少可选方法: {method}")

    return validation_result


# =============================================================================
# 类型别名
# =============================================================================

CoreComponent = ICoreComponent
ServiceComponent = IServiceComponent
LayerComponent = ILayerComponent
BusinessAdapter = IBusinessAdapter
AdapterComponent = IAdapterComponent
ServiceBridge = IServiceBridge
FallbackService = IFallbackService
ComponentManager = IComponentManager
InterfaceManager = IInterfaceManager

# 便捷类型
LayerManager = LayerInterfaceManager
CoreInterface = CoreLayerInterface

__all__ = [
    # 枚举
    'BusinessLayerType',
    'ComponentLifecycle',

    # 核心接口
    'ICoreComponent',
    'IServiceComponent',
    'ILayerComponent',

    # 适配器接口
    'IBusinessAdapter',
    'IAdapterComponent',

    # 服务桥接接口
    'IServiceBridge',
    'IFallbackService',

    # 管理器接口
    'IComponentManager',
    'IInterfaceManager',

    # 实现类
    'LayerInterfaceManager',
    'CoreLayerInterface',

    # 便捷函数
    'create_layer_interface_manager',
    'create_core_layer_interface',
    'validate_component_interface',

    # 类型别名
    'CoreComponent',
    'ServiceComponent',
    'LayerComponent',
    'BusinessAdapter',
    'AdapterComponent',
    'ServiceBridge',
    'FallbackService',
    'ComponentManager',
    'InterfaceManager',
    'LayerManager',
    'CoreInterface'
]
