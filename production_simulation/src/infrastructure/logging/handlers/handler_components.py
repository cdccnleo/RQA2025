"""
handler_components 模块

提供 handler_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, List, Optional, Union
"""
基础设施层 - Handler组件统一实现

使用统一的ComponentFactory基类，提供Handler组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IHandlerComponent(ABC):

    """Handler组件接口"""

    def get_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            组件基本信息字典
        """
        return {
            "component_type": "handler_component",
            "interface": "IHandlerComponent",
            "version": "2.0.0",
            "description": "Handler组件基础接口"
        }

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_handler_id(self) -> int:
        """获取处理器ID"""

    @abstractmethod
    def validate_config(self, config_type: str, config: Dict[str, Any]) -> bool:
        """验证配置"""

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置Schema"""


class HandlerComponent(IHandlerComponent):

    """统一Handler组件实现"""

    def __init__(self, handler_id: int = 1, component_type: str = "LoggingHandler", config: Optional[Dict[str, Any]] = None, name: str = None):
        """初始化组件"""
        self.handler_id = handler_id
        self.component_type = component_type
        self.name = name or "HandlerComponent"
        self.config = config.copy() if config else {}
        self.component_name = f"{component_type}_Component_{handler_id}"
        self.creation_time = datetime.now()
        self.last_used = datetime.now()

    def get_handler_id(self) -> int:
        """获取处理器ID"""
        return self.handler_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "handler_id": self.handler_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_handler_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "handler_id": self.handler_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_handler_processing"
            }
            return result
        except Exception as e:
            return {
                "handler_id": self.handler_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        try:
            uptime = (datetime.now() - self.creation_time).total_seconds()
            creation_time_str = self.creation_time.isoformat()
            last_used_str = self.last_used.isoformat()
            status = "active"
        except Exception as e:
            # 如果时间计算出错，返回错误状态
            uptime = 0.0
            creation_time_str = "unknown"
            last_used_str = "unknown"
            status = "error"
            
        return {
            "handler_id": self.handler_id,
            "name": self.name,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": status,
            "creation_time": creation_time_str,
            "created_at": creation_time_str,  # 添加兼容性字段
            "last_used": last_used_str,
            "uptime_seconds": uptime,
            "health": "good" if status == "active" else "error"
        }

    def validate_config(self, config_type: str, config: Dict[str, Any]) -> bool:
        # 接受 None 和字典类型的配置
        return config is None or isinstance(config, dict)

    def get_config_schema(self) -> Dict[str, Any]:
        return {"type": "object", "properties": {}}

    def update_last_used(self):
        self.last_used = datetime.now()


class LoggingHandlerComponentFactory(ComponentFactory):

    """LoggingHandler组件工厂"""

    # 支持的处理器ID列表
    SUPPORTED_HANDLER_IDS = [1,2,3]
    registered_components = {'file': 1, 'console': 2, 'syslog': 3}

    def __init__(self):
        self._registered_components = {}
        self.registered_components = {
            'file': 1,
            'console': 2,
            # ... add more
        }

    def create_component(self, handler_id: Union[int, str], config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """创建指定ID的handler组件"""
        if isinstance(handler_id, str):
            # 如果是字符串，尝试从注册的组件中获取
            if hasattr(self, '_registered_components') and handler_id in self._registered_components:
                component_class = self._registered_components[handler_id]
                # 检查组件类的构造函数签名，决定如何调用
                import inspect
                sig = inspect.signature(component_class.__init__)
                params = list(sig.parameters.keys())
                # 如果有handler_id参数，传递默认值
                if 'handler_id' in params:
                    return component_class(handler_id=1, component_type="LoggingHandler", config=config, name=name)
                else:
                    return component_class(config=config, name=name)
            # 如果没注册，尝试从默认映射获取ID
            default_id = self.registered_components.get(handler_id, None)
            if default_id is not None:
                return HandlerComponent(default_id, "LoggingHandler", config=config, name=name)
            else:
                raise ValueError(f"Component type '{handler_id}' not registered")
        
        if isinstance(handler_id, int) and handler_id not in self.SUPPORTED_HANDLER_IDS:
            raise ValueError(
                f"不支持的handler ID: {handler_id}。支持的ID: {self.SUPPORTED_HANDLER_IDS}")

        return HandlerComponent(handler_id, "LoggingHandler", config=config, name=name)

    @staticmethod
    def get_available_handlers() -> List[int]:
        """获取所有可用的处理器ID"""
        return sorted(list(LoggingHandlerComponentFactory.SUPPORTED_HANDLER_IDS))

    @staticmethod
    def create_all_handlers() -> Dict[int, HandlerComponent]:
        """创建所有可用处理器"""
        return {
            handler_id: HandlerComponent(handler_id, "LoggingHandler")
            for handler_id in LoggingHandlerComponentFactory.SUPPORTED_HANDLER_IDS
        }

    def get_component_info(self, component_type: str) -> Dict[str, Any]:
        """获取组件信息"""
        try:
            # 首先检查是否有注册的组件
            if hasattr(self, '_registered_components') and component_type in self._registered_components:
                component_class = self._registered_components[component_type]
                return {
                    "component_type": component_type,
                    "class": component_class,
                    "description": f"Registered {component_type} component"
                }

            # 检查静态支持的组件
            if isinstance(component_type, str) and component_type.isdigit():
                component_id = int(component_type)
            elif isinstance(component_type, int):
                component_id = component_type
            else:
                raise ValueError(f"Component type '{component_type}' not registered")

            if component_id in self.SUPPORTED_HANDLER_IDS:
                component = HandlerComponent(component_id, "LoggingHandler")
                info = component.get_info()
                info["component_type"] = component_type
                return info
            raise ValueError(f"Component type '{component_type}' not registered")
        except ValueError:
            raise  # 重新抛出ValueError
        except Exception:
            return {}

    def validate_config(self, component_type: str, config: Dict[str, Any]) -> bool:
        """验证配置"""
        try:
            # 基本配置验证
            if not isinstance(config, dict):
                return False

            # 检查必需字段
            required_fields = ['max_connections', 'timeout']
            for field in required_fields:
                if field not in config:
                    return False

            # 验证数值类型
            if not isinstance(config.get('max_connections'), int) or config['max_connections'] <= 0:
                return False
            if not isinstance(config.get('timeout'), int) or config['timeout'] <= 0:
                return False

            return True
        except Exception:
            return False

    def list_registered_components(self) -> List[str]:
        """列出所有已注册的组件"""
        return list(self._registered_components.keys()) + [str(hid) for hid in self.SUPPORTED_HANDLER_IDS]

    def register_component(self, component_type: str, component_class):
        """注册组件"""
        # 验证component_class是否是类
        if not (isinstance(component_class, type) and hasattr(component_class, '__init__')):
            raise TypeError(f"component_class must be a class, got {type(component_class)}")
        
        # 存储组件类信息用于测试
        if not hasattr(self, '_registered_components'):
            self._registered_components = {}
        self._registered_components[component_type] = component_class
        # 同时设置两个属性以保持兼容性
        self.registered_components[component_type] = component_class

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "LoggingHandlerComponentFactory",
            "version": "2.0.0",
            "total_handlers": len(LoggingHandlerComponentFactory.SUPPORTED_HANDLER_IDS),
            "supported_ids": sorted(list(LoggingHandlerComponentFactory.SUPPORTED_HANDLER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }

# 向后兼容：创建旧的组件实例


def create_logginghandler_handler_component_2():

    return LoggingHandlerComponentFactory.create_component(2)


def create_logginghandler_handler_component_8():

    return LoggingHandlerComponentFactory.create_component(8)


def create_logginghandler_handler_component_14():

    return LoggingHandlerComponentFactory.create_component(14)


def create_logginghandler_handler_component_20():

    return LoggingHandlerComponentFactory.create_component(20)


def create_logginghandler_handler_component_26():

    return LoggingHandlerComponentFactory.create_component(26)


def create_logginghandler_handler_component_32():

    return LoggingHandlerComponentFactory.create_component(32)


def create_logginghandler_handler_component_38():

    return LoggingHandlerComponentFactory.create_component(38)


def create_logginghandler_handler_component_44():

    return LoggingHandlerComponentFactory.create_component(44)


def create_logginghandler_handler_component_50():

    return LoggingHandlerComponentFactory.create_component(50)


def create_logginghandler_handler_component_56():

    return LoggingHandlerComponentFactory.create_component(56)


def create_logginghandler_handler_component_62():

    return LoggingHandlerComponentFactory.create_component(62)


def create_logginghandler_handler_component_68():

    return LoggingHandlerComponentFactory.create_component(68)


def create_logginghandler_handler_component_74():

    return LoggingHandlerComponentFactory.create_component(74)


__all__ = [
    "IHandlerComponent",
    "HandlerComponent",
    "LoggingHandlerComponentFactory",
    "create_logginghandler_handler_component_2",
    "create_logginghandler_handler_component_8",
    "create_logginghandler_handler_component_14",
    "create_logginghandler_handler_component_20",
    "create_logginghandler_handler_component_26",
    "create_logginghandler_handler_component_32",
    "create_logginghandler_handler_component_38",
    "create_logginghandler_handler_component_44",
    "create_logginghandler_handler_component_50",
    "create_logginghandler_handler_component_56",
    "create_logginghandler_handler_component_62",
    "create_logginghandler_handler_component_68",
    "create_logginghandler_handler_component_74",
]
# Logger setup
logger = logging.getLogger(__name__)
