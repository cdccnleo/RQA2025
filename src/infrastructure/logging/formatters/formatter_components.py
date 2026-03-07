"""
formatter_components 模块

提供 formatter_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, List, Optional, Union
"""
基础设施层 - Formatter组件统一实现

使用统一的ComponentFactory基类，提供Formatter组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IFormatterComponent(ABC):

    """Formatter组件接口"""

    def get_info(self) -> Dict[str, Any]:
        """
        获取组件信息

        Returns:
            组件基本信息字典
        """
        return {
            "component_type": "formatter_component",
            "interface": "IFormatterComponent",
            "version": "2.0.0",
            "description": "Formatter组件基础接口"
        }

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_formatter_id(self) -> int:
        """获取formatter ID"""

    @abstractmethod
    def validate_config(self, config_type: str, config: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    def get_config_schema(self) -> Dict[str, Any]:
        pass


class FormatterComponent(IFormatterComponent):

    """统一Formatter组件实现"""

    def __init__(self, formatter_id: int = 1, component_type: str = "Formatter", config: Optional[Dict[str, Any]] = None, name: str = None):
        """初始化组件"""
        self.formatter_id = formatter_id
        self.component_type = component_type
        self.name = name or "FormatterComponent"
        self.config = config.copy() if config else {}
        self.component_name = f"{component_type}_Component_{formatter_id}"
        self.creation_time = datetime.now()
        self.last_used = datetime.now()

    def get_formatter_id(self) -> int:
        """获取formatter ID"""
        return self.formatter_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "formatter_id": self.formatter_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_logging_system_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"processed": data}

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
            "formatter_id": self.formatter_id,
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


class FormatterComponentFactory(ComponentFactory):

    """Formatter组件工厂"""

    # 支持的formatter ID列表
    SUPPORTED_FORMATTER_IDS = [1, 2, 3]
    registered_components = {'json': 1, 'text': 2, 'structured': 3}

    def __init__(self):
        self._registered_components = {}
        self.registered_components = {
            'json': 1,
            'text': 2,
            # ... add more
        }
        self.component_instances = {}
        # existing integer IDs

    def create_component(self, formatter_id: Union[int, str], config: Optional[Dict[str, Any]] = None, name: Optional[str] = None):
        """创建指定ID的formatter组件"""
        if isinstance(formatter_id, str):
            # 如果是字符串，尝试从注册的组件中获取
            if hasattr(self, '_registered_classes') and formatter_id in self._registered_classes:
                component_class = self._registered_classes[formatter_id]
                return component_class(config=config, name=name)
            # 如果没注册，尝试从默认映射获取ID
            default_id = self.registered_components.get(formatter_id, None)
            if default_id is not None:
                return FormatterComponent(default_id, config=config, name=name)
            else:
                raise ValueError(f"Component type '{formatter_id}' not registered")
        
        if formatter_id not in FormatterComponentFactory.SUPPORTED_FORMATTER_IDS:
            raise ValueError(
                f"不支持的formatter ID: {formatter_id}。支持的ID: {FormatterComponentFactory.SUPPORTED_FORMATTER_IDS}")

        return FormatterComponent(formatter_id, config=config, name=name)

    @staticmethod
    def get_available_formatters() -> List[int]:
        """获取所有可用的formatter ID"""
        return sorted(list(FormatterComponentFactory.SUPPORTED_FORMATTER_IDS))

    @staticmethod
    def create_all_formatters() -> Dict[int, FormatterComponent]:
        """创建所有可用formatter"""
        return {
            formatter_id: FormatterComponent(formatter_id, "Formatter")
            for formatter_id in FormatterComponentFactory.SUPPORTED_FORMATTER_IDS
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

            if component_id in self.SUPPORTED_FORMATTER_IDS:
                component = FormatterComponent(component_id, "Formatter")
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
        return list(self._registered_components.keys()) + [str(fid) for fid in self.SUPPORTED_FORMATTER_IDS]

    def register_component(self, component_type: str, component_class):
        """注册组件"""
        # 验证component_class是否是类
        if not (isinstance(component_class, type) and hasattr(component_class, '__init__')):
            raise TypeError(f"component_class must be a class, got {type(component_class)}")
        
        # 存储组件类信息用于测试
        if not hasattr(self, '_registered_classes'):
            self._registered_classes = {}
        self._registered_classes[component_type] = component_class
        # 同时设置两个属性以保持兼容性
        self._registered_components[component_type] = component_class
        self.registered_components[component_type] = component_class


def get_factory_info() -> Dict[str, Any]:
    """获取工厂信息"""
    return {
        "factory_name": "FormatterComponentFactory",
        "version": "2.0.0",
        "total_formatters": len(FormatterComponentFactory.SUPPORTED_FORMATTER_IDS),
        "supported_ids": sorted(list(FormatterComponentFactory.SUPPORTED_FORMATTER_IDS)),
        "created_at": datetime.now().isoformat(),
        "description": f"统一Formatter组件工厂，替代原有的模板化文件"
    }

# 向后兼容：创建旧的组件实例


def create_formatter_formatter_component_3():

    return FormatterComponentFactory.create_component(3)


def create_formatter_formatter_component_9():

    return FormatterComponentFactory.create_component(9)


def create_formatter_formatter_component_15():

    return FormatterComponentFactory.create_component(15)


def create_formatter_formatter_component_21():

    return FormatterComponentFactory.create_component(21)


def create_formatter_formatter_component_27():

    return FormatterComponentFactory.create_component(27)


def create_formatter_formatter_component_33():

    return FormatterComponentFactory.create_component(33)


def create_formatter_formatter_component_39():

    return FormatterComponentFactory.create_component(39)


def create_formatter_formatter_component_45():

    return FormatterComponentFactory.create_component(45)


def create_formatter_formatter_component_51():

    return FormatterComponentFactory.create_component(51)


def create_formatter_formatter_component_57():

    return FormatterComponentFactory.create_component(57)


def create_formatter_formatter_component_63():

    return FormatterComponentFactory.create_component(63)


def create_formatter_formatter_component_69():

    return FormatterComponentFactory.create_component(69)


__all__ = [
    "IFormatterComponent",
    "FormatterComponent",
    "FormatterComponentFactory",
    "create_formatter_formatter_component_3",
    "create_formatter_formatter_component_9",
    "create_formatter_formatter_component_15",
    "create_formatter_formatter_component_21",
    "create_formatter_formatter_component_27",
    "create_formatter_formatter_component_33",
    "create_formatter_formatter_component_39",
    "create_formatter_formatter_component_45",
    "create_formatter_formatter_component_51",
    "create_formatter_formatter_component_57",
    "create_formatter_formatter_component_63",
    "create_formatter_formatter_component_69",
]

# Logger setup
logger = logging.getLogger(__name__)
