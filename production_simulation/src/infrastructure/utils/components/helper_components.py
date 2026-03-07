"""
helper_components 模块

提供 helper_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, Optional, List, Union
"""
基础设施层 - Helper组件统一实现

使用统一的ComponentFactory基类，提供Helper组件的工厂模式实现。
"""

# ComponentFactory, IComponentFactory 已通过其他方式获取

logger = logging.getLogger(__name__)

# Helper组件常量


class HelperComponentConstants:
    """Helper组件相关常量"""

    # 组件版本
    COMPONENT_VERSION = "2.0.0"

    # 支持的helper ID列表
    SUPPORTED_HELPER_IDS = [2, 8, 14, 20, 26, 32, 38, 44, 50, 56, 62, 68, 74, 80, 86]

    # 组件类型
    DEFAULT_COMPONENT_TYPE = "Helper"

    # 状态常量
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"
    STATUS_ERROR = "error"

    # 优先级常量
    DEFAULT_PRIORITY = 1
    MIN_PRIORITY = 0
    MAX_PRIORITY = 10


class IHelperComponent(ABC):
    """Helper组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""

    @abstractmethod
    def get_helper_id(self) -> int:
        """获取helper ID"""


class HelperComponent(IHelperComponent):
    """统一Helper组件实现"""

    def __init__(self, helper_id: int, component_type: str = HelperComponentConstants.DEFAULT_COMPONENT_TYPE):
        """初始化组件"""
        self.helper_id = helper_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{helper_id}"
        self.creation_time = datetime.now()

    def get_helper_id(self) -> int:
        """获取helper ID"""
        return self.helper_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "helper_id": self.helper_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": HelperComponentConstants.COMPONENT_VERSION,
            "type": "unified_infrastructure_utils_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "helper_id": self.helper_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_helper_processing",
            }
            return result
        except Exception as e:
            return {
                "helper_id": self.helper_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "helper_id": self.helper_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
        }


class HelperComponentFactory(ComponentFactory):
    """Helper组件工厂 - 继承统一ComponentFactory"""

    # 支持的helper ID列表
    SUPPORTED_HELPER_IDS = HelperComponentConstants.SUPPORTED_HELPER_IDS

    def __init__(self):
        super().__init__()
        # 注册Helper组件工厂函数
        for helper_id in self.SUPPORTED_HELPER_IDS:
            self.register_factory(
                f"helper_{helper_id}",
                lambda config, hid=helper_id: HelperComponent(hid, "Helper")
            )

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """重写创建方法，支持helper_id参数"""
        config = config or {}

        # 如果是数字类型，转换为helper_前缀
        if component_type.isdigit():
            helper_id = int(component_type)
            component_type = f"helper_{helper_id}"

        # 如果是helper_前缀，直接使用
        if component_type.startswith("helper_"):
            helper_id = int(component_type.split("_")[1])
            if helper_id not in self.SUPPORTED_HELPER_IDS:
                raise ValueError(f"不支持的helper ID: {helper_id}。支持的ID: {self.SUPPORTED_HELPER_IDS}")
            return HelperComponent(helper_id, "Helper")

        # 其他情况使用父类方法
        return super().create_component(component_type, config)

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """实现父类的抽象方法"""
        # Helper组件工厂主要通过注册的工厂函数创建
        return None

    @staticmethod
    def create_component(helper_id: Union[int, str]) -> HelperComponent:
        """创建指定ID的helper组件"""
        if isinstance(helper_id, str):
            if helper_id.startswith("helper_"):
                helper_id = helper_id.split("_", 1)[1]
            helper_id = int(helper_id)
        if helper_id not in HelperComponentFactory.SUPPORTED_HELPER_IDS:
            raise ValueError(
                f"不支持的helper ID: {helper_id}。支持的ID: {HelperComponentFactory.SUPPORTED_HELPER_IDS}")

        return HelperComponent(helper_id, "Helper")

    @staticmethod
    def get_available_helpers() -> List[int]:
        """获取所有可用的helper ID"""
        return sorted(list(HelperComponentFactory.SUPPORTED_HELPER_IDS))

    @staticmethod
    def create_all_helpers() -> Dict[int, HelperComponent]:
        """创建所有可用helper"""
        return {
            helper_id: HelperComponent(helper_id, "Helper") for helper_id in HelperComponentFactory.SUPPORTED_HELPER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "HelperComponentFactory",
            "version": HelperComponentConstants.COMPONENT_VERSION,
            "total_helpers": len(HelperComponentFactory.SUPPORTED_HELPER_IDS),
            "supported_ids": sorted(list(HelperComponentFactory.SUPPORTED_HELPER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件",
        }

# 向后兼容：创建旧的组件实例


def create_helper_helper_component_2():

    return HelperComponentFactory.create_component(2)


def create_helper_helper_component_8():

    return HelperComponentFactory.create_component(8)


def create_helper_helper_component_14():

    return HelperComponentFactory.create_component(14)


def create_helper_helper_component_20():

    return HelperComponentFactory.create_component(20)


def create_helper_helper_component_26():

    return HelperComponentFactory.create_component(26)


def create_helper_helper_component_32():

    return HelperComponentFactory.create_component(32)


def create_helper_helper_component_38():

    return HelperComponentFactory.create_component(38)


def create_helper_helper_component_44():

    return HelperComponentFactory.create_component(44)


def create_helper_helper_component_50():

    return HelperComponentFactory.create_component(50)


def create_helper_helper_component_56():

    return HelperComponentFactory.create_component(56)


def create_helper_helper_component_62():

    return HelperComponentFactory.create_component(62)


def create_helper_helper_component_68():

    return HelperComponentFactory.create_component(68)


def create_helper_helper_component_74():

    return HelperComponentFactory.create_component(74)


def create_helper_helper_component_80():

    return HelperComponentFactory.create_component(80)


def create_helper_helper_component_86():

    return HelperComponentFactory.create_component(86)


__all__ = [
    "IHelperComponent",
    "HelperComponent",
    "HelperComponentFactory",
    "create_helper_helper_component_2",
    "create_helper_helper_component_8",
    "create_helper_helper_component_14",
    "create_helper_helper_component_20",
    "create_helper_helper_component_26",
    "create_helper_helper_component_32",
    "create_helper_helper_component_38",
    "create_helper_helper_component_44",
    "create_helper_helper_component_50",
    "create_helper_helper_component_56",
    "create_helper_helper_component_62",
    "create_helper_helper_component_68",
    "create_helper_helper_component_74",
    "create_helper_helper_component_80",
    "create_helper_helper_component_86",
]
