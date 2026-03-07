"""
factory_components 模块

提供 factory_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, List
"""
基础设施层 - Factory组件统一实现

使用统一的ComponentFactory基类，提供Factory组件的工厂模式实现。
"""

# ComponentFactory, IComponentFactory 已通过其他方式获取

logger = logging.getLogger(__name__)

# Factory组件常量


class FactoryComponentConstants:
    """Factory组件相关常量"""

    # 组件版本
    COMPONENT_VERSION = "2.0.0"

    # 支持的factory ID列表
    SUPPORTED_FACTORY_IDS = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84]

    # 组件类型
    DEFAULT_COMPONENT_TYPE = "Factory"

    # 状态常量
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"
    STATUS_ERROR = "error"

    # 优先级常量
    DEFAULT_PRIORITY = 1
    MIN_PRIORITY = 0
    MAX_PRIORITY = 10


class IFactoryComponent(ABC):
    """Factory组件接口"""

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
    def get_factory_id(self) -> int:
        """获取factory ID"""


class FactoryComponent(IFactoryComponent):
    """统一Factory组件实现"""

    def __init__(self, factory_id: int, component_type: str = FactoryComponentConstants.DEFAULT_COMPONENT_TYPE):
        """初始化组件"""
        self.factory_id = factory_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{factory_id}"
        self.creation_time = datetime.now()

    def get_factory_id(self) -> int:
        """获取factory ID"""
        return self.factory_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "factory_id": self.factory_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": FactoryComponentConstants.COMPONENT_VERSION,
            "type": "unified_infrastructure_utils_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "factory_id": self.factory_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_factory_processing",
            }
            return result
        except Exception as e:
            return {
                "factory_id": self.factory_id,
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
            "factory_id": self.factory_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
        }


class FactoryComponentFactory(ComponentFactory):
    """Factory组件工厂"""

    # 支持的factory ID列表
    SUPPORTED_FACTORY_IDS = FactoryComponentConstants.SUPPORTED_FACTORY_IDS

    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    def create_component(self, factory_id: int) -> FactoryComponent:
        """创建指定ID的factory组件"""
        if factory_id not in FactoryComponentFactory.SUPPORTED_FACTORY_IDS:
            raise ValueError(
                f"不支持的factory ID: {factory_id}。支持的ID: {FactoryComponentFactory.SUPPORTED_FACTORY_IDS}"
            )

        return FactoryComponent(factory_id, "Factory")

    @staticmethod
    def get_available_factorys() -> List[int]:
        """获取所有可用的factory ID"""
        return sorted(list(FactoryComponentFactory.SUPPORTED_FACTORY_IDS))

    @staticmethod
    def create_all_factorys() -> Dict[int, FactoryComponent]:
        """创建所有可用factory"""
        return {
            factory_id: FactoryComponent(factory_id, "Factory")
            for factory_id in FactoryComponentFactory.SUPPORTED_FACTORY_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "FactoryComponentFactory",
            "version": FactoryComponentConstants.COMPONENT_VERSION,
            "total_factorys": len(FactoryComponentFactory.SUPPORTED_FACTORY_IDS),
            "supported_ids": sorted(list(FactoryComponentFactory.SUPPORTED_FACTORY_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件",
        }

# 向后兼容：创建旧的组件实例


def create_factory_factory_component_6():
    return FactoryComponentFactory.create_component(6)


def create_factory_factory_component_12():
    return FactoryComponentFactory.create_component(12)


def create_factory_factory_component_18():
    return FactoryComponentFactory.create_component(18)


def create_factory_factory_component_24():
    return FactoryComponentFactory.create_component(24)


def create_factory_factory_component_30():
    return FactoryComponentFactory.create_component(30)


def create_factory_factory_component_36():
    return FactoryComponentFactory.create_component(36)


def create_factory_factory_component_42():
    return FactoryComponentFactory.create_component(42)


def create_factory_factory_component_48():
    return FactoryComponentFactory.create_component(48)


def create_factory_factory_component_54():
    return FactoryComponentFactory.create_component(54)


def create_factory_factory_component_60():
    return FactoryComponentFactory.create_component(60)


def create_factory_factory_component_66():
    return FactoryComponentFactory.create_component(66)


def create_factory_factory_component_72():
    return FactoryComponentFactory.create_component(72)


def create_factory_factory_component_78():
    return FactoryComponentFactory.create_component(78)


def create_factory_factory_component_84():
    return FactoryComponentFactory.create_component(84)


__all__ = [
    "IFactoryComponent",
    "FactoryComponent",
    "FactoryComponentFactory",
    "create_factory_factory_component_6",
    "create_factory_factory_component_12",
    "create_factory_factory_component_18",
    "create_factory_factory_component_24",
    "create_factory_factory_component_30",
    "create_factory_factory_component_36",
    "create_factory_factory_component_42",
    "create_factory_factory_component_48",
    "create_factory_factory_component_54",
    "create_factory_factory_component_60",
    "create_factory_factory_component_66",
    "create_factory_factory_component_72",
    "create_factory_factory_component_78",
    "create_factory_factory_component_84",
]
