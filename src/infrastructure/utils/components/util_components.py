"""
util_components 模块

提供 util_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from typing import Dict, Any, List
"""
基础设施层 - Util组件统一实现

使用统一的ComponentFactory基类，提供Util组件的工厂模式实现。
"""

# ComponentFactory, IComponentFactory 已通过其他方式获取

logger = logging.getLogger(__name__)

# Util组件常量


class UtilComponentConstants:
    """Util组件相关常量"""

    # 组件版本
    COMPONENT_VERSION = "2.0.0"

    # 支持的util ID列表
    SUPPORTED_UTIL_IDS = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61, 67, 73, 79, 85]

    # 组件类型
    DEFAULT_COMPONENT_TYPE = "Util"

    # 状态常量
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"
    STATUS_ERROR = "error"

    # 优先级常量
    DEFAULT_PRIORITY = 1
    MIN_PRIORITY = 0
    MAX_PRIORITY = 10


class IUtilComponent(ABC):
    """Util组件接口"""

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
    def get_util_id(self) -> int:
        """获取util ID"""


class UtilComponent(IUtilComponent):
    """统一Util组件实现"""

    def __init__(self, util_id: int, component_type: str = UtilComponentConstants.DEFAULT_COMPONENT_TYPE):
        """初始化组件"""
        self.util_id = util_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{util_id}"
        self.creation_time = datetime.now()

    def get_util_id(self) -> int:
        """获取util ID"""
        return self.util_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "util_id": self.util_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": UtilComponentConstants.COMPONENT_VERSION,
            "type": "unified_infrastructure_utils_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "util_id": self.util_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_util_processing",
            }

            return result
        except Exception as e:
            return {
                "util_id": self.util_id,
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
            "util_id": self.util_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
        }


class UtilComponentFactory(ComponentFactory):
    """Util组件工厂 - 继承统一ComponentFactory"""

    # 支持的util ID列表
    SUPPORTED_UTIL_IDS = UtilComponentConstants.SUPPORTED_UTIL_IDS

    def __init__(self):
        super().__init__()
        # 注册Util组件工厂函数
        for util_id in self.SUPPORTED_UTIL_IDS:
            self.register_factory(
                f"util_{util_id}",
                lambda config, uid=util_id: UtilComponent(uid, "Util")
            )

    @staticmethod
    def create_component(util_id: int) -> UtilComponent:
        """创建指定ID的util组件"""
        if util_id not in UtilComponentFactory.SUPPORTED_UTIL_IDS:
            raise ValueError(
                f"不支持的util ID: {util_id}。支持的ID: {UtilComponentFactory.SUPPORTED_UTIL_IDS}")

        return UtilComponent(util_id, "Util")

    @staticmethod
    def get_available_utils() -> List[int]:
        """获取所有可用的util ID"""
        return sorted(list(UtilComponentFactory.SUPPORTED_UTIL_IDS))

    @staticmethod
    def create_all_utils() -> Dict[int, UtilComponent]:
        """创建所有可用util"""
        return {util_id: UtilComponent(util_id, "Util") for util_id in UtilComponentFactory.SUPPORTED_UTIL_IDS}

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "UtilComponentFactory",
            "version": UtilComponentConstants.COMPONENT_VERSION,
            "total_utils": len(UtilComponentFactory.SUPPORTED_UTIL_IDS),
            "supported_ids": sorted(list(UtilComponentFactory.SUPPORTED_UTIL_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件",
        }

# 向后兼容：创建旧的组件实例


def create_util_util_component_1():

    return UtilComponentFactory.create_component(1)


def create_util_util_component_7():

    return UtilComponentFactory.create_component(7)


def create_util_util_component_13():

    return UtilComponentFactory.create_component(13)


def create_util_util_component_19():

    return UtilComponentFactory.create_component(19)


def create_util_util_component_25():

    return UtilComponentFactory.create_component(25)


def create_util_util_component_31():

    return UtilComponentFactory.create_component(31)


def create_util_util_component_37():

    return UtilComponentFactory.create_component(37)


def create_util_util_component_43():

    return UtilComponentFactory.create_component(43)


def create_util_util_component_49():

    return UtilComponentFactory.create_component(49)


def create_util_util_component_55():

    return UtilComponentFactory.create_component(55)


def create_util_util_component_61():

    return UtilComponentFactory.create_component(61)


def create_util_util_component_67():

    return UtilComponentFactory.create_component(67)


def create_util_util_component_73():

    return UtilComponentFactory.create_component(73)


def create_util_util_component_79():

    return UtilComponentFactory.create_component(79)


def create_util_util_component_85():

    return UtilComponentFactory.create_component(85)


__all__ = [
    "IUtilComponent",
    "UtilComponent",
    "UtilComponentFactory",
    "create_util_util_component_1",
    "create_util_util_component_7",
    "create_util_util_component_13",
    "create_util_util_component_19",
    "create_util_util_component_25",
    "create_util_util_component_31",
    "create_util_util_component_37",
    "create_util_util_component_43",
    "create_util_util_component_49",
    "create_util_util_component_55",
    "create_util_util_component_61",
    "create_util_util_component_67",
    "create_util_util_component_73",
    "create_util_util_component_79",
    "create_util_util_component_85",
]
