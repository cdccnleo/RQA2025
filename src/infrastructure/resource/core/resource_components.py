"""
resource_components 模块

提供 resource_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类
import time

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.core.base_components import ComponentFactory
from typing import Dict, Any, Optional, List
"""
基础设施层 - Resource组件统一实现

使用统一的ComponentFactory基类，提供Resource组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IResourceProcessorComponent(ABC):

    """Resource处理器组件接口"""

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
    def get_resource_id(self) -> int:
        """获取resource ID"""


class ResourceComponent(IResourceProcessorComponent):

    """统一Resource组件实现"""

    def __init__(self, resource_id: int, component_type: str = "Resource"):
        """初始化组件"""
        self.resource_id = resource_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{resource_id}"
        self.creation_time = datetime.now()

    def get_resource_id(self) -> int:
        """获取resource ID"""
        return self.resource_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "resource_id": self.resource_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_resource_management_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "resource_id": self.resource_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_resource_processing"
            }
            return result
        except Exception as e:
            return {
                "resource_id": self.resource_id,
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
        return {
            "resource_id": self.resource_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class ResourceComponentFactory(ComponentFactory):

    """Resource组件工厂"""

    # 支持的resource ID列表
    SUPPORTED_RESOURCE_IDS = [1, 7, 13, 19, 25, 31, 37, 43, 49, 55, 61]

    def __init__(self):
        super().__init__()
        # 注册组件工厂函数
        for resource_id in self.SUPPORTED_RESOURCE_IDS:
            self.register_factory(
                f"resource_{resource_id}",
                lambda config, rid=resource_id: ResourceComponent(rid, "Resource"),
            )

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """重写创建方法，支持resource_id参数"""
        config = config or {}

        # 如果是数字类型，转换为resource_前缀
        if component_type.isdigit():
            resource_id = int(component_type)
            component_type = f"resource_{resource_id}"

        # 如果是resource_前缀，直接使用
        if component_type.startswith("resource_"):
            resource_id = int(component_type.split("_")[1])
            if resource_id not in self.SUPPORTED_RESOURCE_IDS:
                raise ValueError(
                    f"不支持的resource ID: {resource_id}。支持的ID: {self.SUPPORTED_RESOURCE_IDS}")
            component = ResourceComponent(resource_id, "Resource")
            # 记录创建统计
            self._record_creation(component_type, time.time())
            return component

        # 其他情况使用父类方法
        return super().create_component(component_type, config)

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """实现父类的抽象方法"""
        # Resource组件工厂主要通过注册的工厂函数创建
        return None

    @staticmethod
    def create_component_static(resource_id: int) -> ResourceComponent:
        """静态方法创建指定ID的resource组件"""
        if resource_id not in ResourceComponentFactory.SUPPORTED_RESOURCE_IDS:
            raise ValueError(
                f"不支持的resource ID: {resource_id}。支持的ID: {ResourceComponentFactory.SUPPORTED_RESOURCE_IDS}")

        return ResourceComponent(resource_id, "Resource")

    @staticmethod
    def get_available_resources() -> List[int]:
        """获取所有可用的resource ID"""
        return sorted(list(ResourceComponentFactory.SUPPORTED_RESOURCE_IDS))

    @staticmethod
    def create_all_resources() -> Dict[int, ResourceComponent]:
        """创建所有可用resource"""
        return {
            resource_id: ResourceComponent(resource_id, "Resource")
            for resource_id in ResourceComponentFactory.SUPPORTED_RESOURCE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ResourceComponentFactory",
            "version": "2.0.0",
            "total_resources": len(ResourceComponentFactory.SUPPORTED_RESOURCE_IDS),
            "supported_ids": sorted(list(ResourceComponentFactory.SUPPORTED_RESOURCE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Resource组件工厂，替代原有的模板化文件"
        }

# 向后兼容：创建旧的组件实例


def create_resource_resource_component_1():

    return ResourceComponentFactory.create_component_static(1)


def create_resource_resource_component_7():

    return ResourceComponentFactory.create_component_static(7)


def create_resource_resource_component_13():

    return ResourceComponentFactory.create_component_static(13)


def create_resource_resource_component_19():

    return ResourceComponentFactory.create_component_static(19)


def create_resource_resource_component_25():

    return ResourceComponentFactory.create_component_static(25)


def create_resource_resource_component_31():

    return ResourceComponentFactory.create_component_static(31)


def create_resource_resource_component_37():

    return ResourceComponentFactory.create_component_static(37)


def create_resource_resource_component_43():

    return ResourceComponentFactory.create_component_static(43)


def create_resource_resource_component_49():

    return ResourceComponentFactory.create_component_static(49)


def create_resource_resource_component_55():

    return ResourceComponentFactory.create_component_static(55)


def create_resource_resource_component_61():

    return ResourceComponentFactory.create_component_static(61)


__all__ = [
    "IResourceProcessorComponent",
    "ResourceComponent",
    "ResourceComponentFactory",
    "create_resource_resource_component_1",
    "create_resource_resource_component_7",
    "create_resource_resource_component_13",
    "create_resource_resource_component_19",
    "create_resource_resource_component_25",
    "create_resource_resource_component_31",
    "create_resource_resource_component_37",
    "create_resource_resource_component_43",
    "create_resource_resource_component_49",
    "create_resource_resource_component_55",
    "create_resource_resource_component_61",
]
