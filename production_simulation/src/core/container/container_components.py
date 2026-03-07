from abc import ABC, abstractmethod

from src.core.constants import DEFAULT_BATCH_SIZE
from datetime import datetime
from typing import Dict, Any, List
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class ComponentFactory:

    """组件工厂"""

    def __init__(self):

        self._components = {}

    def create_component(self, component_type: str, config: Dict[str, Any]):
        """创建组件"""
        try:
            component = self._create_component_instance(component_type, config)
            if component and component.initialize(config):
                return component
            return None
        except Exception as e:
            logger.error(f"创建组件失败: {e}")
            return None

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """创建组件实例"""
        return None


#!/usr/bin/env python3
"""
统一Container组件工厂

合并所有container_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:20:10
"""


class IContainerComponent(ABC):

    """Container组件接口"""

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
    def get_container_id(self) -> int:
        """获取container ID"""


class ContainerComponent(IContainerComponent):

    """统一Container组件实现"""

    def __init__(self, container_id: int, component_type: str = "Container"):
        """初始化组件"""
        self.container_id = container_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{container_id}"
        self.creation_time = datetime.now()

    def get_container_id(self) -> int:
        """获取container ID"""
        return self.container_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "container_id": self.container_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_core_service_container_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "container_id": self.container_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_container_processing"
            }
            return result
        except Exception as e:
            return {
                "container_id": self.container_id,
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
            "container_id": self.container_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class ContainerComponentFactory:

    """Container组件工厂"""

    # 支持的container ID列表
    SUPPORTED_CONTAINER_IDS = [1, 6, 11]

    @staticmethod
    def create_component(container_id: int) -> ContainerComponent:
        """创建指定ID的container组件"""
        if container_id not in ContainerComponentFactory.SUPPORTED_CONTAINER_IDS:
            raise ValueError(
                f"不支持的container ID: {container_id}。支持的ID: {ContainerComponentFactory.SUPPORTED_CONTAINER_IDS}")

        return ContainerComponent(container_id, "Container")

    @staticmethod
    def get_available_containers() -> List[int]:
        """获取所有可用的container ID"""
        return sorted(list(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS))

    @staticmethod
    def create_all_containers() -> Dict[int, ContainerComponent]:
        """创建所有可用container"""
        return {
            container_id: ContainerComponent(container_id, "Container")
            for container_id in ContainerComponentFactory.SUPPORTED_CONTAINER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ContainerComponentFactory",
            "version": "2.0.0",
            "total_containers": len(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS),
            "supported_ids": sorted(list(ContainerComponentFactory.SUPPORTED_CONTAINER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_container_container_component_1(): return ContainerComponentFactory.create_component(1)


def create_container_container_component_6(): return ContainerComponentFactory.create_component(6)


def create_container_container_component_11(): return ContainerComponentFactory.create_component(11)


__all__ = [
    "IContainerComponent",
    "ContainerComponent",
    "ContainerComponentFactory",
    "create_container_container_component_1",
    "create_container_container_component_6",
    "create_container_container_component_11",
]
