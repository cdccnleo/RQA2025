from abc import ABC, abstractmethod
from enum import Enum

from src.core.constants import DEFAULT_BATCH_SIZE
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ComponentStatus(Enum):
    """组件状态枚举"""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"


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

    def __init__(self, name: str, version: str = "1.0.0", description: str = "Container Component"):
        """初始化组件"""
        self.name = name
        self.version = version
        self.description = description
        self._status = "CREATED"
        self.creation_time = datetime.now()
        self.container_id = 1  # 默认container ID

    def get_status(self):
        """获取状态"""
        return ComponentStatus(self._status)

    def initialize(self) -> bool:
        """初始化"""
        self._status = "RUNNING"
        return True

    def shutdown(self) -> bool:
        """关闭"""
        self._status = "STOPPED"
        return True

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

    def get_status(self) -> Any:
        """获取组件状态"""
        # 返回一个具有name属性的对象以兼容测试
        class Status:
            def __init__(self, name: str):
                self.name = name
        return Status(self._status)



class ContainerComponentFactory:

    """Container组件工厂"""

    # 支持的container ID列表
    SUPPORTED_CONTAINER_IDS = [1, 6, 11]

    def __init__(self):
        """初始化工厂"""
        self._registered_types = {}
        self._registered_types["Container"] = lambda config: ContainerComponent(
            config.get("container_id", 1), config.get("component_type", "Container"))

    def register_component_type(self, component_type: str, component_factory: callable) -> None:
        """注册组件类型"""
        self._registered_types[component_type] = component_factory

    def create_component(self, component_type: str, config: Dict[str, Any]) -> Optional[ContainerComponent]:
        """创建指定类型和配置的组件"""
        try:
            if component_type not in self._registered_types:
                return None

            factory_func = self._registered_types[component_type]
            return factory_func(config)
        except Exception:
            return None

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
