from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List
import logging
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


# -*- coding: utf-8 -*-
# #!/usr/bin/env python3
"""
统一Coordinator组件工厂

合并所有coordinator_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:18:01
"""


class ICoordinatorComponent(ABC):

    """Coordinator组件接口"""

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
    def get_coordinator_id(self) -> int:
        """获取coordinator ID"""


class CoordinatorComponent(ICoordinatorComponent):

    """统一Coordinator组件实现"""

    def __init__(self, coordinator_id: int, component_type: str = "Coordinator"):
        """初始化组件"""
        self.coordinator_id = coordinator_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{coordinator_id}"
        self.creation_time = datetime.now()

    def get_coordinator_id(self) -> int:
        """获取coordinator ID"""
        return self.coordinator_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "coordinator_id": self.coordinator_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_core_business_process_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "coordinator_id": self.coordinator_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_coordinator_processing"
            }
            return result
        except Exception as e:
            return {
                "coordinator_id": self.coordinator_id,
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
            "coordinator_id": self.coordinator_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class CoordinatorComponentFactory:

    """Coordinator组件工厂"""

    # 支持的coordinator ID列表
    SUPPORTED_COORDINATOR_IDS = [4, 9, 14]

    @staticmethod
    def create_component(coordinator_id: int) -> CoordinatorComponent:
        """创建指定ID的coordinator组件"""
        if coordinator_id not in CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS:
            raise ValueError(
                f"不支持的coordinator ID: {coordinator_id}。支持的ID: {CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS}")

        return CoordinatorComponent(coordinator_id, "Coordinator")

    @staticmethod
    def get_available_coordinators() -> List[int]:
        """获取所有可用的coordinator ID"""
        return sorted(list(CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS))

    @staticmethod
    def create_all_coordinators() -> Dict[int, CoordinatorComponent]:
        """创建所有可用coordinator"""
        return {
            coordinator_id: CoordinatorComponent(coordinator_id, "Coordinator")
            for coordinator_id in CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "CoordinatorComponentFactory",
            "version": "2.0.0",
            "total_coordinators": len(CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS),
            "supported_ids": sorted(list(CoordinatorComponentFactory.SUPPORTED_COORDINATOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_coordinator_coordinator_component_4(): return CoordinatorComponentFactory.create_component(4)


def create_coordinator_coordinator_component_9(): return CoordinatorComponentFactory.create_component(9)


def create_coordinator_coordinator_component_14(): return CoordinatorComponentFactory.create_component(14)


__all__ = [
    "ICoordinatorComponent",
    "CoordinatorComponent",
    "CoordinatorComponentFactory",
    "create_coordinator_coordinator_component_4",
    "create_coordinator_coordinator_component_9",
    "create_coordinator_coordinator_component_14",
]
