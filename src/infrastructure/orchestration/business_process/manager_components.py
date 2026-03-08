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
统一BusinessProcessManager组件工厂

合并所有manager_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 09:29:26
"""


class IManagerComponent(ABC):

    """Manager组件接口"""

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
    def get_manager_id(self) -> int:
        """获取管理器ID"""


class ManagerComponent(IManagerComponent):

    """统一Manager组件实现"""

    def __init__(self, manager_id: int, component_type: str = "BusinessProcessManager"):
        """初始化组件"""
        self.manager_id = manager_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{manager_id}"
        self.creation_time = datetime.now()

    def get_manager_id(self) -> int:
        """获取管理器ID"""
        return self.manager_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "manager_id": self.manager_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_manager_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "manager_id": self.manager_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_manager_processing"
            }
            return result
        except Exception as e:
            return {
                "manager_id": self.manager_id,
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
            "manager_id": self.manager_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class BusinessProcessManagerComponentFactory:

    """BusinessProcessManager组件工厂"""

    # 支持的管理器ID列表
    SUPPORTED_MANAGER_IDS = [5, 10]

    @staticmethod
    def create_component(manager_id: int) -> ManagerComponent:
        """创建指定ID的管理器组件"""
        if manager_id not in BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS:
            raise ValueError(
                f"不支持的管理器ID: {manager_id}。支持的ID: {BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS}")

        return ManagerComponent(manager_id, "BusinessProcessManager")

    @staticmethod
    def get_available_managers() -> List[int]:
        """获取所有可用的管理器ID"""
        return sorted(list(BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS))

    @staticmethod
    def create_all_managers() -> Dict[int, ManagerComponent]:
        """创建所有可用管理器"""
        return {
            manager_id: ManagerComponent(manager_id, "BusinessProcessManager")
            for manager_id in BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "BusinessProcessManagerComponentFactory",
            "version": "2.0.0",
            "total_managers": len(BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS),
            "supported_ids": sorted(list(BusinessProcessManagerComponentFactory.SUPPORTED_MANAGER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_businessprocessmanager_manager_component_5(
): return BusinessProcessManagerComponentFactory.create_component(5)


def create_businessprocessmanager_manager_component_10(
): return BusinessProcessManagerComponentFactory.create_component(10)


__all__ = [
    "IManagerComponent",
    "ManagerComponent",
    "BusinessProcessManagerComponentFactory",
    "create_businessprocessmanager_manager_component_5",
    "create_businessprocessmanager_manager_component_10",
]
