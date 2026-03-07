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
统一Workflow组件工厂

合并所有workflow_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:18:01
"""


class IWorkflowComponent(ABC):

    """Workflow组件接口"""

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
    def get_workflow_id(self) -> int:
        """获取workflow ID"""


class WorkflowComponent(IWorkflowComponent):

    """统一Workflow组件实现"""

    def __init__(self, workflow_id: int, component_type: str = "Workflow"):
        """初始化组件"""
        self.workflow_id = workflow_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{workflow_id}"
        self.creation_time = datetime.now()

    def get_workflow_id(self) -> int:
        """获取workflow ID"""
        return self.workflow_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "workflow_id": self.workflow_id,
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
                "workflow_id": self.workflow_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_workflow_processing"
            }
            return result
        except Exception as e:
            return {
                "workflow_id": self.workflow_id,
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
            "workflow_id": self.workflow_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class WorkflowComponentFactory:

    """Workflow组件工厂"""

    # 支持的workflow ID列表
    SUPPORTED_WORKFLOW_IDS = [2, 7, 12]

    @staticmethod
    def create_component(workflow_id: int) -> WorkflowComponent:
        """创建指定ID的workflow组件"""
        if workflow_id not in WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS:
            raise ValueError(
                f"不支持的workflow ID: {workflow_id}。支持的ID: {WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS}")

        return WorkflowComponent(workflow_id, "Workflow")

    @staticmethod
    def get_available_workflows() -> List[int]:
        """获取所有可用的workflow ID"""
        return sorted(list(WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS))

    @staticmethod
    def create_all_workflows() -> Dict[int, WorkflowComponent]:
        """创建所有可用workflow"""
        return {
            workflow_id: WorkflowComponent(workflow_id, "Workflow")
            for workflow_id in WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "WorkflowComponentFactory",
            "version": "2.0.0",
            "total_workflows": len(WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS),
            "supported_ids": sorted(list(WorkflowComponentFactory.SUPPORTED_WORKFLOW_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_workflow_workflow_component_2(): return WorkflowComponentFactory.create_component(2)


def create_workflow_workflow_component_7(): return WorkflowComponentFactory.create_component(7)


def create_workflow_workflow_component_12(): return WorkflowComponentFactory.create_component(12)


__all__ = [
    "IWorkflowComponent",
    "WorkflowComponent",
    "WorkflowComponentFactory",
    "create_workflow_workflow_component_2",
    "create_workflow_workflow_component_7",
    "create_workflow_workflow_component_12",
]
