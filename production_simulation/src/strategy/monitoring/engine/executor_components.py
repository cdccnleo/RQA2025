from abc import ABC, abstractmethod
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
统一Executor组件工厂

合并所有executor_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:06
"""


class IExecutorComponent(ABC):

    """Executor组件接口"""

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
    def get_executor_id(self) -> int:
        """获取executor ID"""


class ExecutorComponent(IExecutorComponent):

    """统一Executor组件实现"""

    def __init__(self, executor_id: int, component_type: str = "Executor"):
        """初始化组件"""
        self.executor_id = executor_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{executor_id}"
        self.creation_time = datetime.now()

    def get_executor_id(self) -> int:
        """获取executor ID"""
        return self.executor_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "executor_id": self.executor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_backtest_engine_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "executor_id": self.executor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_executor_processing"
            }
            return result
        except Exception as e:
            return {
                "executor_id": self.executor_id,
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
            "executor_id": self.executor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class ExecutorComponentFactory:

    """Executor组件工厂"""

    # 支持的executor ID列表
    SUPPORTED_EXECUTOR_IDS = [5, 10, 15]

    @staticmethod
    def create_component(executor_id: int) -> ExecutorComponent:
        """创建指定ID的executor组件"""
        if executor_id not in ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS:
            raise ValueError(
                f"不支持的executor ID: {executor_id}。支持的ID: {ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS}")

        return ExecutorComponent(executor_id, "Executor")

    @staticmethod
    def get_available_executors() -> List[int]:
        """获取所有可用的executor ID"""
        return sorted(list(ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS))

    @staticmethod
    def create_all_executors() -> Dict[int, ExecutorComponent]:
        """创建所有可用executor"""
        return {
            executor_id: ExecutorComponent(executor_id, "Executor")
            for executor_id in ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ExecutorComponentFactory",
            "version": "2.0.0",
            "total_executors": len(ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS),
            "supported_ids": sorted(list(ExecutorComponentFactory.SUPPORTED_EXECUTOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_executor_executor_component_5(): return ExecutorComponentFactory.create_component(5)


def create_executor_executor_component_10(): return ExecutorComponentFactory.create_component(10)


def create_executor_executor_component_15(): return ExecutorComponentFactory.create_component(15)


__all__ = [
    "IExecutorComponent",
    "ExecutorComponent",
    "ExecutorComponentFactory",
    "create_executor_executor_component_5",
    "create_executor_executor_component_10",
    "create_executor_executor_component_15",
]
