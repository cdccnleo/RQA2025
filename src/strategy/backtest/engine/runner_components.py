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
统一Runner组件工厂

合并所有runner_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:06
"""


class IRunnerComponent(ABC):

    """Runner组件接口"""

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
    def get_runner_id(self) -> int:
        """获取runner ID"""


class RunnerComponent(IRunnerComponent):

    """统一Runner组件实现"""

    def __init__(self, runner_id: int, component_type: str = "Runner"):
        """初始化组件"""
        self.runner_id = runner_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{runner_id}"
        self.creation_time = datetime.now()

    def get_runner_id(self) -> int:
        """获取runner ID"""
        return self.runner_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "runner_id": self.runner_id,
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
                "runner_id": self.runner_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_runner_processing"
            }
            return result
        except Exception as e:
            return {
                "runner_id": self.runner_id,
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
            "runner_id": self.runner_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class RunnerComponentFactory:

    """Runner组件工厂"""

    # 支持的runner ID列表
    SUPPORTED_RUNNER_IDS = [4, 9, 14, 19]

    @staticmethod
    def create_component(runner_id: int) -> RunnerComponent:
        """创建指定ID的runner组件"""
        if runner_id not in RunnerComponentFactory.SUPPORTED_RUNNER_IDS:
            raise ValueError(
                f"不支持的runner ID: {runner_id}。支持的ID: {RunnerComponentFactory.SUPPORTED_RUNNER_IDS}")

        return RunnerComponent(runner_id, "Runner")

    @staticmethod
    def get_available_runners() -> List[int]:
        """获取所有可用的runner ID"""
        return sorted(list(RunnerComponentFactory.SUPPORTED_RUNNER_IDS))

    @staticmethod
    def create_all_runners() -> Dict[int, RunnerComponent]:
        """创建所有可用runner"""
        return {
            runner_id: RunnerComponent(runner_id, "Runner")
            for runner_id in RunnerComponentFactory.SUPPORTED_RUNNER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "RunnerComponentFactory",
            "version": "2.0.0",
            "total_runners": len(RunnerComponentFactory.SUPPORTED_RUNNER_IDS),
            "supported_ids": sorted(list(RunnerComponentFactory.SUPPORTED_RUNNER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_runner_runner_component_4(): return RunnerComponentFactory.create_component(4)


def create_runner_runner_component_9(): return RunnerComponentFactory.create_component(9)


def create_runner_runner_component_14(): return RunnerComponentFactory.create_component(14)


def create_runner_runner_component_19(): return RunnerComponentFactory.create_component(19)


__all__ = [
    "IRunnerComponent",
    "RunnerComponent",
    "RunnerComponentFactory",
    "create_runner_runner_component_4",
    "create_runner_runner_component_9",
    "create_runner_runner_component_14",
    "create_runner_runner_component_19",
]
