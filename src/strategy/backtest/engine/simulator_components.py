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
统一Simulator组件工厂

合并所有simulator_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:06
"""


class ISimulatorComponent(ABC):

    """Simulator组件接口"""

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
    def get_simulator_id(self) -> int:
        """获取simulator ID"""


class SimulatorComponent(ISimulatorComponent):

    """统一Simulator组件实现"""

    def __init__(self, simulator_id: int, component_type: str = "Simulator"):
        """初始化组件"""
        self.simulator_id = simulator_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{simulator_id}"
        self.creation_time = datetime.now()

    def get_simulator_id(self) -> int:
        """获取simulator ID"""
        return self.simulator_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "simulator_id": self.simulator_id,
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
                "simulator_id": self.simulator_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_simulator_processing"
            }
            return result
        except Exception as e:
            return {
                "simulator_id": self.simulator_id,
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
            "simulator_id": self.simulator_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class SimulatorComponentFactory:

    """Simulator组件工厂"""

    # 支持的simulator ID列表
    SUPPORTED_SIMULATOR_IDS = [3, 8, 13, 18]

    @staticmethod
    def create_component(simulator_id: int) -> SimulatorComponent:
        """创建指定ID的simulator组件"""
        if simulator_id not in SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS:
            raise ValueError(
                f"不支持的simulator ID: {simulator_id}。支持的ID: {SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS}")

        return SimulatorComponent(simulator_id, "Simulator")

    @staticmethod
    def get_available_simulators() -> List[int]:
        """获取所有可用的simulator ID"""
        return sorted(list(SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS))

    @staticmethod
    def create_all_simulators() -> Dict[int, SimulatorComponent]:
        """创建所有可用simulator"""
        return {
            simulator_id: SimulatorComponent(simulator_id, "Simulator")
            for simulator_id in SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "SimulatorComponentFactory",
            "version": "2.0.0",
            "total_simulators": len(SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS),
            "supported_ids": sorted(list(SimulatorComponentFactory.SUPPORTED_SIMULATOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_simulator_simulator_component_3(): return SimulatorComponentFactory.create_component(3)


def create_simulator_simulator_component_8(): return SimulatorComponentFactory.create_component(8)


def create_simulator_simulator_component_13(): return SimulatorComponentFactory.create_component(13)


def create_simulator_simulator_component_18(): return SimulatorComponentFactory.create_component(18)


__all__ = [
    "ISimulatorComponent",
    "SimulatorComponent",
    "SimulatorComponentFactory",
    "create_simulator_simulator_component_3",
    "create_simulator_simulator_component_8",
    "create_simulator_simulator_component_13",
    "create_simulator_simulator_component_18",
]
