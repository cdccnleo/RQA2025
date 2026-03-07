import logging
from typing import Dict, Any, List
from datetime import datetime
from abc import ABC, abstractmethod

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


# !/usr/bin/env python3
"""
统一Speed组件工厂

合并所有speed_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:33:40
"""


class ISpeedComponent(ABC):

    """Speed组件接口"""

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
    def get_speed_id(self) -> int:
        """获取speed ID"""


class SpeedComponent(ISpeedComponent):

    """统一Speed组件实现"""

    def __init__(self, speed_id: int, component_type: str = "Speed"):
        """初始化组件"""
        self.speed_id = speed_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{speed_id}"
        self.creation_time = datetime.now()

    def get_speed_id(self) -> int:
        """获取speed ID"""
        return self.speed_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "speed_id": self.speed_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_engine_optimization_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "speed_id": self.speed_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_speed_processing"
            }
            return result
        except Exception as e:
            return {
                "speed_id": self.speed_id,
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
            "speed_id": self.speed_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class SpeedComponentFactory:

    """Speed组件工厂"""

    # 支持的speed ID列表
    SUPPORTED_SPEED_IDS = [4, 9, 14, 19]

    @staticmethod
    def create_component(speed_id: int) -> SpeedComponent:
        """创建指定ID的speed组件"""
        if speed_id not in SpeedComponentFactory.SUPPORTED_SPEED_IDS:
            raise ValueError(
                f"不支持的speed ID: {speed_id}。支持的ID: {SpeedComponentFactory.SUPPORTED_SPEED_IDS}")

        return SpeedComponent(speed_id, "Speed")

    @staticmethod
    def get_available_speeds() -> List[int]:
        """获取所有可用的speed ID"""
        return sorted(list(SpeedComponentFactory.SUPPORTED_SPEED_IDS))

    @staticmethod
    def create_all_speeds() -> Dict[int, SpeedComponent]:
        """创建所有可用speed"""
        return {
            speed_id: SpeedComponent(speed_id, "Speed")
            for speed_id in SpeedComponentFactory.SUPPORTED_SPEED_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "SpeedComponentFactory",
            "version": "2.0.0",
            "total_speeds": len(SpeedComponentFactory.SUPPORTED_SPEED_IDS),
            "supported_ids": sorted(list(SpeedComponentFactory.SUPPORTED_SPEED_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_speed_speed_component_4(): return SpeedComponentFactory.create_component(4)


def create_speed_speed_component_9(): return SpeedComponentFactory.create_component(9)


def create_speed_speed_component_14(): return SpeedComponentFactory.create_component(14)


def create_speed_speed_component_19(): return SpeedComponentFactory.create_component(19)


__all__ = [
    "ISpeedComponent",
    "SpeedComponent",
    "SpeedComponentFactory",
    "create_speed_speed_component_4",
    "create_speed_speed_component_9",
    "create_speed_speed_component_14",
    "create_speed_speed_component_19",
]

