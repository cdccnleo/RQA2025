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
统一Engine组件工厂

合并所有engine_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:06
"""


class IEngineComponent(ABC):

    """Engine组件接口"""

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
    def get_engine_id(self) -> int:
        """获取engine ID"""


class EngineComponent(IEngineComponent):

    """统一Engine组件实现"""

    def __init__(self, engine_id: int, component_type: str = "Engine"):
        """初始化组件"""
        self.engine_id = engine_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{engine_id}"
        self.creation_time = datetime.now()

    def get_engine_id(self) -> int:
        """获取engine ID"""
        return self.engine_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "engine_id": self.engine_id,
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
                "engine_id": self.engine_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_engine_processing"
            }
            return result
        except Exception as e:
            return {
                "engine_id": self.engine_id,
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
            "engine_id": self.engine_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class EngineComponentFactory:

    """Engine组件工厂"""

    # 支持的engine ID列表
    SUPPORTED_ENGINE_IDS = [1, 6, 11, 16]

    @staticmethod
    def create_component(engine_id: int) -> EngineComponent:
        """创建指定ID的engine组件"""
        if engine_id not in EngineComponentFactory.SUPPORTED_ENGINE_IDS:
            raise ValueError(
                f"不支持的engine ID: {engine_id}。支持的ID: {EngineComponentFactory.SUPPORTED_ENGINE_IDS}")

        return EngineComponent(engine_id, "Engine")

    @staticmethod
    def get_available_engines() -> List[int]:
        """获取所有可用的engine ID"""
        return sorted(list(EngineComponentFactory.SUPPORTED_ENGINE_IDS))

    @staticmethod
    def create_all_engines() -> Dict[int, EngineComponent]:
        """创建所有可用engine"""
        return {
            engine_id: EngineComponent(engine_id, "Engine")
            for engine_id in EngineComponentFactory.SUPPORTED_ENGINE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "EngineComponentFactory",
            "version": "2.0.0",
            "total_engines": len(EngineComponentFactory.SUPPORTED_ENGINE_IDS),
            "supported_ids": sorted(list(EngineComponentFactory.SUPPORTED_ENGINE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_engine_engine_component_1(): return EngineComponentFactory.create_component(1)


def create_engine_engine_component_6(): return EngineComponentFactory.create_component(6)


def create_engine_engine_component_11(): return EngineComponentFactory.create_component(11)


def create_engine_engine_component_16(): return EngineComponentFactory.create_component(16)


__all__ = [
    "IEngineComponent",
    "EngineComponent",
    "EngineComponentFactory",
    "create_engine_engine_component_1",
    "create_engine_engine_component_6",
    "create_engine_engine_component_11",
    "create_engine_engine_component_16",
]
