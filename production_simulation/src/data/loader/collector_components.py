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
统一Collector组件工厂

合并所有collector_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:23:44
"""


class ICollectorComponent(ABC):

    """Collector组件接口"""

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
    def get_collector_id(self) -> int:
        """获取collector ID"""


class CollectorComponent(ICollectorComponent):

    """统一Collector组件实现"""

    def __init__(self, collector_id: int, component_type: str = "Collector"):
        """初始化组件"""
        self.collector_id = collector_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{collector_id}"
        self.creation_time = datetime.now()

    def get_collector_id(self) -> int:
        """获取collector ID"""
        return self.collector_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "collector_id": self.collector_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_data_loader_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "collector_id": self.collector_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_collector_processing"
            }
            return result
        except Exception as e:
            return {
                "collector_id": self.collector_id,
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
            "collector_id": self.collector_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class CollectorComponentFactory:

    """Collector组件工厂"""

    # 支持的collector ID列表
    SUPPORTED_COLLECTOR_IDS = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    @staticmethod
    def create_component(collector_id: int) -> CollectorComponent:
        """创建指定ID的collector组件"""
        if collector_id not in CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS:
            raise ValueError(
                f"不支持的collector ID: {collector_id}。支持的ID: {CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS}")

        return CollectorComponent(collector_id, "Collector")

    @staticmethod
    def get_available_collectors() -> List[int]:
        """获取所有可用的collector ID"""
        return sorted(list(CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS))

    @staticmethod
    def create_all_collectors() -> Dict[int, CollectorComponent]:
        """创建所有可用collector"""
        return {
            collector_id: CollectorComponent(collector_id, "Collector")
            for collector_id in CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "CollectorComponentFactory",
            "version": "2.0.0",
            "total_collectors": len(CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS),
            "supported_ids": sorted(list(CollectorComponentFactory.SUPPORTED_COLLECTOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_collector_collector_component_5(): return CollectorComponentFactory.create_component(5)


def create_collector_collector_component_10(): return CollectorComponentFactory.create_component(10)


def create_collector_collector_component_15(): return CollectorComponentFactory.create_component(15)


def create_collector_collector_component_20(): return CollectorComponentFactory.create_component(20)


def create_collector_collector_component_25(): return CollectorComponentFactory.create_component(25)


def create_collector_collector_component_30(): return CollectorComponentFactory.create_component(30)


def create_collector_collector_component_35(): return CollectorComponentFactory.create_component(35)


def create_collector_collector_component_40(): return CollectorComponentFactory.create_component(40)


def create_collector_collector_component_45(): return CollectorComponentFactory.create_component(45)


__all__ = [
    "ICollectorComponent",
    "CollectorComponent",
    "CollectorComponentFactory",
    "create_collector_collector_component_5",
    "create_collector_collector_component_10",
    "create_collector_collector_component_15",
    "create_collector_collector_component_20",
    "create_collector_collector_component_25",
    "create_collector_collector_component_30",
    "create_collector_collector_component_35",
    "create_collector_collector_component_40",
    "create_collector_collector_component_45",
]
