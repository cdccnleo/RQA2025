"""
monitor_components 模块

提供 monitor_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from infrastructure.utils.common.core.base_components import ComponentFactory
from typing import Dict, Any, List
"""
基础设施层 - Monitor组件统一实现

使用统一的ComponentFactory基类，提供Monitor组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class IMonitorComponent(ABC):

    """Monitor组件接口"""

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
    def get_monitor_id(self) -> int:
        """获取monitor ID"""


class MonitorComponent(IMonitorComponent):

    """统一Monitor组件实现"""

    def __init__(self, monitor_id: int, component_type: str = "Monitor"):
        """初始化组件"""
        self.monitor_id = monitor_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{monitor_id}"
        self.creation_time = datetime.now()

    def get_monitor_id(self) -> int:
        """获取monitor ID"""
        return self.monitor_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "monitor_id": self.monitor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_resource_management_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "monitor_id": self.monitor_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_monitor_processing"
            }
            return result
        except Exception as e:
            return {
                "monitor_id": self.monitor_id,
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
            "monitor_id": self.monitor_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class MonitorComponentFactory(ComponentFactory):

    """Monitor组件工厂"""

    # 支持的monitor ID列表
    def __init__(self):
        super().__init__()
        # 注册组件工厂函数

    SUPPORTED_MONITOR_IDS = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63]

    @staticmethod
    def create_component(monitor_id: int) -> MonitorComponent:
        """创建指定ID的monitor组件"""
        if monitor_id not in MonitorComponentFactory.SUPPORTED_MONITOR_IDS:
            raise ValueError(
                f"不支持的monitor ID: {monitor_id}。支持的ID: {MonitorComponentFactory.SUPPORTED_MONITOR_IDS}")

        return MonitorComponent(monitor_id, "Monitor")

    @staticmethod
    def get_available_monitors() -> List[int]:
        """获取所有可用的monitor ID"""
        return sorted(list(MonitorComponentFactory.SUPPORTED_MONITOR_IDS))

    @staticmethod
    def create_all_monitors() -> Dict[int, MonitorComponent]:
        """创建所有可用monitor"""
        return {
            monitor_id: MonitorComponent(monitor_id, "Monitor")
            for monitor_id in MonitorComponentFactory.SUPPORTED_MONITOR_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "MonitorComponentFactory",
            "version": "2.0.0",
            "total_monitors": len(MonitorComponentFactory.SUPPORTED_MONITOR_IDS),
            "supported_ids": sorted(list(MonitorComponentFactory.SUPPORTED_MONITOR_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }

# 向后兼容：创建旧的组件实例


def create_monitor_monitor_component_3():

    return MonitorComponentFactory.create_component(3)


def create_monitor_monitor_component_9():

    return MonitorComponentFactory.create_component(9)


def create_monitor_monitor_component_15():

    return MonitorComponentFactory.create_component(15)


def create_monitor_monitor_component_21():

    return MonitorComponentFactory.create_component(21)


def create_monitor_monitor_component_27():

    return MonitorComponentFactory.create_component(27)


def create_monitor_monitor_component_33():

    return MonitorComponentFactory.create_component(33)


def create_monitor_monitor_component_39():

    return MonitorComponentFactory.create_component(39)


def create_monitor_monitor_component_45():

    return MonitorComponentFactory.create_component(45)


def create_monitor_monitor_component_51():

    return MonitorComponentFactory.create_component(51)


def create_monitor_monitor_component_57():

    return MonitorComponentFactory.create_component(57)


def create_monitor_monitor_component_63():

    return MonitorComponentFactory.create_component(63)


__all__ = [
    "IMonitorComponent",
    "MonitorComponent",
    "MonitorComponentFactory",
    "create_monitor_monitor_component_3",
    "create_monitor_monitor_component_9",
    "create_monitor_monitor_component_15",
    "create_monitor_monitor_component_21",
    "create_monitor_monitor_component_27",
    "create_monitor_monitor_component_33",
    "create_monitor_monitor_component_39",
    "create_monitor_monitor_component_45",
    "create_monitor_monitor_component_51",
    "create_monitor_monitor_component_57",
    "create_monitor_monitor_component_63",
]
