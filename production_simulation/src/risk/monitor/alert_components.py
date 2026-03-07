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


class IAlertComponent(ABC):

    """Alert组件接口"""

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
    def get_alert_id(self) -> int:
        """获取alert ID"""


class AlertComponent(IAlertComponent):

    """统一Alert组件实现"""

    def __init__(self, alert_id: int, component_type: str = "Alert"):
        """初始化组件"""
        self.alert_id = alert_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{alert_id}"
        self.creation_time = datetime.now()

    def get_alert_id(self) -> int:
        """获取alert ID"""
        return self.alert_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "alert_id": self.alert_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_risk_component",
            "category": "monitor"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "alert_id": self.alert_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_alert_processing"
            }
            return result
        except Exception as e:
            return {
                "alert_id": self.alert_id,
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
            "alert_id": self.alert_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class AlertComponentFactory:

    """Alert组件工厂"""

    # 支持的alert ID列表
    SUPPORTED_ALERT_IDS = [5]

    @staticmethod
    def create_component(alert_id: int) -> AlertComponent:
        """创建指定ID的alert组件"""
        if alert_id not in AlertComponentFactory.SUPPORTED_ALERT_IDS:
            raise ValueError(
                f"不支持的alert ID: {alert_id}。支持的ID: {AlertComponentFactory.SUPPORTED_ALERT_IDS}")

        return AlertComponent(alert_id, "Alert")

    @staticmethod
    def get_available_alerts() -> List[int]:
        """获取所有可用的alert ID"""
        return sorted(list(AlertComponentFactory.SUPPORTED_ALERT_IDS))

    @staticmethod
    def create_all_alerts() -> Dict[int, AlertComponent]:
        """创建所有可用alert"""
        return {
            alert_id: AlertComponent(alert_id, "Alert")
            for alert_id in AlertComponentFactory.SUPPORTED_ALERT_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "AlertComponentFactory",
            "version": "2.0.0",
            "total_alerts": len(AlertComponentFactory.SUPPORTED_ALERT_IDS),
            "supported_ids": sorted(list(AlertComponentFactory.SUPPORTED_ALERT_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Alert组件工厂，替代原有的模板化文件"
        }

# 向后兼容：创建旧的组件实例


def create_alert_alert_component_5():

    return AlertComponentFactory.create_component(5)


__all__ = [
    "IAlertComponent",
    "AlertComponent",
    "AlertComponentFactory",
    "create_alert_alert_component_5",
]
