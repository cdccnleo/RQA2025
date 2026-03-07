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
统一Gateway组件工厂

合并所有gateway_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:38:46
"""


class IGatewayComponent(ABC):

    """Gateway组件接口"""

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
    def get_gateway_id(self) -> int:
        """获取gateway ID"""


class GatewayComponent(IGatewayComponent):

    """统一Gateway组件实现"""

    def __init__(self, gateway_id: int, component_type: str = "Gateway"):
        """初始化组件"""
        self.gateway_id = gateway_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{gateway_id}"
        self.creation_time = datetime.now()

    def get_gateway_id(self) -> int:
        """获取gateway ID"""
        return self.gateway_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "gateway_id": self.gateway_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_gateway_api_gateway_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "gateway_id": self.gateway_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_gateway_processing"
            }
            return result
        except Exception as e:
            return {
                "gateway_id": self.gateway_id,
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
            "gateway_id": self.gateway_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class GatewayComponentFactory:

    """Gateway组件工厂"""

    # 支持的gateway ID列表
    SUPPORTED_GATEWAY_IDS = [1, 7]

    @staticmethod
    def create_component(gateway_id: int) -> GatewayComponent:
        """创建指定ID的gateway组件"""
        if gateway_id not in GatewayComponentFactory.SUPPORTED_GATEWAY_IDS:
            raise ValueError(
                f"不支持的gateway ID: {gateway_id}。支持的ID: {GatewayComponentFactory.SUPPORTED_GATEWAY_IDS}")

        return GatewayComponent(gateway_id, "Gateway")

    @staticmethod
    def get_available_gateways() -> List[int]:
        """获取所有可用的gateway ID"""
        return sorted(list(GatewayComponentFactory.SUPPORTED_GATEWAY_IDS))

    @staticmethod
    def create_all_gateways() -> Dict[int, GatewayComponent]:
        """创建所有可用gateway"""
        return {
            gateway_id: GatewayComponent(gateway_id, "Gateway")
            for gateway_id in GatewayComponentFactory.SUPPORTED_GATEWAY_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "GatewayComponentFactory",
            "version": "2.0.0",
            "total_gateways": len(GatewayComponentFactory.SUPPORTED_GATEWAY_IDS),
            "supported_ids": sorted(list(GatewayComponentFactory.SUPPORTED_GATEWAY_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_gateway_gateway_component_1(): return GatewayComponentFactory.create_component(1)


def create_gateway_gateway_component_7(): return GatewayComponentFactory.create_component(7)


__all__ = [
    "IGatewayComponent",
    "GatewayComponent",
    "GatewayComponentFactory",
    "create_gateway_gateway_component_1",
    "create_gateway_gateway_component_7",
]
