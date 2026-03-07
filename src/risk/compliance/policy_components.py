from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
import time

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


"""
统一Policy组件工厂

合并所有policy_*.py模板文件为统一的管理架构
生成时间: 2025-08-24 10:13:48
"""


class IPolicyComponent(ABC):

    """Policy组件接口"""

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        pass

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        pass

    @abstractmethod
    def get_policy_id(self) -> int:
        """获取policy ID"""
        pass


class PolicyComponent(IPolicyComponent):

    """统一Policy组件实现"""


    def __init__(self, policy_id: int, component_type: str = "Policy"):
        """初始化组件"""
        self.policy_id = policy_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{policy_id}"
        self.creation_time = datetime.now()

    def get_policy_id(self) -> int:
        """获取policy ID"""
        return self.policy_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "policy_id": self.policy_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_risk_component",
            "category": "compliance"
        }


    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "policy_id": self.policy_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_policy_processing"
            }
            return result
        except Exception as e:
            return {
                "policy_id": self.policy_id,
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
            "policy_id": self.policy_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class PolicyComponentFactory:

    """Policy组件工厂"""

    # 支持的policy ID列表
    SUPPORTED_POLICY_IDS = [3, 8]

    @staticmethod
    def create_component(policy_id: int) -> PolicyComponent:
        """创建指定ID的policy组件"""
        if policy_id not in PolicyComponentFactory.SUPPORTED_POLICY_IDS:
            raise ValueError(
                f"不支持的policy ID: {policy_id}。支持的ID: {PolicyComponentFactory.SUPPORTED_POLICY_IDS}")

        return PolicyComponent(policy_id, "Policy")

    @staticmethod
    def get_available_policys() -> List[int]:
        """获取所有可用的policy ID"""
        return sorted(list(PolicyComponentFactory.SUPPORTED_POLICY_IDS))

    @staticmethod
    def create_all_policys() -> Dict[int, PolicyComponent]:
        """创建所有可用policy"""
        return {
            policy_id: PolicyComponent(policy_id, "Policy")
            for policy_id in PolicyComponentFactory.SUPPORTED_POLICY_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "PolicyComponentFactory",
            "version": "2.0.0",
            "total_policys": len(PolicyComponentFactory.SUPPORTED_POLICY_IDS),
            "supported_ids": sorted(list(PolicyComponentFactory.SUPPORTED_POLICY_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": f"统一Policy组件工厂，替代原有的{len(PolicyComponentFactory.SUPPORTED_POLICY_IDS)}个模板化文件"
        }

# 向后兼容：创建旧的组件实误


def create_policy_policy_component_3():
    return PolicyComponentFactory.create_component(3)


def create_policy_policy_component_8():
    return PolicyComponentFactory.create_component(8)

__all__ = [
    "IPolicyComponent",
    "PolicyComponent",
    "PolicyComponentFactory",
    "create_policy_policy_component_3",
    "create_policy_policy_component_8",
]
