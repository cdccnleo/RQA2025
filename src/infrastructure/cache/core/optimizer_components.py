"""
optimizer_components 模块

提供 optimizer_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类和基础缓存组件

from .base import BaseCacheComponent
from datetime import datetime
try:
    from src.infrastructure.utils.components.core.base_components import ComponentFactory
except ImportError:
    # Fallback for cases where the import path might be different
    ComponentFactory = None
from typing import Dict, Any, List, Protocol
"""
基础设施层 - Optimizer组件统一实现

使用统一的ComponentFactory基类，提供Optimizer组件的工厂模式实现。
"""

# from abc import ABC, abstractmethod  # Removed for Protocol conversion
logger = logging.getLogger(__name__)


class IOptimizerComponent(Protocol):

    """OptimizerComponent协议

    定义优化器组件的标准接口。
    使用Protocol模式支持结构化子类型。
    """

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        ...

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        ...

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        ...

    def get_component_id(self) -> int:
        """获取组件ID"""
        ...


class OptimizerComponent(BaseCacheComponent):

    """统一OptimizerComponent实现"""

    def __init__(self, component_id: int):
        """初始化组件"""
        super().__init__(
            component_id=component_id,
            component_type="Cache"
        )
        self._description = "optimizer_templates的统一组件实现"

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        info = {
            "component_id": self.component_id,
            "component_type": self.component_type,
            "component_name": self.component_name,
            "description": self._description,
            "type": "unified_optimizer_component",
            "status": self._status,
            "initialized": self._initialized
        }
        return info

    def get_processing_type(self) -> str:
        """获取处理类型"""
        return "optimizer_processing"

    def get_component_type_name(self) -> str:
        """获取组件类型名称"""
        return "optimizer"


if ComponentFactory is not None:
    class OptimizerComponentFactory(ComponentFactory):
        """OptimizerComponent工厂"""
        
        # 支持的组件ID列表
        SUPPORTED_COMPONENT_IDS = [11, 17, 23]
        
        def __init__(self):
            super().__init__()
            # 注册组件工厂函数
        
        @staticmethod
        def create_component(component_id: int) -> OptimizerComponent:
            """创建指定ID的组件"""
            if component_id not in OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS:
                raise ValueError(
                    f"不支持的组件ID: {component_id}。支持的ID: {OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS}")

            return OptimizerComponent(component_id)

        @staticmethod
        def get_available_components() -> List[int]:
            """获取所有可用的组件ID"""
            return sorted(list(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS))

        @staticmethod
        def create_all_components() -> Dict[int, OptimizerComponent]:
            """创建所有可用组件"""
            return {
                component_id: OptimizerComponent(component_id)
                for component_id in OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS
            }

        @staticmethod
        def get_component_info() -> Dict[str, Any]:
            """获取组件工厂信息"""
            return {
                "factory_name": "OptimizerComponentFactory",
                "version": "2.0.0",
                "total_components": len(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS),
                "supported_ids": sorted(list(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS)),
                "created_at": datetime.now().isoformat(),
                "description": "统一optimizer_templates工厂，替代原有的模板化文件"
            }
else:
    # Fallback when ComponentFactory is not available
    class OptimizerComponentFactory:
        """OptimizerComponent工厂 - 独立实现"""
        
        # 支持的组件ID列表
        SUPPORTED_COMPONENT_IDS = [11, 17, 23]
        
        def __init__(self):
            # 独立实现，不需要调用super()
            pass

        @staticmethod
        def create_component(component_id: int) -> OptimizerComponent:
            """创建指定ID的组件"""
            if component_id not in OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS:
                raise ValueError(
                    f"不支持的组件ID: {component_id}。支持的ID: {OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS}")

            return OptimizerComponent(component_id)

        @staticmethod
        def get_available_components() -> List[int]:
            """获取所有可用的组件ID"""
            return sorted(list(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS))

        @staticmethod
        def create_all_components() -> Dict[int, OptimizerComponent]:
            """创建所有可用组件"""
            return {
                component_id: OptimizerComponent(component_id)
                for component_id in OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS
            }

        @staticmethod
        def get_component_info() -> Dict[str, Any]:
            """获取组件工厂信息"""
            return {
                "factory_name": "OptimizerComponentFactory",
                "version": "2.0.0",
                "total_components": len(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS),
                "supported_ids": sorted(list(OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS)),
                "created_at": datetime.now().isoformat(),
                "description": "统一optimizer_templates工厂，替代原有的模板化文件"
            }

# 向后兼容：创建旧的组件实例


def create_optimizer_component_11():

    return OptimizerComponentFactory.create_component(11)


def create_optimizer_component_17():

    return OptimizerComponentFactory.create_component(17)


def create_optimizer_component_23():

    return OptimizerComponentFactory.create_component(23)


__all__ = [
    "IOptimizerComponent",
    "OptimizerComponent",
    "OptimizerComponentFactory",
    "create_optimizer_component_11",
    "create_optimizer_component_17",
    "create_optimizer_component_23",
]
