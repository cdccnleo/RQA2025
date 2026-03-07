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

# 合并后的统一组件工厂实现


class IUnifiedOptimizationComponent(ABC):

    """统一优化组件接口"""

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
    def get_component_id(self) -> int:
        """获取组件ID"""


class UnifiedOptimizationComponent(IUnifiedOptimizationComponent):

    """统一优化组件实现"""

    def __init__(self, component_id: int, component_type: str = "UnifiedOptimization"):
        """初始化组件"""
        self.component_id = component_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{component_id}"
        self.creation_time = datetime.now()

    def get_component_id(self) -> int:
        """获取组件ID"""
        return self.component_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_optimization_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "component_id": self.component_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_optimization_processing"
            }
            return result
        except Exception as e:
            return {
                "component_id": self.component_id,
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
            "component_id": self.component_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class UnifiedOptimizationFactory:

    """统一优化组件工厂

    合并了原有的OptimizationComponentFactory和EngineOptimizerComponentFactory
    支持所有组件类型的统一管理
    """

    # 合并后的支持ID列表
    SUPPORTED_COMPONENT_IDS = {
        # 原Optimization组件ID
        1, 6, 11, 16,
        # 原EngineOptimizer组件ID
        2, 7, 12, 17
    }

    # 组件类型映射
    COMPONENT_TYPE_MAPPING = {
        # Optimization组件
        1: "Optimization",
        6: "Optimization",
        11: "Optimization",
        16: "Optimization",
        # EngineOptimizer组件
        2: "EngineOptimizer",
        7: "EngineOptimizer",
        12: "EngineOptimizer",
        17: "EngineOptimizer"
    }

    @staticmethod
    def create_component(component_id: int) -> UnifiedOptimizationComponent:
        """创建指定ID的组件"""
        if component_id not in UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS:
            raise ValueError(
                f"不支持的组件ID: {component_id}。"
                f"支持的ID: {sorted(UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS)}"
            )

        component_type = UnifiedOptimizationFactory.COMPONENT_TYPE_MAPPING.get(
            component_id, "UnifiedOptimization"
        )

        return UnifiedOptimizationComponent(component_id, component_type)

    @staticmethod
    def get_available_components() -> List[int]:
        """获取所有可用的组件ID"""
        return sorted(list(UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS))

    @staticmethod
    def get_components_by_type(component_type: str) -> List[int]:
        """按类型获取组件ID"""
        return [
            cid for cid, ctype in UnifiedOptimizationFactory.COMPONENT_TYPE_MAPPING.items()
            if ctype == component_type
        ]

    @staticmethod
    def create_all_components() -> Dict[int, UnifiedOptimizationComponent]:
        """创建所有可用组件"""
        return {
            component_id: UnifiedOptimizationFactory.create_component(component_id)
            for component_id in UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS
        }

    @staticmethod
    def create_components_by_type(component_type: str) -> Dict[int, UnifiedOptimizationComponent]:
        """按类型创建组件"""
        component_ids = UnifiedOptimizationFactory.get_components_by_type(component_type)
        return {
            cid: UnifiedOptimizationFactory.create_component(cid)
            for cid in component_ids
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "UnifiedOptimizationFactory",
            "version": "2.0.0",
            "total_components": len(UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS),
            "supported_ids": sorted(list(UnifiedOptimizationFactory.SUPPORTED_COMPONENT_IDS)),
            "component_types": list(set(UnifiedOptimizationFactory.COMPONENT_TYPE_MAPPING.values())),
            "created_at": datetime.now().isoformat(),
            "description": "统一优化组件工厂，合并了原有的Optimization和EngineOptimizer工厂",
            "merge_info": {
                "merged_from": ["OptimizationComponentFactory", "EngineOptimizerComponentFactory"],
                "merge_time": datetime.now().isoformat(),
                "merge_purpose": "消除代码重复，提高维护效率"
            }
        }


# 向后兼容：创建旧的组件实例
# 原Optimization组件


def create_optimization_optimization_component_1():

    return UnifiedOptimizationFactory.create_component(1)


def create_optimization_optimization_component_6():

    return UnifiedOptimizationFactory.create_component(6)


def create_optimization_optimization_component_11():

    return UnifiedOptimizationFactory.create_component(11)


def create_optimization_optimization_component_16():

    return UnifiedOptimizationFactory.create_component(16)

# 原EngineOptimizer组件


def create_engineoptimizer_optimizer_component_2():

    return UnifiedOptimizationFactory.create_component(2)


def create_engineoptimizer_optimizer_component_7():

    return UnifiedOptimizationFactory.create_component(7)


def create_engineoptimizer_optimizer_component_12():

    return UnifiedOptimizationFactory.create_component(12)


def create_engineoptimizer_optimizer_component_17():

    return UnifiedOptimizationFactory.create_component(17)


__all__ = [
    "ComponentFactory",
    "IUnifiedOptimizationComponent",
    "UnifiedOptimizationComponent",
    "UnifiedOptimizationFactory",
    # 向后兼容函数
    "create_optimization_optimization_component_1",
    "create_optimization_optimization_component_6",
    "create_optimization_optimization_component_11",
    "create_optimization_optimization_component_16",
    "create_engineoptimizer_optimizer_component_2",
    "create_engineoptimizer_optimizer_component_7",
    "create_engineoptimizer_optimizer_component_12",
    "create_engineoptimizer_optimizer_component_17",
]

