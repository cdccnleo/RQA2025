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
统一Scorer组件工厂

合并所有scorer_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:39
"""


class IScorerComponent(ABC):

    """Scorer组件接口"""

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
    def get_scorer_id(self) -> int:
        """获取scorer ID"""


class ScorerComponent(IScorerComponent):

    """统一Scorer组件实现"""

    def __init__(self, scorer_id: int, component_type: str = "Scorer"):
        """初始化组件"""
        self.scorer_id = scorer_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{scorer_id}"
        self.creation_time = datetime.now()

    def get_scorer_id(self) -> int:
        """获取scorer ID"""
        return self.scorer_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "scorer_id": self.scorer_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_backtest_evaluation_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "scorer_id": self.scorer_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_scorer_processing"
            }
            return result
        except Exception as e:
            return {
                "scorer_id": self.scorer_id,
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
            "scorer_id": self.scorer_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class ScorerComponentFactory:

    """Scorer组件工厂"""

    # 支持的scorer ID列表
    SUPPORTED_SCORER_IDS = [3, 8]

    @staticmethod
    def create_component(scorer_id: int) -> ScorerComponent:
        """创建指定ID的scorer组件"""
        if scorer_id not in ScorerComponentFactory.SUPPORTED_SCORER_IDS:
            raise ValueError(
                f"不支持的scorer ID: {scorer_id}。支持的ID: {ScorerComponentFactory.SUPPORTED_SCORER_IDS}")

        return ScorerComponent(scorer_id, "Scorer")

    @staticmethod
    def get_available_scorers() -> List[int]:
        """获取所有可用的scorer ID"""
        return sorted(list(ScorerComponentFactory.SUPPORTED_SCORER_IDS))

    @staticmethod
    def create_all_scorers() -> Dict[int, ScorerComponent]:
        """创建所有可用scorer"""
        return {
            scorer_id: ScorerComponent(scorer_id, "Scorer")
            for scorer_id in ScorerComponentFactory.SUPPORTED_SCORER_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "ScorerComponentFactory",
            "version": "2.0.0",
            "total_scorers": len(ScorerComponentFactory.SUPPORTED_SCORER_IDS),
            "supported_ids": sorted(list(ScorerComponentFactory.SUPPORTED_SCORER_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_scorer_scorer_component_3(): return ScorerComponentFactory.create_component(3)


def create_scorer_scorer_component_8(): return ScorerComponentFactory.create_component(8)


__all__ = [
    "IScorerComponent",
    "ScorerComponent",
    "ScorerComponentFactory",
    "create_scorer_scorer_component_3",
    "create_scorer_scorer_component_8",
]
