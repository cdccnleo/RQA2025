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
统一Evaluation组件工厂

合并所有evaluation_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:39
"""


class IEvaluationComponent(ABC):

    """Evaluation组件接口"""

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
    def get_evaluation_id(self) -> int:
        """获取evaluation ID"""


class EvaluationComponent(IEvaluationComponent):

    """统一Evaluation组件实现"""

    def __init__(self, evaluation_id: int, component_type: str = "Evaluation"):
        """初始化组件"""
        self.evaluation_id = evaluation_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{evaluation_id}"
        self.creation_time = datetime.now()

    def get_evaluation_id(self) -> int:
        """获取evaluation ID"""
        return self.evaluation_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "evaluation_id": self.evaluation_id,
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
                "evaluation_id": self.evaluation_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_evaluation_processing"
            }
            return result
        except Exception as e:
            return {
                "evaluation_id": self.evaluation_id,
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
            "evaluation_id": self.evaluation_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class EvaluationComponentFactory:

    """Evaluation组件工厂"""

    # 支持的evaluation ID列表
    SUPPORTED_EVALUATION_IDS = [1, 6]

    @staticmethod
    def create_component(evaluation_id: int) -> EvaluationComponent:
        """创建指定ID的evaluation组件"""
        if evaluation_id not in EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS:
            raise ValueError(
                f"不支持的evaluation ID: {evaluation_id}。支持的ID: {EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS}")

        return EvaluationComponent(evaluation_id, "Evaluation")

    @staticmethod
    def get_available_evaluations() -> List[int]:
        """获取所有可用的evaluation ID"""
        return sorted(list(EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS))

    @staticmethod
    def create_all_evaluations() -> Dict[int, EvaluationComponent]:
        """创建所有可用evaluation"""
        return {
            evaluation_id: EvaluationComponent(evaluation_id, "Evaluation")
            for evaluation_id in EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "EvaluationComponentFactory",
            "version": "2.0.0",
            "total_evaluations": len(EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS),
            "supported_ids": sorted(list(EvaluationComponentFactory.SUPPORTED_EVALUATION_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_evaluation_evaluation_component_1(): return EvaluationComponentFactory.create_component(1)


def create_evaluation_evaluation_component_6(): return EvaluationComponentFactory.create_component(6)


__all__ = [
    "IEvaluationComponent",
    "EvaluationComponent",
    "EvaluationComponentFactory",
    "create_evaluation_evaluation_component_1",
    "create_evaluation_evaluation_component_6",
]
