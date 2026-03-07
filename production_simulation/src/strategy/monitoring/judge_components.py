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
统一Judge组件工厂

合并所有judge_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:39
"""


class IJudgeComponent(ABC):

    """Judge组件接口"""

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
    def get_judge_id(self) -> int:
        """获取judge ID"""


class JudgeComponent(IJudgeComponent):

    """统一Judge组件实现"""

    def __init__(self, judge_id: int, component_type: str = "Judge"):
        """初始化组件"""
        self.judge_id = judge_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{judge_id}"
        self.creation_time = datetime.now()

    def get_judge_id(self) -> int:
        """获取judge ID"""
        return self.judge_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "judge_id": self.judge_id,
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
                "judge_id": self.judge_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_judge_processing"
            }
            return result
        except Exception as e:
            return {
                "judge_id": self.judge_id,
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
            "judge_id": self.judge_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class JudgeComponentFactory:

    """Judge组件工厂"""

    # 支持的judge ID列表
    SUPPORTED_JUDGE_IDS = [4, 9]

    @staticmethod
    def create_component(judge_id: int) -> JudgeComponent:
        """创建指定ID的judge组件"""
        if judge_id not in JudgeComponentFactory.SUPPORTED_JUDGE_IDS:
            raise ValueError(
                f"不支持的judge ID: {judge_id}。支持的ID: {JudgeComponentFactory.SUPPORTED_JUDGE_IDS}")

        return JudgeComponent(judge_id, "Judge")

    @staticmethod
    def get_available_judges() -> List[int]:
        """获取所有可用的judge ID"""
        return sorted(list(JudgeComponentFactory.SUPPORTED_JUDGE_IDS))

    @staticmethod
    def create_all_judges() -> Dict[int, JudgeComponent]:
        """创建所有可用judge"""
        return {
            judge_id: JudgeComponent(judge_id, "Judge")
            for judge_id in JudgeComponentFactory.SUPPORTED_JUDGE_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "JudgeComponentFactory",
            "version": "2.0.0",
            "total_judges": len(JudgeComponentFactory.SUPPORTED_JUDGE_IDS),
            "supported_ids": sorted(list(JudgeComponentFactory.SUPPORTED_JUDGE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_judge_judge_component_4(): return JudgeComponentFactory.create_component(4)


def create_judge_judge_component_9(): return JudgeComponentFactory.create_component(9)


__all__ = [
    "IJudgeComponent",
    "JudgeComponent",
    "JudgeComponentFactory",
    "create_judge_judge_component_4",
    "create_judge_judge_component_9",
]
