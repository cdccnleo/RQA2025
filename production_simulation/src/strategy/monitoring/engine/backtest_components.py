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
统一Backtest组件工厂

合并所有backtest_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:16:06
"""


class IBacktestComponent(ABC):

    """Backtest组件接口"""

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
    def get_backtest_id(self) -> int:
        """获取backtest ID"""


class BacktestComponent(IBacktestComponent):

    """统一Backtest组件实现"""

    def __init__(self, backtest_id: int, component_type: str = "Backtest"):
        """初始化组件"""
        self.backtest_id = backtest_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{backtest_id}"
        self.creation_time = datetime.now()

    def get_backtest_id(self) -> int:
        """获取backtest ID"""
        return self.backtest_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "backtest_id": self.backtest_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_backtest_engine_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "backtest_id": self.backtest_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_backtest_processing"
            }
            return result
        except Exception as e:
            return {
                "backtest_id": self.backtest_id,
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
            "backtest_id": self.backtest_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class BacktestComponentFactory:

    """Backtest组件工厂"""

    # 支持的backtest ID列表
    SUPPORTED_BACKTEST_IDS = [2, 7, 12, 17]

    @staticmethod
    def create_component(backtest_id: int) -> BacktestComponent:
        """创建指定ID的backtest组件"""
        if backtest_id not in BacktestComponentFactory.SUPPORTED_BACKTEST_IDS:
            raise ValueError(
                f"不支持的backtest ID: {backtest_id}。支持的ID: {BacktestComponentFactory.SUPPORTED_BACKTEST_IDS}")

        return BacktestComponent(backtest_id, "Backtest")

    @staticmethod
    def get_available_backtests() -> List[int]:
        """获取所有可用的backtest ID"""
        return sorted(list(BacktestComponentFactory.SUPPORTED_BACKTEST_IDS))

    @staticmethod
    def create_all_backtests() -> Dict[int, BacktestComponent]:
        """创建所有可用backtest"""
        return {
            backtest_id: BacktestComponent(backtest_id, "Backtest")
            for backtest_id in BacktestComponentFactory.SUPPORTED_BACKTEST_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "BacktestComponentFactory",
            "version": "2.0.0",
            "total_backtests": len(BacktestComponentFactory.SUPPORTED_BACKTEST_IDS),
            "supported_ids": sorted(list(BacktestComponentFactory.SUPPORTED_BACKTEST_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_backtest_backtest_component_2(): return BacktestComponentFactory.create_component(2)


def create_backtest_backtest_component_7(): return BacktestComponentFactory.create_component(7)


def create_backtest_backtest_component_12(): return BacktestComponentFactory.create_component(12)


def create_backtest_backtest_component_17(): return BacktestComponentFactory.create_component(17)


__all__ = [
    "IBacktestComponent",
    "BacktestComponent",
    "BacktestComponentFactory",
    "create_backtest_backtest_component_2",
    "create_backtest_backtest_component_7",
    "create_backtest_backtest_component_12",
    "create_backtest_backtest_component_17",
]
