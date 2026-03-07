import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComponentFactory:

    """组件工厂（占位，实际项目中由统一生成器覆盖）"""

    def __init__(self):
        self._components: Dict[str, Any] = {}

    def create_component(self, component_type: str, config: Dict[str, Any]):
        return None


class IGridComponent(ABC):

    """Grid组件接口"""

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
    def get_grid_id(self) -> int:
        """获取grid ID"""
        pass


class GridComponent(IGridComponent):

    """统一Grid组件实现"""

    def __init__(self, grid_id: int, component_type: str = "Grid"):
        """初始化组件"""
        self.grid_id = grid_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{grid_id}"
        self.creation_time = datetime.now()

    def get_grid_id(self) -> int:
        """获取grid ID"""
        return self.grid_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
        "grid_id": self.grid_id,
        "component_name": self.component_name,
        "component_type": self.component_type,
        "creation_time": self.creation_time.isoformat(),
        "description": "统一{self.component_type}组件实现",
        "version": "2.0.0",
        "type": "unified_ml_tuning_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "grid_id": self.grid_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_grid_processing"
            }
            return result
        except Exception as e:
            return {
            "grid_id": self.grid_id,
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
        "grid_id": self.grid_id,
        "component_name": self.component_name,
        "component_type": self.component_type,
        "status": "active",
        "creation_time": self.creation_time.isoformat(),
        "health": "good"
        }


class GridComponentFactory:

    """Grid组件工厂"""

    SUPPORTED_GRID_IDS = [5, 10, 15, 20]

    @staticmethod
    def create_component(grid_id: int) -> GridComponent:
        """创建指定ID的grid组件"""
        if grid_id not in GridComponentFactory.SUPPORTED_GRID_IDS:
            raise ValueError(
                f"不支持的grid ID: {grid_id}。支持的ID: {GridComponentFactory.SUPPORTED_GRID_IDS}")

        return GridComponent(grid_id, "Grid")

    @staticmethod
    def get_available_grids() -> List[int]:
        """获取所有可用的grid ID"""
        return sorted(list(GridComponentFactory.SUPPORTED_GRID_IDS))

    @staticmethod
    def create_all_grids() -> Dict[int, GridComponent]:
        """创建所有可用grid"""
        return {
            grid_id: GridComponent(grid_id, "Grid")
            for grid_id in GridComponentFactory.SUPPORTED_GRID_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "GridComponentFactory",
            "version": "2.0.0",
            "total_grids": len(GridComponentFactory.SUPPORTED_GRID_IDS),
            "supported_ids": sorted(list(GridComponentFactory.SUPPORTED_GRID_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一Grid组件工厂，替代原模板文件"
        }

def create_grid_grid_component_5():
    """向后兼容工厂，创建 grid_id=5"""
    return GridComponentFactory.create_component(5)


def create_grid_grid_component_10():
    """向后兼容工厂，创建 grid_id=10"""
    return GridComponentFactory.create_component(10)


def create_grid_grid_component_15():
    """向后兼容工厂，创建 grid_id=15"""
    return GridComponentFactory.create_component(15)


def create_grid_grid_component_20():
    """向后兼容工厂，创建 grid_id=20"""
    return GridComponentFactory.create_component(20)

__all__ = [
    "IGridComponent",
    "GridComponent",
    "GridComponentFactory",
    "create_grid_grid_component_5",
    "create_grid_grid_component_10",
    "create_grid_grid_component_15",
    "create_grid_grid_component_20",
        ]
