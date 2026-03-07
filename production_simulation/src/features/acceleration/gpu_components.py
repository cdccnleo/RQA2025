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
统一Gpu组件工厂

合并所有gpu_*.py模板文件为统一的管理架构
生成时间: 2025 - 08 - 24 10:24:21
"""


class IGpuComponent(ABC):

    """Gpu组件接口"""

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
    def get_gpu_id(self) -> int:
        """获取gpu ID"""


class GpuComponent(IGpuComponent):

    """统一Gpu组件实现"""

    def __init__(self, gpu_id: int, component_type: str = "Gpu"):
        """初始化组件"""
        self.gpu_id = gpu_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{gpu_id}"
        self.creation_time = datetime.now()

    def get_gpu_id(self) -> int:
        """获取gpu ID"""
        return self.gpu_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "gpu_id": self.gpu_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": "统一{self.component_type}组件实现",
            "version": "2.0.0",
            "type": "unified_features_acceleration_component"
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "gpu_id": self.gpu_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_gpu_processing"
            }
            return result
        except Exception as e:
            return {
                "gpu_id": self.gpu_id,
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
            "gpu_id": self.gpu_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good"
        }


class GpuComponentFactory:

    """Gpu组件工厂"""

    # 支持的gpu ID列表
    SUPPORTED_GPU_IDS = [1, 6, 11, 16, 21, 26]

    @staticmethod
    def create_component(gpu_id: int) -> GpuComponent:
        """创建指定ID的gpu组件"""
        if gpu_id not in GpuComponentFactory.SUPPORTED_GPU_IDS:
            raise ValueError(f"不支持的gpu ID: {gpu_id}。支持的ID: {GpuComponentFactory.SUPPORTED_GPU_IDS}")

        return GpuComponent(gpu_id, "Gpu")

    @staticmethod
    def get_available_gpus() -> List[int]:
        """获取所有可用的gpu ID"""
        return sorted(list(GpuComponentFactory.SUPPORTED_GPU_IDS))

    @staticmethod
    def create_all_gpus() -> Dict[int, GpuComponent]:
        """创建所有可用gpu"""
        return {
            gpu_id: GpuComponent(gpu_id, "Gpu")
            for gpu_id in GpuComponentFactory.SUPPORTED_GPU_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "GpuComponentFactory",
            "version": "2.0.0",
            "total_gpus": len(GpuComponentFactory.SUPPORTED_GPU_IDS),
            "supported_ids": sorted(list(GpuComponentFactory.SUPPORTED_GPU_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "统一{component_type}组件工厂，替代原有的{len(files)}个模板化文件"
        }


# 向后兼容：创建旧的组件实例

def create_gpu_gpu_component_1(): return GpuComponentFactory.create_component(1)


def create_gpu_gpu_component_6(): return GpuComponentFactory.create_component(6)


def create_gpu_gpu_component_11(): return GpuComponentFactory.create_component(11)


def create_gpu_gpu_component_16(): return GpuComponentFactory.create_component(16)


def create_gpu_gpu_component_21(): return GpuComponentFactory.create_component(21)


def create_gpu_gpu_component_26(): return GpuComponentFactory.create_component(26)


__all__ = [
    "IGpuComponent",
    "GpuComponent",
    "GpuComponentFactory",
    "create_gpu_gpu_component_1",
    "create_gpu_gpu_component_6",
    "create_gpu_gpu_component_11",
    "create_gpu_gpu_component_16",
    "create_gpu_gpu_component_21",
    "create_gpu_gpu_component_26",
]
