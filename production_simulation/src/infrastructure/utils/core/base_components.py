"""
base_components 模块

提供 base_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类
import time

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

# 定义基础ComponentFactory类以避免导入错误
class ComponentFactory:
    """组件工厂基类"""
    
    def __init__(self):
        self._factories = {}
        self._creation_times = []
        self._statistics = {
            "total_creations": 0,
            "successful_creations": 0,
            "failed_creations": 0
        }
    
    def register_factory(self, component_type, factory_func):
        """注册组件工厂函数"""
        self._factories[component_type] = factory_func
    
    def unregister_factory(self, component_type):
        """注销组件工厂函数"""
        if component_type in self._factories:
            del self._factories[component_type]
    
    def get_registered_types(self):
        """获取所有已注册的组件类型"""
        return list(self._factories.keys())
    
    def create_component(self, component_type, config=None):
        """创建组件"""
        config = config or {}
        self._statistics["total_creations"] += 1
        start_time = time.time()
        
        try:
            if component_type in self._factories:
                result = self._factories[component_type](config)
                if result:
                    self._statistics["successful_creations"] += 1
                else:
                    self._statistics["failed_creations"] += 1
                self._record_creation(component_type, start_time)
                return result
            
            result = self._create_component_instance(component_type, config)
            if result:
                self._statistics["successful_creations"] += 1
            else:
                self._statistics["failed_creations"] += 1
            self._record_creation(component_type, start_time)
            return result
        except Exception:
            self._statistics["failed_creations"] += 1
            return None
    
    def _create_component_instance(self, component_type, config):
        """创建组件实例的抽象方法"""
        pass
    
    def _record_creation(self, component_type, creation_time):
        """记录组件创建时间"""
        self._creation_times.append((component_type, creation_time))
    
    def get_statistics(self):
        """获取组件创建统计信息"""
        return self._statistics.copy()
"""
基础设施层 - Base组件统一实现

使用统一的ComponentFactory基类，提供Base组件的工厂模式实现。
"""

logger = logging.getLogger(__name__)

# Base组件常量


class BaseComponentConstants:
    """Base组件相关常量"""

    # 组件版本
    COMPONENT_VERSION = "2.0.0"

    # 支持的base ID列表
    SUPPORTED_BASE_IDS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59, 65, 71, 77, 83, 89]

    # 组件类型
    DEFAULT_COMPONENT_TYPE = "Base"

    # 状态常量
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"
    STATUS_ERROR = "error"

    # 优先级常量
    DEFAULT_PRIORITY = 1
    MIN_PRIORITY = 0
    MAX_PRIORITY = 10


class IComponentFactory(ABC):
    """组件工厂接口"""

    @abstractmethod
    def create_component(self, component_type: str, config: Dict[str, Any]) -> Any:
        """创建组件"""

    @abstractmethod
    def register_factory(self, component_type: str, factory_func: Callable) -> None:
        """注册工厂函数"""


class IBaseComponent(ABC):
    """Base组件接口"""

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
    def get_base_id(self) -> int:
        """获取base ID"""


class BaseComponent(IBaseComponent):
    """统一Base组件实现"""

    def __init__(self, base_id: int, component_type: str = BaseComponentConstants.DEFAULT_COMPONENT_TYPE):
        """初始化组件"""
        self.base_id = base_id
        self.component_type = component_type
        self.component_name = f"{component_type}_Component_{base_id}"
        self.creation_time = datetime.now()

    def get_base_id(self) -> int:
        """获取base ID"""
        return self.base_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "base_id": self.base_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.component_type}组件实现",
            "version": BaseComponentConstants.COMPONENT_VERSION,
            "type": "unified_infrastructure_utils_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "base_id": self.base_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.component_name}",
                "processing_type": "unified_base_processing",
            }
            return result
        except Exception as e:
            return {
                "base_id": self.base_id,
                "component_name": self.component_name,
                "component_type": self.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def get_status(self) -> Dict[str, Any]:
        """获取组件状态"""
        return {
            "base_id": self.base_id,
            "component_name": self.component_name,
            "component_type": self.component_type,
            "status": "active",
            "creation_time": self.creation_time.isoformat(),
            "health": "good",
        }


class BaseComponentFactory(ComponentFactory):
    """Base组件工厂 - 继承统一ComponentFactory"""

    # 支持的base ID列表
    SUPPORTED_BASE_IDS = BaseComponentConstants.SUPPORTED_BASE_IDS

    def __init__(self):
        super().__init__()
        # 注册Base组件工厂函数
        for base_id in self.SUPPORTED_BASE_IDS:
            self.register_factory(
                f"base_{base_id}",
                lambda config, bid=base_id: BaseComponent(bid, "Base"),
            )

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """重写创建方法，支持base_id参数"""
        config = config or {}

        # 如果是数字类型，转换为base_前缀
        if component_type.isdigit():
            base_id = int(component_type)
            component_type = f"base_{base_id}"

        # 如果是base_前缀，直接使用
        if component_type.startswith("base_"):
            base_id = int(component_type.split("_")[1])
            self._statistics["total_creations"] += 1
            start_time = time.time()
            
            if base_id not in self.SUPPORTED_BASE_IDS:
                self._statistics["failed_creations"] += 1
                raise ValueError(f"不支持的base ID: {base_id}。支持的ID: {self.SUPPORTED_BASE_IDS}")
            
            component = BaseComponent(base_id, "Base")
            # 记录创建统计
            self._statistics["successful_creations"] += 1
            self._record_creation(component_type, start_time)
            return component

        # 其他情况使用父类方法
        return super().create_component(component_type, config)

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """实现父类的抽象方法"""
        # Base组件工厂主要通过注册的工厂函数创建
        return None

    @staticmethod
    def create_component_static(base_id: int) -> BaseComponent:
        """静态方法创建指定ID的base组件"""
        if base_id not in BaseComponentFactory.SUPPORTED_BASE_IDS:
            raise ValueError(
                f"不支持的base ID: {base_id}。支持的ID: {BaseComponentFactory.SUPPORTED_BASE_IDS}")
        return BaseComponent(base_id, "Base")

    @staticmethod
    def get_available_bases() -> List[int]:
        """获取所有可用的base ID"""
        return sorted(list(BaseComponentFactory.SUPPORTED_BASE_IDS))

    @staticmethod
    def create_all_bases() -> Dict[int, BaseComponent]:
        """创建所有可用base"""
        return {base_id: BaseComponent(base_id, "Base") for base_id in BaseComponentFactory.SUPPORTED_BASE_IDS}

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "BaseComponentFactory",
            "version": BaseComponentConstants.COMPONENT_VERSION,
            "total_bases": len(BaseComponentFactory.SUPPORTED_BASE_IDS),
            "supported_ids": sorted(list(BaseComponentFactory.SUPPORTED_BASE_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "基于统一ComponentFactory的Base组件工厂",
        }

    # 向后兼容：创建旧的组件实例
    def create_base_base_component_5():
        return BaseComponentFactory.create_component_static(5)

    def create_base_base_component_11():
        return BaseComponentFactory.create_component_static(11)

    def create_base_base_component_17():
        return BaseComponentFactory.create_component_static(17)

    def create_base_base_component_23():
        return BaseComponentFactory.create_component_static(23)

    def create_base_base_component_29():
        return BaseComponentFactory.create_component_static(29)

    def create_base_base_component_35():
        return BaseComponentFactory.create_component_static(35)

    def create_base_base_component_41():
        return BaseComponentFactory.create_component_static(41)

    def create_base_base_component_47():
        return BaseComponentFactory.create_component_static(47)

    def create_base_base_component_53():
        return BaseComponentFactory.create_component_static(53)

    def create_base_base_component_59():
        return BaseComponentFactory.create_component_static(59)

    def create_base_base_component_65():
        return BaseComponentFactory.create_component_static(65)

    def create_base_base_component_71():
        return BaseComponentFactory.create_component_static(71)

    def create_base_base_component_77():
        return BaseComponentFactory.create_component_static(77)

    def create_base_base_component_83():
        return BaseComponentFactory.create_component_static(83)

    def create_base_base_component_89():
        return BaseComponentFactory.create_component_static(89)


__all__ = [
    "IBaseComponent",
    "BaseComponent",
    "BaseComponentFactory",
]
