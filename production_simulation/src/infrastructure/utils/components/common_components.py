"""
common_components 模块

提供 common_components 相关功能和接口。
"""

import logging

# 导入统一的ComponentFactory基类

from abc import ABC, abstractmethod
from datetime import datetime
from src.infrastructure.utils.components.core.base_components import ComponentFactory
from src.infrastructure.utils.core.duplicate_resolver import BaseComponentWithStatus
from typing import Dict, Any, Optional, List
"""
基础设施层 - Common组件统一实现

使用统一的ComponentFactory基类，提供Common组件的工厂模式实现。
"""

# 导入语句已在文件顶部注释掉，避免循环导入
# ComponentFactory, IComponentFactory 已通过其他方式获取
logger = logging.getLogger(__name__)

# Common组件常量


class CommonComponentConstants:
    """Common组件相关常量"""

    # 组件版本
    COMPONENT_VERSION = "2.0.0"

    # 支持的common ID列表
    SUPPORTED_COMMON_IDS = [4, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88]

    # 组件类型
    DEFAULT_COMPONENT_TYPE = "Common"

    # 状态常量
    STATUS_ACTIVE = "active"
    STATUS_INACTIVE = "inactive"
    STATUS_ERROR = "error"

    # 优先级常量
    DEFAULT_PRIORITY = 1
    MIN_PRIORITY = 0
    MAX_PRIORITY = 10


class ICommonComponent(ABC):
    """Common组件接口"""

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
    def get_common_id(self) -> int:
        """获取common ID"""


class CommonComponent(BaseComponentWithStatus, ICommonComponent):
    """统一Common组件实现 - 使用统一状态管理"""

    def __init__(self, common_id: int, component_type: str = CommonComponentConstants.DEFAULT_COMPONENT_TYPE):
        """初始化组件"""
        # 初始化统一状态管理 - BaseComponentWithStatus不接受参数
        super().__init__()

        # 组件特定属性
        self.common_id = common_id
        self.creation_time = datetime.now()
        self.component_name = f"{component_type}_Component_{common_id}"
        self.component_type = component_type
        
        # 创建状态管理器（如果需要）
        if not hasattr(self, 'status_manager'):
            # 创建一个简单的状态管理器对象
            class SimpleStatusManager:
                def __init__(self, name, comp_type):
                    self.component_name = name
                    self.component_type = comp_type
                    self._metadata = {}
                
                def add_metadata(self, key, value):
                    self._metadata[key] = value
            
            self.status_manager = SimpleStatusManager(self.component_name, self.component_type)

        # 初始化状态元数据
        self.status_manager.add_metadata("common_id", common_id)
        self.status_manager.add_metadata("creation_time", self.creation_time.isoformat())

    def get_common_id(self) -> int:
        """获取common ID"""
        return self.common_id

    def get_info(self) -> Dict[str, Any]:
        """获取组件信息"""
        return {
            "common_id": self.common_id,
            "component_name": self.status_manager.component_name,
            "component_type": self.status_manager.component_type,
            "creation_time": self.creation_time.isoformat(),
            "description": f"统一{self.status_manager.component_type}组件实现",
            "version": CommonComponentConstants.COMPONENT_VERSION,
            "type": "unified_infrastructure_utils_component",
        }

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """处理数据"""
        try:
            result = {
                "common_id": self.common_id,
                "component_name": self.status_manager.component_name,
                "component_type": self.status_manager.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "success",
                "result": f"Processed by {self.status_manager.component_name}",
                "processing_type": "unified_common_processing",
            }
            return result
        except Exception as e:
            return {
                "common_id": self.common_id,
                "component_name": self.status_manager.component_name,
                "component_type": self.status_manager.component_type,
                "input_data": data,
                "processed_at": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    def _get_component_specific_status(self) -> Dict[str, Any]:
        """获取组件特定的状态信息"""
        return {
            "health": "good",
            "active": True,
            "last_activity": datetime.now().isoformat()
        }


class CommonComponentFactory(ComponentFactory):
    """Common组件工厂 - 继承统一ComponentFactory"""

    # 支持的common ID列表
    SUPPORTED_COMMON_IDS = CommonComponentConstants.SUPPORTED_COMMON_IDS

    def __init__(self):
        super().__init__()
        # 注册Common组件工厂函数
        for common_id in self.SUPPORTED_COMMON_IDS:
            self.register_factory(
                f"common_{common_id}",
                lambda config, cid=common_id: CommonComponent(cid, "Common")
            )

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """重写创建方法，支持common_id参数"""
        config = config or {}

        # 如果是数字类型，转换为common_前缀
        if component_type.isdigit():
            common_id = int(component_type)
            component_type = f"common_{common_id}"

        # 如果是common_前缀，直接使用
        if component_type.startswith("common_"):
            common_id = int(component_type.split("_")[1])
            if common_id not in self.SUPPORTED_COMMON_IDS:
                raise ValueError(f"不支持的common ID: {common_id}。支持的ID: {self.SUPPORTED_COMMON_IDS}")
            return CommonComponent(common_id, "Common")

        # 其他情况使用父类方法
        return super().create_component(component_type, config)

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """实现父类的抽象方法"""
        # Common组件工厂主要通过注册的工厂函数创建
        return None

    @staticmethod
    def create_component_static(common_id: int) -> CommonComponent:
        """静态方法创建指定ID的common组件"""
        if common_id not in CommonComponentFactory.SUPPORTED_COMMON_IDS:
            raise ValueError(
                f"不支持的common ID: {common_id}。支持的ID: {CommonComponentFactory.SUPPORTED_COMMON_IDS}")
        return CommonComponent(common_id, "Common")

    @staticmethod
    def get_available_commons() -> List[int]:
        """获取所有可用的common ID"""
        return sorted(list(CommonComponentFactory.SUPPORTED_COMMON_IDS))

    @staticmethod
    def create_all_commons() -> Dict[int, CommonComponent]:
        """创建所有可用common"""
        return {
            common_id: CommonComponent(common_id, "Common") for common_id in CommonComponentFactory.SUPPORTED_COMMON_IDS
        }

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """获取工厂信息"""
        return {
            "factory_name": "CommonComponentFactory",
            "version": CommonComponentConstants.COMPONENT_VERSION,
            "total_commons": len(CommonComponentFactory.SUPPORTED_COMMON_IDS),
            "supported_ids": sorted(list(CommonComponentFactory.SUPPORTED_COMMON_IDS)),
            "created_at": datetime.now().isoformat(),
            "description": "基于统一ComponentFactory的Common组件工厂",
        }

# 向后兼容：创建旧的组件实例


def create_common_common_component_4():
    return CommonComponentFactory.create_component_static(4)


def create_common_common_component_10():
    return CommonComponentFactory.create_component_static(10)


def create_common_common_component_16():
    return CommonComponentFactory.create_component_static(16)


def create_common_common_component_22():
    return CommonComponentFactory.create_component_static(22)


def create_common_common_component_28():
    return CommonComponentFactory.create_component_static(28)


def create_common_common_component_34():
    return CommonComponentFactory.create_component_static(34)


def create_common_common_component_40():
    return CommonComponentFactory.create_component_static(40)


def create_common_common_component_46():
    return CommonComponentFactory.create_component_static(46)


def create_common_common_component_52():
    return CommonComponentFactory.create_component_static(52)


def create_common_common_component_58():
    return CommonComponentFactory.create_component_static(58)


def create_common_common_component_64():
    return CommonComponentFactory.create_component_static(64)


def create_common_common_component_70():
    return CommonComponentFactory.create_component_static(70)


def create_common_common_component_76():
    return CommonComponentFactory.create_component_static(76)


def create_common_common_component_82():
    return CommonComponentFactory.create_component_static(82)


def create_common_common_component_88():
    return CommonComponentFactory.create_component_static(88)


__all__ = [
    "ICommonComponent",
    "CommonComponent",
    "CommonComponentFactory",
    "create_common_common_component_4",
    "create_common_common_component_10",
    "create_common_common_component_16",
    "create_common_common_component_22",
    "create_common_common_component_28",
    "create_common_common_component_34",
    "create_common_common_component_40",
    "create_common_common_component_46",
    "create_common_common_component_52",
    "create_common_common_component_58",
    "create_common_common_component_64",
    "create_common_common_component_70",
    "create_common_common_component_76",
    "create_common_common_component_82",
    "create_common_common_component_88",
]
