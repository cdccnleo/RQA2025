"""
base_components 模块

提供 base_components 相关功能和接口。
"""

import logging

import time

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional, List
"""
基础设施层 - 统一组件工厂基类

解决灾难性代码重复问题，为所有组件提供统一的工厂模式实现。
"""

logger = logging.getLogger(__name__)


class ComponentFactory:
    """统一组件工厂 - 消除所有重复定义

    该类提供统一的组件创建和管理功能，所有组件工厂都应该继承此类
    而不是重复定义相同的ComponentFactory类。

    主要特性:
    - 统一的组件创建接口
    - 工厂方法注册机制
    - 异常处理和日志记录
    - 性能监控和统计
    """

    def __init__(self):
        self._components = {}  # 组件缓存
        self._factories = {}  # 工厂函数注册表
        self._statistics = {}  # 性能统计
        self._creation_times = {}  # 创建时间记录

    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """创建组件 - 统一入口

        Args:
            component_type: 组件类型标识
            config: 组件配置参数

        Returns:
            创建的组件实例或None
        """
        config = config or {}
        start_time = time.time()

        try:
            # 首先尝试使用注册的工厂函数
            if component_type in self._factories:
                component = self._factories[component_type](config)
                self._record_creation(component_type, start_time)
                return component

            # 回退到通用创建逻辑
            component = self._create_component_instance(component_type, config)
            if component and hasattr(component, "initialize"):
                if component.initialize(config):
                    self._record_creation(component_type, start_time)
                    return component

            self._record_creation(component_type, start_time, success=False)
            return component

        except Exception as e:
            self._record_creation(component_type, start_time, success=False)
            logger.error(f"创建组件失败 {component_type}: {e}")
            return None

    def register_factory(self, component_type: str, factory_func):
        """注册组件工厂函数

        Args:
            component_type: 组件类型标识
            factory_func: 工厂函数，接受config参数返回组件实例
        """
        self._factories[component_type] = factory_func
        logger.debug(f"注册组件工厂: {component_type}")

    def unregister_factory(self, component_type: str):
        """注销组件工厂函数"""
        if component_type in self._factories:
            del self._factories[component_type]
            logger.debug(f"注销组件工厂: {component_type}")

    def get_registered_types(self) -> List[str]:
        """获取所有已注册的组件类型"""
        return list(self._factories.keys())

    def get_statistics(self) -> Dict[str, Any]:
        """获取组件创建统计信息"""
        return {
            "total_creations": sum(stats["count"] for stats in self._statistics.values()),
            "success_rate": self._calculate_success_rate(),
            "average_creation_time": self._calculate_average_time(),
            "component_stats": self._statistics.copy(),
        }

    def clear_cache(self):
        """清空组件缓存"""
        self._components.clear()
        logger.info("组件缓存已清空")

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """创建组件实例的具体实现

        子类应该重写此方法来实现具体的组件创建逻辑。
        """
        raise NotImplementedError("子类必须实现_create_component_instance方法")

    def _record_creation(self, component_type: str, start_time: float, success: bool = True):
        """记录组件创建统计"""
        duration = time.time() - start_time

        if component_type not in self._statistics:
            self._statistics[component_type] = {
                "count": 0,
                "success_count": 0,
                "total_time": 0.0,
                "last_creation": None,
            }

        stats = self._statistics[component_type]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["last_creation"] = datetime.now().isoformat()

        if success:
            stats["success_count"] += 1

    def _calculate_success_rate(self) -> float:
        """计算总体成功率"""
        total = sum(stats["count"] for stats in self._statistics.values())
        if total == 0:
            return 1.0
        success = sum(stats["success_count"] for stats in self._statistics.values())
        return success / total

    def _calculate_average_time(self) -> float:
        """计算平均创建时间"""
        total_time = sum(stats["total_time"] for stats in self._statistics.values())
        total_count = sum(stats["count"] for stats in self._statistics.values())
        return total_time / total_count if total_count > 0 else 0.0


class IComponentFactory(ABC):
    """组件工厂接口定义"""

    @abstractmethod
    def create_component(self, component_type: str, config: Optional[Dict[str, Any]] = None):
        """创建组件"""

    @abstractmethod
    def register_factory(self, component_type: str, factory_func):
        """注册工厂函数"""

    @abstractmethod
    def get_registered_types(self) -> List[str]:
        """获取注册类型"""


class BaseComponentFactory(ComponentFactory, IComponentFactory):
    """基础组件工厂实现

    提供ComponentFactory的默认实现，可以直接使用或继承扩展。
    """

    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _create_component_instance(self, component_type: str, config: Dict[str, Any]):
        """默认组件创建实现

        尝试根据组件类型动态创建组件类实例。
        """
        try:
            # 尝试动态导入和创建组件
            module_name = f"src.infrastructure.components.{component_type}"
            class_name = component_type.title() + "Component"

            # 这里可以根据需要实现动态导入逻辑
            # 目前返回None，子类应该重写此方法
            self._logger.debug(f"尝试创建组件: {component_type}")
            return None

        except Exception as e:
            self._logger.error(f"动态创建组件失败 {component_type}: {e}")
            return None
