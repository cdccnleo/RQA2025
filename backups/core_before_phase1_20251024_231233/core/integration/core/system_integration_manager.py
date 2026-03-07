"""
系统集成管理器

此模块提供了系统集成管理器的核心功能，包括适配器注册、组件协作和集成编排。
"""

from typing import Any, Dict, List, Optional, Type, Callable
from abc import ABC, abstractmethod
import logging
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class IntegrationComponent(ABC):
    """集成组件抽象基类"""

    @abstractmethod
    def initialize(self) -> bool:
        """初始化组件"""

    @abstractmethod
    def shutdown(self) -> bool:
        """关闭组件"""

    @property
    @abstractmethod
    def component_type(self) -> str:
        """组件类型"""

    @property
    @abstractmethod
    def dependencies(self) -> List[str]:
        """依赖的组件类型列表"""


class AdapterRegistry:
    """适配器注册表"""

    def __init__(self):
        self._adapters: Dict[str, Dict[str, Type]] = defaultdict(dict)
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()

    def register_adapter(self, adapter_type: str, adapter_name: str, adapter_class: Type) -> None:
        """注册适配器"""
        with self._lock:
            self._adapters[adapter_type][adapter_name] = adapter_class
            logger.info(f"注册适配器: {adapter_type}.{adapter_name}")

    def get_adapter_class(self, adapter_type: str, adapter_name: str) -> Optional[Type]:
        """获取适配器类"""
        return self._adapters.get(adapter_type, {}).get(adapter_name)

    def create_adapter_instance(self, adapter_type: str, adapter_name: str, **kwargs) -> Optional[Any]:
        """创建适配器实例"""
        adapter_class = self.get_adapter_class(adapter_type, adapter_name)
        if adapter_class:
            try:
                instance = adapter_class(**kwargs)
                instance_key = f"{adapter_type}.{adapter_name}"
                self._instances[instance_key] = instance
                return instance
            except Exception as e:
                logger.error(f"创建适配器实例失败 {adapter_type}.{adapter_name}: {e}")
        return None

    def get_adapter_instance(self, adapter_type: str, adapter_name: str) -> Optional[Any]:
        """获取适配器实例"""
        instance_key = f"{adapter_type}.{adapter_name}"
        return self._instances.get(instance_key)

    def list_adapters(self, adapter_type: Optional[str] = None) -> Dict[str, List[str]]:
        """列出注册的适配器"""
        if adapter_type:
            return {adapter_type: list(self._adapters.get(adapter_type, {}).keys())}
        return {atype: list(adapters.keys()) for atype, adapters in self._adapters.items()}


class ComponentCoordinator:
    """组件协调器"""

    def __init__(self):
        self._components: Dict[str, IntegrationComponent] = {}
        self._component_order: List[str] = []
        self._lock = threading.RLock()

    def register_component(self, component_id: str, component: IntegrationComponent) -> None:
        """注册组件"""
        with self._lock:
            self._components[component_id] = component
            self._rebuild_initialization_order()
            logger.info(f"注册组件: {component_id} ({component.component_type})")

    def unregister_component(self, component_id: str) -> bool:
        """注销组件"""
        with self._lock:
            if component_id in self._components:
                del self._components[component_id]
                self._rebuild_initialization_order()
                logger.info(f"注销组件: {component_id}")
                return True
        return False

    def initialize_components(self) -> bool:
        """初始化所有组件"""
        with self._lock:
            success_count = 0
            for component_id in self._component_order:
                component = self._components[component_id]
                try:
                    if component.initialize():
                        success_count += 1
                        logger.info(f"组件初始化成功: {component_id}")
                    else:
                        logger.error(f"组件初始化失败: {component_id}")
                except Exception as e:
                    logger.error(f"组件初始化异常 {component_id}: {e}")

            total_count = len(self._components)
            success = success_count == total_count
            logger.info(f"组件初始化完成: {success_count}/{total_count}")
            return success

    def shutdown_components(self) -> bool:
        """关闭所有组件"""
        with self._lock:
            # 反序关闭
            success_count = 0
            for component_id in reversed(self._component_order):
                component = self._components[component_id]
                try:
                    if component.shutdown():
                        success_count += 1
                        logger.info(f"组件关闭成功: {component_id}")
                    else:
                        logger.error(f"组件关闭失败: {component_id}")
                except Exception as e:
                    logger.error(f"组件关闭异常 {component_id}: {e}")

            total_count = len(self._components)
            success = success_count == total_count
            logger.info(f"组件关闭完成: {success_count}/{total_count}")
            return success

    def _rebuild_initialization_order(self) -> None:
        """重建初始化顺序"""
        # 简单的拓扑排序实现
        visited = set()
        temp_visited = set()
        order = []

        def visit(component_id: str) -> bool:
            if component_id in temp_visited:
                logger.error(f"检测到循环依赖: {component_id}")
                return False
            if component_id in visited:
                return True

            temp_visited.add(component_id)
            component = self._components[component_id]

            # 访问依赖
            for dep_type in component.dependencies:
                dep_components = [cid for cid, c in self._components.items()
                                  if c.component_type == dep_type]
                for dep_id in dep_components:
                    if not visit(dep_id):
                        return False

            temp_visited.remove(component_id)
            visited.add(component_id)
            order.append(component_id)
            return True

        # 对所有组件执行拓扑排序
        for component_id in self._components:
            if component_id not in visited:
                if not visit(component_id):
                    logger.error("无法解决组件依赖关系")
                    return

        self._component_order = order


class SystemIntegrationManager:
    """系统集成管理器 - 增强版"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化系统集成管理器

        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.adapter_registry = AdapterRegistry()
        self.component_coordinator = ComponentCoordinator()
        self._integration_pipelines: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        logger.info("初始化系统集成管理器")

    def register_adapter(self, adapter_type: str, adapter_name: str, adapter_class: Type) -> None:
        """注册适配器"""
        self.adapter_registry.register_adapter(adapter_type, adapter_name, adapter_class)

    def create_adapter(self, adapter_type: str, adapter_name: str, **kwargs) -> Optional[Any]:
        """创建适配器实例"""
        return self.adapter_registry.create_adapter_instance(adapter_type, adapter_name, **kwargs)

    def get_adapter(self, adapter_type: str, adapter_name: str) -> Optional[Any]:
        """获取适配器实例"""
        return self.adapter_registry.get_adapter_instance(adapter_type, adapter_name)

    def register_component(self, component_id: str, component: IntegrationComponent) -> None:
        """注册集成组件"""
        self.component_coordinator.register_component(component_id, component)

    def unregister_component(self, component_id: str) -> bool:
        """注销集成组件"""
        return self.component_coordinator.unregister_component(component_id)

    def initialize_components(self) -> bool:
        """初始化所有组件"""
        return self.component_coordinator.initialize_components()

    def shutdown_components(self) -> bool:
        """关闭所有组件"""
        return self.component_coordinator.shutdown_components()

    def create_integration_pipeline(self, pipeline_name: str, steps: List[Callable]) -> None:
        """创建集成管道"""
        with self._lock:
            self._integration_pipelines[pipeline_name] = steps
            logger.info(f"创建集成管道: {pipeline_name} ({len(steps)} 步骤)")

    def execute_pipeline(self, pipeline_name: str, initial_data: Any) -> Any:
        """执行集成管道"""
        pipeline = self._integration_pipelines.get(pipeline_name)
        if not pipeline:
            raise ValueError(f"集成管道不存在: {pipeline_name}")

        data = initial_data
        for step in pipeline:
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"管道步骤执行失败 {pipeline_name}: {e}")
                raise

        return data

    def process(self, data: Any, pipeline_name: str = "default") -> Any:
        """处理数据

        Args:
            data: 输入数据
            pipeline_name: 管道名称

        Returns:
            处理后的数据
        """
        logger.info(f"处理数据: {type(data)} 使用管道: {pipeline_name}")
        if pipeline_name in self._integration_pipelines:
            return self.execute_pipeline(pipeline_name, data)
        return data

    def validate(self) -> bool:
        """验证配置和组件

        Returns:
            验证结果
        """
        logger.info("验证配置和组件")

        # 验证适配器注册
        adapters = self.adapter_registry.list_adapters()
        if not adapters:
            logger.warning("未注册任何适配器")
            return False

        # 验证组件依赖
        try:
            self.component_coordinator._rebuild_initialization_order()
        except Exception as e:
            logger.error(f"组件依赖验证失败: {e}")
            return False

        logger.info("配置和组件验证通过")
        return True

    def get_status(self) -> Dict[str, Any]:
        """获取集成管理器状态"""
        return {
            'adapters': self.adapter_registry.list_adapters(),
            'components': list(self.component_coordinator._components.keys()),
            'pipelines': list(self._integration_pipelines.keys()),
            'config': self.config
        }


# 导出主要类
__all__ = ['SystemIntegrationManager']
