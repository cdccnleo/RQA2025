#!/usr/bin/env python3
"""
RQA2025 基础设施层组件注册表

提供组件的动态注册、发现、生命周期管理和热插拔功能。
"""

from typing import Dict, Any, Optional, List, Type, Callable, Set
import threading
import logging
import importlib
import inspect
from datetime import datetime
import json

from .component_bus import global_component_bus, Message, MessageType, publish_event

logger = logging.getLogger(__name__)


class ComponentMetadata:
    """组件元数据"""

    def __init__(self, name: str, version: str, description: str = "",
                 dependencies: Optional[List[str]] = None,
                 capabilities: Optional[List[str]] = None):
        """
        初始化组件元数据

        Args:
            name: 组件名称
            version: 组件版本
            description: 组件描述
            dependencies: 依赖的其他组件
            capabilities: 组件提供的功能
        """
        self.name = name
        self.version = version
        self.description = description
        self.dependencies = dependencies or []
        self.capabilities = capabilities or []
        self.registered_at = datetime.now()
        self.last_health_check = None
        self.health_status = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'dependencies': self.dependencies,
            'capabilities': self.capabilities,
            'registered_at': self.registered_at.isoformat(),
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'health_status': self.health_status
        }

    def update_health_status(self, status: str):
        """更新健康状态"""
        self.health_status = status
        self.last_health_check = datetime.now()


class ComponentInstance:
    """组件实例"""

    def __init__(self, component_class: Type, metadata: ComponentMetadata,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化组件实例

        Args:
            component_class: 组件类
            metadata: 组件元数据
            config: 组件配置
        """
        self.component_class = component_class
        self.metadata = metadata
        self.config = config or {}
        self.instance = None
        self.is_running = False
        self.created_at = datetime.now()
        self.last_started = None
        self.start_count = 0

    def create_instance(self) -> Any:
        """
        创建组件实例

        Returns:
            Any: 组件实例
        """
        try:
            # 检查构造函数参数
            sig = inspect.signature(self.component_class.__init__)
            params = {}

            # 映射配置参数到构造函数参数
            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue
                if param_name in self.config:
                    params[param_name] = self.config[param_name]
                elif param.default != inspect.Parameter.empty:
                    params[param_name] = param.default

            self.instance = self.component_class(**params)
            logger.info(f"组件 {self.metadata.name} 实例创建成功")
            return self.instance

        except Exception as e:
            logger.error(f"创建组件 {self.metadata.name} 实例失败: {e}")
            raise

    def start(self) -> bool:
        """
        启动组件

        Returns:
            bool: 是否成功启动
        """
        try:
            if self.instance is None:
                self.create_instance()

            if hasattr(self.instance, 'start_monitoring'):
                self.instance.start_monitoring()
            elif hasattr(self.instance, 'start'):
                self.instance.start()

            self.is_running = True
            self.last_started = datetime.now()
            self.start_count += 1

            # 发布组件启动事件
            publish_event("component.lifecycle.started", {
                'component': self.metadata.name,
                'version': self.metadata.version,
                'start_time': self.last_started.isoformat()
            })

            logger.info(f"组件 {self.metadata.name} 启动成功")
            return True

        except Exception as e:
            logger.error(f"启动组件 {self.metadata.name} 失败: {e}")
            return False

    def stop(self) -> bool:
        """
        停止组件

        Returns:
            bool: 是否成功停止
        """
        try:
            if self.instance and self.is_running:
                if hasattr(self.instance, 'stop_monitoring'):
                    self.instance.stop_monitoring()
                elif hasattr(self.instance, 'stop'):
                    self.instance.stop()

            self.is_running = False

            # 发布组件停止事件
            publish_event("component.lifecycle.stopped", {
                'component': self.metadata.name,
                'version': self.metadata.version,
                'stop_time': datetime.now().isoformat()
            })

            logger.info(f"组件 {self.metadata.name} 停止成功")
            return True

        except Exception as e:
            logger.error(f"停止组件 {self.metadata.name} 失败: {e}")
            return False

    def restart(self) -> bool:
        """
        重启组件

        Returns:
            bool: 是否成功重启
        """
        logger.info(f"正在重启组件 {self.metadata.name}")

        if not self.stop():
            return False

        # 短暂等待
        import time
        time.sleep(0.5)

        if not self.start():
            return False

        logger.info(f"组件 {self.metadata.name} 重启成功")
        return True

    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            new_config: 新配置

        Returns:
            bool: 是否成功更新
        """
        try:
            self.config.update(new_config)

            # 如果组件支持热更新配置
            if self.instance and hasattr(self.instance, 'update_config'):
                self.instance.update_config(new_config)

            # 发布配置更新事件
            publish_event("component.config.updated", {
                'component': self.metadata.name,
                'config_keys': list(new_config.keys()),
                'updated_at': datetime.now().isoformat()
            })

            logger.info(f"组件 {self.metadata.name} 配置更新成功")
            return True

        except Exception as e:
            logger.error(f"更新组件 {self.metadata.name} 配置失败: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """
        获取组件状态

        Returns:
            Dict[str, Any]: 组件状态信息
        """
        return {
            'name': self.metadata.name,
            'version': self.metadata.version,
            'is_running': self.is_running,
            'created_at': self.created_at.isoformat(),
            'last_started': self.last_started.isoformat() if self.last_started else None,
            'start_count': self.start_count,
            'config_keys': list(self.config.keys()),
            'has_instance': self.instance is not None
        }


class ComponentRegistry:
    """
    组件注册表

    管理组件的注册、发现、生命周期和依赖关系。
    """

    def __init__(self):
        """初始化组件注册表"""
        self.components: Dict[str, ComponentInstance] = {}
        self.metadata: Dict[str, ComponentMetadata] = {}
        self._lock = threading.RLock()

        # 启动健康检查线程
        self.health_check_thread = threading.Thread(
            target=self._health_check_loop,
            name="ComponentRegistry-HealthCheck",
            daemon=True
        )
        self.health_check_thread.start()

    def register_component(self, name: str, component_class: Type,
                          version: str = "1.0.0", description: str = "",
                          dependencies: Optional[List[str]] = None,
                          capabilities: Optional[List[str]] = None,
                          config: Optional[Dict[str, Any]] = None) -> bool:
        """
        注册组件

        Args:
            name: 组件名称
            component_class: 组件类
            version: 版本号
            description: 描述
            dependencies: 依赖组件列表
            capabilities: 功能列表
            config: 默认配置

        Returns:
            bool: 是否成功注册
        """
        with self._lock:
            try:
                # 检查依赖
                if dependencies:
                    for dep in dependencies:
                        if dep not in self.components:
                            logger.warning(f"组件 {name} 的依赖 {dep} 未注册")

                # 创建元数据
                metadata = ComponentMetadata(
                    name=name,
                    version=version,
                    description=description,
                    dependencies=dependencies or [],
                    capabilities=capabilities or []
                )

                # 创建组件实例
                instance = ComponentInstance(component_class, metadata, config)

                # 注册到注册表
                self.components[name] = instance
                self.metadata[name] = metadata

                # 发布注册事件
                publish_event("component.registry.registered", {
                    'component': name,
                    'version': version,
                    'capabilities': capabilities or []
                })

                logger.info(f"组件 {name} v{version} 注册成功")
                return True

            except Exception as e:
                logger.error(f"注册组件 {name} 失败: {e}")
                return False

    def unregister_component(self, name: str) -> bool:
        """
        注销组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功注销
        """
        with self._lock:
            try:
                if name not in self.components:
                    return False

                # 停止组件
                self.components[name].stop()

                # 检查是否有其他组件依赖此组件
                dependents = self._find_dependents(name)
                if dependents:
                    logger.warning(f"组件 {name} 仍有依赖组件: {dependents}")
                    # 可以选择强制停止或返回错误

                # 从注册表移除
                del self.components[name]
                del self.metadata[name]

                # 发布注销事件
                publish_event("component.registry.unregistered", {
                    'component': name,
                    'unregistered_at': datetime.now().isoformat()
                })

                logger.info(f"组件 {name} 注销成功")
                return True

            except Exception as e:
                logger.error(f"注销组件 {name} 失败: {e}")
                return False

    def get_component(self, name: str) -> Optional[ComponentInstance]:
        """
        获取组件实例

        Args:
            name: 组件名称

        Returns:
            Optional[ComponentInstance]: 组件实例
        """
        return self.components.get(name)

    def get_component_instance(self, name: str) -> Optional[Any]:
        """
        获取组件对象实例

        Args:
            name: 组件名称

        Returns:
            Optional[Any]: 组件对象实例
        """
        component = self.get_component(name)
        return component.instance if component else None

    def list_components(self) -> List[Dict[str, Any]]:
        """
        列出所有组件

        Returns:
            List[Dict[str, Any]]: 组件列表
        """
        with self._lock:
            return [comp.get_status() for comp in self.components.values()]

    def find_components_by_capability(self, capability: str) -> List[str]:
        """
        根据功能查找组件

        Args:
            capability: 功能名称

        Returns:
            List[str]: 具有此功能的组件名称列表
        """
        with self._lock:
            return [name for name, meta in self.metadata.items()
                   if capability in meta.capabilities]

    def start_component(self, name: str) -> bool:
        """
        启动组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功启动
        """
        component = self.get_component(name)
        if component:
            return component.start()
        return False

    def stop_component(self, name: str) -> bool:
        """
        停止组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功停止
        """
        component = self.get_component(name)
        if component:
            return component.stop()
        return False

    def restart_component(self, name: str) -> bool:
        """
        重启组件

        Args:
            name: 组件名称

        Returns:
            bool: 是否成功重启
        """
        component = self.get_component(name)
        if component:
            return component.restart()
        return False

    def update_component_config(self, name: str, config: Dict[str, Any]) -> bool:
        """
        更新组件配置

        Args:
            name: 组件名称
            config: 新配置

        Returns:
            bool: 是否成功更新
        """
        component = self.get_component(name)
        if component:
            return component.update_config(config)
        return False

    def check_dependencies(self, name: str) -> Dict[str, Any]:
        """
        检查组件依赖关系

        Args:
            name: 组件名称

        Returns:
            Dict[str, Any]: 依赖检查结果
        """
        if name not in self.metadata:
            return {'satisfied': False, 'missing': [], 'reason': 'component_not_found'}

        metadata = self.metadata[name]
        missing_deps = []

        for dep in metadata.dependencies:
            if dep not in self.components:
                missing_deps.append(dep)
            elif not self.components[dep].is_running:
                missing_deps.append(f"{dep}(not_running)")

        return {
            'satisfied': len(missing_deps) == 0,
            'missing': missing_deps,
            'total_dependencies': len(metadata.dependencies)
        }

    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统整体健康状态

        Returns:
            Dict[str, Any]: 系统健康状态
        """
        with self._lock:
            total_components = len(self.components)
            running_components = sum(1 for comp in self.components.values() if comp.is_running)
            healthy_components = sum(1 for meta in self.metadata.values() if meta.health_status == 'healthy')

            # 计算依赖满足度
            satisfied_deps = 0
            total_deps = 0
            for name in self.components:
                deps_check = self.check_dependencies(name)
                satisfied_deps += deps_check['total_dependencies'] - len(deps_check['missing'])
                total_deps += deps_check['total_dependencies']

            dependency_satisfaction = (satisfied_deps / total_deps) if total_deps > 0 else 1.0

            return {
                'total_components': total_components,
                'running_components': running_components,
                'healthy_components': healthy_components,
                'dependency_satisfaction': round(dependency_satisfaction, 3),
                'overall_health': 'healthy' if dependency_satisfaction >= 0.8 else 'degraded',
                'timestamp': datetime.now().isoformat()
            }

    def _find_dependents(self, component_name: str) -> List[str]:
        """
        查找依赖指定组件的其他组件

        Args:
            component_name: 组件名称

        Returns:
            List[str]: 依赖此组件的组件列表
        """
        dependents = []
        for name, metadata in self.metadata.items():
            if component_name in metadata.dependencies:
                dependents.append(name)
        return dependents

    def _health_check_loop(self):
        """健康检查循环"""
        while True:
            try:
                # 每30秒进行一次健康检查
                import time
                time.sleep(30)

                with self._lock:
                    for name, component in self.components.items():
                        try:
                            # 简单的健康检查
                            is_healthy = (
                                component.instance is not None and
                                component.is_running and
                                hasattr(component.instance, 'get_health_status')
                            )

                            if is_healthy:
                                health_status = component.instance.get_health_status()
                                health = health_status.get('status', 'unknown')
                            else:
                                health = 'unhealthy'

                            # 更新健康状态
                            self.metadata[name].update_health_status(health)

                        except Exception as e:
                            logger.warning(f"组件 {name} 健康检查失败: {e}")
                            self.metadata[name].update_health_status('error')

            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")

    def save_registry_state(self, file_path: str) -> bool:
        """
        保存注册表状态

        Args:
            file_path: 保存文件路径

        Returns:
            bool: 是否成功保存
        """
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'components': {}
            }

            for name, component in self.components.items():
                state['components'][name] = {
                    'metadata': self.metadata[name].to_dict(),
                    'status': component.get_status(),
                    'config': component.config
                }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)

            logger.info(f"注册表状态已保存到 {file_path}")
            return True

        except Exception as e:
            logger.error(f"保存注册表状态失败: {e}")
            return False

    def load_registry_state(self, file_path: str) -> bool:
        """
        加载注册表状态

        Args:
            file_path: 加载文件路径

        Returns:
            bool: 是否成功加载
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)

            # 这里可以实现状态恢复逻辑
            # 由于组件实例化比较复杂，这里主要用于状态检查

            logger.info(f"注册表状态已从 {file_path} 加载")
            return True

        except Exception as e:
            logger.error(f"加载注册表状态失败: {e}")
            return False


# 全局组件注册表实例
global_component_registry = ComponentRegistry()


def register_component(name: str, component_class: Type, **kwargs) -> bool:
    """
    便捷的组件注册函数

    Args:
        name: 组件名称
        component_class: 组件类
        **kwargs: 其他注册参数

    Returns:
        bool: 是否成功注册
    """
    return global_component_registry.register_component(name, component_class, **kwargs)


def get_component(name: str) -> Optional[Any]:
    """
    便捷的组件获取函数

    Args:
        name: 组件名称

    Returns:
        Optional[Any]: 组件实例
    """
    return global_component_registry.get_component_instance(name)
