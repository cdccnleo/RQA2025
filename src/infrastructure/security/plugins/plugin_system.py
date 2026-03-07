#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 插件系统

提供插件化架构，支持动态加载和扩展安全组件
"""

import importlib
import inspect
import logging
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class PluginInfo:
    """插件信息"""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)


class SecurityPlugin(ABC):
    """安全插件基类"""

    @property
    @abstractmethod
    def plugin_info(self) -> PluginInfo:
        """插件信息"""
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """关闭插件"""
        pass

    def get_capability(self, name: str) -> Optional[Callable]:
        """获取插件能力"""
        return getattr(self, name, None)


class PluginManager:
    """插件管理器"""

    def __init__(self, plugin_dirs: Optional[List[Path]] = None):
        self.plugin_dirs = plugin_dirs or [
            Path("src/infrastructure/security/plugins"),
            Path("plugins/security")
        ]
        self.loaded_plugins: Dict[str, SecurityPlugin] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.capability_registry: Dict[str, List[SecurityPlugin]] = {}

        # 创建插件目录
        for plugin_dir in self.plugin_dirs:
            plugin_dir.mkdir(parents=True, exist_ok=True)

    def load_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """加载插件"""
        if plugin_name in self.loaded_plugins:
            logging.warning(f"插件 {plugin_name} 已经加载")
            return True

        try:
            # 尝试从不同位置加载插件
            plugin_module = None
            for plugin_dir in self.plugin_dirs:
                try:
                    # 从目录导入
                    if (plugin_dir / f"{plugin_name}.py").exists():
                        spec = importlib.util.spec_from_file_location(
                            plugin_name, plugin_dir / f"{plugin_name}.py"
                        )
                        plugin_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(plugin_module)
                        break
                except Exception as e:
                    logging.debug(f"从 {plugin_dir} 加载插件 {plugin_name} 失败: {e}")
                    continue

            if not plugin_module:
                # 尝试作为Python模块导入
                plugin_module = importlib.import_module(f"src.infrastructure.security.plugins.{plugin_name}")

            # 查找插件类
            plugin_class = None
            for name, obj in inspect.getmembers(plugin_module):
                if (inspect.isclass(obj) and
                    issubclass(obj, SecurityPlugin) and
                    obj != SecurityPlugin):
                    plugin_class = obj
                    break

            if not plugin_class:
                raise ValueError(f"在插件 {plugin_name} 中找不到有效的插件类")

            # 实例化插件
            plugin_instance = plugin_class()
            plugin_config = config or {}

            # 初始化插件
            if not plugin_instance.initialize(plugin_config):
                raise RuntimeError(f"插件 {plugin_name} 初始化失败")

            # 注册插件
            self.loaded_plugins[plugin_name] = plugin_instance
            self.plugin_configs[plugin_name] = plugin_config

            # 注册插件能力
            self._register_plugin_capabilities(plugin_instance)

            logging.info(f"成功加载插件: {plugin_name} v{plugin_instance.plugin_info.version}")
            return True

        except Exception as e:
            logging.error(f"加载插件 {plugin_name} 失败: {e}")
            return False

    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        if plugin_name not in self.loaded_plugins:
            logging.warning(f"插件 {plugin_name} 未加载")
            return True

        try:
            plugin = self.loaded_plugins[plugin_name]

            # 关闭插件
            plugin.shutdown()

            # 注销插件能力
            self._unregister_plugin_capabilities(plugin)

            # 移除插件
            del self.loaded_plugins[plugin_name]
            del self.plugin_configs[plugin_name]

            logging.info(f"成功卸载插件: {plugin_name}")
            return True

        except Exception as e:
            logging.error(f"卸载插件 {plugin_name} 失败: {e}")
            return False

    def get_plugin(self, plugin_name: str) -> Optional[SecurityPlugin]:
        """获取插件实例"""
        return self.loaded_plugins.get(plugin_name)

    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出所有已加载的插件"""
        plugins = []
        for name, plugin in self.loaded_plugins.items():
            info = plugin.plugin_info
            plugins.append({
                'name': name,
                'version': info.version,
                'description': info.description,
                'author': info.author,
                'capabilities': info.capabilities,
                'config': self.plugin_configs[name]
            })
        return plugins

    def call_capability(self, capability_name: str, *args, **kwargs) -> List[Any]:
        """调用指定能力的插件"""
        results = []
        if capability_name in self.capability_registry:
            for plugin in self.capability_registry[capability_name]:
                try:
                    capability_func = plugin.get_capability(capability_name)
                    if capability_func:
                        result = capability_func(*args, **kwargs)
                        results.append({
                            'plugin': plugin.plugin_info.name,
                            'result': result
                        })
                except Exception as e:
                    logging.error(f"调用插件 {plugin.plugin_info.name} 的能力 {capability_name} 失败: {e}")
                    results.append({
                        'plugin': plugin.plugin_info.name,
                        'error': str(e)
                    })

        return results

    def reload_plugin(self, plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """重新加载插件"""
        if not self.unload_plugin(plugin_name):
            return False

        return self.load_plugin(plugin_name, config)

    def discover_plugins(self) -> List[str]:
        """发现可用的插件"""
        discovered_plugins = set()

        for plugin_dir in self.plugin_dirs:
            if plugin_dir.exists():
                for plugin_file in plugin_dir.glob("*.py"):
                    if not plugin_file.name.startswith("__"):
                        plugin_name = plugin_file.stem
                        discovered_plugins.add(plugin_name)

        return sorted(list(discovered_plugins))

    def validate_plugin_dependencies(self, plugin_name: str) -> List[str]:
        """验证插件依赖"""
        issues = []

        # 如果插件未加载，先尝试加载它
        if plugin_name not in self.loaded_plugins:
            # 尝试加载插件来获取其依赖信息
            try:
                # 临时加载插件来检查依赖
                for plugin_dir in self.plugin_dirs:
                    plugin_file = plugin_dir / f"{plugin_name}.py"
                    if plugin_file.exists():
                        spec = importlib.util.spec_from_file_location(
                            plugin_name, plugin_file
                        )
                        plugin_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(plugin_module)

                        # 查找插件类
                        plugin_class = None
                        for name, obj in inspect.getmembers(plugin_module):
                            if (inspect.isclass(obj) and
                                issubclass(obj, SecurityPlugin) and
                                obj != SecurityPlugin):
                                plugin_class = obj
                                break

                        if plugin_class:
                            temp_plugin = plugin_class()
                            dependencies = temp_plugin.plugin_info.dependencies

                            for dep in dependencies:
                                if dep not in self.loaded_plugins:
                                    issues.append(f"插件 {plugin_name} 缺少依赖: {dep}")
                        else:
                            issues.append(f"无法加载插件 {plugin_name} 来验证依赖")
                        break
                else:
                    issues.append(f"插件 {plugin_name} 未加载")

            except Exception as e:
                issues.append(f"加载插件 {plugin_name} 失败: {e}")
        else:
            # 插件已加载，检查其依赖
            plugin = self.loaded_plugins[plugin_name]
            dependencies = plugin.plugin_info.dependencies

            for dep in dependencies:
                if dep not in self.loaded_plugins:
                    issues.append(f"插件 {plugin_name} 缺少依赖: {dep}")

        return issues

    def _register_plugin_capabilities(self, plugin: SecurityPlugin) -> List[str]:
        """注册插件能力"""
        registered = []
        for capability in plugin.plugin_info.capabilities:
            if capability not in self.capability_registry:
                self.capability_registry[capability] = []
            if plugin not in self.capability_registry[capability]:
                self.capability_registry[capability].append(plugin)
                registered.append(capability)
        return registered

    def _unregister_plugin_capabilities(self, plugin: SecurityPlugin) -> List[str]:
        """注销插件能力"""
        unregistered = []
        for capability in plugin.plugin_info.capabilities:
            if capability in self.capability_registry:
                if plugin in self.capability_registry[capability]:
                    self.capability_registry[capability].remove(plugin)
                    unregistered.append(capability)
                if not self.capability_registry[capability]:
                    del self.capability_registry[capability]
        return unregistered


# 创建全局插件管理器实例
_global_plugin_manager = PluginManager()

def get_plugin_manager() -> PluginManager:
    """获取全局插件管理器"""
    return _global_plugin_manager

def load_security_plugin(plugin_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """加载安全插件"""
    return _global_plugin_manager.load_plugin(plugin_name, config)

def unload_security_plugin(plugin_name: str) -> bool:
    """卸载安全插件"""
    return _global_plugin_manager.unload_plugin(plugin_name)

def call_plugin_capability(capability_name: str, *args, **kwargs) -> List[Any]:
    """调用插件能力"""
    return _global_plugin_manager.call_capability(capability_name, *args, **kwargs)
