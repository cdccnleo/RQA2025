from .plugin_validator import PluginValidator
import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件管理器

# 使用统一基础设施集成层
try:
    from src.infrastructure.integration import get_features_adapter
    _features_adapter = get_features_adapter()
    logger = logging.getLogger(__name__)
        except ImportError:
    # 降级到直接导入
    from src.infrastructure.logging.core.unified_logger import get_unified_logger
整合注册表、加载器和验证器，提供统一的插件管理接口。
"""

from typing import Dict, List, Optional, Any
import threading
from .base_plugin import BaseFeaturePlugin, PluginType
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader
# 使用标准logging，避免复杂的依赖问题
import logging


def get_unified_logger(name):
    return logging.getLogger(name)


logger = get_unified_logger('__name__')


class FeaturePluginManager:

    """特征插件管理器"""

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        初始化插件管理器

        Args:
            plugin_dirs: 插件目录列表
        """
        self.registry = PluginRegistry()
        self.loader = PluginLoader(plugin_dirs)
        self.validator = PluginValidator()
        self._lock = threading.RLock()
        self._auto_discovery = True
        self._auto_load = True

    def enable_auto_discovery(self, enabled: bool = True):
        """
        启用 / 禁用自动发现

        Args:
            enabled: 是否启用
        """
        self._auto_discovery = enabled
        logger.info(f"自动发现已{'启用' if enabled else '禁用'}")

    def enable_auto_load(self, enabled: bool = True):
        """
        启用 / 禁用自动加载

        Args:
            enabled: 是否启用
        """
        self._auto_load = enabled
        logger.info(f"自动加载已{'启用' if enabled else '禁用'}")

    def discover_and_load_plugins(self) -> List[BaseFeaturePlugin]:
        """
        发现并加载所有插件

        Returns:
            加载的插件列表
        """
        with self._lock:
            try:
                # 发现插件文件
                plugin_files = self.loader.discover_plugins()
                loaded_plugins = []

                for plugin_path in plugin_files:
                    plugin = self.loader.load_plugin_from_file(plugin_path)
                    if plugin is not None:
                        # 验证插件
                        if self.validator.validate_plugin_instance(plugin):
                            # 注册插件
                            if self.registry.register_plugin(plugin):
                                loaded_plugins.append(plugin)
                        else:
                            logger.error(f"插件验证失败: {plugin_path}")

                logger.info(f"发现并加载了 {len(loaded_plugins)} 个插件")
                return loaded_plugins

            except Exception as e:
                logger.error(f"发现和加载插件失败: {e}")
                return []

    def register_plugin(self, plugin: BaseFeaturePlugin) -> bool:
        """
        注册插件

        Args:
            plugin: 插件实例

        Returns:
            注册是否成功
        """
        with self._lock:
            try:
                # 验证插件
                if not self.validator.validate_plugin_instance(plugin):
                    logger.error(f"插件验证失败: {plugin}")
                    return False

                # 注册插件
                success = self.registry.register_plugin(plugin)
                if success:
                    logger.info(f"插件注册成功: {plugin}")
                return success

            except Exception as e:
                logger.error(f"插件注册失败: {e}")
                return False

    def unregister_plugin(self, plugin_name: str) -> bool:
        """
        注销插件

        Args:
            plugin_name: 插件名称

        Returns:
            注销是否成功
        """
        with self._lock:
            try:
                success = self.registry.unregister_plugin(plugin_name)
                if success:
                    logger.info(f"插件注销成功: {plugin_name}")
                return success

            except Exception as e:
                logger.error(f"插件注销失败: {e}")
                return False

    def load_plugin_from_file(self, plugin_path: str) -> Optional[BaseFeaturePlugin]:
        """
        从文件加载插件

        Args:
            plugin_path: 插件文件路径

        Returns:
            插件实例或None
        """
        with self._lock:
            try:
                plugin = self.loader.load_plugin_from_file(plugin_path)
                if plugin is not None:
                    # 验证插件
                    if self.validator.validate_plugin_instance(plugin):
                        # 注册插件
                        if self.registry.register_plugin(plugin):
                            return plugin
                        else:
                            logger.error(f"插件注册失败: {plugin_path}")
                    else:
                        logger.error(f"插件验证失败: {plugin_path}")

                return None

            except Exception as e:
                logger.error(f"从文件加载插件失败: {e}")
                return None

    def load_plugin_from_module(self, module_name: str,


                                plugin_class_name: Optional[str] = None) -> Optional[BaseFeaturePlugin]:
        """
        从模块加载插件

        Args:
            module_name: 模块名
            plugin_class_name: 插件类名（可选）

        Returns:
            插件实例或None
        """
        with self._lock:
            try:
                plugin = self.loader.load_plugin_from_module(module_name, plugin_class_name)
                if plugin is not None:
                    # 验证插件
                    if self.validator.validate_plugin_instance(plugin):
                        # 注册插件
                        if self.registry.register_plugin(plugin):
                            return plugin
                        else:
                            logger.error(f"插件注册失败: {module_name}")
                    else:
                        logger.error(f"插件验证失败: {module_name}")

                return None

            except Exception as e:
                logger.error(f"从模块加载插件失败: {e}")
                return None

    def get_plugin(self, plugin_name: str) -> Optional[BaseFeaturePlugin]:
        """
        获取插件

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例或None
        """
        return self.registry.get_plugin(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BaseFeaturePlugin]:
        """
        按类型获取插件

        Args:
            plugin_type: 插件类型

        Returns:
            插件列表
        """
        return self.registry.get_plugins_by_type(plugin_type)

    def get_plugins_by_tag(self, tag: str) -> List[BaseFeaturePlugin]:
        """
        按标签获取插件

        Args:
            tag: 标签

        Returns:
            插件列表
        """
        return self.registry.get_plugins_by_tag(tag)

    def get_all_plugins(self) -> List[BaseFeaturePlugin]:
        """
        获取所有插件

        Returns:
            插件列表
        """
        return self.registry.get_all_plugins()

    def list_plugins(self, plugin_type: Optional[PluginType] = None,


                     tag: Optional[str] = None) -> List[str]:
        """
        列出插件名称

        Args:
            plugin_type: 插件类型过滤
            tag: 标签过滤

        Returns:
            插件名称列表
        """
        return self.registry.list_plugins(plugin_type, tag)

    def reload_plugin(self, plugin_name: str) -> Optional[BaseFeaturePlugin]:
        """
        重新加载插件

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例或None
        """
        with self._lock:
            try:
                # 注销旧插件
                self.registry.unregister_plugin(plugin_name)

                # 重新加载插件
                plugin = self.loader.reload_plugin(plugin_name)
                if plugin is not None:
                    # 验证插件
                    if self.validator.validate_plugin_instance(plugin):
                        # 注册插件
                        if self.registry.register_plugin(plugin):
                            logger.info(f"插件重新加载成功: {plugin_name}")
                            return plugin
                        else:
                            logger.error(f"插件注册失败: {plugin_name}")
                    else:
                        logger.error(f"插件验证失败: {plugin_name}")

                return None

            except Exception as e:
                logger.error(f"重新加载插件失败: {e}")
                return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        卸载插件

        Args:
            plugin_name: 插件名称

        Returns:
            卸载是否成功
        """
        with self._lock:
            try:
                # 注销插件
                success = self.registry.unregister_plugin(plugin_name)
                if success:
                    # 卸载模块
                    self.loader.unload_plugin(plugin_name)
                    logger.info(f"插件卸载成功: {plugin_name}")
                return success

            except Exception as e:
                logger.error(f"卸载插件失败: {e}")
                return False

    def get_plugin_info(self, plugin_name: str) -> Optional[Dict[str, Any]]:
        """
        获取插件信息

        Args:
            plugin_name: 插件名称

        Returns:
            插件信息字典或None
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is not None:
            return plugin.get_info()
        return None

    def get_plugin_stats(self) -> Dict[str, Any]:
        """
        获取插件统计信息

        Returns:
            统计信息字典
        """
        return self.registry.get_plugin_stats()

    def validate_plugin(self, plugin_name: str) -> bool:
        """
        验证插件

        Args:
            plugin_name: 插件名称

        Returns:
            验证是否通过
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is not None:
            return self.validator.validate_plugin_instance(plugin)
        return False

    def validate_all_plugins(self) -> Dict[str, bool]:
        """
        验证所有插件

        Returns:
            验证结果字典
        """
        results = {}
        for plugin in self.get_all_plugins():
            results[plugin.metadata.name] = self.validator.validate_plugin_instance(plugin)
        return results

    def initialize_plugin(self, plugin_name: str) -> bool:
        """
        初始化插件

        Args:
            plugin_name: 插件名称

        Returns:
            初始化是否成功
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is not None:
            return plugin.initialize()
        return False

    def cleanup_plugin(self, plugin_name: str) -> bool:
        """
        清理插件

        Args:
            plugin_name: 插件名称

        Returns:
            清理是否成功
        """
        plugin = self.get_plugin(plugin_name)
        if plugin is not None:
            return plugin.cleanup()
        return False

    def add_plugin_dir(self, plugin_dir: str):
        """
        添加插件目录

        Args:
            plugin_dir: 插件目录路径
        """
        self.loader.add_plugin_dir(plugin_dir)

    def remove_plugin_dir(self, plugin_dir: str):
        """
        移除插件目录

        Args:
            plugin_dir: 插件目录路径
        """
        self.loader.remove_plugin_dir(plugin_dir)

    def clear(self):
        """清空所有插件"""
        with self._lock:
            self.registry.clear()
            logger.info("所有插件已清空")

    def __len__(self):

        return len(self.registry)

    def __contains__(self, plugin_name: str):

        return plugin_name in self.registry
