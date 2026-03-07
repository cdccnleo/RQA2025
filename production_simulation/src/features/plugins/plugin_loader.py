import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件加载器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
支持从文件系统动态加载插件。
"""

import os
import sys
import importlib
import importlib.util
from typing import Dict, List, Optional, Any, Type
import time
from .base_plugin import BaseFeaturePlugin
from .plugin_validator import PluginValidator


logger = logging.getLogger(__name__)


class PluginLoader:

    """插件加载器"""

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        """
        初始化插件加载器

        Args:
            plugin_dirs: 插件目录列表
        """
        self.plugin_dirs = plugin_dirs or []
        self.validator = PluginValidator()
        self._loaded_modules: Dict[str, Any] = {}
        self._load_times: Dict[str, float] = {}

    def add_plugin_dir(self, plugin_dir: str):
        """
        添加插件目录

        Args:
            plugin_dir: 插件目录路径
        """
        if plugin_dir not in self.plugin_dirs:
            self.plugin_dirs.append(plugin_dir)
            logger.info(f"添加插件目录: {plugin_dir}")

    def remove_plugin_dir(self, plugin_dir: str):
        """
        移除插件目录

        Args:
            plugin_dir: 插件目录路径
        """
        if plugin_dir in self.plugin_dirs:
            self.plugin_dirs.remove(plugin_dir)
            logger.info(f"移除插件目录: {plugin_dir}")

    def discover_plugins(self) -> List[str]:
        """
        发现插件文件

        Returns:
            插件文件路径列表
        """
        plugin_files = []

        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                logger.warning(f"插件目录不存在: {plugin_dir}")
                continue

            for root, dirs, files in os.walk(plugin_dir):
                for file in files:
                    if file.endswith('.py') and not file.startswith('__'):
                        plugin_path = os.path.join(root, file)
                        plugin_files.append(plugin_path)

        logger.info(f"发现 {len(plugin_files)} 个插件文件")
        return plugin_files

    def load_plugin_from_file(self, plugin_path: str) -> Optional[BaseFeaturePlugin]:
        """
        从文件加载插件

        Args:
            plugin_path: 插件文件路径

        Returns:
            插件实例或None
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(plugin_path):
                logger.error(f"插件文件不存在: {plugin_path}")
                return None

            # 生成模块名
            module_name = self._generate_module_name(plugin_path)

            # 加载模块
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            if spec is None or spec.loader is None:
                logger.error(f"无法加载模块: {plugin_path}")
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                logger.error(f"未找到插件类: {plugin_path}")
                return None

            # 验证插件
            if not self.validator.validate_plugin_class(plugin_class):
                logger.error(f"插件验证失败: {plugin_path}")
                return None

            # 创建插件实例
            plugin = plugin_class()

            # 验证插件实例
            if not self.validator.validate_plugin_instance(plugin):
                logger.error(f"插件实例验证失败: {plugin_path}")
                return None

            # 记录加载时间
            self._load_times[plugin.metadata.name] = time.time()
            self._loaded_modules[plugin.metadata.name] = module

            logger.info(f"插件加载成功: {plugin.metadata.name} from {plugin_path}")
            return plugin

        except Exception as e:
            logger.error(f"加载插件失败 {plugin_path}: {e}")
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
        try:
            # 导入模块
            module = importlib.import_module(module_name)

            # 查找插件类
            if plugin_class_name:
                plugin_class = getattr(module, plugin_class_name, None)
            else:
                plugin_class = self._find_plugin_class(module)

            if plugin_class is None:
                logger.error(f"未找到插件类: {module_name}")
                return None

            # 验证插件
            if not self.validator.validate_plugin_class(plugin_class):
                logger.error(f"插件验证失败: {module_name}")
                return None

            # 创建插件实例
            plugin = plugin_class()

            # 验证插件实例
            if not self.validator.validate_plugin_instance(plugin):
                logger.error(f"插件实例验证失败: {module_name}")
                return None

            # 记录加载时间
            self._load_times[plugin.metadata.name] = time.time()
            self._loaded_modules[plugin.metadata.name] = module

            logger.info(f"插件加载成功: {plugin.metadata.name} from {module_name}")
            return plugin

        except Exception as e:
            logger.error(f"加载插件失败 {module_name}: {e}")
            return None

    def load_all_plugins(self) -> List[BaseFeaturePlugin]:
        """
        加载所有插件

        Returns:
            插件实例列表
        """
        plugins = []
        plugin_files = self.discover_plugins()

        for plugin_path in plugin_files:
            plugin = self.load_plugin_from_file(plugin_path)
            if plugin is not None:
                plugins.append(plugin)

        logger.info(f"成功加载 {len(plugins)} 个插件")
        return plugins

    def reload_plugin(self, plugin_name: str) -> Optional[BaseFeaturePlugin]:
        """
        重新加载插件

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例或None
        """
        try:
            # 获取原始模块
            if plugin_name not in self._loaded_modules:
                logger.error(f"插件未加载: {plugin_name}")
                return None

            module = self._loaded_modules[plugin_name]

            # 重新加载模块
            importlib.reload(module)

            # 查找插件类
            plugin_class = self._find_plugin_class(module)
            if plugin_class is None:
                logger.error(f"重新加载后未找到插件类: {plugin_name}")
                return None

            # 创建新实例
            plugin = plugin_class()

            # 更新加载时间
            self._load_times[plugin_name] = time.time()

            logger.info(f"插件重新加载成功: {plugin_name}")
            return plugin

        except Exception as e:
            logger.error(f"重新加载插件失败 {plugin_name}: {e}")
            return None

    def unload_plugin(self, plugin_name: str) -> bool:
        """
        卸载插件

        Args:
            plugin_name: 插件名称

        Returns:
            卸载是否成功
        """
        try:
            if plugin_name in self._loaded_modules:
                # 从sys.modules中移除
                module = self._loaded_modules[plugin_name]
                module_name = module.__name__

                if module_name in sys.modules:
                    del sys.modules[module_name]

                # 清理记录
                del self._loaded_modules[plugin_name]
                if plugin_name in self._load_times:
                    del self._load_times[plugin_name]

                logger.info(f"插件卸载成功: {plugin_name}")
                return True
            else:
                logger.warning(f"插件未加载: {plugin_name}")
                return False

        except Exception as e:
            logger.error(f"卸载插件失败 {plugin_name}: {e}")
            return False

    def get_load_time(self, plugin_name: str) -> Optional[float]:
        """
        获取插件加载时间

        Args:
            plugin_name: 插件名称

        Returns:
            加载时间戳或None
        """
        return self._load_times.get(plugin_name)

    def get_loaded_modules(self) -> List[str]:
        """
        获取已加载的模块列表

        Returns:
            模块名称列表
        """
        return list(self._loaded_modules.keys())

    def _generate_module_name(self, plugin_path: str) -> str:
        """生成模块名"""
        # 移除扩展名
        base_name = os.path.splitext(os.path.basename(plugin_path))[0]
        # 生成唯一模块名
        return f"feature_plugin_{base_name}_{hash(plugin_path)}"

    def _find_plugin_class(self, module: Any) -> Optional[Type[BaseFeaturePlugin]]:
        """查找插件类"""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            # 检查是否是类
            if not isinstance(attr, type):
                continue

            # 检查是否继承自BaseFeaturePlugin
            if (issubclass(attr, BaseFeaturePlugin)
                    and attr != BaseFeaturePlugin):
                return attr

        return None
