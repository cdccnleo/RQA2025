import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件注册表

from src.infrastructure.logging.core.unified_logger import get_unified_logger
管理插件的注册、查找和管理功能。
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading
from .base_plugin import BaseFeaturePlugin, PluginType, PluginMetadata


logger = logging.getLogger(__name__)


class PluginRegistry:

    """插件注册表"""

    def __init__(self):
        """初始化插件注册表"""
        self._plugins: Dict[str, BaseFeaturePlugin] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = defaultdict(list)
        self._plugins_by_tag: Dict[str, List[str]] = defaultdict(list)
        self._metadata_cache: Dict[str, PluginMetadata] = {}
        self._lock = threading.RLock()

    def register_plugin(self, plugin: BaseFeaturePlugin) -> bool:
        """
        注册插件

        Args:
            plugin: 要注册的插件

        Returns:
            注册是否成功
        """
        with self._lock:
            try:
                plugin_name = plugin.metadata.name

                if plugin_name in self._plugins:
                    logger.warning(f"插件已存在: {plugin_name}")
                    return False

                # 注册插件
                self._plugins[plugin_name] = plugin
                self._metadata_cache[plugin_name] = plugin.metadata

                # 按类型分类
                self._plugins_by_type[plugin.metadata.plugin_type].append(plugin_name)

                # 按标签分类
                for tag in plugin.metadata.tags:
                    self._plugins_by_tag[tag].append(plugin_name)

                logger.info(f"插件注册成功: {plugin_name}")
                return True

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
                if plugin_name not in self._plugins:
                    logger.warning(f"插件不存在: {plugin_name}")
                    return False

                plugin = self._plugins[plugin_name]

                # 清理插件资源
                plugin.cleanup()

                # 从注册表中移除
                del self._plugins[plugin_name]
                del self._metadata_cache[plugin_name]

                # 从类型分类中移除
                plugin_type = plugin.metadata.plugin_type
                if plugin_name in self._plugins_by_type[plugin_type]:
                    self._plugins_by_type[plugin_type].remove(plugin_name)

                # 从标签分类中移除
                for tag in plugin.metadata.tags:
                    if plugin_name in self._plugins_by_tag[tag]:
                        self._plugins_by_tag[tag].remove(plugin_name)

                logger.info(f"插件注销成功: {plugin_name}")
                return True

            except Exception as e:
                logger.error(f"插件注销失败: {e}")
                return False

    def get_plugin(self, plugin_name: str) -> Optional[BaseFeaturePlugin]:
        """
        获取插件

        Args:
            plugin_name: 插件名称

        Returns:
            插件实例或None
        """
        with self._lock:
            return self._plugins.get(plugin_name)

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BaseFeaturePlugin]:
        """
        按类型获取插件

        Args:
            plugin_type: 插件类型

        Returns:
            插件列表
        """
        with self._lock:
            plugin_names = self._plugins_by_type.get(plugin_type, [])
            return [self._plugins[name] for name in plugin_names if name in self._plugins]

    def get_plugins_by_tag(self, tag: str) -> List[BaseFeaturePlugin]:
        """
        按标签获取插件

        Args:
            tag: 标签

        Returns:
            插件列表
        """
        with self._lock:
            plugin_names = self._plugins_by_tag.get(tag, [])
            return [self._plugins[name] for name in plugin_names if name in self._plugins]

    def get_all_plugins(self) -> List[BaseFeaturePlugin]:
        """
        获取所有插件

        Returns:
            插件列表
        """
        with self._lock:
            return list(self._plugins.values())

    def get_plugin_metadata(self, plugin_name: str) -> Optional[PluginMetadata]:
        """
        获取插件元数据

        Args:
            plugin_name: 插件名称

        Returns:
            插件元数据或None
        """
        with self._lock:
            return self._metadata_cache.get(plugin_name)

    def get_all_metadata(self) -> List[PluginMetadata]:
        """
        获取所有插件元数据

        Returns:
            元数据列表
        """
        with self._lock:
            return list(self._metadata_cache.values())

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
        with self._lock:
            if plugin_type:
                return self._plugins_by_type.get(plugin_type, [])
            elif tag:
                return self._plugins_by_tag.get(tag, [])
            else:
                return list(self._plugins.keys())

    def get_plugin_count(self) -> int:
        """
        获取插件数量

        Returns:
            插件数量
        """
        with self._lock:
            return len(self._plugins)

    def get_plugin_stats(self) -> Dict[str, Any]:
        """
        获取插件统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                "total_plugins": len(self._plugins),
                "by_type": {pt.value: len(plugins) for pt, plugins in self._plugins_by_type.items()},
                "by_status": defaultdict(int),
                "by_tag": {tag: len(plugins) for tag, plugins in self._plugins_by_tag.items()}
            }

            # 按状态统计
            for metadata in self._metadata_cache.values():
                stats["by_status"][metadata.status.value] += 1

            return stats

    def clear(self):
        """清空注册表"""
        with self._lock:
            # 清理所有插件
            for plugin in self._plugins.values():
                try:
                    plugin.cleanup()
                except Exception as e:
                    logger.error(f"清理插件失败: {e}")

            # 清空所有容器
            self._plugins.clear()
            self._metadata_cache.clear()
            self._plugins_by_type.clear()
            self._plugins_by_tag.clear()

            logger.info("插件注册表已清空")

    def __len__(self):

        return len(self._plugins)

    def __contains__(self, plugin_name: str):

        return plugin_name in self._plugins
