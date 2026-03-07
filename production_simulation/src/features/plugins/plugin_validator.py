import logging
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
插件验证器

from src.infrastructure.logging.core.unified_logger import get_unified_logger
用于验证插件的有效性和兼容性。
"""

from typing import Any, Dict, Type
import inspect
from .base_plugin import BaseFeaturePlugin, PluginMetadata, PluginType


logger = logging.getLogger(__name__)


class PluginValidator:

    """插件验证器"""

    def __init__(self):
        """初始化插件验证器"""
        self.required_methods = ['_get_metadata', 'process']
        self.required_metadata_fields = ['name', 'version', 'description', 'author', 'plugin_type']

    def validate_plugin_class(self, plugin_class: Type[BaseFeaturePlugin]) -> bool:
        """
        验证插件类

        Args:
            plugin_class: 插件类

        Returns:
            验证是否通过
        """
        try:
            # 检查是否继承自BaseFeaturePlugin
            if not issubclass(plugin_class, BaseFeaturePlugin):
                logger.error(f"插件类必须继承自BaseFeaturePlugin: {plugin_class}")
                return False

            # 检查是否是抽象类
            if inspect.isabstract(plugin_class):
                logger.error(f"插件类不能是抽象类: {plugin_class}")
                return False

            # 检查必需方法
            for method_name in self.required_methods:
                if not hasattr(plugin_class, method_name):
                    logger.error(f"插件类缺少必需方法: {method_name}")
                    return False

                method = getattr(plugin_class, method_name)
                if not callable(method):
                    logger.error(f"插件类方法不是可调用的: {method_name}")
                    return False

            # 检查__init__方法
            if not hasattr(plugin_class, '__init__'):
                logger.error(f"插件类缺少__init__方法")
                return False

            logger.info(f"插件类验证通过: {plugin_class}")
            return True

        except Exception as e:
            logger.error(f"插件类验证失败: {e}")
            return False

    def validate_plugin_instance(self, plugin: BaseFeaturePlugin) -> bool:
        """
        验证插件实例

        Args:
            plugin: 插件实例

        Returns:
            验证是否通过
        """
        try:
            # 检查元数据
            if not self._validate_metadata(plugin.metadata):
                return False

            # 检查必需方法
            for method_name in self.required_methods:
                if not hasattr(plugin, method_name):
                    logger.error(f"插件实例缺少必需方法: {method_name}")
                    return False

                method = getattr(plugin, method_name)
                if not callable(method):
                    logger.error(f"插件实例方法不是可调用的: {method_name}")
                    return False

            # 检查可选方法
            optional_methods = ['initialize', 'cleanup', 'get_info', 'validate_input']
            for method_name in optional_methods:
                if hasattr(plugin, method_name):
                    method = getattr(plugin, method_name)
                    if not callable(method):
                        logger.error(f"插件实例方法不是可调用的: {method_name}")
                        return False

            # 测试元数据获取
            try:
                metadata = plugin._get_metadata()
                if not isinstance(metadata, PluginMetadata):
                    logger.error(f"插件元数据类型错误: {type(metadata)}")
                    return False
            except Exception as e:
                logger.error(f"获取插件元数据失败: {e}")
                return False

            logger.info(f"插件实例验证通过: {plugin}")
            return True

        except Exception as e:
            logger.error(f"插件实例验证失败: {e}")
            return False

    def validate_metadata(self, metadata: PluginMetadata) -> bool:
        """
        验证插件元数据

        Args:
            metadata: 插件元数据

        Returns:
            验证是否通过
        """
        return self._validate_metadata(metadata)

    def validate_config(self, plugin: BaseFeaturePlugin, config: Dict[str, Any]) -> bool:
        """
        验证插件配置

        Args:
            plugin: 插件实例
            config: 配置字典

        Returns:
            验证是否通过
        """
        try:
            # 检查配置是否为字典
            if not isinstance(config, dict):
                logger.error(f"配置必须是字典类型: {type(config)}")
                return False

            # 如果有配置模式，进行验证
            if plugin.metadata.config_schema:
                return self._validate_config_schema(config, plugin.metadata.config_schema)

            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def validate_api_compatibility(self, plugin: BaseFeaturePlugin,


                                   min_version: str = "1.0.0",
                                   max_version: str = "2.0.0") -> bool:
        """
        验证API兼容性

        Args:
            plugin: 插件实例
            min_version: 最小API版本
            max_version: 最大API版本

        Returns:
            验证是否通过
        """
        try:
            metadata = plugin.metadata

            # 检查版本范围
            if not self._is_version_in_range(metadata.min_api_version, metadata.max_api_version,
                                             min_version, max_version):
                logger.error(f"API版本不兼容: {metadata.min_api_version}-{metadata.max_api_version}")
                return False

            return True

        except Exception as e:
            logger.error(f"API兼容性验证失败: {e}")
            return False

    def _validate_metadata(self, metadata: PluginMetadata) -> bool:
        """验证元数据"""
        try:
            # 检查必需字段
            for field_name in self.required_metadata_fields:
                if not hasattr(metadata, field_name):
                    logger.error(f"元数据缺少必需字段: {field_name}")
                    return False

                value = getattr(metadata, field_name)
                if value is None or value == "":
                    logger.error(f"元数据字段不能为空: {field_name}")
                    return False

            # 检查名称格式
            if not self._is_valid_name(metadata.name):
                logger.error(f"插件名称格式无效: {metadata.name}")
                return False

            # 检查版本格式
            if not self._is_valid_version(metadata.version):
                logger.error(f"版本格式无效: {metadata.version}")
                return False

            # 检查插件类型
            if not isinstance(metadata.plugin_type, PluginType):
                logger.error(f"插件类型无效: {metadata.plugin_type}")
                return False

            # 检查依赖列表
            if not isinstance(metadata.dependencies, list):
                logger.error(f"依赖必须是列表: {type(metadata.dependencies)}")
                return False

            # 检查标签列表
            if not isinstance(metadata.tags, list):
                logger.error(f"标签必须是列表: {type(metadata.tags)}")
                return False

            return True

        except Exception as e:
            logger.error(f"元数据验证失败: {e}")
            return False

    def _validate_config_schema(self, config: Dict[str, Any],


                                schema: Dict[str, Any]) -> bool:
        """验证配置模式"""
        try:
            # 简单的配置验证
            for key, value in config.items():
                if key not in schema:
                    logger.warning(f"未知配置项: {key}")
                    continue

                expected_type = schema[key].get('type')
                if expected_type and not isinstance(value, expected_type):
                    logger.error(f"配置项类型错误: {key}, 期望 {expected_type}, 实际 {type(value)}")
                    return False

            return True

        except Exception as e:
            logger.error(f"配置模式验证失败: {e}")
            return False

    def _is_version_in_range(self, min_ver: str, max_ver: str,


                             target_min: str, target_max: str) -> bool:
        """检查版本是否在范围内"""
        try:
            # 简单的版本比较

            def version_to_tuple(version: str) -> tuple:

                return tuple(int(x) for x in version.split('.'))

            min_tuple = version_to_tuple(min_ver)
            max_tuple = version_to_tuple(max_ver)
            target_min_tuple = version_to_tuple(target_min)
            target_max_tuple = version_to_tuple(target_max)

            # 检查重叠
            return (min_tuple <= target_max_tuple and max_tuple >= target_min_tuple)

        except Exception:
            return False

    def _is_valid_name(self, name: str) -> bool:
        """检查名称是否有效"""
        if not name or not isinstance(name, str):
            return False

        # 检查长度
        if len(name) < 1 or len(name) > 50:
            return False

        # 检查字符
        import re
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', name):
            return False

        return True

    def _is_valid_version(self, version: str) -> bool:
        """检查版本是否有效"""
        if not version or not isinstance(version, str):
            return False

        # 检查版本格式
        import re
        if not re.match(r'^\d+\.\d+\.\d+$', version):
            return False

        return True
