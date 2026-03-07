#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征层插件系统

提供插件化架构支持，允许动态加载和注册特征处理器。
"""

from .base_plugin import BaseFeaturePlugin, PluginMetadata, PluginType, PluginStatus
from .plugin_manager import FeaturePluginManager
from .plugin_registry import PluginRegistry
from .plugin_loader import PluginLoader
from .plugin_validator import PluginValidator

__all__ = [
    'BaseFeaturePlugin',
    'PluginMetadata',
    'PluginType',
    'PluginStatus',
    'FeaturePluginManager',
    'PluginRegistry',
    'PluginLoader',
    'PluginValidator'
]
