"""
monitor_factory 模块

提供 monitor_factory 相关功能和接口。
"""

import logging

import time

from enum import Enum
from typing import Dict, Any, Optional, Type

from src.infrastructure.logging.core.exceptions import LogMonitorError
from .enums import AlertData


class UnifiedMonitor:
    """统一的监控器基类"""
    pass


class IMonitor:
    """监控器接口"""
    pass


class IMonitorFactory:
    """监控器工厂接口"""
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础设施层 - 日志系统组件monitor_factory 模块日志系统相关的文件提供日志系统相关的功能实现。"""

"""监控器工厂 - 解决重复代码问题
提供统一的监控器创建接口，支持多种监控器类型：
- 统一监控器
- 性能监控器
- 业务监控器
- 系统监控器
- 应用监控器
"""


class MonitorType(Enum):
    """监控器类型"""
    UNIFIED = "unified"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    SYSTEM = "system"
    APPLICATION = "application"


class MonitorFactory(IMonitorFactory):
    """监控器工厂实现"""

    def __init__(self):
        self._monitors: Dict[str, Type[IMonitor]] = {}
        self._monitor_instances: Dict[str, IMonitor] = {}  # 存储监控器实例
        self._logger = logging.getLogger(__name__)
        # 注册默认监控器类型
        self._register_default_monitors()

    def _register_default_monitors(self):
        """
        注册默认的监控器类型

        按照优先级和可用性注册所有默认监控器
        """
        # 定义监控器配置
        monitor_configs = self._get_monitor_configs()

        # 首先尝试批量注册
        self._register_monitors_batch(monitor_configs)

        # 然后逐个注册，确保最大可用性
        self._register_monitors_individual(monitor_configs)

    def _get_monitor_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        获取监控器配置

        Returns:
            监控器类型到配置的映射
        """
        return {
            MonitorType.UNIFIED.value: {
                'class': 'UnifiedMonitor',
                'module': 'infrastructure.logging.core.monitoring',
                'description': '统一监控器'
            },
            MonitorType.PERFORMANCE.value: {
                'class': 'PerformanceOptimizedMonitor',
                'module': 'infrastructure.logging.performance_optimized_monitor',
                'description': '性能优化监控器'
            },
            MonitorType.BUSINESS.value: {
                'class': 'BusinessMetricsMonitor',
                'module': 'infrastructure.logging.business_metrics_monitor',
                'description': '业务指标监控器'
            },
            MonitorType.SYSTEM.value: {
                'class': 'SystemMonitor',
                'module': 'infrastructure.logging.system_monitor',
                'description': '系统监控器'
            },
            MonitorType.APPLICATION.value: {
                'class': 'ApplicationMonitor',
                'module': 'infrastructure.logging.application_monitor',
                'description': '应用监控器'
            },
            'automation': {
                'class': 'AutomationMonitor',
                'module': 'infrastructure.logging.automation_monitor',
                'description': '自动化监控器'
            },
            'prometheus': {
                'class': 'PrometheusMonitor',
                'module': 'infrastructure.logging.prometheus_monitor',
                'description': 'Prometheus监控器'
            }
        }

    def _register_monitors_batch(self, monitor_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        批量注册监控器

        Args:
            monitor_configs: 监控器配置字典
        """
        try:
            # 动态导入所有需要的模块
            imported_modules = self._import_monitor_modules(monitor_configs)

            # 注册所有成功导入的监控器
            for monitor_type, config in monitor_configs.items():
                class_name = config['class']
                if class_name in imported_modules:
                    self.register_monitor(monitor_type, imported_modules[class_name])

        except ImportError as e:
            self._logger.warning(f"批量注册监控器失败，某些监控器类型不可用: {e}")

    def _import_monitor_modules(self, monitor_configs: Dict[str, Dict[str, Any]]) -> Dict[str, type]:
        """
        导入监控器模块

        Args:
            monitor_configs: 监控器配置字典

        Returns:
            类名到类的映射
        """
        imported_classes = {}

        for config in monitor_configs.values():
            try:
                module = __import__(config['module'], fromlist=[config['class']])
                monitor_class = getattr(module, config['class'])
                imported_classes[config['class']] = monitor_class
            except (ImportError, AttributeError):
                # 忽略导入失败的模块
                pass

        return imported_classes

    def _register_monitors_individual(self, monitor_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        逐个注册监控器，确保最大可用性

        Args:
            monitor_configs: 监控器配置字典
        """
        for monitor_type, config in monitor_configs.items():
            try:
                module = __import__(config['module'], fromlist=[config['class']])
                monitor_class = getattr(module, config['class'])
                self.register_monitor(monitor_type, monitor_class)
            except (ImportError, AttributeError):
                # 忽略单个监控器注册失败
                pass

    def create_monitor(self, monitor_type: str, **kwargs) -> IMonitor:
        """
        创建监控器
        Args:
            monitor_type: 监控器类型
            **kwargs: 初始化参数
        Returns:
            监控器实例
        Raises:
            MonitorError: 当监控器类型不存在时
        """
        if monitor_type not in self._monitors:
            available = list(self._monitors.keys())
            raise LogMonitorError(f"未知的监控器类型: {monitor_type}，可用类型: {available}")
        try:
            monitor_class = self._monitors[monitor_type]
            monitor = monitor_class(**kwargs)
            self._logger.info(f"成功创建监控器: {monitor_type}")
            return monitor
        except Exception as e:
            self._logger.error(f"创建监控器失败: {monitor_type}, 错误: {str(e)}")
            raise LogMonitorError(f"创建监控器失败: {str(e)}")

    def register_monitor(self, name: str, monitor_class: Type[IMonitor]) -> None:
        """
        注册监控器类型
        Args:
            name: 监控器名称
            monitor_class: 监控器类
        """
        if not issubclass(monitor_class, IMonitor):
            raise LogMonitorError(f"监控器类必须实现 IMonitor 接口: {monitor_class}")
        self._monitors[name] = monitor_class
        self._logger.info(f"注册监控器: {name}")

    def get_available_monitors(self) -> Dict[str, Type[IMonitor]]:
        """
        获取可用的监控器类型
        Returns:
            可用监控器类型字典
        """
        return self._monitors.copy()

    def get_monitor(self, name: str) -> IMonitor:
        """
        获取指定的监控器实例（单例模式）
        Args:
            name: 监控器名称
        Returns:
            监控器实例
        Raises:
            ValueError: 当监控器类型不存在时
        """
        monitor_class = self._monitors.get(name)
        if monitor_class is None:
            raise ValueError(f"Unknown monitor type: {name}")
        # 如果实例已存在，返回现有实例
        if name in self._monitor_instances:
            return self._monitor_instances[name]
        try:
            # 创建新实例并缓存
            instance = monitor_class()
            self._monitor_instances[name] = instance
            self._logger.info(f"创建新的监控器实例: {name}")
            return instance
        except Exception as e:
            self._logger.error(f"创建监控器实例失败: {e}")
            raise ValueError(f"Failed to create monitor instance: {e}")


class SystemMonitor(UnifiedMonitor):
    """系统监控器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._system_metrics = {}

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录系统指标"""
        if name not in self._system_metrics:
            self._system_metrics[name] = []
        metric_data = {
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        }
        self._system_metrics[name].append(metric_data)

    def get_metrics(self, name: str, time_range: Optional[tuple] = None) -> list:
        """获取系统指标"""
        if name not in self._system_metrics:
            return []
        metrics = self._system_metrics[name]
        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m['timestamp'] <= end_time]
        return metrics


class ApplicationMonitor(UnifiedMonitor):
    """应用监控器"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._app_metrics = {}

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录应用指标"""
        if name not in self._app_metrics:
            self._app_metrics[name] = []
        metric_data = {
            'value': value,
            'timestamp': time.time(),
            'tags': tags or {}
        }
        self._app_metrics[name].append(metric_data)

    def get_metrics(self, name: str, time_range: Optional[tuple] = None) -> list:
        """获取应用指标"""
        if name not in self._app_metrics:
            return []
        metrics = self._app_metrics[name]
        if time_range:
            start_time, end_time = time_range
            metrics = [m for m in metrics if start_time <= m['timestamp'] <= end_time]
        return metrics


def get_monitor_factory() -> MonitorFactory:
    """获取监控器工厂实例"""
    return MonitorFactory()


def create_monitor(monitor_type: str = "unified", **kwargs) -> IMonitor:
    """
    创建监控器的便捷函数
    Args:
        monitor_type: 监控器类型
        **kwargs: 初始化参数
    Returns:
        监控器实例
    """
    factory = get_monitor_factory()
    return factory.create_monitor(monitor_type, **kwargs)


def register_monitor(name: str, monitor_class: Type[IMonitor]) -> None:
    """
    注册监控器类型的便捷函数
    Args:
        name: 监控器名称
        monitor_class: 监控器类
    """
    factory = get_monitor_factory()
    factory.register_monitor(name, monitor_class)
