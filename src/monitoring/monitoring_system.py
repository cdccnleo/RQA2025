#!/usr/bin/env python3
"""
监控系统主模块

整合所有监控组件，提供统一的监控系统接口。
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .core.unified_monitoring_interface import (
    IMonitoringSystem,
    IPerformanceAnalyzer,
    IMonitorComponent,
    MonitorType,
    AlertLevel,
    MetricType
)
from .engine.performance_analyzer import PerformanceAnalyzer
from .engine.monitor_components import MonitorComponentFactory as MonitorComponents
from .intelligent_alert_system import IntelligentAlertSystem

logger = logging.getLogger(__name__)


class MonitoringSystem(IMonitoringSystem):
    """
    监控系统主类

    整合所有监控组件，提供统一的监控接口。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化监控系统

        Args:
            config: 监控系统配置
        """
        self.config = config or {}
        self._initialized = False
        self._components = {}
        self._monitors = {}
        self._performance_analyzer = None
        self._intelligent_alert_system = None
        self._maintenance_mode = False
        self._global_alert_rules = {}

        logger.info("监控系统初始化开始")

    def initialize_monitoring(self, config: Dict[str, Any]) -> bool:
        """
        初始化监控系统

        Args:
            config: 监控配置

        Returns:
            bool: 初始化是否成功
        """
        try:
            self.config.update(config)

            # 初始化性能分析器
            self._performance_analyzer = PerformanceAnalyzer(self.config.get('performance', {}))

            # 初始化智能告警系统
            self._intelligent_alert_system = IntelligentAlertSystem(self.config.get('alert', {}))

            # 初始化监控组件
            self._components = MonitorComponents(self.config.get('components', {}))

            self._initialized = True
            logger.info("监控系统初始化完成")
            return True

        except Exception as e:
            logger.error(f"监控系统初始化失败: {e}")
            return False

    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            bool: 启动是否成功
        """
        if not self._initialized:
            logger.error("监控系统未初始化")
            return False

        try:
            # 启动所有监控组件
            for component_name, component in self._components.items():
                if hasattr(component, 'start'):
                    component.start()

            # 启动智能告警系统
            if self._intelligent_alert_system and hasattr(self._intelligent_alert_system, 'start'):
                self._intelligent_alert_system.start()

            logger.info("监控系统启动完成")
            return True

        except Exception as e:
            logger.error(f"监控系统启动失败: {e}")
            return False

    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            bool: 停止是否成功
        """
        try:
            # 停止所有监控组件
            for component_name, component in self._components.items():
                if hasattr(component, 'stop'):
                    component.stop()

            # 停止智能告警系统
            if self._intelligent_alert_system and hasattr(self._intelligent_alert_system, 'stop'):
                self._intelligent_alert_system.stop()

            logger.info("监控系统停止完成")
            return True

        except Exception as e:
            logger.error(f"监控系统停止失败: {e}")
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            Dict[str, Any]: 系统状态信息
        """
        return {
            'initialized': self._initialized,
            'components_count': len(self._components),
            'performance_analyzer': self._performance_analyzer is not None,
            'intelligent_alert_system': self._intelligent_alert_system is not None,
            'timestamp': datetime.now().isoformat()
        }

    def collect_metrics(self) -> Dict[str, Any]:
        """
        收集监控指标

        Returns:
            Dict[str, Any]: 监控指标数据
        """
        if not self._initialized:
            return {}

        metrics = {}
        try:
            # 收集各组件的指标
            for component_name, component in self._components.items():
                if hasattr(component, 'collect_metrics'):
                    component_metrics = component.collect_metrics()
                    if component_metrics:
                        metrics[component_name] = component_metrics

            # 收集性能指标
            if self._performance_analyzer and hasattr(self._performance_analyzer, 'get_metrics'):
                perf_metrics = self._performance_analyzer.get_metrics()
                if perf_metrics:
                    metrics['performance'] = perf_metrics

        except Exception as e:
            logger.error(f"收集监控指标失败: {e}")

        return metrics

    def get_performance_analyzer(self) -> Optional[IPerformanceAnalyzer]:
        """
        获取性能分析器

        Returns:
            Optional[IPerformanceAnalyzer]: 性能分析器实例
        """
        return self._performance_analyzer

    def get_monitor_components(self) -> Dict[str, IMonitorComponent]:
        """
        获取监控组件

        Returns:
            Dict[str, IMonitorComponent]: 监控组件字典
        """
        return self._components

    def get_intelligent_alert_system(self) -> Optional[IntelligentAlertSystem]:
        """
        获取智能告警系统

        Returns:
            Optional[IntelligentAlertSystem]: 智能告警系统实例
        """
        return self._intelligent_alert_system

    # 实现 IMonitoringSystem 接口的抽象方法

    def register_monitor(self, monitor: Any) -> bool:
        """
        注册监控器

        Args:
            monitor: 监控器实例

        Returns:
            bool: 注册是否成功
        """
        try:
            monitor_type = getattr(monitor, 'monitor_type', str(id(monitor)))
            self._monitors[monitor_type] = monitor
            logger.info(f"监控器注册成功: {monitor_type}")
            return True
        except Exception as e:
            logger.error(f"监控器注册失败: {e}")
            return False

    def unregister_monitor(self, monitor_type: Any) -> bool:
        """
        注销监控器

        Args:
            monitor_type: 监控器类型

        Returns:
            bool: 注销是否成功
        """
        try:
            if monitor_type in self._monitors:
                del self._monitors[monitor_type]
                logger.info(f"监控器注销成功: {monitor_type}")
                return True
            return False
        except Exception as e:
            logger.error(f"监控器注销失败: {e}")
            return False

    def get_monitor(self, monitor_type: Any) -> Optional[Any]:
        """
        获取监控器

        Args:
            monitor_type: 监控器类型

        Returns:
            Optional[Any]: 监控器实例
        """
        return self._monitors.get(monitor_type)

    def get_all_monitors(self) -> Dict[str, Any]:
        """
        获取所有监控器

        Returns:
            Dict[str, Any]: 监控器字典
        """
        return self._monitors.copy()

    def start_all_monitors(self) -> bool:
        """
        启动所有监控器

        Returns:
            bool: 启动是否成功
        """
        try:
            for monitor in self._monitors.values():
                if hasattr(monitor, 'start'):
                    monitor.start()
            logger.info("所有监控器启动完成")
            return True
        except Exception as e:
            logger.error(f"启动监控器失败: {e}")
            return False

    def stop_all_monitors(self) -> bool:
        """
        停止所有监控器

        Returns:
            bool: 停止是否成功
        """
        try:
            for monitor in self._monitors.values():
                if hasattr(monitor, 'stop'):
                    monitor.stop()
            logger.info("所有监控器停止完成")
            return True
        except Exception as e:
            logger.error(f"停止监控器失败: {e}")
            return False

    def get_system_health_score(self) -> float:
        """
        获取系统健康评分

        Returns:
            float: 健康评分 (0-100)
        """
        if not self._initialized:
            return 0.0

        try:
            # 简单的健康评分计算
            component_count = len(self._components)
            monitor_count = len(self._monitors)
            total_components = component_count + monitor_count

            if total_components == 0:
                return 100.0

            # 假设所有组件都是健康的
            healthy_components = total_components
            return (healthy_components / total_components) * 100.0

        except Exception as e:
            logger.error(f"计算健康评分失败: {e}")
            return 0.0

    def enable_maintenance_mode(self) -> bool:
        """
        启用维护模式

        Returns:
            bool: 启用是否成功
        """
        self._maintenance_mode = True
        logger.info("维护模式已启用")
        return True

    def disable_maintenance_mode(self) -> bool:
        """
        禁用维护模式

        Returns:
            bool: 禁用是否成功
        """
        self._maintenance_mode = False
        logger.info("维护模式已禁用")
        return True

    def is_maintenance_mode_active(self) -> bool:
        """
        检查维护模式是否激活

        Returns:
            bool: 维护模式状态
        """
        return self._maintenance_mode

    def set_global_alert_rules(self, rules: Dict[str, Any]) -> bool:
        """
        设置全局告警规则

        Args:
            rules: 告警规则

        Returns:
            bool: 设置是否成功
        """
        try:
            self._global_alert_rules = rules
            logger.info("全局告警规则设置成功")
            return True
        except Exception as e:
            logger.error(f"设置全局告警规则失败: {e}")
            return False

    def get_global_alert_rules(self) -> Dict[str, Any]:
        """
        获取全局告警规则

        Returns:
            Dict[str, Any]: 告警规则
        """
        return self._global_alert_rules.copy()

    def generate_system_report(self) -> Dict[str, Any]:
        """
        生成系统报告

        Returns:
            Dict[str, Any]: 系统报告
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'health_score': self.get_system_health_score(),
            'maintenance_mode': self._maintenance_mode,
            'monitors_count': len(self._monitors),
            'components_count': len(self._components),
            'alert_rules_count': len(self._global_alert_rules)
        }


# 为了向后兼容，提供别名
MonitorComponents = MonitorComponents
PerformanceAnalyzer = PerformanceAnalyzer

__all__ = [
    'MonitoringSystem',
    'PerformanceAnalyzer',
    'MonitorComponents'
]
