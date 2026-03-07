#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层 - Logger池监控组件 (重构版)

提供Logger对象池的实时监控、性能指标收集和告警功能。
集成到统一监控系统中，支持Prometheus指标导出。

重构说明:
- 拆分为多个职责单一的组件
- 使用参数对象模式替换长参数列表
- 提高代码可维护性和可测试性
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from ..core.parameter_objects import (
    MonitoringConfig,
    LoggerPoolStatsConfig,
    PrometheusExportConfig,
    DataPersistenceConfig,
    AlertRuleConfig
)
from ..components.monitoring_coordinator import MonitoringCoordinator
from ..components.stats_collector import StatsCollector
from ..components.alert_manager import AlertManager
from ..components.metrics_exporter import MetricsExporter
from ..components.data_persistor import DataPersistor


class LoggerPoolMonitorRefactored:
    """
    Logger池监控器 (重构版)

    使用组件化架构的Logger池监控器，提供：
    - 模块化的监控组件
    - 统一的参数对象配置
    - 更好的可维护性和可测试性
    """

    def __init__(self, pool_name: str = "default",
                 monitoring_config: Optional[MonitoringConfig] = None,
                 stats_config: Optional[LoggerPoolStatsConfig] = None,
                 export_config: Optional[PrometheusExportConfig] = None,
                 persistence_config: Optional[DataPersistenceConfig] = None):
        """
        初始化Logger池监控器

        Args:
            pool_name: 池名称
            monitoring_config: 监控配置
            stats_config: 统计配置
            export_config: 导出配置
            persistence_config: 持久化配置
        """
        self.pool_name = pool_name

        # 使用默认配置
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.stats_config = stats_config or LoggerPoolStatsConfig()
        self.export_config = export_config or PrometheusExportConfig()
        self.persistence_config = persistence_config or DataPersistenceConfig()

        # 初始化组件
        self._init_components()

        # 兼容性属性（为了向后兼容）
        self.current_stats: Optional[Dict[str, Any]] = None
        self.history_stats: List[Dict[str, Any]] = []

    def _init_components(self):
        """初始化监控组件"""
        # 统计收集器
        self.stats_collector = StatsCollector(self.pool_name, self.stats_config)

        # 告警管理器
        self.alert_manager = AlertManager(self.pool_name, self.monitoring_config.alert_thresholds)

        # 指标导出器
        self.metrics_exporter = MetricsExporter(self.pool_name, self.export_config)

        # 数据持久化器
        self.data_persistor = DataPersistor(self.pool_name, self.persistence_config)

        # 监控协调器
        self.monitoring_coordinator = MonitoringCoordinator(self.pool_name, self.monitoring_config)
        self.monitoring_coordinator.set_components(
            self.stats_collector,
            self.alert_manager,
            self.metrics_exporter
        )

    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            bool: 是否成功启动
        """
        return self.monitoring_coordinator.start_monitoring()

    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            bool: 是否成功停止
        """
        return self.monitoring_coordinator.stop_monitoring()

    def get_monitoring_status(self) -> Dict[str, Any]:
        """
        获取监控状态

        Returns:
            Dict[str, Any]: 监控状态信息
        """
        return self.monitoring_coordinator.get_monitoring_status()

    def collect_current_stats(self) -> Optional[Dict[str, Any]]:
        """
        收集当前统计信息

        Returns:
            Optional[Dict[str, Any]]: 当前统计信息
        """
        stats = self.stats_collector.collect_stats()
        if stats:
            # 持久化数据
            self.data_persistor.persist_data(stats)

            # 更新兼容性属性
            self.current_stats = stats
            self.history_stats = self.stats_collector.get_history_stats()

        return stats

    def get_current_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取当前统计信息

        Returns:
            Optional[Dict[str, Any]]: 当前统计信息
        """
        return self.stats_collector.get_current_stats()

    def get_history_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取历史统计信息

        Args:
            limit: 返回的最大记录数

        Returns:
            List[Dict[str, Any]]: 历史统计信息列表
        """
        return self.stats_collector.get_history_stats(limit)

    def get_metrics_for_prometheus(self) -> str:
        """
        获取Prometheus格式的指标

        Returns:
            str: Prometheus格式的指标字符串
        """
        return self.metrics_exporter.get_prometheus_metrics()

    def get_metrics_for_json(self) -> str:
        """
        获取JSON格式的指标

        Returns:
            str: JSON格式的指标字符串
        """
        return self.metrics_exporter.get_json_metrics()

    def get_alert_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取告警历史

        Args:
            limit: 返回的最大记录数

        Returns:
            List[Dict[str, Any]]: 告警历史列表
        """
        return self.alert_manager.get_alert_history(limit)

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """
        获取活跃告警

        Returns:
            List[Dict[str, Any]]: 活跃告警列表
        """
        return self.alert_manager.get_active_alerts()

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID

        Returns:
            bool: 是否成功确认
        """
        return self.alert_manager.acknowledge_alert(alert_id)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """
        获取告警统计信息

        Returns:
            Dict[str, Any]: 告警统计
        """
        return self.alert_manager.get_alert_statistics()

    def retrieve_historical_data(self, start_time: Optional[datetime] = None,
                                end_time: Optional[datetime] = None,
                                limit: int = 1000) -> List[Dict[str, Any]]:
        """
        检索历史数据

        Args:
            start_time: 开始时间
            end_time: 结束时间
            limit: 最大返回记录数

        Returns:
            List[Dict[str, Any]]: 历史数据列表
        """
        return self.data_persistor.retrieve_data(start_time, end_time, limit)

    def get_data_statistics(self) -> Dict[str, Any]:
        """
        获取数据统计信息

        Returns:
            Dict[str, Any]: 数据统计信息
        """
        return self.data_persistor.get_data_statistics()

    def export_data(self, file_path: str, format_type: str = 'json') -> bool:
        """
        导出数据

        Args:
            file_path: 导出文件路径
            format_type: 导出格式

        Returns:
            bool: 是否成功导出
        """
        return self.data_persistor.export_data(file_path, format_type)

    def cleanup_old_data(self, days_to_keep: int = 30) -> int:
        """
        清理旧数据

        Args:
            days_to_keep: 保留天数

        Returns:
            int: 删除的记录数量
        """
        return self.data_persistor.cleanup_old_data(days_to_keep)

    def add_custom_alert_rule(self, rule: AlertRuleConfig):
        """
        添加自定义告警规则

        Args:
            rule: 告警规则配置
        """
        self.alert_manager.add_alert_rule(rule)

    def remove_alert_rule(self, rule_id: str) -> bool:
        """
        移除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 是否成功移除
        """
        return self.alert_manager.remove_alert_rule(rule_id)

    def get_export_status(self) -> Dict[str, Any]:
        """
        获取导出状态

        Returns:
            Dict[str, Any]: 导出状态信息
        """
        return self.metrics_exporter.get_export_status()

    def export_metrics_to_file(self, format_type: str = 'prometheus',
                              file_path: Optional[str] = None) -> bool:
        """
        导出指标到文件

        Args:
            format_type: 导出格式
            file_path: 文件路径

        Returns:
            bool: 是否成功导出
        """
        return self.metrics_exporter.export_to_file(format_type, file_path)

    def analyze_performance_trends(self, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """
        分析性能趋势

        Args:
            metric_name: 指标名称
            window_size: 分析窗口大小

        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        return self.stats_collector.analyze_trends(metric_name, window_size)

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能汇总信息

        Returns:
            Dict[str, Any]: 性能汇总
        """
        current_stats = self.get_current_stats()
        if not current_stats:
            return {}

        # 计算各种百分位数
        percentiles = self.stats_collector.calculate_percentiles(
            self.stats_collector.get_access_times(),
            [50.0, 95.0, 99.0]
        )

        return {
            'current_stats': current_stats,
            'access_time_percentiles': percentiles,
            'alert_statistics': self.get_alert_statistics(),
            'data_statistics': self.get_data_statistics(),
            'generated_at': datetime.now().isoformat()
        }

    # 上下文管理器支持
    def __enter__(self):
        """上下文管理器入口"""
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_monitoring()
