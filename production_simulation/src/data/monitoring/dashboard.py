#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据层监控可视化面板
实现实时监控面板、性能指标可视化和详细质量报告
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
import tempfile
import os

from src.infrastructure.logging import get_infrastructure_logger
from ..enhanced_integration_manager import EnhancedDataIntegrationManager

logger = get_infrastructure_logger('data_dashboard')


@dataclass
class DashboardConfig:

    """仪表板配置"""
    title: str = "RQA2025 数据层监控面板"
    refresh_interval: int = 30  # 30秒刷新间隔
    enable_auto_refresh: bool = True
    max_history_points: int = 1000
    enable_alerts: bool = True
    enable_export: bool = True
    theme: str = "dark"  # dark, light
    layout: str = "grid"  # grid, list, compact


@dataclass
class MetricWidget:

    """指标组件"""
    id: str
    title: str
    metric_type: str  # gauge, chart, table, status
    data_source: str
    refresh_interval: int = 30
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: str = ""
    description: str = ""


@dataclass
class AlertRule:

    """告警规则"""
    id: str
    name: str
    condition: str
    threshold: float
    level: str  # info, warning, critical
    message_template: str
    enabled: bool = True


class DataDashboard:

    """数据层监控仪表板"""

    def __init__(self, enhanced_manager: EnhancedDataIntegrationManager, config: DashboardConfig = None):
        """
        初始化数据监控仪表板

        Args:
            enhanced_manager: 增强版数据集成管理器
            config: 仪表板配置
        """
        self.enhanced_manager = enhanced_manager
        self.config = config or DashboardConfig()

        # 组件管理
        self.widgets: Dict[str, MetricWidget] = {}
        self.alert_rules: Dict[str, AlertRule] = {}

        # 数据缓存
        self.metric_cache: Dict[str, Any] = {}
        self.history_data: Dict[str, List[Dict]] = {}

        # 回调管理
        self.callbacks: Dict[str, List[Callable]] = {}

        # 自动刷新
        self._refresh_thread = None
        self._stop_refresh = False

        # 初始化默认组件
        self._init_default_widgets()
        self._init_default_alert_rules()

        logger.info("数据层监控仪表板初始化完成")

    def _init_default_widgets(self):
        """初始化默认组件"""
        # 性能指标组件
        self.add_widget(MetricWidget(
            id="performance_overview",
            title="性能概览",
            metric_type="gauge",
            data_source="performance",
            refresh_interval=30,
            unit="%",
            description="系统整体性能指标"
        ))

        # 缓存命中率组件
        self.add_widget(MetricWidget(
            id="cache_hit_rate",
            title="缓存命中率",
            metric_type="gauge",
            data_source="cache_hit_rate",
            refresh_interval=15,
            threshold_warning=0.8,
            threshold_critical=0.6,
            unit="%",
            description="数据缓存命中率"
        ))

        # 节点状态组件
        self.add_widget(MetricWidget(
            id="node_status",
            title="节点状态",
            metric_type="status",
            data_source="node_status",
            refresh_interval=10,
            description="分布式节点运行状态"
        ))

        # 数据流状态组件
        self.add_widget(MetricWidget(
            id="stream_status",
            title="数据流状态",
            metric_type="status",
            data_source="stream_status",
            refresh_interval=10,
            description="实时数据流运行状态"
        ))

        # 质量指标组件
        self.add_widget(MetricWidget(
            id="quality_metrics",
            title="数据质量",
            metric_type="chart",
            data_source="quality_metrics",
            refresh_interval=60,
            description="数据质量监控指标"
        ))

        # 告警历史组件
        self.add_widget(MetricWidget(
            id="alert_history",
            title="告警历史",
            metric_type="table",
            data_source="alert_history",
            refresh_interval=60,
            description="最近告警记录"
        ))

    def _init_default_alert_rules(self):
        """初始化默认告警规则"""
        # 性能告警
        self.add_alert_rule(AlertRule(
            id="performance_degradation",
            name="性能下降告警",
            condition="performance_avg < 0.8",
            threshold=0.8,
            level="warning",
            message_template="系统性能下降: {value:.2f}%",
            enabled=True
        ))

        # 缓存告警
        self.add_alert_rule(AlertRule(
            id="cache_miss_high",
            name="缓存命中率低",
            condition="cache_hit_rate < 0.7",
            threshold=0.7,
            level="warning",
            message_template="缓存命中率过低: {value:.2f}%",
            enabled=True
        ))

        # 节点告警
        self.add_alert_rule(AlertRule(
            id="node_offline",
            name="节点离线",
            condition="offline_nodes > 0",
            threshold=0,
            level="critical",
            message_template="检测到 {value} 个节点离线",
            enabled=True
        ))

    def add_widget(self, widget: MetricWidget):
        """添加指标组件"""
        self.widgets[widget.id] = widget
        self.history_data[widget.id] = []
        logger.info(f"添加监控组件: {widget.title}")

    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.alert_rules[rule.id] = rule
        logger.info(f"添加告警规则: {rule.name}")

    def add_callback(self, event_type: str, callback: Callable):
        """添加事件回调"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    def _trigger_callback(self, event_type: str, data: Any):
        """触发回调"""
        if event_type in self.callbacks:
            for callback in self.callbacks[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"回调执行失败 {event_type}: {e}")

    def _collect_metrics(self) -> Dict[str, Any]:
        """收集指标数据"""
        try:
            # 获取性能指标
            performance_metrics = self.enhanced_manager.get_performance_metrics()

            # 获取质量报告
            quality_report = self.enhanced_manager.get_quality_report(days=1)

            # 获取告警历史
            alert_history = self.enhanced_manager.get_alert_history(hours=24)

            # 计算缓存命中率
            cache_stats = performance_metrics.get("cache", {})
            cache_hit_rate = 0.0
            if cache_stats.get("hits", 0) + cache_stats.get("misses", 0) > 0:
                cache_hit_rate = cache_stats["hits"] / (cache_stats["hits"] + cache_stats["misses"])

            # 计算节点状态
            nodes = performance_metrics.get("nodes", {})
            online_nodes = sum(1 for node in nodes.values() if node.get("status") == "active")
            total_nodes = len(nodes) if nodes else 1
            node_availability = online_nodes / total_nodes if total_nodes > 0 else 0.0

            # 计算数据流状态
            streams = performance_metrics.get("streams", {})
            running_streams = sum(1 for stream in streams.values()
                                  if stream.get("is_running", False))
            total_streams = len(streams) if streams else 0
            stream_availability = running_streams / total_streams if total_streams > 0 else 1.0

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "overall_score": min(node_availability, stream_availability, cache_hit_rate),
                    "node_availability": node_availability,
                    "stream_availability": stream_availability,
                    "cache_hit_rate": cache_hit_rate,
                    "load_time_avg": performance_metrics.get("performance", {}).get("distributed_load_time", {}).get("avg", 0.0)
                },
                "cache": cache_stats,
                "nodes": nodes,
                "streams": streams,
                "quality": quality_report,
                "alerts": alert_history
            }

            return metrics

        except Exception as e:
            logger.error(f"收集指标数据失败: {e}")
            return {"timestamp": datetime.now().isoformat(), "error": str(e)}

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        metrics = self._collect_metrics()

        dashboard_data = {
            "config": asdict(self.config),
            "widgets": {widget_id: asdict(widget) for widget_id, widget in self.widgets.items()},
            "alert_rules": {rule_id: asdict(rule) for rule_id, rule in self.alert_rules.items()},
            "current_metrics": metrics,
            "history_data": self.history_data,
            "status": {
                "auto_refresh": self._refresh_thread is not None,
                "last_update": datetime.now().isoformat(),
                "widget_count": len(self.widgets),
                "alert_rule_count": len(self.alert_rules)
            }
        }

        return dashboard_data

    def export_dashboard_report(self, format: str = "json", file_path: Optional[str] = None) -> str:
        """导出仪表板报告"""
        dashboard_data = self.get_dashboard_data()

        if format.lower() == "json":
            content = json.dumps(dashboard_data, indent=2, ensure_ascii=False)
            extension = ".json"
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        if file_path is None:
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_path = os.path.join(temp_dir, f"dashboard_report_{timestamp}{extension}")

        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"仪表板报告已导出: {file_path}")
        return file_path

    def shutdown(self):
        """关闭仪表板"""
        self.stop_auto_refresh()
        logger.info("数据层监控仪表板已关闭")
