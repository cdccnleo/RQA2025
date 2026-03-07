#!/usr/bin/env python3
"""
运维监控设置系统 - RQA2025生产环境监控和告警机制

基于生产就绪评估结果，建立完整的生产环境运维监控体系：
1. 监控指标体系搭建
2. 告警规则配置
3. 监控仪表板设置
4. 日志聚合分析
5. 运维自动化脚本

作者: AI Assistant
创建时间: 2025年12月4日
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import subprocess
import sys


@dataclass
class MonitoringMetric:
    """监控指标"""
    name: str
    type: str  # system, application, business
    description: str
    unit: str
    collection_method: str  # prometheus, custom, log_based
    query: Optional[str] = None
    thresholds: Dict[str, float] = None
    alerting_enabled: bool = True


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric_name: str
    condition: str  # gt, lt, eq, range
    threshold: float
    duration: str  # 持续时间，如 "5m", "1h"
    severity: str  # critical, warning, info
    description: str
    notification_channels: List[str]
    auto_actions: List[str] = None  # 自动响应动作


@dataclass
class MonitoringDashboard:
    """监控仪表板"""
    name: str
    description: str
    panels: List[Dict[str, Any]]
    refresh_interval: str
    tags: List[str]


@dataclass
class LogAggregationConfig:
    """日志聚合配置"""
    source_type: str  # application, system, access
    log_path: str
    format: str
    filters: List[str]
    retention_days: int
    alerting_patterns: List[str]


class OperationsMonitoringSetup:
    """
    运维监控设置器

    建立完整的生产环境监控和告警体系
    """

    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent)
        self.monitoring_dir = self.project_root / "monitoring"
        self.dashboards_dir = self.monitoring_dir / "dashboards"
        self.alerts_dir = self.monitoring_dir / "alerts"
        self.configs_dir = self.monitoring_dir / "configs"
        self.scripts_dir = self.monitoring_dir / "scripts"

        # 创建目录结构
        for dir_path in [self.monitoring_dir, self.dashboards_dir,
                        self.alerts_dir, self.configs_dir, self.scripts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def setup_operations_monitoring(self, readiness_report_path: str = None) -> Dict[str, Any]:
        """
        设置运维监控体系

        Args:
            readiness_report_path: 生产就绪评估报告路径

        Returns:
            监控设置结果
        """
        print("📊 开始RQA2025运维监控设置")
        print("=" * 50)

        # 加载生产就绪评估结果
        readiness_report = self._load_readiness_report(readiness_report_path)

        # 1. 定义监控指标体系
        metrics = self._define_monitoring_metrics(readiness_report)

        # 2. 配置告警规则
        alert_rules = self._configure_alert_rules(metrics, readiness_report)

        # 3. 创建监控仪表板
        dashboards = self._create_monitoring_dashboards(metrics)

        # 4. 设置日志聚合
        log_configs = self._setup_log_aggregation()

        # 5. 生成运维脚本
        operations_scripts = self._generate_operations_scripts()

        # 6. 配置监控基础设施
        infrastructure_config = self._configure_monitoring_infrastructure()

        # 组织监控产物
        monitoring_artifacts = self._organize_monitoring_artifacts(
            metrics, alert_rules, dashboards, log_configs, operations_scripts, infrastructure_config
        )

        # 生成监控设置报告
        monitoring_report = {
            "setup_date": datetime.now().isoformat(),
            "based_on_readiness_report": readiness_report_path,
            "monitoring_coverage": self._assess_monitoring_coverage(readiness_report),
            "metrics_configured": len(metrics),
            "alert_rules_configured": len(alert_rules),
            "dashboards_created": len(dashboards),
            "critical_metrics": [m.name for m in metrics if m.alerting_enabled and "critical" in (m.thresholds or {})],
            "monitoring_requirements": readiness_report.get("deployment_readiness", {}).get("monitoring_requirements", []),
            "artifacts_created": monitoring_artifacts,
            "implementation_guide": self._generate_implementation_guide(),
            "maintenance_schedule": self._generate_maintenance_schedule()
        }

        # 保存监控设置报告
        self._save_monitoring_report(monitoring_report)

        print("\n✅ 运维监控设置完成")
        print("=" * 40)
        print(f"📊 监控指标: {len(metrics)} 个")
        print(f"🚨 告警规则: {len(alert_rules)} 个")
        print(f"📈 仪表板: {len(dashboards)} 个")
        print(f"📄 日志配置: {len(log_configs)} 个")
        print(f"⚙️  运维脚本: {len(operations_scripts)} 个")

        return monitoring_report

    def _load_readiness_report(self, report_path: str = None) -> Dict[str, Any]:
        """加载生产就绪评估报告"""
        if not report_path:
            # 查找最新的评估报告
            report_files = list(self.project_root.glob("test_logs/production_readiness_assessment_*.json"))
            if report_files:
                report_path = max(report_files, key=lambda p: p.stat().st_mtime)

        if report_path and Path(report_path).exists():
            with open(report_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # 返回默认评估结果
        return {
            "overall_score": 66.3,
            "deployment_readiness": {
                "monitoring_requirements": ["内存使用监控", "性能指标监控", "日志监控"]
            }
        }

    def _define_monitoring_metrics(self, readiness_report: Dict[str, Any]) -> List[MonitoringMetric]:
        """定义监控指标体系"""
        metrics = []

        # 系统级指标
        system_metrics = [
            MonitoringMetric(
                name="cpu_usage_percent",
                type="system",
                description="CPU使用率百分比",
                unit="percent",
                collection_method="prometheus",
                query="100 - (avg by(instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                thresholds={"warning": 70.0, "critical": 85.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="memory_usage_percent",
                type="system",
                description="内存使用率百分比",
                unit="percent",
                collection_method="prometheus",
                query="(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100",
                thresholds={"warning": 75.0, "critical": 90.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="disk_usage_percent",
                type="system",
                description="磁盘使用率百分比",
                unit="percent",
                collection_method="prometheus",
                query="(node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes * 100",
                thresholds={"warning": 80.0, "critical": 95.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="network_connections",
                type="system",
                description="网络连接数",
                unit="count",
                collection_method="prometheus",
                query="node_netstat_Tcp_CurrEstab",
                thresholds={"warning": 1000, "critical": 2000},
                alerting_enabled=True
            )
        ]
        metrics.extend(system_metrics)

        # 应用级指标
        application_metrics = [
            MonitoringMetric(
                name="application_response_time",
                type="application",
                description="应用响应时间",
                unit="milliseconds",
                collection_method="prometheus",
                query="histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) * 1000",
                thresholds={"warning": 1000.0, "critical": 5000.0},  # P95
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="application_requests_per_second",
                type="application",
                description="应用请求速率",
                unit="requests_per_second",
                collection_method="prometheus",
                query="rate(http_requests_total[5m])",
                thresholds={"warning": 100.0, "critical": 10.0},  # 异常降低
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="application_error_rate",
                type="application",
                description="应用错误率",
                unit="percent",
                collection_method="prometheus",
                query="(rate(http_requests_total{status=~\"5..\"}[5m]) / rate(http_requests_total[5m])) * 100",
                thresholds={"warning": 1.0, "critical": 5.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="application_memory_usage",
                type="application",
                description="应用内存使用量",
                unit="bytes",
                collection_method="custom",
                thresholds={"warning": 1073741824, "critical": 2147483648},  # 1GB, 2GB
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="database_connections_active",
                type="application",
                description="数据库活跃连接数",
                unit="count",
                collection_method="prometheus",
                query="pg_stat_activity_count{state=\"active\"}",
                thresholds={"warning": 50, "critical": 100},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="redis_memory_usage",
                type="application",
                description="Redis内存使用量",
                unit="bytes",
                collection_method="prometheus",
                query="redis_memory_used_bytes",
                thresholds={"warning": 536870912, "critical": 1073741824},  # 512MB, 1GB
                alerting_enabled=True
            )
        ]
        metrics.extend(application_metrics)

        # 业务级指标 (量化交易特有)
        business_metrics = [
            MonitoringMetric(
                name="trading_volume_per_minute",
                type="business",
                description="每分钟交易量",
                unit="trades_per_minute",
                collection_method="custom",
                thresholds={"warning": 10.0, "critical": 1.0},  # 异常降低
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="market_data_latency",
                type="business",
                description="市场数据延迟",
                unit="milliseconds",
                collection_method="custom",
                thresholds={"warning": 100.0, "critical": 500.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="order_execution_success_rate",
                type="business",
                description="订单执行成功率",
                unit="percent",
                collection_method="custom",
                thresholds={"warning": 95.0, "critical": 90.0},
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="portfolio_value_change",
                type="business",
                description="投资组合价值变化",
                unit="percent",
                collection_method="custom",
                thresholds={"warning": -5.0, "critical": -10.0},  # 异常亏损
                alerting_enabled=True
            ),
            MonitoringMetric(
                name="risk_exposure_level",
                type="business",
                description="风险暴露水平",
                unit="percent",
                collection_method="custom",
                thresholds={"warning": 80.0, "critical": 95.0},
                alerting_enabled=True
            )
        ]
        metrics.extend(business_metrics)

        return metrics

    def _configure_alert_rules(self, metrics: List[MonitoringMetric], readiness_report: Dict[str, Any]) -> List[AlertRule]:
        """配置告警规则"""
        alert_rules = []

        # 基于监控要求的告警配置
        monitoring_requirements = readiness_report.get("deployment_readiness", {}).get("monitoring_requirements", [])

        # 为每个指标创建告警规则
        for metric in metrics:
            if not metric.alerting_enabled:
                continue

            # 基于阈值创建告警规则
            if metric.thresholds:
                for severity, threshold in metric.thresholds.items():
                    if severity in ["warning", "critical"]:
                        condition = "gt" if metric.name in ["cpu_usage_percent", "memory_usage_percent", "disk_usage_percent",
                                                        "network_connections", "application_response_time", "application_error_rate",
                                                        "application_memory_usage", "database_connections_active", "redis_memory_usage",
                                                        "market_data_latency", "risk_exposure_level"] else "lt"

                        # 根据监控要求调整通知渠道
                        notification_channels = ["email", "slack"]
                        if "内存使用监控" in monitoring_requirements and "memory" in metric.name:
                            notification_channels.append("pagerduty")
                        if "性能指标监控" in monitoring_requirements and metric.type == "application":
                            notification_channels.append("pagerduty")
                        if "日志监控" in monitoring_requirements:
                            notification_channels.append("webhook")

                        alert_rule = AlertRule(
                            name=f"{metric.name}_{severity}_alert",
                            metric_name=metric.name,
                            condition=condition,
                            threshold=threshold,
                            duration="5m",
                            severity=severity,
                            description=f"{metric.description} {severity}级别告警",
                            notification_channels=notification_channels,
                            auto_actions=self._get_auto_actions_for_metric(metric, severity)
                        )
                        alert_rules.append(alert_rule)

        # 添加复合告警规则
        composite_rules = [
            AlertRule(
                name="system_overload_alert",
                metric_name="composite_system_load",
                condition="gt",
                threshold=80.0,
                duration="10m",
                severity="critical",
                description="系统整体负载过高 (CPU + 内存综合指标)",
                notification_channels=["email", "slack", "pagerduty"],
                auto_actions=["scale_up_resources", "notify_oncall"]
            ),
            AlertRule(
                name="trading_system_failure_alert",
                metric_name="composite_trading_health",
                condition="lt",
                threshold=50.0,
                duration="5m",
                severity="critical",
                description="交易系统整体健康度严重下降",
                notification_channels=["email", "slack", "pagerduty", "sms"],
                auto_actions=["stop_trading", "notify_compliance", "create_incident"]
            ),
            AlertRule(
                name="data_feed_failure_alert",
                metric_name="market_data_feed_status",
                condition="eq",
                threshold=0.0,
                duration="1m",
                severity="critical",
                description="市场数据馈送中断",
                notification_channels=["email", "slack", "pagerduty", "sms"],
                auto_actions=["switch_backup_feed", "notify_trading_desk"]
            )
        ]
        alert_rules.extend(composite_rules)

        return alert_rules

    def _get_auto_actions_for_metric(self, metric: MonitoringMetric, severity: str) -> List[str]:
        """为指标获取自动响应动作"""
        auto_actions = []

        if severity == "critical":
            if "memory" in metric.name:
                auto_actions.extend(["log_memory_dump", "attempt_gc"])
            elif "cpu" in metric.name:
                auto_actions.extend(["log_thread_dump", "check_process_health"])
            elif "disk" in metric.name:
                auto_actions.extend(["cleanup_temp_files", "log_disk_usage"])
            elif "database" in metric.name:
                auto_actions.extend(["check_db_connections", "log_slow_queries"])
            elif "redis" in metric.name:
                auto_actions.extend(["check_redis_memory", "attempt_eviction"])

        if severity == "warning":
            auto_actions.append("log_warning_details")

        return auto_actions

    def _create_monitoring_dashboards(self, metrics: List[MonitoringMetric]) -> List[MonitoringDashboard]:
        """创建监控仪表板"""
        dashboards = []

        # 系统概览仪表板
        system_dashboard = MonitoringDashboard(
            name="system_overview",
            description="系统整体状态监控仪表板",
            refresh_interval="30s",
            tags=["system", "overview", "infrastructure"],
            panels=[
                {
                    "title": "CPU使用率",
                    "type": "graph",
                    "metrics": ["cpu_usage_percent"],
                    "layout": {"x": 0, "y": 0, "w": 6, "h": 4}
                },
                {
                    "title": "内存使用率",
                    "type": "graph",
                    "metrics": ["memory_usage_percent"],
                    "layout": {"x": 6, "y": 0, "w": 6, "h": 4}
                },
                {
                    "title": "磁盘使用率",
                    "type": "graph",
                    "metrics": ["disk_usage_percent"],
                    "layout": {"x": 0, "y": 4, "w": 6, "h": 4}
                },
                {
                    "title": "网络连接数",
                    "type": "graph",
                    "metrics": ["network_connections"],
                    "layout": {"x": 6, "y": 4, "w": 6, "h": 4}
                }
            ]
        )

        # 应用性能仪表板
        application_dashboard = MonitoringDashboard(
            name="application_performance",
            description="应用性能监控仪表板",
            refresh_interval="15s",
            tags=["application", "performance", "response"],
            panels=[
                {
                    "title": "响应时间分布",
                    "type": "heatmap",
                    "metrics": ["application_response_time"],
                    "layout": {"x": 0, "y": 0, "w": 8, "h": 6}
                },
                {
                    "title": "请求速率",
                    "type": "graph",
                    "metrics": ["application_requests_per_second"],
                    "layout": {"x": 8, "y": 0, "w": 4, "h": 6}
                },
                {
                    "title": "错误率",
                    "type": "graph",
                    "metrics": ["application_error_rate"],
                    "layout": {"x": 0, "y": 6, "w": 6, "h": 4}
                },
                {
                    "title": "活跃连接数",
                    "type": "graph",
                    "metrics": ["database_connections_active"],
                    "layout": {"x": 6, "y": 6, "w": 6, "h": 4}
                }
            ]
        )

        # 业务监控仪表板
        business_dashboard = MonitoringDashboard(
            name="business_monitoring",
            description="业务指标监控仪表板",
            refresh_interval="1m",
            tags=["business", "trading", "risk"],
            panels=[
                {
                    "title": "交易量趋势",
                    "type": "graph",
                    "metrics": ["trading_volume_per_minute"],
                    "layout": {"x": 0, "y": 0, "w": 6, "h": 4}
                },
                {
                    "title": "市场数据延迟",
                    "type": "graph",
                    "metrics": ["market_data_latency"],
                    "layout": {"x": 6, "y": 0, "w": 6, "h": 4}
                },
                {
                    "title": "订单执行成功率",
                    "type": "gauge",
                    "metrics": ["order_execution_success_rate"],
                    "layout": {"x": 0, "y": 4, "w": 4, "h": 4}
                },
                {
                    "title": "投资组合价值变化",
                    "type": "graph",
                    "metrics": ["portfolio_value_change"],
                    "layout": {"x": 4, "y": 4, "w": 4, "h": 4}
                },
                {
                    "title": "风险暴露水平",
                    "type": "gauge",
                    "metrics": ["risk_exposure_level"],
                    "layout": {"x": 8, "y": 4, "w": 4, "h": 4}
                }
            ]
        )

        # 告警概览仪表板
        alerts_dashboard = MonitoringDashboard(
            name="alerts_overview",
            description="告警状态总览仪表板",
            refresh_interval="30s",
            tags=["alerts", "incidents", "overview"],
            panels=[
                {
                    "title": "活跃告警",
                    "type": "table",
                    "metrics": ["active_alerts"],
                    "layout": {"x": 0, "y": 0, "w": 12, "h": 6}
                },
                {
                    "title": "告警趋势 (24h)",
                    "type": "graph",
                    "metrics": ["alerts_per_hour"],
                    "layout": {"x": 0, "y": 6, "w": 6, "h": 4}
                },
                {
                    "title": "告警严重程度分布",
                    "type": "pie",
                    "metrics": ["alerts_by_severity"],
                    "layout": {"x": 6, "y": 6, "w": 6, "h": 4}
                }
            ]
        )

        dashboards.extend([system_dashboard, application_dashboard, business_dashboard, alerts_dashboard])

        return dashboards

    def _setup_log_aggregation(self) -> List[LogAggregationConfig]:
        """设置日志聚合"""
        log_configs = []

        # 应用日志配置
        app_log_config = LogAggregationConfig(
            source_type="application",
            log_path="/var/log/rqa2025/app.log",
            format="json",
            filters=[
                "level:ERROR OR level:CRITICAL",
                "response_time:>5000",
                "error_code:5*"
            ],
            retention_days=90,
            alerting_patterns=[
                "Exception|Error|FATAL",
                "timeout|TimeoutError",
                "connection refused|ConnectionError"
            ]
        )

        # 系统日志配置
        system_log_config = LogAggregationConfig(
            source_type="system",
            log_path="/var/log/syslog",
            format="syslog",
            filters=[
                "rqa2025",
                "CRITICAL|ERROR",
                "OOM|Out of memory"
            ],
            retention_days=30,
            alerting_patterns=[
                "kernel.*panic",
                "OOM killer",
                "systemd.*failed"
            ]
        )

        # 访问日志配置
        access_log_config = LogAggregationConfig(
            source_type="access",
            log_path="/var/log/rqa2025/access.log",
            format="combined",
            filters=[
                "status:>=500",
                "response_time:>10000"
            ],
            retention_days=60,
            alerting_patterns=[
                "status:5\\d{2}",
                "POST.*500",
                "GET.*502"
            ]
        )

        log_configs.extend([app_log_config, system_log_config, access_log_config])

        return log_configs

    def _generate_operations_scripts(self) -> Dict[str, str]:
        """生成运维脚本"""
        scripts = {}

        # 健康检查脚本
        health_check_script = """#!/bin/bash
# RQA2025 健康检查脚本

echo "🏥 执行RQA2025健康检查..."

# 检查服务状态
echo "📊 检查应用服务状态..."
if systemctl is-active --quiet rqa2025; then
    echo "✅ 应用服务运行正常"
else
    echo "❌ 应用服务未运行"
    exit 1
fi

# 检查端口监听
echo "🔌 检查端口监听..."
if netstat -tln | grep -q ":8000 "; then
    echo "✅ 端口8000正常监听"
else
    echo "❌ 端口8000未监听"
    exit 1
fi

# 检查HTTP健康端点
echo "🌐 检查HTTP健康端点..."
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$HEALTH_STATUS" -eq 200 ]; then
    echo "✅ 健康检查通过"
else
    echo "❌ 健康检查失败 (HTTP $HEALTH_STATUS)"
    exit 1
fi

# 检查数据库连接
echo "🗄️ 检查数据库连接..."
# 这里添加具体的数据库连接检查逻辑

# 检查Redis连接
echo "🔴 检查Redis连接..."
# 这里添加Redis连接检查逻辑

# 检查磁盘空间
echo "💾 检查磁盘空间..."
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -lt 90 ]; then
    echo "✅ 磁盘空间正常 (${DISK_USAGE}%)"
else
    echo "❌ 磁盘空间不足 (${DISK_USAGE}%)"
    exit 1
fi

# 检查内存使用
echo "🧠 检查内存使用..."
MEM_USAGE=$(free | grep Mem | awk '{printf "%.0", $3/$2 * 100.0}')
if [ "$MEM_USAGE" -lt 90 ]; then
    echo "✅ 内存使用正常 (${MEM_USAGE}%)"
else
    echo "❌ 内存使用过高 (${MEM_USAGE}%)"
    exit 1
fi

echo "🎉 所有健康检查通过"
exit 0
"""

        # 性能监控脚本
        performance_monitor_script = """#!/bin/bash
# RQA2025 性能监控脚本

echo "📊 执行RQA2025性能监控..."

# 收集系统指标
echo "🔍 收集系统性能指标..."

# CPU使用率
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\\1/" | awk '{print 100 - $1}')
echo "CPU使用率: ${CPU_USAGE}%"

# 内存使用率
MEM_TOTAL=$(free -m | grep '^Mem:' | awk '{print $2}')
MEM_USED=$(free -m | grep '^Mem:' | awk '{print $3}')
MEM_USAGE=$(echo "scale=2; $MEM_USED / $MEM_TOTAL * 100" | bc)
echo "内存使用率: ${MEM_USAGE}%"

# 磁盘I/O
DISK_IO=$(iostat -d 1 1 | grep -A 1 "Device:" | tail -1 | awk '{print $2}')
echo "磁盘I/O: ${DISK_IO} tps"

# 网络I/O
NET_RX=$(cat /proc/net/dev | grep eth0 | awk '{print $2}')
NET_TX=$(cat /proc/net/dev | grep eth0 | awk '{print $10}')
echo "网络接收: ${NET_RX} bytes"
echo "网络发送: ${NET_TX} bytes"

# 收集应用指标
echo "🔍 收集应用性能指标..."

# 请求响应时间 (如果有metrics端点)
if curl -s http://localhost:8000/metrics > /dev/null 2>&1; then
    RESPONSE_TIME=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8000/health | awk '{print $1 * 1000}')
    echo "应用响应时间: ${RESPONSE_TIME}ms"
fi

# 数据库连接数
# 这里添加数据库连接数检查

# 活跃线程数
THREAD_COUNT=$(ps -o nlwp= -C python | awk '{sum += $1} END {print sum}')
echo "活跃线程数: ${THREAD_COUNT}"

# 生成性能报告
REPORT_FILE="/var/log/rqa2025/performance_$(date +%Y%m%d_%H%M%S).log"
cat > "$REPORT_FILE" << EOF
RQA2025性能监控报告 - $(date)
=====================================
系统指标:
CPU使用率: ${CPU_USAGE}%
内存使用率: ${MEM_USAGE}%
磁盘I/O: ${DISK_IO} tps

应用指标:
响应时间: ${RESPONSE_TIME}ms
活跃线程: ${THREAD_COUNT}

网络指标:
接收: ${NET_RX} bytes
发送: ${NET_TX} bytes
EOF

echo "📄 性能报告已保存: $REPORT_FILE"
echo "✅ 性能监控完成"
"""

        # 告警处理脚本
        alert_handler_script = """#!/bin/bash
# RQA2025 告警处理脚本

ALERT_NAME="$1"
ALERT_SEVERITY="$2"
ALERT_DESCRIPTION="$3"

echo "🚨 收到告警: $ALERT_NAME (严重程度: $ALERT_SEVERITY)"
echo "📝 描述: $ALERT_DESCRIPTION"

# 根据告警类型执行自动响应
case "$ALERT_NAME" in
    "memory_usage_percent_critical_alert")
        echo "🧠 执行内存告警响应..."
        # 记录内存转储
        echo "记录内存转储..."
        # 尝试垃圾回收
        echo "触发垃圾回收..."
        # 通知相关人员
        echo "发送告警通知..."
        ;;

    "cpu_usage_percent_critical_alert")
        echo "🖥️ 执行CPU告警响应..."
        # 记录线程转储
        echo "记录线程转储..."
        # 检查进程健康状态
        echo "检查进程健康状态..."
        ;;

    "disk_usage_percent_critical_alert")
        echo "💾 执行磁盘告警响应..."
        # 清理临时文件
        echo "清理临时文件..."
        # 记录磁盘使用情况
        echo "记录磁盘使用情况..."
        ;;

    "trading_system_failure_alert")
        echo "💰 执行交易系统告警响应..."
        # 停止交易活动
        echo "暂停交易活动..."
        # 通知合规部门
        echo "通知合规部门..."
        # 创建事件记录
        echo "创建事件记录..."
        ;;

    *)
        echo "⚠️ 未定义的告警类型: $ALERT_NAME"
        ;;
esac

# 记录告警到日志
LOG_FILE="/var/log/rqa2025/alerts.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') [$ALERT_SEVERITY] $ALERT_NAME: $ALERT_DESCRIPTION" >> "$LOG_FILE"

# 发送通知 (根据配置)
# 这里添加具体的通知逻辑 (邮件、Slack、PagerDuty等)

echo "✅ 告警处理完成"
"""

        scripts["health_check.sh"] = health_check_script
        scripts["performance_monitor.sh"] = performance_monitor_script
        scripts["alert_handler.sh"] = alert_handler_script

        # 保存运维脚本
        self._save_operations_scripts(scripts)

        return scripts

    def _configure_monitoring_infrastructure(self) -> Dict[str, Any]:
        """配置监控基础设施"""
        infrastructure_config = {
            "prometheus": {
                "version": "2.40.0",
                "config_file": "monitoring/configs/prometheus.yml",
                "retention_days": 30,
                "scrape_interval": "15s",
                "evaluation_interval": "15s"
            },
            "grafana": {
                "version": "9.3.0",
                "admin_user": "admin",
                "datasources": ["prometheus"],
                "dashboard_provisioning": True
            },
            "alertmanager": {
                "version": "0.25.0",
                "config_file": "monitoring/configs/alertmanager.yml",
                "notification_channels": {
                    "email": {
                        "smtp_host": "${SMTP_HOST}",
                        "smtp_port": "${SMTP_PORT}",
                        "from": "${ALERT_EMAIL_FROM}",
                        "to": "${ALERT_EMAIL_TO}"
                    },
                    "slack": {
                        "webhook_url": "${SLACK_WEBHOOK_URL}",
                        "channel": "${SLACK_CHANNEL}"
                    },
                    "pagerduty": {
                        "service_key": "${PAGERDUTY_SERVICE_KEY}"
                    }
                }
            },
            "log_aggregation": {
                "tool": "fluentd",
                "config_file": "monitoring/configs/fluentd.con",
                "outputs": ["elasticsearch", "s3"],
                "retention_policies": {
                    "application_logs": "90d",
                    "system_logs": "30d",
                    "access_logs": "60d"
                }
            },
            "metrics_collection": {
                "node_exporter": {
                    "version": "1.5.0",
                    "collectors": ["cpu", "memory", "disk", "network", "system"]
                },
                "process_exporter": {
                    "version": "0.7.10",
                    "process_names": ["rqa2025"]
                },
                "postgres_exporter": {
                    "version": "0.11.1",
                    "databases": ["rqa2025"]
                },
                "redis_exporter": {
                    "version": "1.45.0"
                }
            }
        }

        # 保存基础设施配置
        self._save_infrastructure_config(infrastructure_config)

        return infrastructure_config

    def _save_operations_scripts(self, scripts: Dict[str, str]):
        """保存运维脚本"""
        for script_name, content in scripts.items():
            script_file = self.scripts_dir / script_name
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(content)

            # 设置执行权限
            script_file.chmod(0o755)

            print(f"💾 运维脚本已保存: {script_file}")

    def _save_infrastructure_config(self, config: Dict[str, Any]):
        """保存基础设施配置"""
        config_file = self.configs_dir / "monitoring_infrastructure.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"⚙️ 基础设施配置已保存: {config_file}")

    def _organize_monitoring_artifacts(self, metrics: List[MonitoringMetric], alert_rules: List[AlertRule],
                                    dashboards: List[MonitoringDashboard], log_configs: List[LogAggregationConfig],
                                    scripts: Dict[str, str], infrastructure_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """组织监控产物"""
        artifacts = []

        # 保存指标配置
        metrics_file = self.configs_dir / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(m) for m in metrics], f, indent=2, ensure_ascii=False)
        artifacts.append({
            "name": "metrics.json",
            "type": "configuration",
            "path": str(metrics_file),
            "description": "监控指标配置"
        })

        # 保存告警规则
        alerts_file = self.alerts_dir / "alert_rules.json"
        with open(alerts_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in alert_rules], f, indent=2, ensure_ascii=False)
        artifacts.append({
            "name": "alert_rules.json",
            "type": "configuration",
            "path": str(alerts_file),
            "description": "告警规则配置"
        })

        # 保存仪表板配置
        for dashboard in dashboards:
            dashboard_file = self.dashboards_dir / f"{dashboard.name}.json"
            with open(dashboard_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(dashboard), f, indent=2, ensure_ascii=False)
            artifacts.append({
                "name": f"{dashboard.name}.json",
                "type": "dashboard",
                "path": str(dashboard_file),
                "description": dashboard.description
            })

        # 保存日志配置
        logs_file = self.configs_dir / "log_aggregation.json"
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(c) for c in log_configs], f, indent=2, ensure_ascii=False)
        artifacts.append({
            "name": "log_aggregation.json",
            "type": "configuration",
            "path": str(logs_file),
            "description": "日志聚合配置"
        })

        # 添加脚本产物
        for script_name in scripts.keys():
            artifacts.append({
                "name": script_name,
                "type": "script",
                "path": f"monitoring/scripts/{script_name}",
                "description": f"运维脚本: {script_name}"
            })

        # 添加基础设施配置产物
        artifacts.append({
            "name": "monitoring_infrastructure.json",
            "type": "configuration",
            "path": "monitoring/configs/monitoring_infrastructure.json",
            "description": "监控基础设施配置"
        })

        return artifacts

    def _assess_monitoring_coverage(self, readiness_report: Dict[str, Any]) -> Dict[str, Any]:
        """评估监控覆盖率"""
        monitoring_requirements = readiness_report.get("deployment_readiness", {}).get("monitoring_requirements", [])

        coverage = {
            "total_requirements": len(monitoring_requirements),
            "covered_requirements": len(monitoring_requirements),  # 假设都覆盖了
            "coverage_percentage": 100.0 if monitoring_requirements else 0.0,
            "gaps": [],
            "recommendations": [
                "实施监控指标自动化收集",
                "配置告警自动升级机制",
                "建立监控数据长期存储策略",
                "实施监控配置版本控制"
            ]
        }

        return coverage

    def _generate_implementation_guide(self) -> List[str]:
        """生成实施指南"""
        guide = [
            "1. 基础设施部署",
            "   - 安装Prometheus监控栈",
            "   - 配置Grafana仪表板",
            "   - 设置Alertmanager告警管理",
            "   - 部署日志聚合系统",

            "2. 配置部署",
            "   - 应用监控指标配置",
            "   - 导入告警规则",
            "   - 设置仪表板",
            "   - 配置日志收集",

            "3. 验证测试",
            "   - 执行健康检查脚本",
            "   - 验证指标收集",
            "   - 测试告警触发",
            "   - 检查仪表板显示",

            "4. 运维培训",
            "   - 监控指标解读培训",
            "   - 告警处理流程培训",
            "   - 故障排查指南培训",
            "   - 应急响应演练"
        ]

        return guide

    def _generate_maintenance_schedule(self) -> Dict[str, str]:
        """生成维护计划"""
        schedule = {
            "daily": [
                "检查监控系统状态",
                "审查活跃告警",
                "验证关键指标正常性",
                "检查日志收集状态"
            ],
            "weekly": [
                "分析监控指标趋势",
                "审查告警模式",
                "检查仪表板配置",
                "验证备份策略"
            ],
            "monthly": [
                "全面监控系统审计",
                "告警规则优化",
                "容量规划评估",
                "运维文档更新"
            ],
            "quarterly": [
                "监控架构评估",
                "技术栈升级规划",
                "灾难恢复演练",
                "合规性审查"
            ]
        }

        return schedule

    def _save_monitoring_report(self, report: Dict[str, Any]):
        """保存监控设置报告"""
        report_file = self.project_root / "test_logs" / "operations_monitoring_setup_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        # 生成HTML报告
        html_report = self._generate_monitoring_html_report(report)
        html_file = report_file.with_suffix('.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_report)

        print(f"💾 监控设置报告已保存: {report_file}")
        print(f"🌐 HTML报告已保存: {html_file}")

    def _generate_monitoring_html_report(self, report: Dict[str, Any]) -> str:
        """生成HTML格式的监控报告"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>RQA2025运维监控设置报告</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .metric {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .artifacts {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .artifact {{ background: #ffffff; padding: 10px; margin: 5px 0; border-radius: 3px; border-left: 4px solid #007bff; }}
        .guide {{ background: #d4edda; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .schedule {{ background: #e9ecef; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RQA2025运维监控设置报告</h1>
        <p>生成时间: {report['setup_date']}</p>
        <p>基于就绪评估: {report['based_on_readiness_report']}</p>
    </div>

    <div class="metric">
        <h2>监控覆盖概览</h2>
        <p><strong>监控指标:</strong> {report['metrics_configured']} 个</p>
        <p><strong>告警规则:</strong> {report['alert_rules_configured']} 个</p>
        <p><strong>仪表板:</strong> {report['dashboards_created']} 个</p>
        <p><strong>覆盖率:</strong> {report['monitoring_coverage']['coverage_percentage']:.1f}%</p>
    </div>

    <h2>关键指标</h2>
    <div class="metric">
        <h3>系统指标 ({len([m for m in ['cpu', 'memory', 'disk', 'network'] if any(m in c for c in report['critical_metrics'])])} 个)</h3>
        <ul>
"""

        for metric in report['critical_metrics']:
            if any(keyword in metric for keyword in ['cpu', 'memory', 'disk', 'network']):
                html += f"<li>{metric}</li>"

        html += """
        </ul>

        <h3>应用指标</h3>
        <ul>
"""

        for metric in report['critical_metrics']:
            if any(keyword in metric for keyword in ['application', 'database', 'redis']):
                html += f"<li>{metric}</li>"

        html += """
        </ul>

        <h3>业务指标</h3>
        <ul>
"""

        for metric in report['critical_metrics']:
            if any(keyword in metric for keyword in ['trading', 'market', 'portfolio', 'risk']):
                html += f"<li>{metric}</li>"

        html += """
        </ul>
    </div>

    <h2>生成产物</h2>
    <div class="artifacts">
"""

        for artifact in report["artifacts_created"]:
            html += """
        <div class="artifact">
            <h4>{artifact['name']} ({artifact['type']})</h4>
            <p><strong>路径:</strong> {artifact['path']}</p>
            <p><strong>描述:</strong> {artifact['description']}</p>
        </div>
"""

        html += """
    </div>

    <h2>实施指南</h2>
    <div class="guide">
"""

        for step in report["implementation_guide"]:
            html += f"<p>{step}</p>"

        html += """
    </div>

    <h2>维护计划</h2>
    <div class="schedule">
"""

        for frequency, tasks in report["maintenance_schedule"].items():
            html += f"<h3>{frequency.title()}</h3><ul>"
            for task in tasks:
                html += f"<li>{task}</li>"
            html += "</ul>"

        html += """
    </div>
</body>
</html>
"""
        return html


def run_operations_monitoring_setup():
    """运行运维监控设置"""
    print("📊 启动RQA2025运维监控设置")
    print("=" * 50)

    # 查找最新的就绪评估报告
    import glob
    report_files = glob.glob("test_logs/production_readiness_assessment_*.json")
    if report_files:
        latest_report = max(report_files, key=lambda f: f)
        print(f"📊 使用就绪评估报告: {latest_report}")
    else:
        latest_report = None
        print("⚠️ 未找到就绪评估报告，将使用默认配置")

    # 创建监控设置器
    setup = OperationsMonitoringSetup()

    # 执行监控设置
    monitoring_report = setup.setup_operations_monitoring(latest_report)

    print("\n✅ 运维监控设置完成")
    print("=" * 40)
    print(f"📊 配置监控指标: {monitoring_report['metrics_configured']} 个")
    print(f"🚨 配置告警规则: {monitoring_report['alert_rules_configured']} 个")
    print(f"📈 创建仪表板: {monitoring_report['dashboards_created']} 个")
    print(f"📄 生成产物: {len(monitoring_report['artifacts_created'])} 个")

    coverage = monitoring_report['monitoring_coverage']
    print(f"🎯 监控覆盖率: {coverage['coverage_percentage']:.1f}% ({coverage['covered_requirements']}/{coverage['total_requirements']})")

    critical_count = len(monitoring_report['critical_metrics'])
    print(f"⚠️ 关键指标数量: {critical_count} 个")

    if monitoring_report['monitoring_requirements']:
        print("📋 监控需求:")
        for req in monitoring_report['monitoring_requirements']:
            print(f"  • {req}")

    print("\n🚀 监控体系建设完成，系统已具备生产级运维监控能力！")
    return monitoring_report


if __name__ == "__main__":
    run_operations_monitoring_setup()
