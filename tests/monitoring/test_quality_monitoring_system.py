"""
质量监控体系
提供持续质量监控、预警机制和质量仪表板
支持实时监控、趋势分析和自动预警
"""

import pytest
import time
import json
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue


class AlertSeverity(Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """指标类型"""
    COVERAGE = "coverage"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    SECURITY = "security"
    RELIABILITY = "reliability"


@dataclass
class QualityMetric:
    """质量指标"""
    metric_id: str
    name: str
    type: MetricType
    value: float
    unit: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityAlert:
    """质量告警"""
    alert_id: str
    title: str
    description: str
    severity: AlertSeverity
    metric_id: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None


@dataclass
class QualityDashboard:
    """质量仪表板"""
    dashboard_id: str
    name: str
    description: str
    metrics: List[str]  # 指标ID列表
    alerts: List[str]   # 告警ID列表
    charts: List[Dict[str, Any]]
    refresh_interval: int  # 刷新间隔(秒)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class MonitoringConfig:
    """监控配置"""
    enabled: bool = True
    collection_interval: int = 300  # 5分钟
    retention_days: int = 90
    alert_enabled: bool = True
    dashboard_enabled: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])


class QualityMonitoringSystem:
    """质量监控体系"""

    def __init__(self):
        self.metrics = {}  # metric_id -> list of QualityMetric
        self.alerts = {}   # alert_id -> QualityAlert
        self.dashboards = {}
        self.config = MonitoringConfig()

        # 监控线程
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        self.metric_queue = queue.Queue()

        # 告警配置
        self.alert_rules = self._initialize_alert_rules()

    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """初始化告警规则"""
        return {
            'test_coverage': {
                'metric_type': MetricType.COVERAGE,
                'warning_threshold': 80.0,
                'critical_threshold': 70.0,
                'description': '单元测试覆盖率告警'
            },
            'performance_response_time': {
                'metric_type': MetricType.PERFORMANCE,
                'warning_threshold': 500.0,  # ms
                'critical_threshold': 2000.0,  # ms
                'description': 'API响应时间告警'
            },
            'code_quality_maintainability': {
                'metric_type': MetricType.QUALITY,
                'warning_threshold': 70.0,
                'critical_threshold': 50.0,
                'description': '代码可维护性告警'
            },
            'security_vulnerabilities': {
                'metric_type': MetricType.SECURITY,
                'warning_threshold': 5,
                'critical_threshold': 10,
                'description': '安全漏洞数量告警'
            },
            'reliability_error_rate': {
                'metric_type': MetricType.RELIABILITY,
                'warning_threshold': 1.0,  # %
                'critical_threshold': 5.0,  # %
                'description': '系统错误率告警'
            }
        }

    def start_monitoring(self):
        """启动监控"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return

        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        print("✅ 质量监控系统已启动")

    def stop_monitoring(self):
        """停止监控"""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("✅ 质量监控系统已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while not self.stop_monitoring.is_set():
            try:
                # 收集指标
                self._collect_metrics()

                # 检查告警
                self._check_alerts()

                # 更新仪表板
                self._update_dashboards()

                # 清理过期数据
                self._cleanup_old_data()

            except Exception as e:
                print(f"❌ 监控循环异常: {e}")

            # 等待下一个收集周期
            self.stop_monitoring.wait(self.config.collection_interval)

    def collect_metric(self, metric: QualityMetric):
        """收集指标"""
        if metric.metric_id not in self.metrics:
            self.metrics[metric.metric_id] = []

        self.metrics[metric.metric_id].append(metric)
        self.metric_queue.put(metric)

        # 检查是否需要触发告警
        self._check_metric_alerts(metric)

        print(f"📊 收集指标: {metric.name} = {metric.value}{metric.unit}")

    def _collect_metrics(self):
        """收集所有指标"""
        # 这里应该实现具体的指标收集逻辑
        # 为了演示，我们收集一些模拟指标

        current_time = datetime.now()

        # 测试覆盖率指标
        coverage_metric = QualityMetric(
            metric_id='test_coverage',
            name='单元测试覆盖率',
            type=MetricType.COVERAGE,
            value=85.5,
            unit='%',
            timestamp=current_time,
            threshold_warning=80.0,
            threshold_critical=70.0
        )
        self.collect_metric(coverage_metric)

        # 性能指标
        response_time_metric = QualityMetric(
            metric_id='api_response_time',
            name='API平均响应时间',
            type=MetricType.PERFORMANCE,
            value=450.0,
            unit='ms',
            timestamp=current_time,
            threshold_warning=500.0,
            threshold_critical=2000.0
        )
        self.collect_metric(response_time_metric)

        # 代码质量指标
        maintainability_metric = QualityMetric(
            metric_id='code_maintainability',
            name='代码可维护性指数',
            type=MetricType.QUALITY,
            value=75.0,
            unit='',
            timestamp=current_time,
            threshold_warning=70.0,
            threshold_critical=50.0
        )
        self.collect_metric(maintainability_metric)

    def _check_metric_alerts(self, metric: QualityMetric):
        """检查指标告警"""
        if not metric.threshold_warning and not metric.threshold_critical:
            return

        alert_triggered = False
        severity = AlertSeverity.INFO

        if metric.threshold_critical and metric.value <= metric.threshold_critical:
            severity = AlertSeverity.CRITICAL
            alert_triggered = True
        elif metric.threshold_warning and metric.value <= metric.threshold_warning:
            severity = AlertSeverity.WARNING
            alert_triggered = True

        if alert_triggered:
            alert = QualityAlert(
                alert_id=f"alert_{int(time.time())}_{metric.metric_id}",
                title=f"{metric.name}超出阈值",
                description=f"{metric.name}当前值为{metric.value}{metric.unit}，{'低于' if metric.type == MetricType.COVERAGE else '高于'}阈值",
                severity=severity,
                metric_id=metric.metric_id,
                current_value=metric.value,
                threshold_value=metric.threshold_critical or metric.threshold_warning,
                timestamp=datetime.now()
            )

            self.alerts[alert.alert_id] = alert
            self._send_alert_notification(alert)

            print(f"🚨 触发告警: {alert.title} (严重程度: {alert.severity.value})")

    def _check_alerts(self):
        """检查所有告警规则"""
        for rule_name, rule_config in self.alert_rules.items():
            self._evaluate_alert_rule(rule_name, rule_config)

    def _evaluate_alert_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """评估告警规则"""
        metric_type = rule_config['metric_type']
        metric_id = f"{metric_type.value}_{rule_name.split('_')[-1]}"

        # 获取最新指标值
        if metric_id in self.metrics and self.metrics[metric_id]:
            latest_metric = max(self.metrics[metric_id], key=lambda m: m.timestamp)

            # 检查是否已经存在活跃告警
            active_alerts = [
                alert for alert in self.alerts.values()
                if alert.metric_id == metric_id and not alert.resolved
            ]

            if active_alerts:
                # 检查是否需要升级或保持告警
                self._update_existing_alerts(active_alerts, latest_metric, rule_config)
            else:
                # 检查是否需要创建新告警
                self._create_new_alert_if_needed(latest_metric, rule_config)

    def _update_existing_alerts(self, active_alerts: List[QualityAlert],
                               latest_metric: QualityMetric, rule_config: Dict[str, Any]):
        """更新现有告警"""
        for alert in active_alerts:
            # 如果指标回到正常范围，标记为已解决
            if latest_metric.value > (rule_config.get('warning_threshold', 0)):
                alert.resolved = True
                alert.resolution_notes = f"指标已恢复正常: {latest_metric.value}"
                print(f"✅ 告警已解决: {alert.title}")

    def _create_new_alert_if_needed(self, metric: QualityMetric, rule_config: Dict[str, Any]):
        """根据需要创建新告警"""
        # 这里可以实现更复杂的告警逻辑
        pass

    def _send_alert_notification(self, alert: QualityAlert):
        """发送告警通知"""
        if not self.config.alert_enabled:
            return

        message = f"""
🚨 质量告警

标题: {alert.title}
描述: {alert.description}
严重程度: {alert.severity.value.upper()}
指标: {alert.metric_id}
当前值: {alert.current_value}
阈值: {alert.threshold_value}
时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

请及时处理！
"""

        for channel in self.config.notification_channels:
            self._send_to_channel(channel, message)

    def _send_to_channel(self, channel: str, message: str):
        """发送到指定渠道"""
        # 这里应该实现具体的通知逻辑
        print(f"📤 发送告警到 {channel}: {message[:100]}...")

    def create_dashboard(self, name: str, description: str, metric_ids: List[str]) -> str:
        """创建仪表板"""
        dashboard_id = f"dashboard_{int(time.time())}"

        charts = []
        for metric_id in metric_ids:
            if metric_id in self.metrics:
                chart = {
                    'metric_id': metric_id,
                    'type': 'line_chart',
                    'title': f"{metric_id}趋势图",
                    'data_points': len(self.metrics[metric_id])
                }
                charts.append(chart)

        dashboard = QualityDashboard(
            dashboard_id=dashboard_id,
            name=name,
            description=description,
            metrics=metric_ids,
            alerts=[],
            charts=charts,
            refresh_interval=300
        )

        self.dashboards[dashboard_id] = dashboard
        print(f"📊 创建仪表板: {name} ({dashboard_id})")

        return dashboard_id

    def get_dashboard_data(self, dashboard_id: str) -> Dict[str, Any]:
        """获取仪表板数据"""
        if dashboard_id not in self.dashboards:
            raise ValueError(f"仪表板不存在: {dashboard_id}")

        dashboard = self.dashboards[dashboard_id]

        dashboard_data = {
            'dashboard_id': dashboard_id,
            'name': dashboard.name,
            'description': dashboard.description,
            'last_updated': dashboard.last_updated.isoformat(),
            'metrics': {},
            'alerts': [],
            'charts': dashboard.charts
        }

        # 获取指标数据
        for metric_id in dashboard.metrics:
            if metric_id in self.metrics:
                metrics_data = self.metrics[metric_id][-20:]  # 最近20个数据点
                dashboard_data['metrics'][metric_id] = [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'value': m.value,
                        'unit': m.unit
                    } for m in metrics_data
                ]

        # 获取活跃告警
        active_alerts = [
            alert for alert in self.alerts.values()
            if not alert.resolved
        ]
        dashboard_data['alerts'] = [
            {
                'alert_id': alert.alert_id,
                'title': alert.title,
                'severity': alert.severity.value,
                'timestamp': alert.timestamp.isoformat()
            } for alert in active_alerts
        ]

        return dashboard_data

    def _update_dashboards(self):
        """更新所有仪表板"""
        for dashboard in self.dashboards.values():
            dashboard.last_updated = datetime.now()

    def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        # 清理过期指标
        for metric_id in list(self.metrics.keys()):
            self.metrics[metric_id] = [
                m for m in self.metrics[metric_id]
                if m.timestamp > cutoff_date
            ]
            if not self.metrics[metric_id]:
                del self.metrics[metric_id]

        # 清理过期告警（保留已解决的告警30天）
        resolved_cutoff = datetime.now() - timedelta(days=30)
        for alert_id in list(self.alerts.keys()):
            alert = self.alerts[alert_id]
            if alert.resolved and alert.timestamp < resolved_cutoff:
                del self.alerts[alert_id]

    def get_monitoring_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        report = {
            'summary': {
                'total_metrics': len(self.metrics),
                'total_alerts': len(self.alerts),
                'active_alerts': len([a for a in self.alerts.values() if not a.resolved]),
                'total_dashboards': len(self.dashboards),
                'monitoring_status': 'active' if self.monitoring_thread and self.monitoring_thread.is_alive() else 'inactive'
            },
            'metrics_summary': self._get_metrics_summary(),
            'alerts_summary': self._get_alerts_summary(),
            'trends': self._calculate_trends(),
            'recommendations': self._generate_monitoring_recommendations()
        }

        return report

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """获取指标汇总"""
        summary = {
            'by_type': {},
            'total_data_points': 0,
            'latest_values': {}
        }

        for metric_id, metrics_list in self.metrics.items():
            if not metrics_list:
                continue

            latest_metric = max(metrics_list, key=lambda m: m.timestamp)
            metric_type = latest_metric.type.value

            if metric_type not in summary['by_type']:
                summary['by_type'][metric_type] = 0
            summary['by_type'][metric_type] += 1

            summary['total_data_points'] += len(metrics_list)
            summary['latest_values'][metric_id] = {
                'value': latest_metric.value,
                'unit': latest_metric.unit,
                'timestamp': latest_metric.timestamp.isoformat()
            }

        return summary

    def _get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警汇总"""
        summary = {
            'by_severity': {},
            'total_resolved': 0,
            'total_active': 0,
            'resolution_rate': 0.0
        }

        total_alerts = len(self.alerts)
        if total_alerts == 0:
            return summary

        resolved_alerts = [a for a in self.alerts.values() if a.resolved]

        summary['total_resolved'] = len(resolved_alerts)
        summary['total_active'] = total_alerts - len(resolved_alerts)
        summary['resolution_rate'] = (len(resolved_alerts) / total_alerts) * 100

        # 按严重程度统计
        for alert in self.alerts.values():
            severity = alert.severity.value
            if severity not in summary['by_severity']:
                summary['by_severity'][severity] = 0
            summary['by_severity'][severity] += 1

        return summary

    def _calculate_trends(self) -> Dict[str, Any]:
        """计算趋势"""
        trends = {}

        for metric_id, metrics_list in self.metrics.items():
            if len(metrics_list) < 2:
                continue

            # 计算最近7天的数据
            week_ago = datetime.now() - timedelta(days=7)
            recent_metrics = [m for m in metrics_list if m.timestamp > week_ago]

            if len(recent_metrics) >= 2:
                values = [m.value for m in recent_metrics]
                trend_slope = self._calculate_trend_slope(recent_metrics)

                trend_direction = 'stable'
                if trend_slope > 0.1:
                    trend_direction = 'improving' if metric_id.endswith('coverage') or metric_id.endswith('maintainability') else 'worsening'
                elif trend_slope < -0.1:
                    trend_direction = 'worsening' if metric_id.endswith('coverage') or metric_id.endswith('maintainability') else 'improving'

                trends[metric_id] = {
                    'direction': trend_direction,
                    'slope': trend_slope,
                    'data_points': len(recent_metrics),
                    'avg_value': statistics.mean(values),
                    'min_value': min(values),
                    'max_value': max(values)
                }

        return trends

    def _calculate_trend_slope(self, metrics: List[QualityMetric]) -> float:
        """计算趋势斜率"""
        if len(metrics) < 2:
            return 0.0

        # 简化的趋势计算
        sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)
        n = len(sorted_metrics)

        # 计算时间差（天数）
        time_diffs = [(m.timestamp - sorted_metrics[0].timestamp).total_seconds() / 86400
                     for m in sorted_metrics]

        # 计算斜率
        sum_x = sum(time_diffs)
        sum_y = sum(m.value for m in sorted_metrics)
        sum_xy = sum(x * m.value for x, m in zip(time_diffs, sorted_metrics))
        sum_x2 = sum(x * x for x in time_diffs)

        if n * sum_x2 - sum_x * sum_x == 0:
            return 0.0

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return slope

    def _generate_monitoring_recommendations(self) -> List[str]:
        """生成监控建议"""
        recommendations = []

        # 基于指标分析生成建议
        metrics_summary = self._get_metrics_summary()
        alerts_summary = self._get_alerts_summary()
        trends = self._calculate_trends()

        # 告警相关建议
        if alerts_summary['total_active'] > 5:
            recommendations.append("⚠️ 活跃告警数量过多，建议优先处理严重告警")
        elif alerts_summary['resolution_rate'] < 80:
            recommendations.append("📈 告警解决率偏低，建议改进问题处理流程")

        # 指标覆盖建议
        if metrics_summary['total_metrics'] < 5:
            recommendations.append("📊 监控指标覆盖不足，建议增加更多质量指标")

        # 趋势分析建议
        worsening_trends = [k for k, v in trends.items() if v['direction'] == 'worsening']
        if worsening_trends:
            recommendations.append(f"📉 {len(worsening_trends)}个指标呈下降趋势，需要关注")

        # 通用建议
        recommendations.extend([
            "🔄 定期审查告警阈值，确保其合理性",
            "📈 扩展监控覆盖范围，增加更多关键指标",
            "🤖 考虑引入AI辅助的异常检测",
            "📋 建立定期质量审查机制"
        ])

        return recommendations

    def acknowledge_alert(self, alert_id: str, user: str) -> bool:
        """确认告警"""
        if alert_id not in self.alerts:
            return False

        self.alerts[alert_id].acknowledged = True
        print(f"✅ 告警已确认: {alert_id} by {user}")
        return True

    def resolve_alert(self, alert_id: str, resolution_notes: str, user: str) -> bool:
        """解决告警"""
        if alert_id not in self.alerts:
            return False

        alert = self.alerts[alert_id]
        alert.resolved = True
        alert.resolution_notes = resolution_notes
        print(f"✅ 告警已解决: {alert_id} by {user}")
        return True


class TestQualityMonitoringSystem:
    """质量监控体系测试"""

    def setup_method(self):
        """测试前准备"""
        self.monitoring_system = QualityMonitoringSystem()

    def test_metric_collection_and_alerting(self):
        """测试指标收集和告警"""
        # 创建测试指标
        metric = QualityMetric(
            metric_id='test_coverage',
            name='单元测试覆盖率',
            type=MetricType.COVERAGE,
            value=75.0,  # 低于警告阈值80
            unit='%',
            timestamp=datetime.now(),
            threshold_warning=80.0,
            threshold_critical=70.0
        )

        # 收集指标
        self.monitoring_system.collect_metric(metric)

        # 验证指标已收集
        assert 'test_coverage' in self.monitoring_system.metrics
        assert len(self.monitoring_system.metrics['test_coverage']) == 1
        collected_metric = self.monitoring_system.metrics['test_coverage'][0]
        assert collected_metric.value == 75.0
        assert collected_metric.unit == '%'

        # 验证告警已触发（因为值低于警告阈值）
        assert len(self.monitoring_system.alerts) > 0
        alert_found = False
        for alert in self.monitoring_system.alerts.values():
            if alert.metric_id == 'test_coverage':
                alert_found = True
                assert alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
                assert alert.current_value == 75.0
                break
        assert alert_found, "应该为低覆盖率触发告警"

        print(f"✅ 指标收集和告警测试通过 - 收集指标: {len(self.monitoring_system.metrics)}, 触发告警: {len(self.monitoring_system.alerts)}")

    def test_dashboard_creation_and_data_retrieval(self):
        """测试仪表板创建和数据检索"""
        # 创建一些测试指标
        metrics = [
            QualityMetric('coverage', '覆盖率', MetricType.COVERAGE, 85.0, '%', datetime.now()),
            QualityMetric('performance', '性能', MetricType.PERFORMANCE, 450.0, 'ms', datetime.now()),
            QualityMetric('quality', '质量', MetricType.QUALITY, 75.0, '', datetime.now())
        ]

        for metric in metrics:
            self.monitoring_system.collect_metric(metric)

        # 创建仪表板
        dashboard_id = self.monitoring_system.create_dashboard(
            name='质量概览',
            description='系统质量指标总览',
            metric_ids=['coverage', 'performance', 'quality']
        )

        # 验证仪表板已创建
        assert dashboard_id in self.monitoring_system.dashboards
        dashboard = self.monitoring_system.dashboards[dashboard_id]
        assert dashboard.name == '质量概览'
        assert len(dashboard.metrics) == 3
        assert len(dashboard.charts) == 3

        # 获取仪表板数据
        dashboard_data = self.monitoring_system.get_dashboard_data(dashboard_id)

        # 验证数据结构
        assert dashboard_data['dashboard_id'] == dashboard_id
        assert dashboard_data['name'] == '质量概览'
        assert 'metrics' in dashboard_data
        assert 'alerts' in dashboard_data
        assert 'charts' in dashboard_data

        # 验证指标数据
        assert len(dashboard_data['metrics']) == 3
        for metric_id in ['coverage', 'performance', 'quality']:
            assert metric_id in dashboard_data['metrics']
            assert len(dashboard_data['metrics'][metric_id]) > 0

        print(f"✅ 仪表板创建和数据检索测试通过 - 仪表板ID: {dashboard_id}, 指标数: {len(dashboard_data['metrics'])}")

    def test_monitoring_report_generation(self):
        """测试监控报告生成"""
        # 添加一些测试数据
        metrics = [
            QualityMetric('test_coverage', '覆盖率', MetricType.COVERAGE, 82.0, '%', datetime.now()),
            QualityMetric('api_response_time', '响应时间', MetricType.PERFORMANCE, 420.0, 'ms', datetime.now()),
            QualityMetric('code_maintainability', '可维护性', MetricType.QUALITY, 78.0, '', datetime.now())
        ]

        for metric in metrics:
            self.monitoring_system.collect_metric(metric)

        # 创建一些告警
        alert = QualityAlert(
            alert_id='alert_1',
            title='覆盖率告警',
            description='测试覆盖率低于阈值',
            severity=AlertSeverity.WARNING,
            metric_id='test_coverage',
            current_value=82.0,
            threshold_value=85.0,
            timestamp=datetime.now()
        )
        self.monitoring_system.alerts[alert.alert_id] = alert

        # 生成监控报告
        report = self.monitoring_system.get_monitoring_report()

        # 验证报告结构
        assert 'summary' in report
        assert 'metrics_summary' in report
        assert 'alerts_summary' in report
        assert 'trends' in report
        assert 'recommendations' in report

        # 验证摘要
        summary = report['summary']
        assert summary['total_metrics'] == 3
        assert summary['total_alerts'] == 1
        assert summary['active_alerts'] == 1

        # 验证指标汇总
        metrics_summary = report['metrics_summary']
        assert 'by_type' in metrics_summary
        assert 'total_data_points' in metrics_summary
        assert 'latest_values' in metrics_summary

        # 验证告警汇总
        alerts_summary = report['alerts_summary']
        assert 'by_severity' in alerts_summary
        assert 'total_active' in alerts_summary

        print(f"✅ 监控报告生成测试通过 - 指标数: {summary['total_metrics']}, 告警数: {summary['total_alerts']}, 建议数: {len(report['recommendations'])}")

    def test_alert_acknowledgement_and_resolution(self):
        """测试告警确认和解决"""
        # 创建测试告警
        alert = QualityAlert(
            alert_id='test_alert_1',
            title='测试告警',
            description='这是一个测试告警',
            severity=AlertSeverity.WARNING,
            metric_id='test_metric',
            current_value=50.0,
            threshold_value=60.0,
            timestamp=datetime.now()
        )
        self.monitoring_system.alerts[alert.alert_id] = alert

        # 确认告警
        result = self.monitoring_system.acknowledge_alert(alert.alert_id, 'test_user')
        assert result is True
        assert self.monitoring_system.alerts[alert.alert_id].acknowledged is True

        # 解决告警
        result = self.monitoring_system.resolve_alert(alert.alert_id, '问题已修复', 'test_user')
        assert result is True
        assert self.monitoring_system.alerts[alert.alert_id].resolved is True
        assert self.monitoring_system.alerts[alert.alert_id].resolution_notes == '问题已修复'

        print("✅ 告警确认和解决测试通过")

    def test_trend_analysis(self):
        """测试趋势分析"""
        # 创建时间序列数据
        base_time = datetime.now()
        trend_metrics = []

        # 创建7天的趋势数据（逐渐改善）
        for i in range(7):
            metric_time = base_time - timedelta(days=6-i)
            # 覆盖率从70%逐渐提升到85%
            coverage_value = 70 + (i * 2.14)

            metric = QualityMetric(
                metric_id='coverage_trend',
                name='覆盖率趋势',
                type=MetricType.COVERAGE,
                value=round(coverage_value, 1),
                unit='%',
                timestamp=metric_time
            )
            trend_metrics.append(metric)

        # 添加指标到系统
        for metric in trend_metrics:
            self.monitoring_system.collect_metric(metric)

        # 计算趋势
        trends = self.monitoring_system._calculate_trends()

        # 验证趋势分析
        assert 'coverage_trend' in trends
        trend_data = trends['coverage_trend']
        assert 'direction' in trend_data
        assert 'slope' in trend_data
        assert trend_data['direction'] == 'improving'  # 因为覆盖率在提升
        assert trend_data['slope'] > 0  # 斜率应该是正的
        assert trend_data['data_points'] == 7

        print(f"✅ 趋势分析测试通过 - 趋势方向: {trend_data['direction']}, 斜率: {trend_data['slope']:.3f}")

    def test_monitoring_configuration(self):
        """测试监控配置"""
        # 验证默认配置
        config = self.monitoring_system.config
        assert config.enabled is True
        assert config.collection_interval == 300  # 5分钟
        assert config.retention_days == 90
        assert config.alert_enabled is True
        assert config.dashboard_enabled is True
        assert 'email' in config.notification_channels
        assert 'slack' in config.notification_channels

        # 修改配置
        config.collection_interval = 600  # 10分钟
        config.notification_channels.append('webhook')

        assert config.collection_interval == 600
        assert 'webhook' in config.notification_channels

        print("✅ 监控配置测试通过")

    def test_data_cleanup(self):
        """测试数据清理"""
        # 添加一些过期数据
        old_time = datetime.now() - timedelta(days=100)  # 超过90天 retention

        old_metric = QualityMetric(
            metric_id='old_metric',
            name='过期指标',
            type=MetricType.COVERAGE,
            value=80.0,
            unit='%',
            timestamp=old_time
        )
        self.monitoring_system.collect_metric(old_metric)

        # 创建过期告警
        old_alert = QualityAlert(
            alert_id='old_alert',
            title='过期告警',
            description='测试过期告警',
            severity=AlertSeverity.WARNING,
            metric_id='old_metric',
            current_value=80.0,
            threshold_value=85.0,
            timestamp=old_time,
            resolved=True
        )
        self.monitoring_system.alerts[old_alert.alert_id] = old_alert

        # 执行清理
        self.monitoring_system._cleanup_old_data()

        # 验证过期数据已被清理
        assert 'old_metric' not in self.monitoring_system.metrics

        # 已解决的过期告警也应该被清理（如果超过30天）
        if (datetime.now() - old_time).days > 30:
            assert old_alert.alert_id not in self.monitoring_system.alerts

        print("✅ 数据清理测试通过")

    def test_monitoring_recommendations(self):
        """测试监控建议生成"""
        # 创建一些测试场景
        recommendations = self.monitoring_system._generate_monitoring_recommendations()

        # 验证建议生成
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        # 验证包含通用建议
        general_recs = [r for r in recommendations if '🔄' in r or '📈' in r or '🤖' in r or '📋' in r]
        assert len(general_recs) > 0

        print(f"✅ 监控建议生成测试通过 - 生成 {len(recommendations)} 条建议")
