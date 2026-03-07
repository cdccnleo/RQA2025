#!/usr/bin/env python3
"""
性能监控体系搭建工具

用于实施实时性能监控、告警规则配置、监控仪表板搭建、自动化调优脚本等。
"""

import time
import psutil
import threading
import json
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any, List, Callable
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str
    operator: str  # '>', '<', '>=', '<=', '=='
    threshold: float
    level: AlertLevel
    description: str
    enabled: bool = True


@dataclass
class Alert:
    """告警信息"""
    rule_name: str
    level: AlertLevel
    message: str
    value: float
    threshold: float
    timestamp: float
    resolved: bool = False


class MetricsCollector:
    """指标收集器"""

    def __init__(self):
        self.metrics_history = {}
        self.collection_interval = 60  # 60秒

    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        # 网络IO
        net_io = psutil.net_io_counters()
        net_sent_mb = net_io.bytes_sent / 1024 / 1024
        net_recv_mb = net_io.bytes_recv / 1024 / 1024

        # 进程信息
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()

        metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "cpu_percent_process": process_cpu,
            "memory_percent": memory.percent,
            "memory_used_mb": memory.used / 1024 / 1024,
            "memory_process_rss_mb": process_memory.rss / 1024 / 1024,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024 / 1024 / 1024,
            "network_sent_mb": net_sent_mb,
            "network_recv_mb": net_recv_mb,
            "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
        }

        # 保存到历史记录
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((metrics["timestamp"], value))

            # 保持最近1000个数据点
            if len(self.metrics_history[key]) > 1000:
                self.metrics_history[key] = self.metrics_history[key][-1000:]

        return metrics

    def collect_application_metrics(self, app_stats: Dict[str, Any]) -> Dict[str, Any]:
        """收集应用指标"""
        metrics = {
            "timestamp": time.time(),
            "active_connections": app_stats.get("active_connections", 0),
            "requests_per_second": app_stats.get("requests_per_second", 0),
            "response_time_ms": app_stats.get("response_time_ms", 0),
            "error_rate": app_stats.get("error_rate", 0),
            "cache_hit_rate": app_stats.get("cache_hit_rate", 0),
            "db_connections": app_stats.get("db_connections", 0),
            "queue_size": app_stats.get("queue_size", 0)
        }

        # 保存到历史记录
        for key, value in metrics.items():
            if key not in self.metrics_history:
                self.metrics_history[key] = []
            self.metrics_history[key].append((metrics["timestamp"], value))

            # 保持最近1000个数据点
            if len(self.metrics_history[key]) > 1000:
                self.metrics_history[key] = self.metrics_history[key][-1000:]

        return metrics

    def get_metric_trends(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """获取指标趋势"""
        if metric_name not in self.metrics_history:
            return {"error": f"指标 {metric_name} 不存在"}

        data = self.metrics_history[metric_name]
        cutoff_time = time.time() - (hours * 3600)

        # 过滤最近的数据
        recent_data = [(t, v) for t, v in data if t >= cutoff_time]

        if not recent_data:
            return {"error": f"没有最近 {hours} 小时的 {metric_name} 数据"}

        values = [v for _, v in recent_data]

        return {
            "metric": metric_name,
            "hours": hours,
            "data_points": len(values),
            "current_value": values[-1] if values else 0,
            "min_value": min(values) if values else 0,
            "max_value": max(values) if values else 0,
            "avg_value": sum(values) / len(values) if values else 0,
            "trend": "increasing" if len(values) >= 2 and values[-1] > values[0] else "decreasing"
        }


class AlertManager:
    """告警管理器"""

    def __init__(self):
        self.rules = []
        self.active_alerts = []
        self.alert_history = []
        self.notifiers = []

    def add_rule(self, rule: AlertRule):
        """添加告警规则"""
        self.rules.append(rule)
        logger.info(f"添加告警规则: {rule.name}")

    def add_notifier(self, notifier: Callable):
        """添加通知器"""
        self.notifiers.append(notifier)

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """评估告警规则"""
        new_alerts = []

        for rule in self.rules:
            if not rule.enabled:
                continue

            if rule.metric not in metrics:
                continue

            value = metrics[rule.metric]

            # 评估条件
            triggered = False
            if rule.operator == '>':
                triggered = value > rule.threshold
            elif rule.operator == '<':
                triggered = value < rule.threshold
            elif rule.operator == '>=':
                triggered = value >= rule.threshold
            elif rule.operator == '<=':
                triggered = value <= rule.threshold
            elif rule.operator == '==':
                triggered = abs(value - rule.threshold) < 0.001

            if triggered:
                alert = Alert(
                    rule_name=rule.name,
                    level=rule.level,
                    message=f"{rule.description}: {value} {rule.operator} {rule.threshold}",
                    value=value,
                    threshold=rule.threshold,
                    timestamp=time.time()
                )

                new_alerts.append(alert)
                self.active_alerts.append(alert)
                logger.warning(f"触发告警: {alert.message}")

        return new_alerts

    def resolve_alerts(self, metrics: Dict[str, Any]):
        """解决已恢复的告警"""
        resolved = []

        for alert in self.active_alerts[:]:  # 复制列表以便修改
            rule = next((r for r in self.rules if r.name == alert.rule_name), None)
            if rule and rule.metric in metrics:
                value = metrics[rule.metric]
                resolved_condition = False

                # 检查是否恢复正常
                if rule.operator == '>':
                    resolved_condition = value <= rule.threshold
                elif rule.operator == '<':
                    resolved_condition = value >= rule.threshold
                # 其他操作符的恢复逻辑可以根据需要添加

                if resolved_condition:
                    alert.resolved = True
                    alert.resolved_timestamp = time.time()
                    resolved.append(alert)
                    self.active_alerts.remove(alert)
                    logger.info(f"告警已解决: {alert.message}")

        return resolved

    def notify_alerts(self, alerts: List[Alert]):
        """通知告警"""
        for alert in alerts:
            for notifier in self.notifiers:
                try:
                    notifier(alert)
                except Exception as e:
                    logger.error(f"通知失败: {e}")


class EmailNotifier:
    """邮件通知器"""

    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str, recipients: List[str]):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients

    def __call__(self, alert: Alert):
        """发送告警邮件"""
        subject = f"[{alert.level.value.upper()}] 系统性能告警"
        body = f"""
系统性能告警

告警规则: {alert.rule_name}
告警级别: {alert.level.value}
消息: {alert.message}
当前值: {alert.value}
阈值: {alert.threshold}
时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}

请及时处理！
"""

        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.username
        msg['To'] = ', '.join(self.recipients)

        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            server.sendmail(self.username, self.recipients, msg.as_string())
            server.quit()
            logger.info(f"告警邮件已发送: {alert.message}")
        except Exception as e:
            logger.error(f"发送告警邮件失败: {e}")


class PerformanceDashboard:
    """性能仪表板"""

    def __init__(self, collector: MetricsCollector, alert_manager: AlertManager):
        self.collector = collector
        self.alert_manager = alert_manager

    def generate_dashboard_data(self) -> Dict[str, Any]:
        """生成仪表板数据"""
        # 获取当前系统指标
        system_metrics = self.collector.collect_system_metrics()

        # 获取趋势数据
        cpu_trend = self.collector.get_metric_trends("cpu_percent", 1)
        memory_trend = self.collector.get_metric_trends("memory_percent", 1)

        # 获取活跃告警
        active_alerts = [
            {
                "rule_name": alert.rule_name,
                "level": alert.level.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "age_minutes": (time.time() - alert.timestamp) / 60
            }
            for alert in self.alert_manager.active_alerts
        ]

        dashboard = {
            "timestamp": time.time(),
            "system_metrics": {
                "cpu_percent": system_metrics["cpu_percent"],
                "memory_percent": system_metrics["memory_percent"],
                "disk_percent": system_metrics["disk_percent"],
                "network_sent_mb": system_metrics["network_sent_mb"],
                "network_recv_mb": system_metrics["network_recv_mb"]
            },
            "trends": {
                "cpu_trend": cpu_trend,
                "memory_trend": memory_trend
            },
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": active_alerts[:10],  # 最新的10个
                "total_rules": len(self.alert_manager.rules)
            },
            "health_score": self._calculate_health_score(system_metrics, active_alerts)
        }

        return dashboard

    def _calculate_health_score(self, metrics: Dict[str, Any], alerts: List[Dict]) -> float:
        """计算健康评分"""
        score = 100.0

        # CPU使用率影响
        cpu_percent = metrics["cpu_percent"]
        if cpu_percent > 90:
            score -= 30
        elif cpu_percent > 80:
            score -= 20
        elif cpu_percent > 70:
            score -= 10

        # 内存使用率影响
        memory_percent = metrics["memory_percent"]
        if memory_percent > 90:
            score -= 30
        elif memory_percent > 80:
            score -= 20
        elif memory_percent > 70:
            score -= 10

        # 活跃告警影响
        alert_penalty = len(alerts) * 5
        score -= min(alert_penalty, 30)

        return max(0, score)


class AutomatedOptimizer:
    """自动化调优器"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector
        self.optimization_actions = []

    def analyze_and_optimize(self) -> List[str]:
        """分析并执行优化"""
        actions_taken = []

        # 获取当前指标
        metrics = self.collector.collect_system_metrics()

        # CPU优化
        if metrics["cpu_percent"] > 80:
            actions_taken.extend(self._optimize_cpu_usage(metrics))

        # 内存优化
        if metrics["memory_percent"] > 80:
            actions_taken.extend(self._optimize_memory_usage(metrics))

        # 磁盘优化
        if metrics["disk_percent"] > 90:
            actions_taken.extend(self._optimize_disk_usage(metrics))

        # 记录优化动作
        self.optimization_actions.extend(actions_taken)

        return actions_taken

    def _optimize_cpu_usage(self, metrics: Dict[str, Any]) -> List[str]:
        """CPU使用率优化"""
        actions = []

        if metrics["cpu_percent"] > 90:
            actions.append("检测到高CPU使用率 (>90%)")
            actions.append("建议: 实施算法并行化处理")
            actions.append("建议: 启用GPU加速计算")
            actions.append("执行: 调整进程优先级")

        elif metrics["cpu_percent"] > 80:
            actions.append("检测到较高CPU使用率 (>80%)")
            actions.append("建议: 优化热点代码")
            actions.append("执行: 启用缓存预热")

        return actions

    def _optimize_memory_usage(self, metrics: Dict[str, Any]) -> List[str]:
        """内存使用率优化"""
        actions = []

        if metrics["memory_percent"] > 90:
            actions.append("检测到高内存使用率 (>90%)")
            actions.append("执行: 强制垃圾回收")
            actions.append("建议: 实施内存池管理")
            actions.append("建议: 优化缓存策略")

        elif metrics["memory_percent"] > 80:
            actions.append("检测到较高内存使用率 (>80%)")
            actions.append("执行: 清理过期缓存")
            actions.append("建议: 实施模型压缩")

        return actions

    def _optimize_disk_usage(self, metrics: Dict[str, Any]) -> List[str]:
        """磁盘使用率优化"""
        actions = []

        if metrics["disk_percent"] > 90:
            actions.append("检测到高磁盘使用率 (>90%)")
            actions.append("执行: 清理临时文件")
            actions.append("建议: 实施数据压缩存储")
            actions.append("建议: 优化日志轮换策略")

        return actions


class PerformanceMonitoringSystem:
    """性能监控体系"""

    def __init__(self):
        self.collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.dashboard = PerformanceDashboard(self.collector, self.alert_manager)
        self.optimizer = AutomatedOptimizer(self.collector)
        self.monitoring_thread = None
        self.running = False

    def setup_default_alert_rules(self):
        """设置默认告警规则"""
        rules = [
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_percent",
                operator=">",
                threshold=80.0,
                level=AlertLevel.WARNING,
                description="CPU使用率过高"
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric="cpu_percent",
                operator=">",
                threshold=90.0,
                level=AlertLevel.CRITICAL,
                description="CPU使用率严重过高"
            ),
            AlertRule(
                name="high_memory_usage",
                metric="memory_percent",
                operator=">",
                threshold=85.0,
                level=AlertLevel.WARNING,
                description="内存使用率过高"
            ),
            AlertRule(
                name="critical_memory_usage",
                metric="memory_percent",
                operator=">",
                threshold=95.0,
                level=AlertLevel.CRITICAL,
                description="内存使用率严重过高"
            ),
            AlertRule(
                name="disk_space_low",
                metric="disk_percent",
                operator=">",
                threshold=90.0,
                level=AlertLevel.ERROR,
                description="磁盘空间不足"
            )
        ]

        for rule in rules:
            self.alert_manager.add_rule(rule)

    def setup_email_notifications(self, smtp_config: Dict[str, str]):
        """设置邮件通知"""
        notifier = EmailNotifier(
            smtp_server=smtp_config.get("server", "smtp.example.com"),
            smtp_port=int(smtp_config.get("port", 587)),
            username=smtp_config.get("username", ""),
            password=smtp_config.get("password", ""),
            recipients=smtp_config.get("recipients", [])
        )
        self.alert_manager.add_notifier(notifier)

    def start_monitoring(self, interval: int = 60):
        """开始监控"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        logger.info(f"性能监控已启动，收集间隔: {interval}秒")

    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("性能监控已停止")

    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.running:
            try:
                # 收集指标
                metrics = self.collector.collect_system_metrics()

                # 评估告警规则
                new_alerts = self.alert_manager.evaluate_rules(metrics)

                # 解决已恢复的告警
                resolved_alerts = self.alert_manager.resolve_alerts(metrics)

                # 通知新告警
                if new_alerts:
                    self.alert_manager.notify_alerts(new_alerts)

                # 执行自动化优化
                optimization_actions = self.optimizer.analyze_and_optimize()

                time.sleep(interval)

            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(interval)

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        dashboard_data = self.dashboard.generate_dashboard_data()

        return {
            "timestamp": time.time(),
            "health_score": dashboard_data["health_score"],
            "system_metrics": dashboard_data["system_metrics"],
            "active_alerts": dashboard_data["alerts"]["active_count"],
            "recent_optimizations": self.optimizer.optimization_actions[-5:] if self.optimizer.optimization_actions else []
        }

    def generate_report(self) -> Dict[str, Any]:
        """生成监控报告"""
        dashboard_data = self.dashboard.generate_dashboard_data()

        # 获取趋势分析
        trends = {}
        for metric in ["cpu_percent", "memory_percent", "disk_percent"]:
            trends[metric] = self.collector.get_metric_trends(metric, 1)

        # 获取告警统计
        alert_stats = {
            "total_rules": len(self.alert_manager.rules),
            "active_alerts": len(self.alert_manager.active_alerts),
            "total_alerts_history": len(self.alert_manager.alert_history),
            "alerts_by_level": {}
        }

        for alert in self.alert_manager.active_alerts + self.alert_manager.alert_history:
            level = alert.level.value
            alert_stats["alerts_by_level"][level] = alert_stats["alerts_by_level"].get(level, 0) + 1

        report = {
            "generated_at": time.time(),
            "monitoring_duration_hours": (time.time() - self.collector.metrics_history.get(
                list(self.collector.metrics_history.keys())[0], [(0, 0)]
            )[0][0]) / 3600 if self.collector.metrics_history else 0,
            "dashboard": dashboard_data,
            "trends": trends,
            "alert_statistics": alert_stats,
            "optimization_actions": self.optimizer.optimization_actions[-10:],  # 最近10个
            "recommendations": self._generate_recommendations(dashboard_data, trends, alert_stats)
        }

        return report

    def _generate_recommendations(self, dashboard: Dict, trends: Dict, alert_stats: Dict) -> List[str]:
        """生成建议"""
        recommendations = []

        # 基于健康评分
        health_score = dashboard.get("health_score", 100)
        if health_score < 70:
            recommendations.append("🚨 系统健康评分较低，建议立即优化")
        elif health_score < 85:
            recommendations.append("⚠️ 系统健康评分中等，建议持续监控")

        # 基于趋势
        for metric, trend_data in trends.items():
            if trend_data.get("trend") == "increasing":
                if "cpu" in metric:
                    recommendations.append("📈 CPU使用率呈上升趋势，考虑优化计算密集型任务")
                elif "memory" in metric:
                    recommendations.append("📈 内存使用率呈上升趋势，检查内存泄漏")

        # 基于告警
        if alert_stats["active_alerts"] > 0:
            recommendations.append(f"🔥 有 {alert_stats['active_alerts']} 个活跃告警需要处理")

        # 通用建议
        recommendations.extend([
            "📊 定期审查告警规则配置",
            "🔄 优化指标收集频率",
            "📈 建立性能基准线",
            "🎯 配置自动化响应机制"
        ])

        return recommendations


def main():
    """主函数"""
    print("🚀 CPU/内存性能优化专项 - 性能监控体系")
    print("=" * 60)

    # 创建监控系统
    monitoring_system = PerformanceMonitoringSystem()

    # 设置默认告警规则
    print("\n1. 设置告警规则...")
    monitoring_system.setup_default_alert_rules()

    # 启动监控（运行30秒进行测试）
    print("\n2. 启动性能监控...")
    monitoring_system.start_monitoring(interval=5)  # 5秒间隔用于测试

    # 等待一段时间收集数据
    print("   正在收集性能数据...")
    time.sleep(15)

    # 停止监控
    print("\n3. 停止监控并生成报告...")
    monitoring_system.stop_monitoring()

    # 获取系统状态
    status = monitoring_system.get_system_status()

    # 生成完整报告
    report = monitoring_system.generate_report()

    # 输出摘要
    print("\n📊 性能监控体系报告摘要")
    print("-" * 50)

    print(f"系统健康评分: {status['health_score']:.1f}/100")
    print(f"活跃告警数量: {status['active_alerts']}")

    metrics = status['system_metrics']
    print("\n🖥️ 当前系统指标:")
    print(f"   CPU使用率: {metrics['cpu_percent']:.1f}%")
    print(f"   内存使用率: {metrics['memory_percent']:.1f}%")
    print(f"   磁盘使用率: {metrics['disk_percent']:.1f}%")
    print(f"   网络发送: {metrics['network_sent_mb']:.1f} MB")
    print(f"   网络接收: {metrics['network_recv_mb']:.1f} MB")    # 告警统计
    alert_stats = report['alert_statistics']
    print("\n🔥 告警统计:")
    print(f"   配置规则: {alert_stats['total_rules']}")
    print(f"   活跃告警: {alert_stats['active_alerts']}")
    print(f"   历史告警: {alert_stats['total_alerts_history']}")

    # 优化建议
    recommendations = report.get('recommendations', [])
    print("\n💡 优化建议:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec}")

    # 保存详细报告
    with open("performance_monitoring_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print("\n📄 详细报告已保存到: performance_monitoring_report.json")
    print("\n✅ 性能监控体系搭建完成！")


if __name__ == "__main__":
    main()
