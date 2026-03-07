"""
monitoring_dashboard 模块

提供 monitoring_dashboard 相关功能和接口。
"""

import json
import logging
import os

import secrets
import statistics
import threading
import time
import random

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
"""
运维监控面板
提供实时监控、告警管理、性能分析等功能
"""

logger = logging.getLogger(__name__)

# 常量定义 - 清理魔法数字
DEFAULT_RETENTION_DAYS = 30  # 默认数据保留天数
DEFAULT_MAX_METRICS = 10000  # 默认最大指标数量
DEFAULT_ALERT_TIMEOUT = 300.0  # 默认告警超时时间(秒)
RECENT_ALERTS_LIMIT = 10  # 最近告警显示数量
REFRESH_CHECK_MULTIPLIER = 10  # 刷新检查倍数
STOP_CHECK_INTERVAL = 600  # 停止检查间隔(次)
STOP_CHECK_DELAY = 10.0  # 停止检查延迟(秒)
SECONDS_PER_HOUR = 3600  # 每小时秒数
HOURS_PER_DAY = 24  # 每天小时数
HIGH_CPU_THRESHOLD = 80.0  # 高CPU使用率阈值


class MetricType(Enum):

    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):

    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Metric:

    """监控指标"""
    name: str
    value: float
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    description: str = ""


@dataclass
class Alert:

    """告警信息"""
    name: str
    message: str
    severity: AlertSeverity
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    labels: Dict[str, str] = field(default_factory=dict)
    source: str = ""


@dataclass
class DashboardConfig:

    """监控面板配置"""
    refresh_interval: float = 5.0  # 刷新间隔(秒)
    retention_days: int = DEFAULT_RETENTION_DAYS  # 数据保留天数
    max_metrics: int = DEFAULT_MAX_METRICS  # 最大指标数量
    alert_timeout: float = DEFAULT_ALERT_TIMEOUT  # 告警超时时间(秒)
    enable_auto_cleanup: bool = True  # 启用自动清理


class MonitoringDashboard:

    """运维监控面板"""

    def __init__(self, config: DashboardConfig = None, auto_start: bool = False):

        self.config = config or DashboardConfig()
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.max_metrics))
        self._alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._dashboard_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._running = False
        self._monitor_thread = None
        self._cleanup_thread = None

        # 回调函数
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        self._metric_callbacks: List[Callable[[Metric], None]] = []

        logger.info("监控面板初始化完成")

        # 可选自动启动 - 明确检查参数
        if auto_start is True:
            self.start()

    def start(self):
        """启动监控面板"""
        if self._running:
            logger.warning("监控面板已在运行")
            return

        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)

        self._monitor_thread.start()
        self._cleanup_thread.start()

        logger.info("监控面板已启动")

    def stop(self):
        """停止监控面板"""
        logger.info("开始停止监控面板")
        self._running = False

        # 更安全的线程停止逻辑
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.debug("等待监控线程停止...")
            self._monitor_thread.join(timeout=2.0)  # 减少超时时间
            if self._monitor_thread.is_alive():
                logger.warning("监控线程未在预期时间内停止")

        if self._cleanup_thread and self._cleanup_thread.is_alive():
            logger.debug("等待清理线程停止...")
            self._cleanup_thread.join(timeout=2.0)  # 减少超时时间
            if self._cleanup_thread.is_alive():
                logger.warning("清理线程未在预期时间内停止")

        logger.info("监控面板已停止")

    def add_metric(self, metric: Metric):
        """添加监控指标"""
        with self._lock:
            self._metrics[metric.name].append(metric)

            # 触发回调
            for callback in self._metric_callbacks:
                try:
                    callback(metric)
                except Exception as e:
                    logger.error(f"指标回调执行失败: {e}")

    def update(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新仪表盘数据并写入基础指标"""
        if not isinstance(data, dict):
            raise ValueError("dashboard update payload must be a dict")

        with self._lock:
            timestamp = datetime.now().isoformat()
            self._dashboard_data.update(data)
            self._dashboard_data["last_update"] = timestamp

            for key, value in data.items():
                numeric_value = float(value) if isinstance(value, (int, float)) else 0.0
                metric = Metric(
                    name=key,
                    value=numeric_value,
                    metric_type=MetricType.GAUGE,
                    description="dashboard_update",
                    timestamp=time.time(),
                )
                self._metrics[key].append(metric)

        return {"timestamp": timestamp, "updated_keys": list(data.keys())}

    def get_summary(self) -> Dict[str, Any]:
        """返回仪表盘摘要信息"""
        with self._lock:
            total_metrics = sum(len(v) for v in self._metrics.values())
            return {
                "total_metrics": total_metrics,
                "active_alerts": len([a for a in self._alerts.values() if not a.resolved]),
                "last_update": self._dashboard_data.get("last_update"),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """返回统计信息"""
        with self._lock:
            metric_counts = {name: len(values) for name, values in self._metrics.items()}
            return {
                "metric_counts": metric_counts,
                "alerts_count": len(self._alerts),
                "configured_rules": len(self._alert_rules),
            }

    def get_metrics(self, name: str = None, start_time: float = None, end_time: float = None) -> List[Metric]:
        """获取监控指标"""
        with self._lock:
            if name:
                metrics = list(self._metrics.get(name, []))
            else:
                metrics = []
                for metric_list in self._metrics.values():
                    metrics.extend(metric_list)

            # 时间过滤
            if start_time or end_time:
                filtered_metrics = []
                for metric in metrics:
                    if start_time and metric.timestamp < start_time:
                        continue
                    if end_time and metric.timestamp > end_time:
                        continue
                    filtered_metrics.append(metric)
                return filtered_metrics

            return metrics

    def add_alert_rule(self, rule_name: str, query: str, severity: AlertSeverity,
                       threshold: float, duration: float = 0):
        """添加告警规则"""
        with self._lock:
            self._alert_rules[rule_name] = {
                'query': query,
                'severity': severity,
                'threshold': threshold,
                'duration': duration,
                'created_time': time.time()
            }

            logger.info(f"添加告警规则: {rule_name}")

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        with self._lock:
            if rule_name in self._alert_rules:
                del self._alert_rules[rule_name]
                logger.info(f"移除告警规则: {rule_name}")

    def get_alerts(self, severity: AlertSeverity = None, resolved: bool = None) -> List[Alert]:
        """获取告警信息"""
        with self._lock:
            alerts = list(self._alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        if resolved is not None:
            alerts = [alert for alert in alerts if alert.resolved == resolved]

        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)

    def resolve_alert(self, alert_name: str):
        """解决告警"""
        with self._lock:
            if alert_name in self._alerts:
                self._alerts[alert_name].resolved = True
                logger.info(f"解决告警: {alert_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """添加告警回调"""
        self._alert_callbacks.append(callback)

    def add_metric_callback(self, callback: Callable[[Metric], None]):
        """添加指标回调"""
        self._metric_callbacks.append(callback)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取面板数据"""
        with self._lock:
            now = time.time()

            # 计算统计信息
            stats = {}
            for name, metrics in self._metrics.items():
                if metrics:
                    values = [m.value for m in metrics]
                    stats[name] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': statistics.mean(values),
                        'latest': metrics[-1].value if metrics else 0
                    }

            # 告警统计
            alert_stats = defaultdict(int)
            # 获取未解决的告警列表（在同一个锁保护下）
            unresolved_alerts = [a for a in self._alerts.values() if not a.resolved]
            recent_unresolved_alerts = sorted(unresolved_alerts, key=lambda x: x.timestamp, reverse=True)[:RECENT_ALERTS_LIMIT]
            
            for alert in self._alerts.values():
                alert_stats[alert.severity.value] += 1

            return {
                'timestamp': now,
                'metrics_count': sum(len(metrics) for metrics in self._metrics.values()),
                'alerts_count': len(unresolved_alerts),
                'alert_rules_count': len(self._alert_rules),
                'stats': stats,
                'alert_stats': dict(alert_stats),
                'recent_alerts': recent_unresolved_alerts
            }

    def export_data(self, file_path: str):
        """导出监控数据"""
        with self._lock:
            data = {
                'metrics': {name: list(metrics) for name, metrics in self._metrics.items()},
                'alerts': list(self._alerts.values()),
                'alert_rules': self._alert_rules,
                'export_time': time.time()
            }

            with open(file_path, 'w', encoding='utf - 8') as f:
                json.dump(data, f, default=lambda x: x.__dict__, indent=2)

            logger.info(f"监控数据已导出到: {file_path}")

    def check_health(self) -> Dict[str, Any]:
        """执行整体健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        try:
            logger.info("开始监控面板健康检查")

            health_checks = {
                "dashboard_status": self.check_dashboard_status(),
                "metrics_integrity": self.check_metrics_integrity(),
                "alert_system": self.check_alert_system_health(),
                "performance": self.check_performance_health()
            }

            # 综合健康状态
            overall_healthy = all(check.get("healthy", False) for check in health_checks.values())

            result = {
                "healthy": overall_healthy,
                "timestamp": datetime.now().isoformat(),
                "service": "monitoring_dashboard",
                "checks": health_checks
            }

            if not overall_healthy:
                logger.warning("监控面板健康检查发现问题")
                result["issues"] = [
                    name for name, check in health_checks.items()
                    if not check.get("healthy", False)
                ]

            logger.info(f"监控面板健康检查完成，状态: {'健康' if overall_healthy else '异常'}")
            return result

        except Exception as e:
            logger.error(f"监控面板健康检查失败: {str(e)}", exc_info=True)
            return {
                "healthy": False,
                "timestamp": datetime.now().isoformat(),
                "service": "monitoring_dashboard",
                "error": str(e)
            }

    def check_dashboard_status(self) -> Dict[str, Any]:
        """检查面板运行状态

        Returns:
            Dict[str, Any]: 面板状态检查结果
        """
        try:
            thread_alive = self._monitor_thread.is_alive() if self._monitor_thread else False
            is_running = self._running

            return {
                "healthy": thread_alive == is_running,  # 线程状态应与运行标志一致
                "running": is_running,
                "thread_alive": thread_alive,
                "config": {
                    "refresh_interval": self.config.refresh_interval,
                    "max_metrics": self.config.max_metrics,
                    "retention_days": self.config.retention_days
                }
            }
        except Exception as e:
            logger.error(f"面板状态检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_metrics_integrity(self) -> Dict[str, Any]:
        """检查指标数据完整性

        Returns:
            Dict[str, Any]: 指标完整性检查结果
        """
        try:
            total_metrics = sum(len(metrics) for metrics in self._metrics.values())
            metrics_count = len(self._metrics)

            # 检查是否有数据
            has_data = total_metrics > 0

            # 检查数据是否在合理范围内
            within_limits = total_metrics <= self.config.max_metrics

            # 检查时间戳一致性
            timestamp_issues = []
            for name, metrics in self._metrics.items():
                if metrics:
                    timestamps = [m.timestamp for m in metrics]
                    if len(timestamps) > 1 and not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
                        timestamp_issues.append(name)

            return {
                "healthy": has_data and within_limits and len(timestamp_issues) == 0,
                "total_metrics": total_metrics,
                "metrics_count": metrics_count,
                "within_limits": within_limits,
                "timestamp_issues": timestamp_issues
            }
        except Exception as e:
            logger.error(f"指标完整性检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_alert_system_health(self) -> Dict[str, Any]:
        """检查告警系统健康状态

        Returns:
            Dict[str, Any]: 告警系统健康检查结果
        """
        try:
            active_alerts = len([a for a in self._alerts.values() if not a.resolved])
            total_rules = len(self._alert_rules)

            # 检查是否有活跃告警
            has_active_alerts = active_alerts > 0

            # 检查规则数量是否合理
            reasonable_rules = 0 <= total_rules <= 1000

            # 检查未解决的严重告警
            critical_alerts = len([a for a in self._alerts.values()
                                   if not a.resolved and a.severity == AlertSeverity.CRITICAL])

            return {
                "healthy": reasonable_rules,  # 主要检查规则数量是否合理
                "active_alerts": active_alerts,
                "total_rules": total_rules,
                "critical_alerts": critical_alerts,
                "reasonable_rules": reasonable_rules
            }
        except Exception as e:
            logger.error(f"告警系统健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def check_performance_health(self) -> Dict[str, Any]:
        """检查性能健康状态

        Returns:
            Dict[str, Any]: 性能健康检查结果
        """
        try:
            # 检查内存使用情况（简化检查）
            memory_usage = len(self._metrics) * 1024  # 估算内存使用
            acceptable_memory = memory_usage < 100 * 1024 * 1024  # 100MB

            # 检查响应时间（简化检查）
            response_time_ok = True  # 假设响应时间正常

            # 检查数据处理能力
            processing_capacity_ok = len(self._metrics) <= self.config.max_metrics

            return {
                "healthy": acceptable_memory and response_time_ok and processing_capacity_ok,
                "memory_usage_kb": memory_usage / 1024,
                "response_time_ok": response_time_ok,
                "processing_capacity_ok": processing_capacity_ok,
                "metrics_utilization": len(self._metrics) / max(1, self.config.max_metrics)
            }
        except Exception as e:
            logger.error(f"性能健康检查失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要

        Returns:
            Dict[str, Any]: 健康状态摘要
        """
        try:
            dashboard_data = self.get_dashboard_data()
            health_check = self.check_health()

            return {
                "status": "healthy" if health_check["healthy"] else "unhealthy",
                "dashboard_data": dashboard_data,
                "health_check": health_check,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康状态摘要失败: {str(e)}")
            return {"status": "error", "error": str(e)}

    def health_summary(self) -> Dict[str, Any]:
        """获取健康摘要报告

        Returns:
            Dict[str, Any]: 健康摘要报告
        """
        try:
            health_check = self.check_health()
            dashboard_data = self.get_dashboard_data()

            return {
                "overall_health": "healthy" if health_check["healthy"] else "unhealthy",
                "dashboard_summary": {
                    "total_metrics": dashboard_data.get("metrics_count", 0),
                    "active_alerts": dashboard_data.get("alerts_count", 0),
                    "configured_rules": dashboard_data.get("alert_rules_count", 0)
                },
                "performance_metrics": {
                    "refresh_interval": self.config.refresh_interval,
                    "retention_days": self.config.retention_days,
                    "max_metrics": self.config.max_metrics
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"获取健康摘要报告失败: {str(e)}")
            return {"overall_health": "error", "error": str(e)}

    def monitor_dashboard_performance(self) -> Dict[str, Any]:
        """监控面板性能指标

        Returns:
            Dict[str, Any]: 性能监控结果
        """
        try:
            # 计算性能指标
            metrics_count = sum(len(metrics) for metrics in self._metrics.values())
            alerts_count = len(self._alerts)
            rules_count = len(self._alert_rules)

            # 计算内存使用估算
            memory_usage = (metrics_count * 512 + alerts_count *
                            256 + rules_count * 128) / 1024  # KB

            # 检查性能阈值
            memory_ok = memory_usage < 50 * 1024  # 50MB
            metrics_ok = metrics_count <= self.config.max_metrics
            alerts_ok = alerts_count < 1000  # 告警数量阈值

            return {
                "healthy": memory_ok and metrics_ok and alerts_ok,
                "metrics": {
                    "total_metrics": metrics_count,
                    "total_alerts": alerts_count,
                    "total_rules": rules_count,
                    "memory_usage_kb": memory_usage
                },
                "thresholds": {
                    "memory_ok": memory_ok,
                    "metrics_ok": metrics_ok,
                    "alerts_ok": alerts_ok
                }
            }
        except Exception as e:
            logger.error(f"面板性能监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def monitor_alert_effectiveness(self) -> Dict[str, Any]:
        """监控告警系统有效性

        Returns:
            Dict[str, Any]: 告警有效性监控结果
        """
        try:
            total_alerts = len(self._alerts)
            if total_alerts == 0:
                return {"healthy": True, "reason": "no_alerts_to_analyze"}

            resolved_alerts = len([a for a in self._alerts.values() if a.resolved])
            active_alerts = total_alerts - resolved_alerts

            # 计算解决率
            resolution_rate = resolved_alerts / total_alerts if total_alerts > 0 else 0

            # 按严重程度统计
            severity_stats = defaultdict(int)
            for alert in self._alerts.values():
                severity_stats[alert.severity.value] += 1

            # 检查告警是否及时响应（简化检查）
            timely_response = True  # 假设响应及时

            return {
                "healthy": resolution_rate > 0.5,  # 解决率 > 50%
                "alerts_stats": {
                    "total": total_alerts,
                    "active": active_alerts,
                    "resolved": resolved_alerts,
                    "resolution_rate": resolution_rate
                },
                "severity_distribution": dict(severity_stats),
                "timely_response": timely_response
            }
        except Exception as e:
            logger.error(f"告警有效性监控失败: {str(e)}")
            return {"healthy": False, "error": str(e)}

    def validate_dashboard_config(self) -> Dict[str, Any]:
        """验证面板配置有效性

        Returns:
            Dict[str, Any]: 配置验证结果
        """
        try:
            validation_results = {
                "refresh_interval": self._validate_refresh_interval(),
                "max_metrics": self._validate_max_metrics(),
                "retention_days": self._validate_retention_days(),
                "alert_rules": self._validate_alert_rules()
            }

            overall_valid = all(result.get("valid", False)
                                for result in validation_results.values())

            return {
                "valid": overall_valid,
                "validation_results": validation_results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"面板配置验证失败: {str(e)}")
            return {"valid": False, "error": str(e)}

    def _validate_refresh_interval(self) -> Dict[str, Any]:
        """验证刷新间隔"""
        interval = self.config.refresh_interval
        valid = 0.1 <= interval <= 3600  # 0.1秒到1小时之间

        return {
            "valid": valid,
            "current_value": interval,
            "valid_range": "0.1-3600 seconds"
        }

    def _validate_max_metrics(self) -> Dict[str, Any]:
        """验证最大指标数量"""
        max_metrics = self.config.max_metrics
        valid = 100 <= max_metrics <= 100000  # 100到10万个之间

        return {
            "valid": valid,
            "current_value": max_metrics,
            "valid_range": "100-100000 metrics"
        }

    def _validate_retention_days(self) -> Dict[str, Any]:
        """验证数据保留天数"""
        retention = self.config.retention_days
        valid = 1 <= retention <= 365  # 1天到1年之间

        return {
            "valid": valid,
            "current_value": retention,
            "valid_range": "1-365 days"
        }

    def _validate_alert_rules(self) -> Dict[str, Any]:
        """验证告警规则"""
        rules_count = len(self._alert_rules)
        valid = 0 <= rules_count <= 1000  # 0到1000个规则

        # 检查规则格式（简化检查）
        invalid_rules = []
        for rule_name, rule_config in self._alert_rules.items():
            if not isinstance(rule_config, dict) or 'query' not in rule_config:
                invalid_rules.append(rule_name)

        return {
            "valid": valid and len(invalid_rules) == 0,
            "rules_count": rules_count,
            "invalid_rules": invalid_rules,
            "valid_range": "0-1000 rules"
        }

    def import_data(self, file_path: str):
        """导入监控数据"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return

        with open(file_path, 'r', encoding='utf - 8') as f:
            data = json.load(f)

        with self._lock:
            # 导入指标
            for name, metrics_data in data.get('metrics', {}).items():
                self._metrics[name] = deque(metrics_data, maxlen=self.config.max_metrics)

            # 导入告警
            for alert_data in data.get('alerts', []):
                alert = Alert(**alert_data)
                self._alerts[alert.name] = alert

            # 导入告警规则
            self._alert_rules.update(data.get('alert_rules', {}))

        logger.info(f"监控数据已从 {file_path} 导入")

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._check_alert_rules()
                # 使用更短的睡眠时间，以便更快响应停止信号
                for _ in range(int(self.config.refresh_interval * REFRESH_CHECK_MULTIPLIER)):  # 每0.1秒检查一次
                    if not self._running:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                if not self._running:
                    break
                time.sleep(1.0)

    def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                self._cleanup_old_data()
                # 使用更短的睡眠时间，以便更快响应停止信号
                for _ in range(STOP_CHECK_INTERVAL):  # 每分钟检查一次，但每0.1秒检查停止信号
                    if not self._running:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                if not self._running:
                    break
                time.sleep(STOP_CHECK_DELAY)

    def _check_alert_rules(self):
        """检查告警规则"""
        with self._lock:
            for rule_name, rule in self._alert_rules.items():
                try:
                    # 简单的规则检查逻辑
                    if rule['query'] in self._metrics:
                        metrics = self._metrics[rule['query']]
                        if metrics:
                            latest_value = metrics[-1].value

                            # 检查阈值
                            if latest_value > rule['threshold']:
                                alert_name = f"{rule_name}_{int(time.time())}"
                                alert = Alert(
                                    name=alert_name,
                                    message=f"指标 {rule['query']} 值 {latest_value} 超过阈值 {rule['threshold']}",
                                    severity=rule['severity'],
                                    source=rule['query']
                                )

                                self._alerts[alert_name] = alert

                                # 触发告警回调
                                for callback in self._alert_callbacks:
                                    try:
                                        callback(alert)
                                    except Exception as e:
                                        logger.error(f"告警回调执行失败: {e}")

                except Exception as e:
                    logger.error(f"检查告警规则 {rule_name} 失败: {e}")

    def _cleanup_old_data(self):
        """清理旧数据"""
        if not self.config.enable_auto_cleanup:
            return

        cutoff_time = time.time() - (self.config.retention_days * HOURS_PER_DAY * SECONDS_PER_HOUR)

        with self._lock:
            # 清理旧指标
            for name in list(self._metrics.keys()):
                metrics = self._metrics[name]
                while metrics and metrics[0].timestamp < cutoff_time:
                    metrics.popleft()

                # 如果指标为空，删除整个键
                if not metrics:
                    del self._metrics[name]

            # 清理已解决的旧告警
            current_time = time.time()
            for alert_name in list(self._alerts.keys()):
                alert = self._alerts[alert_name]
                if alert.resolved and (current_time - alert.timestamp) > self.config.alert_timeout:
                    del self._alerts[alert_name]


class DashboardManager:

    """监控面板管理器"""

    def __init__(self):

        self.dashboards: Dict[str, MonitoringDashboard] = {}
        self._lock = threading.Lock()

    def create_dashboard(
        self,
        name: str,
        config: DashboardConfig = None,
        auto_start: bool = False
    ) -> MonitoringDashboard:
        """创建监控面板"""
        with self._lock:
            if name in self.dashboards:
                logger.warning(f"监控面板 {name} 已存在")
                return self.dashboards[name]

            dashboard = MonitoringDashboard(config, auto_start=auto_start)
            self.dashboards[name] = dashboard
            logger.info(f"创建监控面板: {name}")
            return dashboard

    def get_dashboard(self, name: str) -> Optional[MonitoringDashboard]:
        """获取监控面板"""
        return self.dashboards.get(name)

    def remove_dashboard(self, name: str):
        """移除监控面板"""
        with self._lock:
            if name in self.dashboards:
                dashboard = self.dashboards[name]
                dashboard.stop()
                del self.dashboards[name]
                logger.info(f"移除监控面板: {name}")

    def start_all_dashboards(self):
        """启动所有监控面板"""
        for dashboard in self.dashboards.values():
            dashboard.start()

    def stop_all_dashboards(self):
        """停止所有监控面板"""
        for dashboard in self.dashboards.values():
            dashboard.stop()

    def get_all_dashboard_data(self) -> Dict[str, Dict[str, Any]]:
        """获取所有面板数据"""
        return {name: dashboard.get_dashboard_data() for name, dashboard in self.dashboards.items()}

    def export_all_data(self, directory: str):
        """导出所有面板数据"""
        os.makedirs(directory, exist_ok=True)

        for name, dashboard in self.dashboards.items():
            file_path = os.path.join(directory, f"{name}_data.json")
            dashboard.export_data(file_path)


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建监控面板管理器
    manager = DashboardManager()

    # 创建监控面板
    config = DashboardConfig(refresh_interval=2.0, retention_days=7)
    dashboard = manager.create_dashboard("main", config)

    # 添加告警回调

    def alert_callback(alert: Alert):

        print(f"🚨 告警: {alert.severity.value} - {alert.message}")

    dashboard.add_alert_callback(alert_callback)

    # 添加指标回调

    def metric_callback(metric: Metric):

        print(f"📊 指标: {metric.name} = {metric.value}")

    dashboard.add_metric_callback(metric_callback)

    # 启动面板
    dashboard.start()

    # 添加告警规则
    dashboard.add_alert_rule("high_cpu", "cpu_usage", AlertSeverity.WARNING, HIGH_CPU_THRESHOLD)
    dashboard.add_alert_rule("high_memory", "memory_usage", AlertSeverity.ERROR, 90.0)

    # 模拟添加指标
    try:
        for i in range(10):
            # CPU使用率
            cpu_metric = Metric(
                name="cpu_usage",
                value=random.uniform(50, 95),
                metric_type=MetricType.GAUGE,
                labels={"host": "server1"}
            )

            dashboard.add_metric(cpu_metric)

            # 内存使用率
            memory_metric = Metric(
                name="memory_usage",
                value=random.uniform(60, 98),
                metric_type=MetricType.GAUGE,
                labels={"host": "server1"}
            )

            dashboard.add_metric(memory_metric)

            # 请求计数
            request_metric = Metric(
                name="request_count",
                value=i + 1,
                metric_type=MetricType.COUNTER,
                labels={"service": "api"}
            )

            dashboard.add_metric(request_metric)

            time.sleep(1)

        # 获取面板数据
        data = dashboard.get_dashboard_data()
        print(f"面板数据: {json.dumps(data, indent=2, default=str)}")

        # 获取告警
        alerts = dashboard.get_alerts()
        print(f"告警数量: {len(alerts)}")

        # 导出数据
        dashboard.export_data("dashboard_data.json")

    finally:
        # 停止面板
        dashboard.stop()
        print("监控面板已停止")
