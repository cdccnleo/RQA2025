
import time

from ..monitoring.health_evaluator import HealthEvaluator
from ..monitoring.health_metrics_collector import HealthMetricsCollector
from ..monitoring.health_reporter import HealthReporter
from .shared_interfaces import ILogger, IErrorHandler, StandardLogger, BaseErrorHandler
from typing import Dict, Optional, Any
"""
系统健康监控器 - 重构版本

负责协调各个健康监控组件，提供统一的健康监控接口
"""


class SystemHealthMonitor:
    """系统健康监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None,
                 error_handler: Optional[IErrorHandler] = None):

        self.logger = logger or StandardLogger(self.__class__.__name__)
        self.error_handler = error_handler or BaseErrorHandler()
        self.config = config or {}

        # 初始化专用组件
        self.metrics_collector = HealthMetricsCollector(config, self.logger)
        self.evaluator = HealthEvaluator(self._get_health_thresholds(), self.logger)
        self.reporter = HealthReporter(self.logger)

        # 组件引用 (保持向后兼容)
        self.performance_monitor = None
        self.alert_manager = None
        self.test_monitor = None
        self.system_coordinator = None

    def _get_health_thresholds(self) -> Dict[str, float]:
        """获取健康阈值配置"""
        return {
            "cpu_critical": self.config.get("cpu_critical_threshold", 90.0),
            "cpu_warning": self.config.get("cpu_warning_threshold", 80.0),
            "memory_critical": self.config.get("memory_critical_threshold", 90.0),
            "memory_warning": self.config.get("memory_warning_threshold", 85.0),
            "disk_critical": self.config.get("disk_critical_threshold", 95.0),
            "disk_warning": self.config.get("disk_warning_threshold", 90.0),
            "alerts_critical": self.config.get("alerts_critical_threshold", 10),
            "alerts_warning": self.config.get("alerts_warning_threshold", 5),
            "test_success_rate_critical": self.config.get("test_success_rate_critical", 0.5),
            "test_success_rate_warning": self.config.get("test_success_rate_warning", 0.8)
        }

    def set_components(self, performance_monitor, alert_manager,
                       test_monitor, system_coordinator):
        """设置组件引用"""
        self.performance_monitor = performance_monitor
        self.alert_manager = alert_manager
        self.test_monitor = test_monitor
        self.system_coordinator = system_coordinator

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        try:
            status = {
                "timestamp": time.time(),
                "running": self.system_coordinator.is_running() if self.system_coordinator else False
            }

            # 组件状态
            if self.performance_monitor:
                status["performance_monitoring"] = getattr(
                    self.performance_monitor, 'monitoring', False)

            if self.test_monitor:
                status["test_monitoring"] = getattr(self.test_monitor, 'monitoring', False)
                status["active_tests_count"] = len(self.test_monitor.get_active_tests())

            if self.alert_manager:
                status["active_alerts_count"] = len(self.alert_manager.get_active_alerts())
                status["alert_rules_count"] = len(self.alert_manager.alert_rules)

            # 通知渠道状态（需要从其他组件获取）
            status["notification_channels"] = {}

            return status

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取系统状态失败"})
            return {"error": str(e), "timestamp": time.time()}

    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        try:
            # 获取当前指标
            metrics = self.metrics_collector.collect_current_metrics()
            alert_stats = self._get_alert_stats()
            test_stats = self._get_test_stats()

            # 使用评估器评估健康状态
            health_assessment = self.evaluator.evaluate_overall_health(
                metrics, alert_stats, test_stats)

            # 使用报告器生成完整报告
            return self.reporter.generate_health_report(health_assessment, metrics, alert_stats, test_stats)

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "生成系统健康报告失败"})
            return {
                "error": str(e),
                "generated_at": time.time(),
                "health_score": 0.0,
                "health_status": "未知"
            }

    def _get_alert_stats(self) -> Optional[Dict[str, Any]]:
        """获取告警统计"""
        try:
            if self.alert_manager:
                return self.alert_manager.get_alert_statistics()
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取告警统计失败"})
        return {}

    def _get_test_stats(self) -> Optional[Dict[str, Any]]:
        """获取测试统计"""
        try:
            if self.test_monitor:
                return self.test_monitor.get_test_statistics()
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取测试统计失败"})
        return {}

    def get_health_trends(self, hours: int = 24) -> Dict[str, Any]:
        """获取健康趋势分析"""
        try:
            # 使用报告器生成趋势报告
            # 这里需要历史健康数据，暂时返回占位符
            health_history = []  # 需要从存储系统获取历史数据
            return self.reporter.generate_trend_report(health_history, hours)
        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取健康趋势失败"})
            return {"error": str(e)}

    def get_component_health(self) -> Dict[str, Any]:
        """获取各组件健康状态"""
        try:
            component_statuses = {}

            # 检查各组件
            if self.performance_monitor:
                component_statuses["performance_monitor"] = {
                    "status": "healthy" if getattr(self.performance_monitor, 'monitoring', False) else "stopped",
                    "last_check": getattr(self.performance_monitor, 'last_update', None),
                    "details": {"monitoring": getattr(self.performance_monitor, 'monitoring', False)}
                }

            if self.alert_manager:
                active_alerts = len(getattr(self.alert_manager, 'get_active_alerts', lambda: [])())
                component_statuses["alert_manager"] = {
                    "status": "healthy",
                    "last_check": None,
                    "details": {
                        "active_alerts": active_alerts,
                        "rules_count": len(getattr(self.alert_manager, 'alert_rules', []))
                    }
                }

            if self.test_monitor:
                test_stats = getattr(self.test_monitor, 'get_test_statistics', lambda: {})()
                success_rate = test_stats.get('success_rate', 0)
                component_statuses["test_monitor"] = {
                    "status": "healthy" if success_rate > 0.5 else "degraded",
                    "last_check": None,
                    "details": {
                        "success_rate": success_rate,
                        "active_tests": test_stats.get('active_tests', 0)
                    }
                }

            # 使用报告器生成组件健康报告
            return self.reporter.generate_component_health_report(component_statuses)

        except Exception as e:
            self.error_handler.handle_error(e, {"context": "获取组件健康状态失败"})
            return {"error": str(e)}
