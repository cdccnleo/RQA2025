#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 异常监控告警系统

测试exception_monitoring_alert.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestExceptionMonitoringAlert:
    """测试异常监控告警系统"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.monitoring.exception_monitoring_alert import (
                ExceptionMonitor, ExceptionAlertRule, ExceptionAlert,
                ExceptionAlertNotifier, ExceptionDashboard, ExceptionAnalytics,
                ExceptionPattern, ExceptionSeverity, ExceptionCategory
            )
            self.ExceptionMonitor = ExceptionMonitor
            self.ExceptionAlertRule = ExceptionAlertRule
            self.ExceptionAlert = ExceptionAlert
            self.ExceptionAlertNotifier = ExceptionAlertNotifier
            self.ExceptionDashboard = ExceptionDashboard
            self.ExceptionAnalytics = ExceptionAnalytics
            self.ExceptionPattern = ExceptionPattern
            self.ExceptionSeverity = ExceptionSeverity
            self.ExceptionCategory = ExceptionCategory
        except ImportError as e:
            pytest.skip(f"Exception monitoring components not available: {e}")

    def test_exception_severity_enum(self):
        """测试异常严重程度枚举"""
        if not hasattr(self, 'ExceptionSeverity'):
            pytest.skip("ExceptionSeverity not available")

        assert hasattr(self.ExceptionSeverity, 'LOW')
        assert hasattr(self.ExceptionSeverity, 'MEDIUM')
        assert hasattr(self.ExceptionSeverity, 'HIGH')
        assert hasattr(self.ExceptionSeverity, 'CRITICAL')

    def test_exception_category_enum(self):
        """测试异常类别枚举"""
        if not hasattr(self, 'ExceptionCategory'):
            pytest.skip("ExceptionCategory not available")

        assert hasattr(self.ExceptionCategory, 'SYSTEM')
        assert hasattr(self.ExceptionCategory, 'APPLICATION')
        assert hasattr(self.ExceptionCategory, 'NETWORK')
        assert hasattr(self.ExceptionCategory, 'DATABASE')

    def test_exception_pattern(self):
        """测试异常模式"""
        if not hasattr(self, 'ExceptionPattern'):
            pytest.skip("ExceptionPattern not available")

        pattern = self.ExceptionPattern(
            name="test_pattern",
            pattern=r"ValueError: .*",
            severity=self.ExceptionSeverity.HIGH,
            category=self.ExceptionCategory.APPLICATION
        )

        assert pattern.name == "test_pattern"
        assert pattern.pattern == r"ValueError: .*"
        assert pattern.severity == self.ExceptionSeverity.HIGH
        assert pattern.category == self.ExceptionCategory.APPLICATION

    def test_exception_alert_rule(self):
        """测试异常告警规则"""
        if not hasattr(self, 'ExceptionAlertRule'):
            pytest.skip("ExceptionAlertRule not available")

        rule = self.ExceptionAlertRule(
            name="high_frequency_errors",
            condition="error_count > 100",
            severity=self.ExceptionSeverity.CRITICAL,
            time_window_minutes=5,
            cooldown_minutes=10
        )

        assert rule.name == "high_frequency_errors"
        assert rule.condition == "error_count > 100"
        assert rule.severity == self.ExceptionSeverity.CRITICAL
        assert rule.time_window_minutes == 5
        assert rule.cooldown_minutes == 10

    def test_exception_alert(self):
        """测试异常告警"""
        if not hasattr(self, 'ExceptionAlert'):
            pytest.skip("ExceptionAlert not available")

        alert = self.ExceptionAlert(
            rule_name="test_rule",
            severity=self.ExceptionSeverity.HIGH,
            message="High frequency of ValueError exceptions",
            exception_details={
                "type": "ValueError",
                "count": 150,
                "time_window": "5 minutes"
            }
        )

        assert alert.rule_name == "test_rule"
        assert alert.severity == self.ExceptionSeverity.HIGH
        assert "ValueError" in alert.message
        assert alert.exception_details["count"] == 150
        assert isinstance(alert.timestamp, datetime)

    def test_exception_monitor(self):
        """测试异常监控器"""
        if not hasattr(self, 'ExceptionMonitor'):
            pytest.skip("ExceptionMonitor not available")

        monitor = self.ExceptionMonitor()

        assert monitor is not None
        assert hasattr(monitor, 'patterns')
        assert hasattr(monitor, 'alert_rules')
        assert hasattr(monitor, 'exception_history')

        # 测试记录异常
        exception_info = {
            "type": "ValueError",
            "message": "Invalid input",
            "traceback": "Traceback...",
            "timestamp": datetime.now()
        }

        monitor.record_exception(exception_info)
        assert len(monitor.exception_history) > 0

        # 测试模式匹配
        pattern = self.ExceptionPattern(
            name="value_error",
            pattern=r"ValueError: .*",
            severity=self.ExceptionSeverity.MEDIUM,
            category=self.ExceptionCategory.APPLICATION
        )
        monitor.add_pattern(pattern)

        matched = monitor.match_patterns("ValueError: Invalid input")
        assert len(matched) > 0
        assert matched[0].name == "value_error"

    def test_exception_alert_notifier(self):
        """测试异常告警通知器"""
        if not hasattr(self, 'ExceptionAlertNotifier'):
            pytest.skip("ExceptionAlertNotifier not available")

        notifier = self.ExceptionAlertNotifier(
            email_recipients=["admin@example.com"],
            slack_webhook="https://hooks.slack.com/...",
            alert_threshold=self.ExceptionSeverity.HIGH
        )

        assert notifier.email_recipients == ["admin@example.com"]
        assert notifier.slack_webhook == "https://hooks.slack.com/..."
        assert notifier.alert_threshold == self.ExceptionSeverity.HIGH

        # 测试发送告警（mock）
        with patch('smtplib.SMTP') as mock_smtp, \
             patch('requests.post') as mock_post:

            mock_response = Mock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            alert = self.ExceptionAlert(
                rule_name="test",
                severity=self.ExceptionSeverity.CRITICAL,
                message="Critical exception detected"
            )

            result = notifier.notify(alert)
            assert result is True

    def test_exception_dashboard(self):
        """测试异常仪表板"""
        if not hasattr(self, 'ExceptionDashboard'):
            pytest.skip("ExceptionDashboard not available")

        dashboard = self.ExceptionDashboard()

        assert dashboard is not None

        # 测试添加异常数据
        exception_data = {
            "type": "ConnectionError",
            "count": 25,
            "severity": self.ExceptionSeverity.HIGH,
            "category": self.ExceptionCategory.NETWORK
        }

        dashboard.add_exception_data(exception_data)

        # 测试生成报告
        report = dashboard.generate_report()
        assert isinstance(report, dict)
        assert "total_exceptions" in report
        assert "severity_breakdown" in report

    def test_exception_analytics(self):
        """测试异常分析器"""
        if not hasattr(self, 'ExceptionAnalytics'):
            pytest.skip("ExceptionAnalytics not available")

        analytics = self.ExceptionAnalytics()

        assert analytics is not None

        # 测试分析异常趋势
        historical_data = [
            {"date": "2024-01-01", "count": 10},
            {"date": "2024-01-02", "count": 15},
            {"date": "2024-01-03", "count": 8},
            {"date": "2024-01-04", "count": 22},
            {"date": "2024-01-05", "count": 18}
        ]

        trend = analytics.analyze_trend(historical_data)
        assert isinstance(trend, dict)
        assert "trend_direction" in trend
        assert "average_count" in trend

        # 测试预测
        prediction = analytics.predict_future_exceptions(historical_data, days=3)
        assert isinstance(prediction, dict)
        assert "predicted_counts" in prediction

    def test_exception_monitor_integration(self):
        """测试异常监控器集成"""
        if not all(hasattr(self, cls) for cls in [
            'ExceptionMonitor', 'ExceptionAlertRule', 'ExceptionAlertNotifier'
        ]):
            pytest.skip("Required components not available")

        monitor = self.ExceptionMonitor()
        notifier = self.ExceptionAlertNotifier()
        rule = self.ExceptionAlertRule(
            name="critical_errors",
            condition="error_count > 50",
            severity=self.ExceptionSeverity.CRITICAL
        )

        monitor.add_alert_rule(rule)
        monitor.set_notifier(notifier)

        # 模拟大量异常
        for i in range(60):
            exception_info = {
                "type": "CriticalError",
                "message": f"Critical error {i}",
                "timestamp": datetime.now()
            }
            monitor.record_exception(exception_info)

        # 检查是否触发了告警
        alerts = monitor.check_alerts()
        assert len(alerts) > 0

    def test_exception_monitor_concurrent(self):
        """测试异常监控器的并发处理"""
        if not hasattr(self, 'ExceptionMonitor'):
            pytest.skip("ExceptionMonitor not available")

        monitor = self.ExceptionMonitor()
        results = []
        errors = []

        def worker_thread(thread_id):
            """工作线程"""
            try:
                for i in range(100):
                    exception_info = {
                        "type": f"ThreadError_{thread_id}",
                        "message": f"Error {i} from thread {thread_id}",
                        "timestamp": datetime.now()
                    }
                    monitor.record_exception(exception_info)
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker_thread, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30.0)

        # 验证结果
        assert len(results) == 5  # 所有线程都完成了
        assert len(errors) == 0   # 没有错误
        assert len(monitor.exception_history) == 500  # 5线程 * 100异常

    def test_exception_monitor_error_handling(self):
        """测试异常监控器的错误处理"""
        if not hasattr(self, 'ExceptionMonitor'):
            pytest.skip("ExceptionMonitor not available")

        monitor = self.ExceptionMonitor()

        # 测试无效异常记录
        monitor.record_exception(None)  # 应该不会崩溃
        monitor.record_exception({})    # 空字典

        # 测试无效模式
        monitor.add_pattern(None)  # 应该不会崩溃

        # 测试无效规则
        monitor.add_alert_rule(None)  # 应该不会崩溃

        # 监控器应该仍然正常工作
        assert monitor.exception_history is not None
        assert monitor.patterns is not None
        assert monitor.alert_rules is not None


if __name__ == '__main__':
    pytest.main([__file__])

