"""
测试告警数据类

验证AlertRule、Alert、PerformanceMetrics、TestExecutionInfo等数据类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from datetime import datetime, timedelta
from src.infrastructure.resource.models.alert_dataclasses import (
    AlertRule, Alert, PerformanceMetrics, TestExecutionInfo
)
from src.infrastructure.resource.models.alert_enums import AlertType, AlertLevel


class TestAlertRule:
    """测试AlertRule数据类"""

    def test_alert_rule_creation(self):
        """测试告警规则创建"""
        rule = AlertRule(
            name="CPU高负载",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.WARNING,
            condition="cpu_usage > threshold",
            threshold=80.0,
            enabled=True,
            cooldown=300
        )

        assert rule.name == "CPU高负载"
        assert rule.alert_type == AlertType.SYSTEM_ERROR
        assert rule.alert_level == AlertLevel.WARNING
        assert rule.condition == "cpu_usage > threshold"
        assert rule.threshold == 80.0
        assert rule.enabled is True
        assert rule.cooldown == 300
        assert rule.last_triggered is None

    def test_alert_rule_defaults(self):
        """测试告警规则默认值"""
        rule = AlertRule(
            name="测试规则",
            alert_type=AlertType.TEST_FAILURE,
            alert_level=AlertLevel.ERROR,
            condition="test_success_rate < threshold",
            threshold=90.0
        )

        assert rule.enabled is True
        assert rule.cooldown == 300
        assert rule.last_triggered is None


class TestAlert:
    """测试Alert数据类"""

    def test_alert_creation(self):
        """测试告警创建"""
        timestamp = datetime.now()
        alert = Alert(
            id="alert_001",
            alert_type=AlertType.SYSTEM_ERROR,
            alert_level=AlertLevel.CRITICAL,
            message="系统错误",
            details={"cpu": 95.0, "memory": 90.0},
            timestamp=timestamp,
            source="performance_monitor",
            resolved=False
        )

        assert alert.id == "alert_001"
        assert alert.alert_type == AlertType.SYSTEM_ERROR
        assert alert.alert_level == AlertLevel.CRITICAL
        assert alert.message == "系统错误"
        assert alert.details == {"cpu": 95.0, "memory": 90.0}
        assert alert.timestamp == timestamp
        assert alert.source == "performance_monitor"
        assert alert.resolved is False
        assert alert.resolved_at is None

    def test_alert_resolution(self):
        """测试告警解决"""
        alert = Alert(
            id="alert_002",
            alert_type=AlertType.NETWORK_ISSUE,
            alert_level=AlertLevel.WARNING,
            message="网络延迟",
            details={"latency": 150.0},
            timestamp=datetime.now(),
            source="network_monitor",
            resolved=False
        )

        # 模拟解决告警
        alert.resolved = True
        alert.resolved_at = datetime.now()

        assert alert.resolved is True
        assert alert.resolved_at is not None


class TestPerformanceMetrics:
    """测试PerformanceMetrics数据类"""

    def test_performance_metrics_creation(self):
        """测试性能指标创建"""
        timestamp = datetime.now()
        metrics = PerformanceMetrics(
            cpu_usage=75.5,
            memory_usage=82.3,
            disk_usage=65.0,
            network_latency=25.0,
            test_execution_time=120.5,
            test_success_rate=95.0,
            active_threads=8,
            timestamp=timestamp
        )

        assert metrics.cpu_usage == 75.5
        assert metrics.memory_usage == 82.3
        assert metrics.disk_usage == 65.0
        assert metrics.network_latency == 25.0
        assert metrics.test_execution_time == 120.5
        assert metrics.test_success_rate == 95.0
        assert metrics.active_threads == 8
        assert metrics.timestamp == timestamp

    def test_performance_metrics_defaults(self):
        """测试性能指标默认值"""
        metrics = PerformanceMetrics()

        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.disk_usage == 0.0
        assert metrics.network_latency == 0.0
        assert metrics.test_execution_time == 0.0
        assert metrics.test_success_rate == 0.0
        assert metrics.active_threads == 0
        assert isinstance(metrics.timestamp, datetime)


class TestTestExecutionInfo:
    """测试TestExecutionInfo数据类"""

    def test_test_execution_info_creation(self):
        """测试测试执行信息创建"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=120)
        perf_metrics = PerformanceMetrics(cpu_usage=50.0, memory_usage=60.0)

        test_info = TestExecutionInfo(
            test_id="test_001",
            test_name="用户登录测试",
            start_time=start_time,
            end_time=end_time,
            status="completed",
            execution_time=120.0,
            error_message=None,
            performance_metrics=perf_metrics
        )

        assert test_info.test_id == "test_001"
        assert test_info.test_name == "用户登录测试"
        assert test_info.start_time == start_time
        assert test_info.end_time == end_time
        assert test_info.status == "completed"
        assert test_info.execution_time == 120.0
        assert test_info.error_message is None
        assert test_info.performance_metrics == perf_metrics

    def test_test_execution_info_running(self):
        """测试运行中的测试执行信息"""
        start_time = datetime.now()

        test_info = TestExecutionInfo(
            test_id="test_002",
            test_name="数据处理测试",
            start_time=start_time
        )

        assert test_info.status == "running"
        assert test_info.end_time is None
        assert test_info.execution_time is None
        assert test_info.error_message is None
        assert test_info.performance_metrics is None

    def test_test_execution_info_failed(self):
        """测试失败的测试执行信息"""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)

        test_info = TestExecutionInfo(
            test_id="test_003",
            test_name="API测试",
            start_time=start_time,
            end_time=end_time,
            status="failed",
            execution_time=30.0,
            error_message="Connection timeout"
        )

        assert test_info.status == "failed"
        assert test_info.execution_time == 30.0
        assert test_info.error_message == "Connection timeout"
