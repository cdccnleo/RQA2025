"""
MonitoringDashboard最小测试套件

只测试最基本的功能，避免线程相关问题
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest

# 只导入必要的常量和枚举
from src.infrastructure.health.services.monitoring_dashboard import (
    MetricType,
    AlertSeverity,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_MAX_METRICS,
    DEFAULT_ALERT_TIMEOUT
)


class TestMonitoringDashboardMinimal:
    """MonitoringDashboard最小测试"""

    def test_constants(self):
        """测试常量定义"""
        assert DEFAULT_RETENTION_DAYS == 30
        assert DEFAULT_MAX_METRICS == 10000
        assert DEFAULT_ALERT_TIMEOUT == 300.0

    def test_enums(self):
        """测试枚举定义"""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"

        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"
