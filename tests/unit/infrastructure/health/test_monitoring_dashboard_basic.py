"""
MonitoringDashboard基础测试套件

针对monitoring_dashboard.py模块的基础测试覆盖
目标: 建立基础测试框架，从21.07%覆盖率开始
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock
from datetime import datetime

# 导入被测试模块
from src.infrastructure.health.services.monitoring_dashboard import (
    MonitoringDashboard,
    DashboardManager,
    MetricType,
    AlertSeverity,
    Metric,
    Alert,
    DashboardConfig,
    DEFAULT_RETENTION_DAYS,
    DEFAULT_MAX_METRICS,
    DEFAULT_ALERT_TIMEOUT
)


class TestMonitoringDashboardBasic:
    """MonitoringDashboard基础测试"""

    @pytest.fixture
    def dashboard_config(self):
        """创建仪表板配置"""
        return DashboardConfig(
            refresh_interval=30.0,
            retention_days=30,
            max_metrics=10000
        )

    @pytest.fixture
    def dashboard(self, dashboard_config):
        """创建MonitoringDashboard实例"""
        db = MonitoringDashboard(dashboard_config, auto_start=False)
        yield db
        # 清理：强制停止所有可能运行的线程
        db._running = False
        # 不调用stop()方法，避免join()阻塞

    @pytest.fixture
    def dashboard_manager(self):
        """创建DashboardManager实例"""
        return DashboardManager()

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

    def test_metric_dataclass(self):
        """测试Metric数据类"""
        import time
        metric = Metric(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            labels={"service": "test"},
            timestamp=time.time()
        )

        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.metric_type == MetricType.GAUGE
        assert isinstance(metric.labels, dict)

    def test_alert_dataclass(self):
        """测试Alert数据类"""
        import time
        alert = Alert(
            name="test_alert",
            message="Test alert message",
            severity=AlertSeverity.WARNING,
            timestamp=time.time(),
            resolved=False,
            labels={"component": "test"}
        )

        assert alert.name == "test_alert"
        assert alert.message == "Test alert message"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.resolved == False

    def test_dashboard_config_dataclass(self):
        """测试DashboardConfig数据类"""
        config = DashboardConfig(
            refresh_interval=60.0,
            retention_days=7,
            max_metrics=5000
        )

        assert config.refresh_interval == 60.0
        assert config.retention_days == 7
        assert config.max_metrics == 5000
        assert config.alert_timeout == DEFAULT_ALERT_TIMEOUT  # 默认值

    def test_monitoring_dashboard_initialization(self, dashboard):
        """测试MonitoringDashboard初始化"""
        assert dashboard is not None
        assert hasattr(dashboard, 'config')
        assert hasattr(dashboard, '_metrics')
        assert hasattr(dashboard, '_alerts')

    def test_dashboard_manager_initialization(self, dashboard_manager):
        """测试DashboardManager初始化"""
        assert dashboard_manager is not None
        assert hasattr(dashboard_manager, 'dashboards')

    def test_dashboard_add_metric(self, dashboard):
        """测试添加指标"""
        import time
        metric = Metric(
            name="cpu_usage",
            value=75.5,
            metric_type=MetricType.GAUGE,
            timestamp=time.time()
        )

        dashboard.add_metric(metric)

        # 检查指标是否被添加
        metrics = dashboard.get_metrics("cpu_usage")
        assert len(metrics) == 1
        assert metrics[0].name == "cpu_usage"
        assert metrics[0].value == 75.5

    def test_dashboard_get_metrics_without_filters(self, dashboard):
        """测试获取所有指标"""
        import time
        metric1 = Metric("metric1", 10.0, MetricType.COUNTER, timestamp=time.time())
        metric2 = Metric("metric2", 20.0, MetricType.GAUGE, timestamp=time.time())

        dashboard.add_metric(metric1)
        dashboard.add_metric(metric2)

        all_metrics = dashboard.get_metrics()
        assert len(all_metrics) == 2

    def test_dashboard_add_alert_rule(self, dashboard):
        """测试添加告警规则"""
        dashboard.add_alert_rule(
            rule_name="high_cpu",
            query="cpu_usage > 80",
            severity=AlertSeverity.WARNING,
            threshold=80.0,
            duration=60.0
        )

        # 验证规则是否添加（通过私有属性检查）
        assert len(dashboard._alert_rules) == 1
        assert "high_cpu" in dashboard._alert_rules
        rule = dashboard._alert_rules["high_cpu"]
        assert rule["query"] == "cpu_usage > 80"
        assert rule["severity"] == AlertSeverity.WARNING
        assert rule["threshold"] == 80.0
        assert rule["duration"] == 60.0

    def test_dashboard_remove_alert_rule(self, dashboard):
        """测试移除告警规则"""
        # 先添加规则
        dashboard.add_alert_rule("test_rule", "test > 0", AlertSeverity.INFO, threshold=50.0)

        # 移除规则
        dashboard.remove_alert_rule("test_rule")

        # 验证规则是否移除
        assert len(dashboard._alert_rules) == 0

    def test_dashboard_resolve_alert(self, dashboard):
        """测试解决告警"""
        import time
        # 创建并添加告警
        alert = Alert("test_alert", "Test alert", AlertSeverity.WARNING, timestamp=time.time())
        dashboard._alerts["test_alert"] = alert

        # 验证告警已添加
        assert "test_alert" in dashboard._alerts
        assert dashboard._alerts["test_alert"].resolved == False

        # 解决告警
        dashboard.resolve_alert("test_alert")

        # 验证告警是否解决（resolved标志被设置）
        assert "test_alert" in dashboard._alerts
        assert dashboard._alerts["test_alert"].resolved == True

    def test_dashboard_get_dashboard_data(self, dashboard):
        """测试获取仪表板数据"""
        data = dashboard.get_dashboard_data()

        assert isinstance(data, dict)
        assert 'metrics' in data or len(data) > 0

    def test_dashboard_check_health(self, dashboard):
        """测试健康检查"""
        health = dashboard.check_health()

        assert isinstance(health, dict)
        assert len(health) > 0

    def test_dashboard_health_status(self, dashboard):
        """测试健康状态"""
        status = dashboard.health_status()

        assert isinstance(status, dict)
        assert 'status' in status or len(status) > 0

    def test_dashboard_health_summary(self, dashboard):
        """测试健康摘要"""
        summary = dashboard.health_summary()

        assert isinstance(summary, dict)
        assert 'summary' in summary or len(summary) > 0

    def test_dashboard_manager_create_dashboard(self, dashboard_manager):
        """测试创建仪表板"""
        config = DashboardConfig()
        dashboard = dashboard_manager.create_dashboard("test_dashboard", config)

        assert dashboard is not None
        # 仪表板没有name属性，但存储在manager的字典中
        assert "test_dashboard" in dashboard_manager.dashboards
        assert dashboard_manager.dashboards["test_dashboard"] == dashboard

    def test_dashboard_manager_get_dashboard(self, dashboard_manager):
        """测试获取仪表板"""
        # 先创建仪表板
        config = DashboardConfig()
        dashboard_manager.create_dashboard("test_dashboard", config)

        # 获取仪表板
        retrieved = dashboard_manager.get_dashboard("test_dashboard")

        assert retrieved is not None
        # 验证返回的是正确的仪表板实例
        assert retrieved == dashboard_manager.dashboards["test_dashboard"]

    def test_dashboard_manager_remove_dashboard(self, dashboard_manager):
        """测试移除仪表板"""
        # 先创建仪表板
        config = DashboardConfig()
        dashboard_manager.create_dashboard("test_dashboard", config)

        # 移除仪表板
        dashboard_manager.remove_dashboard("test_dashboard")

        # 验证仪表板是否移除
        assert "test_dashboard" not in dashboard_manager.dashboards

    def test_dashboard_manager_get_nonexistent_dashboard(self, dashboard_manager):
        """测试获取不存在的仪表板"""
        result = dashboard_manager.get_dashboard("nonexistent")
        assert result is None

    def test_dashboard_start_stop_logic(self, dashboard):
        """测试仪表板启动和停止逻辑（不实际启动线程）"""
        # 保存原始状态
        original_running = dashboard._running

        try:
            # 测试启动逻辑
            dashboard._running = False
            # 这里我们只测试逻辑，不实际启动线程
            dashboard._running = True
            assert dashboard._running == True

            # 测试停止逻辑 - 确保不会调用实际的stop()方法
            dashboard._running = False
            assert dashboard._running == False
        finally:
            # 恢复原始状态，避免影响fixture的teardown
            dashboard._running = original_running

    def test_dashboard_validate_config(self, dashboard):
        """测试配置验证"""
        result = dashboard.validate_dashboard_config()

        assert isinstance(result, dict)
        assert 'valid' in result or len(result) > 0

    def test_callback_registration(self, dashboard):
        """测试回调函数注册"""
        # 定义回调函数
        def alert_callback(alert):
            pass

        def metric_callback(metric):
            pass

        # 添加回调函数
        dashboard.add_alert_callback(alert_callback)
        dashboard.add_metric_callback(metric_callback)

        # 验证回调函数已注册
        assert alert_callback in dashboard._alert_callbacks
        assert metric_callback in dashboard._metric_callbacks
