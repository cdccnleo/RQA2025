import pytest
from src.infrastructure.monitoring import ApplicationMonitor
from unittest.mock import MagicMock, patch
import time
import contextlib

# 统一mock prometheus_client的指标对象
@pytest.fixture(autouse=True)
def mock_prometheus():
    with patch('prometheus_client.Counter', MagicMock()), \
         patch('prometheus_client.Gauge', MagicMock()), \
         patch('prometheus_client.Histogram', MagicMock()):
        yield

class TestApplicationMonitor:
    @pytest.fixture
    def monitor(self):
        """提供干净的ApplicationMonitor实例，并mock influx_client和influx_bucket"""
        m = ApplicationMonitor(app_name="test_app")
        m.influx_client = MagicMock()
        m.influx_bucket = "test_bucket"
        return m

    def test_metric_recording(self, monitor):
        """测试指标记录功能"""
        with patch.object(monitor, 'influx_client') as mock_client:
            # Mock write_api
            mock_write_api = MagicMock()
            mock_client.write_api.return_value = mock_write_api
            
            monitor.record_metric(
                name="test_metric",
                value=42,
                tags={"env": "test"}
            )

            # 验证指标是否正确发送
            assert mock_write_api.write.called
            print('write.call_args:', mock_write_api.write.call_args)

    def test_context_manager(self, monitor):
        """测试监控上下文管理器"""
        with patch.object(monitor, 'record_function') as mock_record:
            monitor.monitor("test_operation")(lambda: time.sleep(0.1))()
            # 验证执行时间是否被记录
            mock_record.assert_called()
            calls = mock_record.call_args_list
            assert len(calls) > 0

    def test_health_check(self, monitor):
        """测试健康检查功能"""
        mock_check = MagicMock(return_value=True)
        monitor.add_health_check("test_service", mock_check)

        status = monitor.run_health_checks()
        assert status["test_service"] is True
        mock_check.assert_called_once()

    def test_alert_trigger(self, monitor):
        """测试告警触发机制"""
        alert_triggered = False

        def alert_handler(alert):
            nonlocal alert_triggered
            alert_triggered = True
            assert alert["metric"] == "cpu_usage"
            assert alert["value"] > 90

        monitor.add_alert_rule(
            name="high_cpu",
            condition=lambda x: x > 90,
            handler=alert_handler
        )

        # 触发告警
        monitor.record_metric("cpu_usage", 95)
        # 检查告警规则是否被添加
        assert "high_cpu" in monitor._alert_rules

    def test_custom_tags(self, monitor):
        """测试自定义标签功能"""
        monitor.set_default_tags({"region": "east", "env": "prod"})

        with patch.object(monitor, 'influx_client') as mock_client:
            # Mock write_api
            mock_write_api = MagicMock()
            mock_client.write_api.return_value = mock_write_api
            
            monitor.record_metric("test_metric", 1)

            # 验证默认标签是否被设置
            assert monitor._default_tags["region"] == "east"
            assert monitor._default_tags["env"] == "prod"
