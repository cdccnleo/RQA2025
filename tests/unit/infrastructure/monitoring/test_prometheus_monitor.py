import pytest
from unittest.mock import patch, MagicMock
from src.infrastructure.monitoring.prometheus_monitor import PrometheusMonitor

class TestPrometheusMonitor:
    @pytest.fixture
    def monitor(self):
        return PrometheusMonitor(gateway_url="http://mock-gateway:9091")

    def test_metric_registration(self, monitor):
        # 检查常用指标注册
        assert hasattr(monitor, 'alert_count')
        assert hasattr(monitor, 'experiment_duration')
        # 指标名在注册表中
        assert 'alerts_total' in monitor.registry._names_to_collectors
        assert 'chaos_experiment_duration_seconds' in monitor.registry._names_to_collectors

    @patch('src.infrastructure.monitoring.prometheus_monitor.push_to_gateway')
    def test_alert_and_push(self, mock_push, monitor):
        # 发送告警
        monitor.alert('test_source', 'test message', severity='critical')
        # 指标自增
        value = monitor.alert_count.labels(severity='critical', source='test_source')._value.get()
        assert value == 1
        # 推送被调用
        mock_push.assert_called_once()

    def test_send_metric_dynamic(self, monitor):
        # 动态发送新指标
        monitor.send_metric('custom_metric', 42.0, labels={'tag': 'A'})
        metric = monitor.registry._names_to_collectors['custom_metric']
        assert metric.labels(tag='A')._value.get() == 42.0
        # 再次发送同名指标
        monitor.send_metric('custom_metric', 100.0, labels={'tag': 'A'})
        assert metric.labels(tag='A')._value.get() == 100.0

    @patch('src.infrastructure.monitoring.prometheus_monitor.push_to_gateway', side_effect=Exception('push error'))
    def test_push_metrics_exception(self, mock_push, monitor):
        # 异常分支覆盖
        monitor.push_metrics()  # 不抛异常
        mock_push.assert_called_once()

    @patch('prometheus_client.delete_from_gateway')
    def test_cleanup_success(self, mock_delete, monitor):
        monitor.cleanup()
        mock_delete.assert_called_once_with(monitor.gateway_url, job='rqa_monitoring')

    @patch('prometheus_client.delete_from_gateway', side_effect=Exception('delete error'))
    def test_cleanup_exception(self, mock_delete, monitor):
        monitor.cleanup()  # 不抛异常
        mock_delete.assert_called_once()

    @patch('prometheus_client.delete_from_gateway', side_effect=ImportError)
    def test_cleanup_import_error(self, mock_delete, monitor):
        monitor.cleanup()  # 不抛异常
        mock_delete.assert_called_once() 