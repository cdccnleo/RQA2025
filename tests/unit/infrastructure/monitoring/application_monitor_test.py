import unittest
import time
from unittest.mock import patch, MagicMock
from src.infrastructure.monitoring import ApplicationMonitor

class TestApplicationMonitor(unittest.TestCase):
    """应用监控系统单元测试"""

    def setUp(self):
        self.monitor = ApplicationMonitor(app_name="test_app")
        self.mock_client = MagicMock()
        self.monitor.influx_client = self.mock_client

    def test_metric_recording(self):
        """测试指标记录功能"""
        test_metric = {
            'name': 'order_processing_time',
            'value': 0.15,
            'tags': {'strategy': 'momentum'}
        }

        self.monitor.record_metric(**test_metric)

        # 验证指标格式正确
        self.mock_client.write_points.assert_called_once()
        call_args = self.mock_client.write_points.call_args[0][0]
        self.assertEqual(call_args[0]['measurement'], 'order_processing_time')
        self.assertEqual(call_args[0]['tags']['strategy'], 'momentum')

    def test_monitor_decorator(self):
        """测试监控装饰器"""
        @self.monitor.monitor("test_operation")
        def test_func():
            time.sleep(0.01)
            return "result"

        result = test_func()

        # 验证函数执行和结果
        self.assertEqual(result, "result")

        # 验证指标记录
        self.mock_client.write_points.assert_called()
        call_args = self.mock_client.write_points.call_args[0][0]
        self.assertEqual(call_args[0]['measurement'], 'test_operation_duration')

    def test_health_check(self):
        """测试健康检查机制"""
        # 初始状态应为健康
        self.assertTrue(self.monitor.is_healthy())

        # 报告错误应改变健康状态
        self.monitor.report_error("database_connection")
        self.assertFalse(self.monitor.is_healthy())

        # 重置后应恢复健康
        self.monitor.reset_health()
        self.assertTrue(self.monitor.is_healthy())

    @patch('src.infrastructure.monitoring.AlertManager.send_alert')
    def test_threshold_alert(self, mock_send_alert):
        """测试阈值告警触发"""
        # 设置阈值
        self.monitor.set_threshold(
            metric='cpu_usage',
            warning=80,
            critical=90
        )

        # 触发警告
        self.monitor.check_threshold('cpu_usage', 85)
        mock_send_alert.assert_called_with(
            "Warning: cpu_usage exceeds 80% (current: 85%)",
            level='warning'
        )

        # 触发严重告警
        self.monitor.check_threshold('cpu_usage', 95)
        mock_send_alert.assert_called_with(
            "Critical: cpu_usage exceeds 90% (current: 95%)",
            level='critical'
        )

    def test_prometheus_integration(self):
        """测试Prometheus指标导出"""
        from prometheus_client import REGISTRY

        # 记录自定义指标
        self.monitor.record_prometheus_metric(
            name='orders_processed',
            description='Total orders processed',
            value=100,
            labels={'type': 'market'}
        )

        # 验证指标存在
        metric = REGISTRY.get_sample_value(
            'test_app_orders_processed_total',
            labels={'type': 'market'}
        )
        self.assertEqual(metric, 100)

if __name__ == '__main__':
    unittest.main()
