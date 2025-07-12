import unittest
from unittest.mock import MagicMock
from src.infrastructure.monitoring import MonitoringSystem

class TestMonitoringSystem(unittest.TestCase):
    def setUp(self):
        self.monitor = MonitoringSystem()
        self.monitor.alert_service = MagicMock()

    def test_metric_collection(self):
        """测试指标收集功能"""
        self.monitor.record_metric("cpu_usage", 75)
        self.assertEqual(self.monitor.get_metric("cpu_usage"), 75)

    def test_alert_triggering(self):
        """测试告警触发机制"""
        self.monitor.set_threshold("memory", 90)
        self.monitor.record_metric("memory", 95)
        self.monitor.alert_service.send_alert.assert_called_once()

    def test_performance_aggregation(self):
        """测试性能数据聚合"""
        for i in range(5):
            self.monitor.record_metric("latency", i*10)
        stats = self.monitor.get_stats("latency")
        self.assertEqual(stats["avg"], 20)
        self.assertEqual(stats["max"], 40)

if __name__ == '__main__':
    unittest.main()
