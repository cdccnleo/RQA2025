import unittest
from unittest.mock import patch, MagicMock
import time
import psutil
from src.infrastructure.monitoring import (
    ApplicationMonitor,
    PerformanceMonitor,
    SystemMonitor
)

class TestMonitoringSystems(unittest.TestCase):
    """统一后的基础设施监控测试"""

    def setUp(self):
        # 初始化各监控器
        self.app_monitor = ApplicationMonitor()
        self.perf_monitor = PerformanceMonitor()
        self.sys_monitor = SystemMonitor()

        # 模拟数据
        self.sample_metrics = {
            'cpu': 25.5,
            'memory': 45.2,
            'disk': 10.8
        }

        # 配置模拟
        self.app_monitor._get_metrics = MagicMock(return_value=self.sample_metrics)
        self.perf_monitor._collect = MagicMock(return_value=self.sample_metrics)
        self.sys_monitor._check = MagicMock(return_value=self.sample_metrics)

    def test_basic_monitoring(self):
        """测试基础监控功能"""
        metrics = self.app_monitor.collect()
        self.assertEqual(metrics['cpu'], 25.5)
        self.assertEqual(metrics['memory'], 45.2)

    # 从test_application_monitor.py合并的测试
    def test_threshold_alert(self):
        """测试阈值告警触发"""
        # 设置CPU阈值告警
        self.app_monitor.set_threshold('cpu', warning=80, critical=90)

        # 模拟高负载
        with patch.object(self.app_monitor, '_get_metrics') as mock_metrics:
            mock_metrics.return_value = {'cpu': 95.0, 'memory': 45.2}
            alerts = self.app_monitor.check()
            self.assertEqual(alerts['cpu']['level'], 'CRITICAL')

    # 从test_performance_monitor.py合并的测试
    def test_performance_tracking(self):
        """测试性能指标跟踪"""
        # 模拟性能数据收集
        with patch('psutil.cpu_percent') as mock_cpu:
            mock_cpu.return_value = 30.0
            metrics = self.perf_monitor.collect_performance()
            self.assertAlmostEqual(metrics['cpu'], 30.0, delta=0.1)

    # 从test_system_monitor.py合并的测试
    @patch('psutil.disk_usage')
    def test_disk_monitoring(self, mock_disk):
        """测试磁盘空间监控"""
        # 模拟磁盘使用情况
        mock_disk.return_value = MagicMock(percent=75.0)
        metrics = self.sys_monitor.check_disk()
        self.assertEqual(metrics['disk'], 75.0)

    # 新增告警抑制测试
    def test_alert_suppression(self):
        """测试重复告警抑制"""
        # 设置内存阈值
        self.app_monitor.set_threshold('memory', warning=50, critical=70)

        # 第一次触发告警
        with patch.object(self.app_monitor, '_get_metrics') as mock_metrics:
            mock_metrics.return_value = {'memory': 75.0}
            alerts = self.app_monitor.check()
            self.assertEqual(alerts['memory']['level'], 'CRITICAL')

        # 短时间内相同告警应被抑制
        with patch.object(self.app_monitor, '_get_metrics') as mock_metrics:
            mock_metrics.return_value = {'memory': 76.0}
            alerts = self.app_monitor.check()
            self.assertNotIn('memory', alerts)

    # 新增性能基准测试
    def test_monitoring_overhead(self):
        """测试监控系统自身性能开销"""
        # 测试100次收集的性能
        start = time.time()
        for _ in range(100):
            _ = self.app_monitor.collect()
        elapsed = time.time() - start

        # 单次收集应小于5ms
        self.assertLess(elapsed/100, 0.005)

if __name__ == '__main__':
    unittest.main()
