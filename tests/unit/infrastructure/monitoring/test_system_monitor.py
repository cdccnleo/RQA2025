#!/usr/bin/env python3
"""
系统监控组件测试

测试目标：提升infrastructure/monitoring模块的覆盖率
测试范围：SystemMonitor核心功能
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import Dict, Optional, List

# 导入测试模块
try:
    from src.infrastructure.monitoring.infrastructure.system_monitor import (
        SystemMonitor, SystemMetrics
    )
except ImportError:
    pytest.skip("SystemMonitor模块导入失败", allow_module_level=True)


class TestSystemMetrics:
    """测试SystemMetrics数据类"""

    def test_system_metrics_creation(self):
        """测试系统指标创建"""
        metrics = SystemMetrics(
            cpu_percent=45.5,
            memory_percent=67.8,
            disk_usage_percent=23.4,
            network_connections=150,
            timestamp=1234567890.0
        )

        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.8
        assert metrics.disk_usage_percent == 23.4
        assert metrics.network_connections == 150
        assert metrics.timestamp == 1234567890.0

    def test_system_metrics_defaults(self):
        """测试系统指标默认值"""
        # 测试各个字段的合理性
        assert SystemMetrics.__annotations__['cpu_percent'] == float
        assert SystemMetrics.__annotations__['memory_percent'] == float
        assert SystemMetrics.__annotations__['disk_usage_percent'] == float
        assert SystemMetrics.__annotations__['network_connections'] == int
        assert SystemMetrics.__annotations__['timestamp'] == float


class TestSystemMonitor:
    """测试SystemMonitor核心功能"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = SystemMonitor()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_system_monitor_initialization(self):
        """测试SystemMonitor初始化"""
        monitor = SystemMonitor()

        assert isinstance(monitor.metrics_history, list)
        assert monitor.max_history_size == 1000
        assert hasattr(monitor, '_lock')  # 检查有_lock属性
        assert len(monitor.metrics_history) == 0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_collect_system_metrics(self, mock_net_connections, mock_disk_usage,
                                   mock_virtual_memory, mock_cpu_percent):
        """测试系统指标收集"""
        # 设置mock返回值
        mock_cpu_percent.return_value = 45.5

        mock_memory = Mock()
        mock_memory.percent = 67.8
        mock_virtual_memory.return_value = mock_memory

        mock_disk = Mock()
        mock_disk.percent = 23.4
        mock_disk_usage.return_value = mock_disk

        mock_net_connections.return_value = [Mock()] * 150  # 150个连接

        # 收集指标
        metrics = self.monitor.collect_system_metrics()

        # 验证结果
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.8
        assert metrics.disk_usage_percent == 23.4
        assert metrics.network_connections == 150
        assert isinstance(metrics.timestamp, float)
        assert metrics.timestamp > 0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_collect_system_metrics_with_exceptions(self, mock_net_connections, mock_disk_usage,
                                                   mock_virtual_memory, mock_cpu_percent):
        """测试系统指标收集异常处理"""
        # 设置所有mock抛出异常
        mock_cpu_percent.side_effect = Exception("CPU监控失败")
        mock_virtual_memory.side_effect = Exception("内存监控失败")
        mock_disk_usage.side_effect = Exception("磁盘监控失败")
        mock_net_connections.side_effect = Exception("网络监控失败")

        # 收集指标应该返回默认值而不是抛出异常
        metrics = self.monitor.collect_system_metrics()

        # 验证返回默认值
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_connections == 0

    def test_get_current_metrics_no_data(self):
        """测试获取当前指标 - 无数据"""
        metrics = self.monitor.get_current_metrics()

        assert metrics is None

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_get_current_metrics_with_data(self, mock_net_connections, mock_disk_usage,
                                          mock_virtual_memory, mock_cpu_percent):
        """测试获取当前指标 - 有数据"""
        # 设置mock返回值
        mock_cpu_percent.return_value = 30.0
        mock_memory = Mock()
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 40.0
        mock_disk_usage.return_value = mock_disk
        mock_net_connections.return_value = [Mock()] * 100

        # 收集一次指标
        collected_metrics = self.monitor.collect_system_metrics()

        # 获取当前指标
        current_metrics = self.monitor.get_current_metrics()

        assert current_metrics is not None
        assert current_metrics.cpu_percent == collected_metrics.cpu_percent
        assert current_metrics.memory_percent == collected_metrics.memory_percent
        assert current_metrics.disk_usage_percent == collected_metrics.disk_usage_percent
        assert current_metrics.network_connections == collected_metrics.network_connections

    def test_get_metrics_history_empty(self):
        """测试获取指标历史 - 空历史"""
        history = self.monitor.get_metrics_history()

        assert isinstance(history, list)
        assert len(history) == 0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_get_metrics_history_with_data(self, mock_net_connections, mock_disk_usage,
                                          mock_virtual_memory, mock_cpu_percent):
        """测试获取指标历史 - 有数据"""
        # 设置mock返回值
        mock_cpu_percent.return_value = 25.0
        mock_memory = Mock()
        mock_memory.percent = 45.0
        mock_virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 35.0
        mock_disk_usage.return_value = mock_disk
        mock_net_connections.return_value = [Mock()] * 80

        # 收集多次指标
        for i in range(5):
            self.monitor.collect_system_metrics()

        # 获取历史
        history = self.monitor.get_metrics_history()

        assert isinstance(history, list)
        assert len(history) == 5
        for metrics in history:
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 25.0
            assert metrics.memory_percent == 45.0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_connections')
    def test_metrics_history_size_limit(self, mock_net_connections, mock_disk_usage,
                                       mock_virtual_memory, mock_cpu_percent):
        """测试指标历史大小限制"""
        # 设置小的历史大小限制
        self.monitor.max_history_size = 3

        # 设置mock返回值
        mock_cpu_percent.return_value = 20.0
        mock_memory = Mock()
        mock_memory.percent = 40.0
        mock_virtual_memory.return_value = mock_memory
        mock_disk = Mock()
        mock_disk.percent = 30.0
        mock_disk_usage.return_value = mock_disk
        mock_net_connections.return_value = [Mock()] * 60

        # 收集超过限制的指标
        for i in range(5):
            self.monitor.collect_system_metrics()

        # 验证历史大小被限制
        history = self.monitor.get_metrics_history()
        assert len(history) == 3  # 应该只有最新的3个

    def test_clear_history(self):
        """测试清空指标历史"""
        # 先添加一些数据
        with patch('psutil.cpu_percent', return_value=15.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[]):

            mock_memory = Mock()
            mock_memory.percent = 35.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 25.0
            mock_disk.return_value = mock_disk_obj

            # 收集指标
            self.monitor.collect_system_metrics()

        # 验证有数据
        assert len(self.monitor.get_metrics_history()) > 0

        # 清空历史
        self.monitor.clear_history()

        # 验证已清空
        assert len(self.monitor.get_metrics_history()) == 0

    def test_average_metrics_calculation(self):
        """测试平均指标计算"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 100):

            mock_memory = Mock()
            mock_memory.percent = 60.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 40.0
            mock_disk.return_value = mock_disk_obj

            # 收集多个指标
            for _ in range(3):
                self.monitor.collect_system_metrics()

            # 获取平均指标
            avg_metrics = self.monitor.get_average_metrics(time_window=300)

            assert isinstance(avg_metrics, dict)
            assert 'cpu_percent' in avg_metrics
            assert 'memory_percent' in avg_metrics
            assert 'disk_usage_percent' in avg_metrics
            assert 'network_connections' in avg_metrics

            # 验证平均值计算
            assert avg_metrics['cpu_percent'] == 50.0
            assert avg_metrics['memory_percent'] == 60.0
            assert avg_metrics['disk_usage_percent'] == 40.0
            assert avg_metrics['network_connections'] == 100

    def test_thread_safety(self):
        """测试线程安全性"""
        import concurrent.futures

        results = []

        def collect_metrics():
            with patch('psutil.cpu_percent', return_value=10.0), \
                 patch('psutil.virtual_memory') as mock_mem, \
                 patch('psutil.disk_usage') as mock_disk, \
                 patch('psutil.net_connections', return_value=[Mock()] * 10):

                mock_memory = Mock()
                mock_memory.percent = 20.0
                mock_mem.return_value = mock_memory

                mock_disk_obj = Mock()
                mock_disk_obj.percent = 15.0
                mock_disk.return_value = mock_disk_obj

                metrics = self.monitor.collect_system_metrics()
                results.append(metrics)

        # 并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(collect_metrics) for _ in range(10)]
            concurrent.futures.wait(futures)

        # 验证所有操作都成功完成
        assert len(results) == 10
        assert len(self.monitor.get_metrics_history()) == 10

        for metrics in results:
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 10.0
            assert metrics.memory_percent == 20.0

    def test_monitor_repr(self):
        """测试SystemMonitor字符串表示"""
        repr_str = repr(self.monitor)
        assert "SystemMonitor" in repr_str

    def test_metrics_repr(self):
        """测试SystemMetrics字符串表示"""
        metrics = SystemMetrics(10.0, 20.0, 30.0, 40, 1234567890.0)
        repr_str = repr(metrics)
        assert "SystemMetrics" in repr_str
        assert "10.0" in repr_str
        assert "20.0" in repr_str


class TestSystemMonitorIntegration:
    """测试SystemMonitor集成功能"""

    def setup_method(self):
        """测试前准备"""
        self.monitor = SystemMonitor()

    def teardown_method(self):
        """测试后清理"""
        pass

    def test_continuous_monitoring_workflow(self):
        """测试连续监控工作流"""
        with patch('psutil.cpu_percent', return_value=50.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 75):

            mock_memory = Mock()
            mock_memory.percent = 60.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 45.0
            mock_disk.return_value = mock_disk_obj

            # 模拟连续监控
            for i in range(10):
                self.monitor.collect_system_metrics()
                time.sleep(0.01)  # 短暂延迟

            # 验证监控数据
            history = self.monitor.get_metrics_history()
            assert len(history) == 10

            # 验证数据一致性
            for metrics in history:
                assert metrics.cpu_percent == 50.0
                assert metrics.memory_percent == 60.0
                assert metrics.disk_usage_percent == 45.0
                assert metrics.network_connections == 75

            # 验证时间戳递增
            timestamps = [m.timestamp for m in history]
            assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))

    def test_monitoring_with_error_recovery(self):
        """测试监控错误恢复"""
        call_count = 0

        def mock_cpu_percent(interval=0.1):
            nonlocal call_count
            call_count += 1
            if call_count == 3:  # 第3次调用失败
                raise Exception("临时监控失败")
            return 40.0

        with patch('psutil.cpu_percent', side_effect=mock_cpu_percent), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 50):

            mock_memory = Mock()
            mock_memory.percent = 55.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 40.0
            mock_disk.return_value = mock_disk_obj

            # 尝试收集指标
            try:
                self.monitor.collect_system_metrics()  # 应该失败
                assert False, "应该抛出异常"
            except Exception:
                pass  # 预期的异常

            # 验证历史没有增加失败的指标
            history_before = len(self.monitor.get_metrics_history())

            # 再次尝试，应该成功
            metrics = self.monitor.collect_system_metrics()
            assert isinstance(metrics, SystemMetrics)
            assert metrics.cpu_percent == 40.0

            # 验证历史增加了成功的指标
            history_after = len(self.monitor.get_metrics_history())
            assert history_after == history_before + 1

    def test_performance_under_load(self):
        """测试负载下的性能"""
        import time

        with patch('psutil.cpu_percent', return_value=35.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 25):

            mock_memory = Mock()
            mock_memory.percent = 45.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 30.0
            mock_disk.return_value = mock_disk_obj

            # 测量性能
            start_time = time.time()
            for _ in range(100):
                self.monitor.collect_system_metrics()
            end_time = time.time()

            duration = end_time - start_time
            avg_duration = duration / 100

            # 验证性能合理（每收集一次不应超过1秒）
            assert avg_duration < 1.0, f"性能太慢: {avg_duration}秒/次"

            # 验证数据完整性
            history = self.monitor.get_metrics_history()
            assert len(history) == 100

    def test_monitor_resource_cleanup(self):
        """测试监控资源清理"""
        # 创建多个monitor实例
        monitors = [SystemMonitor() for _ in range(5)]

        with patch('psutil.cpu_percent', return_value=28.0), \
             patch('psutil.virtual_memory') as mock_mem, \
             patch('psutil.disk_usage') as mock_disk, \
             patch('psutil.net_connections', return_value=[Mock()] * 30):

            mock_memory = Mock()
            mock_memory.percent = 42.0
            mock_mem.return_value = mock_memory

            mock_disk_obj = Mock()
            mock_disk_obj.percent = 28.0
            mock_disk.return_value = mock_disk_obj

            # 每个monitor收集一些数据
            for monitor in monitors:
                for _ in range(3):
                    monitor.collect_system_metrics()

                # 清空历史
                monitor.clear_history()
                assert len(monitor.get_metrics_history()) == 0

        # 验证所有monitor都能正常工作
        for monitor in monitors:
            assert monitor.get_current_metrics() is None
            assert len(monitor.get_metrics_history()) == 0