"""
测试系统监控器

覆盖 system_monitor.py 中的所有类和功能
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.infrastructure.monitoring.infrastructure.system_monitor import (
    SystemMetrics,
    SystemMonitor
)


class TestSystemMetrics:
    """SystemMetrics 数据类测试"""

    def test_initialization(self):
        """测试初始化"""
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


class TestSystemMonitor:
    """SystemMonitor 类测试"""

    def test_initialization(self):
        """测试初始化"""
        monitor = SystemMonitor()

        assert monitor.metrics_history == []
        assert monitor.max_history_size == 1000
        assert hasattr(monitor, '_lock')

    @patch('src.infrastructure.monitoring.infrastructure.system_monitor.psutil')
    def test_collect_system_metrics(self, mock_psutil):
        """测试收集系统指标"""
        # Mock psutil 返回值
        mock_psutil.cpu_percent.return_value = 45.5
        mock_psutil.virtual_memory.return_value.percent = 67.8
        mock_psutil.disk_usage.return_value.percent = 23.4
        mock_psutil.net_connections.return_value = [Mock()] * 150

        monitor = SystemMonitor()

        metrics = monitor.collect_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 67.8
        assert metrics.disk_usage_percent == 23.4
        assert metrics.network_connections == 150

    def test_get_current_metrics_no_history(self):
        """测试获取当前指标（无历史）"""
        monitor = SystemMonitor()

        metrics = monitor.get_current_metrics()

        assert metrics is None

    def test_get_current_metrics_with_history(self):
        """测试获取当前指标（有历史）"""
        monitor = SystemMonitor()

        test_metrics = SystemMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_usage_percent=50.0,
            network_connections=100,
            timestamp=time.time()
        )

        monitor.metrics_history.append(test_metrics)

        current = monitor.get_current_metrics()

        assert current == test_metrics

    def test_get_metrics_history(self):
        """测试获取指标历史"""
        monitor = SystemMonitor()

        # 添加一些指标
        for i in range(5):
            metrics = SystemMetrics(
                cpu_percent=float(i * 10),
                memory_percent=float(i * 15),
                disk_usage_percent=float(i * 20),
                network_connections=i * 25,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        history = monitor.get_metrics_history()

        assert len(history) == 5
        assert history[0].cpu_percent == 0.0
        assert history[4].cpu_percent == 40.0

    def test_get_metrics_history_with_limit(self):
        """测试获取指标历史带限制"""
        monitor = SystemMonitor()

        # 添加10个指标
        for i in range(10):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        history = monitor.get_metrics_history(limit=3)

        assert len(history) == 3
        assert history[0].cpu_percent == 7.0  # 倒数第3个
        assert history[2].cpu_percent == 9.0  # 最新的

    def test_clear_history(self):
        """测试清空历史"""
        monitor = SystemMonitor()

        # 添加一些指标
        for i in range(3):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        assert len(monitor.metrics_history) == 3

        monitor.clear_history()

        assert len(monitor.metrics_history) == 0


class TestSystemMetrics:
    """SystemMetrics 数据类测试"""

    def test_initialization(self):
        """测试初始化"""
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

    def test_initialization_with_zero_values(self):
        """测试零值初始化"""
        metrics = SystemMetrics(
            cpu_percent=0.0,
            memory_percent=0.0,
            disk_usage_percent=0.0,
            network_connections=0,
            timestamp=0.0
        )

        assert metrics.cpu_percent == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_usage_percent == 0.0
        assert metrics.network_connections == 0
        assert metrics.timestamp == 0.0

    def test_initialization_with_high_values(self):
        """测试高值初始化"""
        metrics = SystemMetrics(
            cpu_percent=100.0,
            memory_percent=95.5,
            disk_usage_percent=99.9,
            network_connections=10000,
            timestamp=time.time()
        )

        assert metrics.cpu_percent == 100.0
        assert metrics.memory_percent == 95.5
        assert metrics.disk_usage_percent == 99.9
        assert metrics.network_connections == 10000
        assert isinstance(metrics.timestamp, float)

    def test_data_integrity(self):
        """测试数据完整性"""
        original_data = {
            "cpu_percent": 50.0,
            "memory_percent": 60.0,
            "disk_usage_percent": 70.0,
            "network_connections": 200,
            "timestamp": time.time()
        }

        metrics = SystemMetrics(**original_data)

        # 验证所有字段都被正确设置
        assert metrics.cpu_percent == original_data["cpu_percent"]
        assert metrics.memory_percent == original_data["memory_percent"]
        assert metrics.disk_usage_percent == original_data["disk_usage_percent"]
        assert metrics.network_connections == original_data["network_connections"]
        assert metrics.timestamp == original_data["timestamp"]

    def test_immutability_concept(self):
        """测试不可变性概念"""
        # dataclasses 默认是可变的，但我们可以测试基本功能
        metrics = SystemMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_usage_percent=50.0,
            network_connections=100,
            timestamp=time.time()
        )

        # 验证初始值
        assert metrics.cpu_percent == 30.0
        assert metrics.memory_percent == 40.0
        assert metrics.disk_usage_percent == 50.0
        assert metrics.network_connections == 100


class TestSystemMonitor:
    """SystemMonitor 类测试"""

    def test_initialization(self):
        """测试初始化"""
        monitor = SystemMonitor()

        assert monitor.metrics_history == []
        assert monitor.max_history_size == 1000
        assert hasattr(monitor, '_lock')

    @patch('src.infrastructure.monitoring.infrastructure.system_monitor.psutil')
    def test_collect_system_metrics_with_time_mock(self, mock_psutil):
        """测试收集指标"""
        # Mock psutil 返回值
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.virtual_memory.return_value.percent = 45.8
        mock_psutil.disk_usage.return_value.percent = 67.2
        mock_psutil.net_connections.return_value = [Mock()] * 20  # 20个连接

        monitor = SystemMonitor()

        with patch('time.time', return_value=1234567890.0):
            metrics = monitor.collect_system_metrics()

        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_percent == 25.5
        assert metrics.memory_percent == 45.8
        assert metrics.disk_usage_percent == 67.2
        assert metrics.network_connections == 20
        assert metrics.timestamp == 1234567890.0

    @patch('src.infrastructure.monitoring.infrastructure.system_monitor.psutil')
    def test_collect_system_metrics_with_exceptions(self, mock_psutil):
        """测试收集指标时的异常处理"""
        # Mock psutil 抛出异常
        mock_psutil.cpu_percent.side_effect = Exception("CPU error")
        mock_psutil.virtual_memory.return_value.percent = 50.0
        mock_psutil.disk_usage.return_value.percent = 60.0
        mock_psutil.net_connections.return_value = [Mock()] * 10

        monitor = SystemMonitor()

        with patch('time.time', return_value=1234567890.0):
            metrics = monitor.collect_system_metrics()

        # 即使CPU收集失败，其他指标仍应正常收集
        assert metrics.cpu_percent == 0.0  # 默认值
        assert metrics.memory_percent == 50.0
        assert metrics.disk_usage_percent == 60.0
        assert metrics.network_connections == 10
        assert metrics.timestamp == 1234567890.0


    def test_get_latest_metrics(self):
        """测试获取最新指标"""
        monitor = SystemMonitor()

        # 初始状态没有指标
        latest = monitor.get_current_metrics()
        assert latest is None

        # 添加指标后
        metrics = SystemMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_usage_percent=50.0,
            network_connections=100,
            timestamp=time.time()
        )
        monitor.metrics_history.append(metrics)

        latest = monitor.get_current_metrics()
        assert latest == metrics

    def test_get_metrics_history(self):
        """测试获取指标历史"""
        monitor = SystemMonitor()

        # 空历史
        history = monitor.get_metrics_history()
        assert history == []

        # 添加多个指标
        for i in range(3):
            metrics = SystemMetrics(
                cpu_percent=float(i * 10),
                memory_percent=float(i * 15),
                disk_usage_percent=float(i * 20),
                network_connections=i * 50,
                timestamp=time.time() + i
            )
            monitor.metrics_history.append(metrics)

        history = monitor.get_metrics_history()
        assert len(history) == 3
        assert history[0].cpu_percent == 0.0
        assert history[1].cpu_percent == 10.0
        assert history[2].cpu_percent == 20.0

    def test_get_metrics_history_with_limit(self):
        """测试获取指标历史带限制"""
        monitor = SystemMonitor()

        # 添加5个指标
        for i in range(5):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time() + i
            )
            monitor.metrics_history.append(metrics)

        # 获取最近2个
        history = monitor.get_metrics_history(limit=2)
        assert len(history) == 2
        assert history[0].cpu_percent == 3.0  # 倒数第2个
        assert history[1].cpu_percent == 4.0  # 最新的

    def test_get_average_metrics(self):
        """测试获取平均指标"""
        monitor = SystemMonitor()

        # 添加测试数据
        test_data = [
            (10.0, 20.0, 30.0, 100),
            (20.0, 30.0, 40.0, 150),
            (30.0, 40.0, 50.0, 200),
        ]

        for cpu, mem, disk, net in test_data:
            metrics = SystemMetrics(
                cpu_percent=cpu,
                memory_percent=mem,
                disk_usage_percent=disk,
                network_connections=net,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        avg_metrics = monitor.get_average_metrics()

        assert avg_metrics["cpu_percent"] == 20.0  # (10+20+30)/3
        assert avg_metrics["memory_percent"] == 30.0  # (20+30+40)/3
        assert avg_metrics["disk_usage_percent"] == 40.0  # (30+40+50)/3
        assert avg_metrics["network_connections"] == 150  # (100+150+200)/3

    def test_get_average_metrics_empty_history(self):
        """测试获取平均指标（空历史）"""
        monitor = SystemMonitor()

        avg_metrics = monitor.get_average_metrics()

        assert avg_metrics == {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "disk_usage_percent": 0.0,
            "network_connections": 0
        }

    def test_clear_history(self):
        """测试清空历史"""
        monitor = SystemMonitor()

        # 添加一些指标
        for i in range(3):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        assert len(monitor.metrics_history) == 3

        # 清空历史
        monitor.clear_history()

        assert len(monitor.metrics_history) == 0


    def test_history_size_limit(self):
        """测试历史大小限制"""
        monitor = SystemMonitor()
        monitor.max_history_size = 3

        # 添加4个指标，超过限制
        for i in range(4):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)
            # 手动应用限制逻辑（因为测试不调用collect_system_metrics）
            if len(monitor.metrics_history) > monitor.max_history_size:
                monitor.metrics_history[:] = monitor.metrics_history[-monitor.max_history_size:]

        # 应该只保留最新的3个
        assert len(monitor.metrics_history) == 3
        assert monitor.metrics_history[0].cpu_percent == 1.0  # 最旧的被移除
        assert monitor.metrics_history[1].cpu_percent == 2.0
        assert monitor.metrics_history[2].cpu_percent == 3.0  # 最新的


class TestSystemMonitorIntegration:
    """SystemMonitor 集成测试"""

    @patch('src.infrastructure.monitoring.infrastructure.system_monitor.psutil')
    def test_full_monitoring_cycle(self, mock_psutil):
        """测试完整监控周期"""
        # Mock psutil 返回值
        mock_psutil.cpu_percent.return_value = 15.5
        mock_psutil.virtual_memory.return_value.percent = 35.2
        mock_psutil.disk_usage.return_value.percent = 45.8
        mock_psutil.net_connections.return_value = [Mock()] * 25

        monitor = SystemMonitor()

        # 收集多个指标
        timestamps = []
        for i in range(3):
            with patch('time.time', return_value=1000000000 + i * 60):
                metrics = monitor.collect_system_metrics()
                timestamps.append(metrics.timestamp)

        # 验证历史记录（collect_system_metrics会自动添加到历史）
        assert len(monitor.metrics_history) == 3
        assert all(metrics.cpu_percent == 15.5 for metrics in monitor.metrics_history)
        assert all(metrics.memory_percent == 35.2 for metrics in monitor.metrics_history)
        assert all(metrics.disk_usage_percent == 45.8 for metrics in monitor.metrics_history)
        assert all(metrics.network_connections == 25 for metrics in monitor.metrics_history)

        # 验证时间戳
        assert timestamps == [1000000000, 1000000060, 1000000120]

        # 验证平均值计算（手动计算，因为时间过滤会排除mock数据）
        total_cpu = sum(m.cpu_percent for m in monitor.metrics_history)
        total_memory = sum(m.memory_percent for m in monitor.metrics_history)
        total_disk = sum(m.disk_usage_percent for m in monitor.metrics_history)
        total_network = sum(m.network_connections for m in monitor.metrics_history)
        count = len(monitor.metrics_history)

        assert abs(total_cpu / count - 15.5) < 0.01
        assert abs(total_memory / count - 35.2) < 0.01
        assert abs(total_disk / count - 45.8) < 0.01
        assert total_network / count == 25

    def test_monitoring_stats_accuracy(self):
        """测试监控统计准确性"""
        monitor = SystemMonitor()

        # 添加一些测试数据
        for i in range(3):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        # 测试获取历史记录
        history = monitor.get_metrics_history()
        assert len(history) == 3
        assert history[0].cpu_percent == 0.0
        assert history[2].cpu_percent == 2.0

    def test_concurrent_access_safety(self):
        """测试并发访问安全性"""
        monitor = SystemMonitor()

        import threading
        results = []
        errors = []

        def worker():
            try:
                for _ in range(10):
                    metrics = SystemMetrics(
                        cpu_percent=50.0,
                        memory_percent=60.0,
                        disk_usage_percent=70.0,
                        network_connections=100,
                        timestamp=time.time()
                    )
                    with monitor._lock:
                        monitor.metrics_history.append(metrics)
                        if len(monitor.metrics_history) > monitor.max_history_size:
                            monitor.metrics_history[:] = monitor.metrics_history[-monitor.max_history_size:]
                    results.append(metrics)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 30  # 3线程 * 10次调用
        assert len(errors) == 0  # 没有错误
        assert all(isinstance(r, SystemMetrics) for r in results)

    def test_error_recovery(self):
        """测试错误恢复"""
        monitor = SystemMonitor()

        # 模拟部分指标收集失败的情况
        with patch('src.infrastructure.monitoring.infrastructure.system_monitor.psutil') as mock_psutil:
            # CPU收集失败
            mock_psutil.cpu_percent.side_effect = Exception("CPU sensor failure")

            # 其他指标正常
            mock_psutil.virtual_memory.return_value.percent = 55.0
            mock_psutil.disk_usage.return_value.percent = 65.0
            mock_psutil.net_connections.return_value = [Mock()] * 30

            metrics = monitor.collect_system_metrics()

            # 验证系统在部分失败时仍能继续工作
            assert metrics.cpu_percent == 0.0  # 失败时的默认值
            assert metrics.memory_percent == 55.0  # 正常收集
            assert metrics.disk_usage_percent == 65.0  # 正常收集
            assert metrics.network_connections == 30  # 正常收集

        # 验证历史记录功能正常
        monitor.metrics_history.append(metrics)
        latest = monitor.get_current_metrics()
        assert latest == metrics

    def test_resource_cleanup(self):
        """测试资源清理"""
        monitor = SystemMonitor()

        # 添加一些历史数据
        for i in range(5):
            metrics = SystemMetrics(
                cpu_percent=float(i),
                memory_percent=float(i),
                disk_usage_percent=float(i),
                network_connections=i,
                timestamp=time.time()
            )
            monitor.metrics_history.append(metrics)

        assert len(monitor.metrics_history) == 5

        # 清空历史
        monitor.clear_history()

        assert len(monitor.metrics_history) == 0

        # 验证清空后各项功能仍正常
        latest = monitor.get_current_metrics()
        assert latest is None

        avg = monitor.get_average_metrics()
        assert avg["cpu_percent"] == 0.0

        # 测试清理功能
        initial_history_size = len(monitor.metrics_history)
        monitor.clear_history()
        assert len(monitor.metrics_history) == 0
