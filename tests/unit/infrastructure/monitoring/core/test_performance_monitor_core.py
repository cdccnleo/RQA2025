#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 性能监控器

测试 core/performance_monitor.py 中的所有类和方法
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta


@pytest.fixture
def module():
    """导入模块"""
    from src.infrastructure.monitoring.core import performance_monitor
    return performance_monitor


@pytest.fixture
def metrics(module):
    """创建性能指标实例"""
    return module.PerformanceMetrics("test_metric")


@pytest.fixture
def monitor(module):
    """创建性能监控器实例"""
    return module.PerformanceMonitor(collection_interval=1)


class TestPerformanceMetrics:
    """测试性能指标类"""

    def test_initialization(self, metrics):
        """测试初始化"""
        assert metrics.name == "test_metric"
        assert len(metrics.values) == 0
        assert len(metrics.timestamps) == 0

    def test_add_value(self, metrics):
        """测试添加指标值"""
        metrics.add_value(10.5)
        assert len(metrics.values) == 1
        assert metrics.values[0] == 10.5

    def test_add_value_with_timestamp(self, metrics):
        """测试添加指标值 - 带时间戳"""
        timestamp = datetime.now()
        metrics.add_value(20.0, timestamp)
        assert metrics.timestamps[0] == timestamp

    def test_add_value_multiple(self, metrics):
        """测试添加多个指标值"""
        for i in range(5):
            metrics.add_value(float(i))
        assert len(metrics.values) == 5

    def test_get_recent_values(self, metrics):
        """测试获取最近的值"""
        # 添加一些旧值
        old_time = datetime.now() - timedelta(minutes=10)
        metrics.add_value(1.0, old_time)
        
        # 添加一些新值
        for i in range(3):
            metrics.add_value(float(i + 2))
        
        recent = metrics.get_recent_values(minutes=5)
        assert len(recent) >= 3

    def test_get_stats_empty(self, metrics):
        """测试获取统计信息 - 空数据"""
        stats = metrics.get_stats()
        assert stats['count'] == 0
        assert stats['mean'] == 0.0

    def test_get_stats(self, metrics):
        """测试获取统计信息"""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        for v in values:
            metrics.add_value(v)
        
        stats = metrics.get_stats()
        assert stats['count'] == 5
        assert stats['mean'] == 30.0
        assert stats['min'] == 10.0
        assert stats['max'] == 50.0
        assert stats['latest'] == 50.0

    def test_get_stats_single_value(self, metrics):
        """测试获取统计信息 - 单个值"""
        metrics.add_value(10.0)
        stats = metrics.get_stats()
        assert stats['count'] == 1
        assert stats['std_dev'] == 0.0


class TestPerformanceMonitor:
    """测试性能监控器"""

    def test_initialization(self, monitor):
        """测试初始化"""
        assert monitor.collection_interval == 1
        assert monitor.is_running is False
        assert monitor.monitor_thread is None
        assert 'cpu_usage' in monitor.system_metrics
        assert 'cpu_usage' in monitor.thresholds

    def test_start(self, monitor, monkeypatch):
        """测试启动监控"""
        created_threads = []
        original_thread = threading.Thread

        def mock_thread(*args, **kwargs):
            thread = original_thread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        monkeypatch.setattr(threading, "Thread", mock_thread)

        monitor.start()

        assert monitor.is_running is True
        assert len(created_threads) == 1

    def test_start_already_running(self, monitor):
        """测试启动监控 - 已经运行"""
        monitor.is_running = True
        original_thread = monitor.monitor_thread

        monitor.start()

        # 不应该创建新线程
        assert monitor.monitor_thread == original_thread

    def test_stop(self, monitor, monkeypatch):
        """测试停止监控"""
        mock_thread = MagicMock()
        mock_thread.is_alive.return_value = True
        mock_thread.join = MagicMock()

        monitor.is_running = True
        monitor.monitor_thread = mock_thread

        monitor.stop()

        assert monitor.is_running is False
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_not_running(self, monitor):
        """测试停止监控 - 未运行"""
        monitor.is_running = False
        monitor.stop()  # 不应该抛出异常

    def test_collect_system_metrics_success(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 成功"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        monitor._collect_system_metrics()

        # 验证指标被记录
        assert 'cpu_usage' in monitor.metrics
        assert 'memory_usage' in monitor.metrics
        assert 'disk_usage' in monitor.metrics

    def test_collect_system_metrics_exception(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 异常处理"""
        # Mock psutil 抛出异常
        monkeypatch.setattr(module.psutil, "cpu_percent", MagicMock(side_effect=Exception("Test error")))

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)

        monitor._collect_system_metrics()

        # 异常应该被捕获
        assert len(warnings) > 0

    def test_collect_system_metrics_no_network(self, monitor, module, monkeypatch):
        """测试收集系统指标 - 无网络数据"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=None))

        monitor._collect_system_metrics()

        # 应该不记录网络指标
        assert 'cpu_usage' in monitor.metrics

    def test_record_metric(self, monitor):
        """测试记录指标"""
        monitor.record_metric("test_metric", 10.5)
        assert "test_metric" in monitor.metrics
        assert len(monitor.metrics["test_metric"].values) == 1

    def test_record_metric_with_component(self, monitor):
        """测试记录指标 - 带组件"""
        monitor.record_metric("test_metric", 10.5, component="TestComponent")
        assert "TestComponent" in monitor.component_metrics
        assert "test_metric" in monitor.component_metrics["TestComponent"]

    def test_record_component_metric(self, monitor):
        """测试记录组件指标"""
        monitor.record_component_metric("TestComponent", "test_metric", 10.5)
        assert "TestComponent" in monitor.component_metrics
        assert "test_metric" in monitor.component_metrics["TestComponent"]

    def test_get_metric_stats(self, monitor):
        """测试获取指标统计"""
        monitor.record_metric("test_metric", 10.0)
        monitor.record_metric("test_metric", 20.0)
        monitor.record_metric("test_metric", 30.0)

        stats = monitor.get_metric_stats("test_metric")
        assert stats['count'] == 3
        assert stats['mean'] == 20.0

    def test_get_metric_stats_component(self, monitor):
        """测试获取组件指标统计"""
        monitor.record_component_metric("TestComponent", "test_metric", 10.0)
        monitor.record_component_metric("TestComponent", "test_metric", 20.0)

        stats = monitor.get_metric_stats("test_metric", component="TestComponent")
        assert stats['count'] == 2

    def test_get_metric_stats_not_found(self, monitor):
        """测试获取指标统计 - 不存在"""
        stats = monitor.get_metric_stats("nonexistent")
        assert stats == {}

    def test_get_metric_latest(self, monitor):
        """测试获取最新指标值"""
        monitor.record_metric("test_metric", 10.0)
        monitor.record_metric("test_metric", 20.0)

        latest = monitor.get_metric_latest("test_metric")
        assert latest == 20.0

    def test_get_metric_latest_not_found(self, monitor):
        """测试获取最新指标值 - 不存在"""
        latest = monitor.get_metric_latest("nonexistent")
        assert latest is None

    def test_get_recent_metrics(self, monitor, module, monkeypatch):
        """测试获取最近的指标"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        monitor._collect_system_metrics()
        monitor.record_component_metric("TestComponent", "response_time", 0.5)

        recent = monitor.get_recent_metrics()
        assert 'cpu_usage' in recent
        assert 'memory_usage' in recent
        assert 'TestComponent.response_time' in recent

    def test_get_component_performance_report(self, monitor):
        """测试获取组件性能报告"""
        monitor.record_component_metric("TestComponent", "response_time", 0.5)
        monitor.record_component_metric("TestComponent", "response_time", 0.6)
        monitor.record_component_metric("TestComponent", "throughput", 100.0)

        report = monitor.get_component_performance_report("TestComponent")
        assert report['component'] == "TestComponent"
        assert 'metrics' in report
        assert 'summary' in report

    def test_get_component_performance_report_not_found(self, monitor):
        """测试获取组件性能报告 - 不存在"""
        report = monitor.get_component_performance_report("Nonexistent")
        assert report['component'] == "Nonexistent"
        assert len(report['metrics']) == 0

    def test_set_threshold(self, monitor):
        """测试设置阈值"""
        monitor.set_threshold("test_metric", 50.0)
        assert monitor.thresholds["test_metric"] == 50.0

    def test_get_performance_summary(self, monitor, module, monkeypatch):
        """测试获取性能摘要"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        monitor._collect_system_metrics()
        summary = monitor.get_performance_summary()

        assert 'timestamp' in summary
        assert 'system_metrics' in summary
        assert 'component_count' in summary
        assert 'alerts' in summary

    def test_check_thresholds_no_exceed(self, monitor, module, monkeypatch):
        """测试检查阈值 - 未超过"""
        # Mock psutil
        mock_cpu_percent = MagicMock(return_value=50.0)
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        monitor._collect_system_metrics()
        
        # Mock component_bus 模块（动态导入）
        import sys
        mock_bus = MagicMock()
        mock_message_class = MagicMock()
        mock_message_type = MagicMock()
        mock_message_type.PERFORMANCE_ALERT = "performance_alert"
        
        mock_component_bus_module = MagicMock()
        mock_component_bus_module.global_component_bus = mock_bus
        mock_component_bus_module.Message = mock_message_class
        mock_component_bus_module.MessageType = mock_message_type
        
        # Mock 相对导入路径
        sys.modules['src.infrastructure.monitoring.core.component_bus'] = mock_component_bus_module
        sys.modules['infrastructure.monitoring.core.component_bus'] = mock_component_bus_module

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)

        monitor._check_thresholds()

        # 不应该有警告（因为值未超过阈值）
        assert len(warnings) == 0

    def test_check_thresholds_exceed(self, monitor, module, monkeypatch):
        """测试检查阈值 - 超过阈值"""
        # Mock psutil - CPU使用率超过阈值
        mock_cpu_percent = MagicMock(return_value=90.0)  # 超过阈值80
        mock_virtual_memory = MagicMock()
        mock_virtual_memory.percent = 60.0
        mock_disk_usage = MagicMock()
        mock_disk_usage.percent = 70.0
        mock_net_io = MagicMock()
        mock_net_io.bytes_sent = 1000
        mock_net_io.bytes_recv = 2000

        monkeypatch.setattr(module.psutil, "cpu_percent", mock_cpu_percent)
        monkeypatch.setattr(module.psutil, "virtual_memory", MagicMock(return_value=mock_virtual_memory))
        monkeypatch.setattr(module.psutil, "disk_usage", MagicMock(return_value=mock_disk_usage))
        monkeypatch.setattr(module.psutil, "net_io_counters", MagicMock(return_value=mock_net_io))

        monitor._collect_system_metrics()
        
        # Mock component_bus 模块（动态导入）
        import sys
        mock_bus = MagicMock()
        mock_message_class = MagicMock()
        mock_message_type = MagicMock()
        mock_message_type.PERFORMANCE_ALERT = "performance_alert"
        
        mock_component_bus_module = MagicMock()
        mock_component_bus_module.global_component_bus = mock_bus
        mock_component_bus_module.Message = mock_message_class
        mock_component_bus_module.MessageType = mock_message_type
        
        # Mock 相对导入路径
        sys.modules['src.infrastructure.monitoring.core.component_bus'] = mock_component_bus_module
        sys.modules['infrastructure.monitoring.core.component_bus'] = mock_component_bus_module

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)

        monitor._check_thresholds()

        # 应该有警告（因为值超过阈值）
        assert len(warnings) > 0
        # 验证 publish 被调用（如果导入成功）
        # 注意：如果导入失败，代码会继续执行但不会调用 publish

    def test_monitoring_loop_exception_handling(self, monitor, module, monkeypatch):
        """测试监控循环异常处理"""
        call_count = {"count": 0}
        def mock_collect():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Collect error")
            monitor.is_running = False
        
        monkeypatch.setattr(monitor, "_collect_system_metrics", mock_collect)
        monkeypatch.setattr(monitor, "_check_thresholds", lambda: None)
        monkeypatch.setattr(module.time, "sleep", lambda *_: None)

        errors = []
        def mock_error(msg):
            errors.append(msg)
        
        monkeypatch.setattr(module.logger, "error", mock_error)

        monitor.is_running = True
        monitor._monitoring_loop()

        # 验证异常被记录
        assert len(errors) > 0


class TestGlobalFunctions:
    """测试全局函数"""

    def test_monitor_performance_decorator_success(self, module, monkeypatch):
        """测试性能监控装饰器 - 成功"""
        @module.monitor_performance("test_operation")
        def test_func():
            return "success"

        # Mock global_performance_monitor
        mock_monitor = MagicMock()
        monkeypatch.setattr(module, "global_performance_monitor", mock_monitor)

        result = test_func()

        assert result == "success"
        assert mock_monitor.record_metric.called

    def test_monitor_performance_decorator_exception(self, module, monkeypatch):
        """测试性能监控装饰器 - 异常"""
        @module.monitor_performance("test_operation")
        def test_func():
            raise ValueError("Test error")

        # Mock global_performance_monitor
        mock_monitor = MagicMock()
        monkeypatch.setattr(module, "global_performance_monitor", mock_monitor)

        warnings = []
        def mock_warning(msg):
            warnings.append(msg)
        
        monkeypatch.setattr(module.logger, "warning", mock_warning)

        with pytest.raises(ValueError):
            test_func()

        assert mock_monitor.record_metric.called

    def test_start_performance_monitoring(self, module, monkeypatch):
        """测试启动全局性能监控"""
        mock_monitor = MagicMock()
        monkeypatch.setattr(module, "global_performance_monitor", mock_monitor)

        module.start_performance_monitoring()

        mock_monitor.start.assert_called_once()

    def test_stop_performance_monitoring(self, module, monkeypatch):
        """测试停止全局性能监控"""
        mock_monitor = MagicMock()
        monkeypatch.setattr(module, "global_performance_monitor", mock_monitor)

        module.stop_performance_monitoring()

        mock_monitor.stop.assert_called_once()

    def test_get_performance_report(self, module, monkeypatch):
        """测试获取性能报告"""
        mock_monitor = MagicMock()
        mock_monitor.get_performance_summary.return_value = {"test": "data"}
        monkeypatch.setattr(module, "global_performance_monitor", mock_monitor)

        report = module.get_performance_report()

        assert report == {"test": "data"}
        mock_monitor.get_performance_summary.assert_called_once()

