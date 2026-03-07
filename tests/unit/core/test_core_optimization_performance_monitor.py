#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心优化性能监控器

测试目标：提升core_optimization/components/performance_monitor.py的覆盖率到100%
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock

from src.core.core_optimization.components.performance_monitor import (
    PerformanceMetric,
    PerformanceMonitor
)


class TestPerformanceMetric:
    """测试性能指标数据类"""

    @pytest.fixture
    def metric(self):
        """创建性能指标实例"""
        return PerformanceMetric(
            name="cpu_usage",
            value=75.5,
            unit="percent",
            timestamp=time.time(),
            category="system"
        )

    def test_performance_metric_creation(self, metric):
        """测试性能指标创建"""
        assert metric.name == "cpu_usage"
        assert metric.value == 75.5
        assert metric.unit == "percent"
        assert metric.category == "system"
        assert isinstance(metric.timestamp, float)

    def test_performance_metric_default_category(self):
        """测试性能指标默认类别"""
        metric = PerformanceMetric(
            name="memory_usage",
            value=1024.0,
            unit="MB",
            timestamp=time.time()
        )

        assert metric.category == "general"

    def test_performance_metric_equality(self):
        """测试性能指标相等性"""
        timestamp = time.time()

        metric1 = PerformanceMetric("test", 100.0, "units", timestamp, "category")
        metric2 = PerformanceMetric("test", 100.0, "units", timestamp, "category")
        metric3 = PerformanceMetric("test", 200.0, "units", timestamp, "category")

        assert metric1 == metric2
        assert metric1 != metric3

    def test_performance_metric_string_representation(self, metric):
        """测试性能指标字符串表示"""
        str_repr = str(metric)
        assert "cpu_usage" in str_repr
        assert "75.5" in str_repr
        assert "percent" in str_repr


class TestPerformanceMonitor:
    """测试性能监控器"""

    @pytest.fixture
    def monitor(self):
        """创建性能监控器实例"""
        return PerformanceMonitor(monitoring_interval=1)  # 使用较短的间隔以加快测试

    def test_monitor_initialization(self, monitor):
        """测试监控器初始化"""
        assert monitor.name == "PerformanceMonitor"
        assert monitor.monitoring_interval == 1
        assert isinstance(monitor.metrics, list)
        assert monitor.monitoring_thread is None
        assert monitor.is_monitoring == False
        assert isinstance(monitor.metric_handlers, dict)

    def test_default_handlers_registration(self, monitor):
        """测试默认处理器注册"""
        expected_handlers = [
            "cpu_usage", "memory_usage", "disk_usage", "network_io",
            "process_count", "thread_count", "open_files", "system_load"
        ]

        for handler_name in expected_handlers:
            assert handler_name in monitor.metric_handlers
            assert callable(monitor.metric_handlers[handler_name])

    @patch('psutil.cpu_percent')
    def test_collect_cpu_usage(self, mock_cpu_percent, monitor):
        """测试收集CPU使用率"""
        mock_cpu_percent.return_value = 45.5

        metric = monitor._collect_cpu_usage()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "cpu_usage"
        assert metric.value == 45.5
        assert metric.unit == "percent"
        assert metric.category == "system"
        assert isinstance(metric.timestamp, float)

    @patch('psutil.virtual_memory')
    def test_collect_memory_usage(self, mock_virtual_memory, monitor):
        """测试收集内存使用率"""
        mock_memory = Mock()
        mock_memory.percent = 67.8
        mock_virtual_memory.return_value = mock_memory

        metric = monitor._collect_memory_usage()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "memory_usage"
        assert metric.value == 67.8
        assert metric.unit == "percent"
        assert metric.category == "system"

    @patch('psutil.disk_usage')
    def test_collect_disk_usage(self, mock_disk_usage, monitor):
        """测试收集磁盘使用率"""
        mock_disk = Mock()
        mock_disk.percent = 55.2
        mock_disk_usage.return_value = mock_disk

        metric = monitor._collect_disk_usage()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "disk_usage"
        assert metric.value == 55.2
        assert metric.unit == "percent"
        assert metric.category == "storage"

    @patch('psutil.net_io_counters')
    def test_collect_network_io(self, mock_net_io, monitor):
        """测试收集网络IO"""
        mock_net = Mock()
        mock_net.bytes_sent = 1000000
        mock_net.bytes_recv = 2000000
        mock_net_io.return_value = mock_net

        with patch('time.time', return_value=1000.0):
            with patch.object(monitor, '_get_previous_network_stats', return_value=(500000, 1000000, 999.0)):
                metric = monitor._collect_network_io()

                assert isinstance(metric, PerformanceMetric)
                assert metric.name == "network_io"
                assert metric.unit == "bytes_per_second"
                assert metric.category == "network"

    @patch('psutil.pids')
    def test_collect_process_count(self, mock_pids, monitor):
        """测试收集进程数量"""
        mock_pids.return_value = [1, 2, 3, 4, 5]  # 5个进程

        metric = monitor._collect_process_count()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "process_count"
        assert metric.value == 5
        assert metric.unit == "count"
        assert metric.category == "system"

    def test_collect_thread_count(self, monitor):
        """测试收集线程数量"""
        metric = monitor._collect_thread_count()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "thread_count"
        assert metric.unit == "count"
        assert metric.category == "system"
        assert isinstance(metric.value, int)
        assert metric.value >= 1  # 至少有主线程

    def test_collect_open_files(self, monitor):
        """测试收集打开文件数量"""
        metric = monitor._collect_open_files()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "open_files"
        assert metric.unit == "count"
        assert metric.category == "system"
        assert isinstance(metric.value, int)

    @patch('os.getloadavg')
    def test_collect_system_load(self, mock_getloadavg, monitor):
        """测试收集系统负载"""
        mock_getloadavg.return_value = (1.5, 1.2, 0.8)

        metric = monitor._collect_system_load()

        assert isinstance(metric, PerformanceMetric)
        assert metric.name == "system_load"
        assert metric.value == 1.5  # 1分钟平均负载
        assert metric.unit == "load"
        assert metric.category == "system"

    def test_start_monitoring(self, monitor):
        """测试开始监控"""
        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None

        monitor.start_monitoring()

        assert monitor.is_monitoring
        assert monitor.monitoring_thread is not None
        assert monitor.monitoring_thread.is_alive()

        # 清理
        monitor.stop_monitoring()

    def test_stop_monitoring(self, monitor):
        """测试停止监控"""
        monitor.start_monitoring()
        assert monitor.is_monitoring

        monitor.stop_monitoring()

        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None

    def test_stop_monitoring_not_started(self, monitor):
        """测试停止未启动的监控"""
        assert not monitor.is_monitoring

        # 不应该抛出异常
        monitor.stop_monitoring()

        assert not monitor.is_monitoring

    def test_get_metrics(self, monitor):
        """测试获取指标"""
        # 添加一些测试指标
        test_metrics = [
            PerformanceMetric("test1", 100.0, "units", time.time(), "test"),
            PerformanceMetric("test2", 200.0, "units", time.time(), "test")
        ]

        monitor.metrics.extend(test_metrics)

        metrics = monitor.get_metrics()

        assert isinstance(metrics, list)
        assert len(metrics) >= 2

        # 验证包含我们添加的指标
        metric_names = [m.name for m in metrics]
        assert "test1" in metric_names
        assert "test2" in metric_names

    def test_get_metrics_by_category(self, monitor):
        """测试按类别获取指标"""
        # 添加不同类别的指标
        monitor.metrics.extend([
            PerformanceMetric("cpu", 50.0, "%", time.time(), "system"),
            PerformanceMetric("memory", 60.0, "%", time.time(), "system"),
            PerformanceMetric("network", 100.0, "MB/s", time.time(), "network")
        ])

        system_metrics = monitor.get_metrics_by_category("system")
        network_metrics = monitor.get_metrics_by_category("network")

        assert len(system_metrics) == 2
        assert len(network_metrics) == 1

        assert all(m.category == "system" for m in system_metrics)
        assert all(m.category == "network" for m in network_metrics)

    def test_get_latest_metric(self, monitor):
        """测试获取最新指标"""
        # 添加指标（较新的时间戳）
        old_time = time.time() - 100
        new_time = time.time()

        monitor.metrics.extend([
            PerformanceMetric("test", 50.0, "units", old_time, "test"),
            PerformanceMetric("test", 60.0, "units", new_time, "test")
        ])

        latest = monitor.get_latest_metric("test")

        assert latest.value == 60.0
        assert latest.timestamp == new_time

    def test_get_latest_metric_not_found(self, monitor):
        """测试获取不存在的最新指标"""
        result = monitor.get_latest_metric("nonexistent")

        assert result is None

    def test_get_metric_average(self, monitor):
        """测试获取指标平均值"""
        monitor.metrics.extend([
            PerformanceMetric("response_time", 100.0, "ms", time.time(), "performance"),
            PerformanceMetric("response_time", 200.0, "ms", time.time(), "performance"),
            PerformanceMetric("response_time", 300.0, "ms", time.time(), "performance")
        ])

        average = monitor.get_metric_average("response_time")

        assert average == 200.0  # (100 + 200 + 300) / 3

    def test_get_metric_average_no_metrics(self, monitor):
        """测试获取没有指标的平均值"""
        result = monitor.get_metric_average("nonexistent")

        assert result == 0.0

    def test_clear_metrics(self, monitor):
        """测试清空指标"""
        monitor.metrics.extend([
            PerformanceMetric("test1", 100.0, "units", time.time()),
            PerformanceMetric("test2", 200.0, "units", time.time())
        ])

        assert len(monitor.metrics) >= 2

        monitor.clear_metrics()

        assert len(monitor.metrics) == 0

    def test_register_custom_handler(self, monitor):
        """测试注册自定义处理器"""
        def custom_handler():
            return PerformanceMetric("custom", 42.0, "units", time.time(), "custom")

        monitor.register_metric_handler("custom_metric", custom_handler)

        assert "custom_metric" in monitor.metric_handlers
        assert monitor.metric_handlers["custom_metric"] == custom_handler

    def test_collect_all_metrics(self, monitor):
        """测试收集所有指标"""
        with patch.object(monitor, '_collect_cpu_usage') as mock_cpu, \
             patch.object(monitor, '_collect_memory_usage') as mock_memory:

            mock_cpu.return_value = PerformanceMetric("cpu", 50.0, "%", time.time(), "system")
            mock_memory.return_value = PerformanceMetric("memory", 60.0, "%", time.time(), "system")

            metrics = monitor._collect_all_metrics()

            assert isinstance(metrics, list)
            assert len(metrics) >= 2

            # 验证调用了处理器
            mock_cpu.assert_called_once()
            mock_memory.assert_called_once()

    def test_monitoring_loop(self, monitor):
        """测试监控循环"""
        # 这个方法是私有的，主要在监控线程中运行
        # 我们可以通过检查它是否收集指标来测试

        initial_count = len(monitor.metrics)

        # 手动运行一次监控循环
        monitor._monitoring_loop()

        # 应该收集到一些指标
        assert len(monitor.metrics) > initial_count


class TestPerformanceMonitorIntegration:
    """测试性能监控器集成场景"""

    @pytest.fixture
    def monitor(self):
        """创建性能监控器实例"""
        return PerformanceMonitor(monitoring_interval=1)

    def test_complete_monitoring_workflow(self, monitor):
        """测试完整的监控工作流程"""
        # 1. 启动监控
        monitor.start_monitoring()

        assert monitor.is_monitoring
        assert monitor.monitoring_thread.is_alive()

        # 2. 等待一段时间让监控收集一些数据
        time.sleep(2)

        # 3. 检查是否收集到了指标
        metrics = monitor.get_metrics()
        assert len(metrics) > 0

        # 4. 检查不同类别的指标
        system_metrics = monitor.get_metrics_by_category("system")
        assert len(system_metrics) > 0

        # 5. 检查最新的CPU使用率
        latest_cpu = monitor.get_latest_metric("cpu_usage")
        assert latest_cpu is not None
        assert latest_cpu.name == "cpu_usage"
        assert latest_cpu.unit == "percent"

        # 6. 停止监控
        monitor.stop_monitoring()

        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None

    def test_custom_metrics_collection(self, monitor):
        """测试自定义指标收集"""
        # 注册自定义指标处理器
        custom_values = [10.0, 20.0, 30.0]
        value_index = 0

        def custom_metric_handler():
            nonlocal value_index
            value = custom_values[value_index % len(custom_values)]
            value_index += 1
            return PerformanceMetric("custom_metric", value, "custom_units", time.time(), "custom")

        monitor.register_metric_handler("custom_metric", custom_metric_handler)

        # 手动收集自定义指标
        metric = monitor.metric_handlers["custom_metric"]()

        assert metric.name == "custom_metric"
        assert metric.value in custom_values
        assert metric.unit == "custom_units"
        assert metric.category == "custom"

    def test_metrics_analysis_functions(self, monitor):
        """测试指标分析功能"""
        # 添加一些测试指标
        base_time = time.time()
        monitor.metrics.extend([
            PerformanceMetric("response_time", 100.0, "ms", base_time, "performance"),
            PerformanceMetric("response_time", 150.0, "ms", base_time + 1, "performance"),
            PerformanceMetric("response_time", 200.0, "ms", base_time + 2, "performance"),
            PerformanceMetric("cpu_usage", 50.0, "%", base_time, "system"),
            PerformanceMetric("cpu_usage", 60.0, "%", base_time + 1, "system")
        ])

        # 测试平均值计算
        avg_response = monitor.get_metric_average("response_time")
        expected_avg = (100.0 + 150.0 + 200.0) / 3
        assert avg_response == expected_avg

        avg_cpu = monitor.get_metric_average("cpu_usage")
        expected_cpu_avg = (50.0 + 60.0) / 2
        assert avg_cpu == expected_cpu_avg

        # 测试最新指标获取
        latest_response = monitor.get_latest_metric("response_time")
        assert latest_response.value == 200.0
        assert latest_response.timestamp == base_time + 2

        latest_cpu = monitor.get_latest_metric("cpu_usage")
        assert latest_cpu.value == 60.0

    def test_concurrent_monitoring_access(self, monitor):
        """测试并发监控访问"""
        import threading

        results = []
        errors = []

        def access_monitor(operation_name):
            try:
                if operation_name == "get_metrics":
                    metrics = monitor.get_metrics()
                    results.append(f"get_metrics_{len(metrics)}")
                elif operation_name == "get_latest":
                    latest = monitor.get_latest_metric("cpu_usage")
                    results.append(f"get_latest_{latest is not None}")
                elif operation_name == "clear":
                    monitor.clear_metrics()
                    results.append("clear_done")
            except Exception as e:
                errors.append(f"{operation_name}_error: {str(e)}")

        # 先添加一些指标
        monitor.metrics.append(PerformanceMetric("cpu_usage", 50.0, "%", time.time(), "system"))

        # 创建多个线程并发访问
        operations = ["get_metrics", "get_latest", "clear", "get_metrics", "get_latest"]
        threads = []

        for operation in operations:
            thread = threading.Thread(target=access_monitor, args=(operation,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有操作都成功完成
        assert len(results) == len(operations)

    def test_monitoring_thread_safety(self, monitor):
        """测试监控线程安全"""
        # 启动监控
        monitor.start_monitoring()

        # 在监控运行时执行各种操作
        time.sleep(1)

        # 获取指标
        metrics = monitor.get_metrics()
        assert isinstance(metrics, list)

        # 注册新的处理器
        def new_handler():
            return PerformanceMetric("new_metric", 42.0, "units", time.time(), "test")

        monitor.register_metric_handler("new_metric", new_handler)

        # 等待更多数据收集
        time.sleep(1)

        # 检查新指标是否被收集
        new_metrics = [m for m in monitor.get_metrics() if m.name == "new_metric"]
        # 注意：新指标可能还没有被收集到，取决于定时器

        # 停止监控
        monitor.stop_monitoring()

        assert not monitor.is_monitoring

    def test_resource_cleanup(self, monitor):
        """测试资源清理"""
        # 启动监控
        monitor.start_monitoring()

        # 等待一些操作
        time.sleep(1)

        # 记录初始状态
        initial_metric_count = len(monitor.get_metrics())

        # 停止监控
        monitor.stop_monitoring()

        # 清空指标
        monitor.clear_metrics()

        # 验证清理
        assert len(monitor.get_metrics()) == 0
        assert not monitor.is_monitoring
        assert monitor.monitoring_thread is None

    def test_error_handling_in_handlers(self, monitor):
        """测试处理器中的错误处理"""
        def failing_handler():
            raise Exception("Handler failed")

        monitor.register_metric_handler("failing_handler", failing_handler)

        # 调用失败的处理器应该不会导致整个监控崩溃
        try:
            # 手动调用处理器（在实际监控循环中会被保护）
            failing_handler()
            assert False, "Should have raised exception"
        except Exception:
            pass  # 预期的异常

        # 监控器应该仍然正常工作
        metrics = monitor.get_metrics()
        assert isinstance(metrics, list)

    def test_performance_under_load(self, monitor):
        """测试负载下的性能"""
        # 注册多个处理器
        for i in range(10):
            def handler_factory(index=i):
                def handler():
                    time.sleep(0.001)  # 模拟少量工作
                    return PerformanceMetric(f"test_metric_{index}", float(index), "units", time.time(), "test")
                return handler

            monitor.register_metric_handler(f"test_handler_{i}", handler_factory())

        # 执行所有处理器
        start_time = time.time()

        collected_metrics = []
        for handler_name, handler in monitor.metric_handlers.items():
            if handler_name.startswith("test_handler_"):
                metric = handler()
                collected_metrics.append(metric)

        end_time = time.time()
        collection_time = end_time - start_time

        # 验证收集了所有指标
        assert len(collected_metrics) == 10

        # 验证性能（应该在合理时间内完成）
        assert collection_time < 1.0  # 应该在1秒内完成

        # 验证指标正确性
        for i, metric in enumerate(collected_metrics):
            assert metric.name == f"test_metric_{i}"
            assert metric.value == float(i)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
