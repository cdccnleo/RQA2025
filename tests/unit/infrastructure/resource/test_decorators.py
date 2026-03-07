#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
装饰器测试
测试decorators.py中的监控装饰器功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from prometheus_client import CollectorRegistry, Histogram, Counter

from src.infrastructure.resource.utils.decorators import (
    monitor_performance, monitor_errors, monitor_resource,
    _get_resource_metrics, _create_resource_metrics,
    _execute_with_monitoring, _record_error, _record_duration
)


class TestMonitorPerformance:
    """测试性能监控装饰器"""

    def test_monitor_performance_decorator_basic(self):
        """测试基本的性能监控装饰器"""
        registry = CollectorRegistry()

        @monitor_performance("test_operation", registry=registry)
        def test_function():
            time.sleep(0.01)
            return "success"

        result = test_function()

        assert result == "success"

        # 验证指标被创建并有数据
        histogram = registry._names_to_collectors.get('performance_duration_seconds')
        assert histogram is not None
        assert isinstance(histogram, Histogram)

    def test_monitor_performance_decorator_with_exception(self):
        """测试性能监控装饰器处理异常"""
        registry = CollectorRegistry()

        @monitor_performance("test_operation", registry=registry)
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

        # 验证即使有异常，性能指标也被记录
        histogram = registry._names_to_collectors.get('performance_duration_seconds')
        assert histogram is not None

    def test_monitor_performance_decorator_default_name(self):
        """测试性能监控装饰器使用默认操作名称"""
        registry = CollectorRegistry()

        @monitor_performance(registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

        # 验证使用函数名作为操作名称
        histogram = registry._names_to_collectors.get('performance_duration_seconds')
        assert histogram is not None

    def test_monitor_performance_decorator_custom_registry(self):
        """测试性能监控装饰器使用自定义注册表"""
        registry = CollectorRegistry()

        @monitor_performance("custom_operation", registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

        # 验证指标在正确的注册表中
        assert len(registry._collector_to_names) > 0

    def test_monitor_performance_caching(self):
        """测试性能监控装饰器的指标缓存"""
        registry1 = CollectorRegistry()
        registry2 = CollectorRegistry()

        @monitor_performance("test_op1", registry=registry1)
        def func1():
            return "result1"

        @monitor_performance("test_op2", registry=registry1)
        def func2():
            return "result2"

        @monitor_performance("test_op3", registry=registry2)
        def func3():
            return "result3"

        # 执行函数
        func1()
        func2()
        func3()

        # 验证每个注册表都有正确的指标
        assert 'performance_duration_seconds' in registry1._names_to_collectors
        assert 'performance_duration_seconds' in registry2._names_to_collectors


class TestMonitorErrors:
    """测试错误监控装饰器"""

    def test_monitor_errors_decorator_basic(self):
        """测试基本的错误监控装饰器"""
        registry = CollectorRegistry()

        @monitor_errors(registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

        # 验证计数器被创建
        counter = registry._names_to_collectors.get('error_count_total')
        assert counter is not None
        assert isinstance(counter, Counter)

    def test_monitor_errors_decorator_with_exception(self):
        """测试错误监控装饰器处理异常"""
        registry = CollectorRegistry()

        @monitor_errors([ValueError], registry=registry)
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()

        # 验证错误被计数
        counter = registry._names_to_collectors.get('error_count_total')
        assert counter is not None

    def test_monitor_errors_decorator_different_error_types(self):
        """测试错误监控装饰器处理不同类型的异常"""
        registry = CollectorRegistry()

        @monitor_errors([ValueError, RuntimeError], registry=registry)
        def test_function():
            raise RuntimeError("Runtime error")

        with pytest.raises(RuntimeError, match="Runtime error"):
            test_function()

    def test_monitor_errors_decorator_unmonitored_exception(self):
        """测试错误监控装饰器处理未监控的异常"""
        registry = CollectorRegistry()

        @monitor_errors([ValueError], registry=registry)
        def test_function():
            raise RuntimeError("Unmonitored error")

        with pytest.raises(RuntimeError, match="Unmonitored error"):
            test_function()

    def test_monitor_errors_none_error_types(self):
        """测试错误监控装饰器使用None作为错误类型（监控所有异常）"""
        registry = CollectorRegistry()

        @monitor_errors(None, registry=registry)
        def test_function():
            raise RuntimeError("Any error")

        with pytest.raises(RuntimeError, match="Any error"):
            test_function()


class TestMonitorResource:
    """测试资源监控装饰器"""

    def test_monitor_resource_decorator_basic(self):
        """测试基本的资源监控装饰器"""
        registry = CollectorRegistry()

        @monitor_resource("cpu", registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

        # 验证资源监控指标被创建
        histogram = registry._names_to_collectors.get('resource_usage_seconds')
        counter = registry._names_to_collectors.get('resource_errors_total')
        assert histogram is not None
        assert counter is not None

    def test_monitor_resource_decorator_with_exception(self):
        """测试资源监控装饰器处理异常"""
        registry = CollectorRegistry()

        @monitor_resource("disk", registry=registry)
        def test_function():
            raise OSError("Disk error")

        with pytest.raises(OSError, match="Disk error"):
            test_function()

    def test_monitor_resource_decorator_different_resource_types(self):
        """测试资源监控装饰器处理不同资源类型"""
        registry = CollectorRegistry()

        @monitor_resource("network", registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"


class TestResourceMetricsFunctions:
    """测试资源指标相关函数"""

    def test_get_resource_metrics(self):
        """测试获取资源指标"""
        registry = CollectorRegistry()

        metrics = _get_resource_metrics(registry)

        assert 'usage_histogram' in metrics
        assert 'error_counter' in metrics
        assert isinstance(metrics['usage_histogram'], Histogram)
        assert isinstance(metrics['error_counter'], Counter)

    def test_get_resource_metrics_caching(self):
        """测试资源指标缓存"""
        registry = CollectorRegistry()

        # 第一次调用
        metrics1 = _get_resource_metrics(registry)

        # 第二次调用应该返回相同的对象
        metrics2 = _get_resource_metrics(registry)

        assert metrics1 is metrics2

    def test_create_resource_metrics(self):
        """测试创建资源指标"""
        registry = CollectorRegistry()

        metrics = _create_resource_metrics(registry)

        assert 'usage_histogram' in metrics
        assert 'error_counter' in metrics
        assert isinstance(metrics['usage_histogram'], Histogram)
        assert isinstance(metrics['error_counter'], Counter)

    def test_execute_with_monitoring_success(self):
        """测试成功执行的监控"""
        registry = CollectorRegistry()
        metrics = _get_resource_metrics(registry)

        def test_func():
            return "success"

        result = _execute_with_monitoring(test_func, (), {}, "cpu", metrics)

        assert result == "success"

    def test_execute_with_monitoring_failure(self):
        """测试失败执行的监控"""
        registry = CollectorRegistry()
        metrics = _get_resource_metrics(registry)

        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            _execute_with_monitoring(test_func, (), {}, "cpu", metrics)

    def test_record_error(self):
        """测试记录错误"""
        registry = CollectorRegistry()
        metrics = _get_resource_metrics(registry)

        error = ValueError("Test error")
        _record_error(metrics, "cpu", "test_operation", error)

        # 验证错误计数器被更新（这里只是确保不抛出异常）

    def test_record_duration(self):
        """测试记录执行时间"""
        registry = CollectorRegistry()
        metrics = _get_resource_metrics(registry)

        _record_duration(metrics, "cpu", "test_operation", 0.5, "success")

        # 验证直方图被更新（这里只是确保不抛出异常）


class TestDecoratorIntegration:
    """测试装饰器集成"""

    def test_multiple_decorators_on_same_function(self):
        """测试在同一个函数上使用多个装饰器"""
        # 为每个装饰器使用不同的registry以避免指标名称冲突
        perf_registry = CollectorRegistry()
        resource_registry = CollectorRegistry()
        error_registry = CollectorRegistry()

        @monitor_performance("complex_operation", registry=perf_registry)
        @monitor_resource("cpu", registry=resource_registry)
        @monitor_errors([Exception], registry=error_registry)
        def complex_function():
            time.sleep(0.01)
            return "complex_success"

        result = complex_function()

        assert result == "complex_success"

        # 验证所有指标都被创建
        assert 'performance_duration_seconds' in perf_registry._names_to_collectors
        assert 'resource_usage_seconds' in resource_registry._names_to_collectors
        assert 'resource_errors_total' in resource_registry._names_to_collectors
        assert 'error_count_total' in error_registry._names_to_collectors

    def test_decorator_performance_impact(self):
        """测试装饰器对性能的影响"""
        registry = CollectorRegistry()

        @monitor_performance("performance_test", registry=registry)
        def fast_function():
            return sum(range(1000))

        # 执行多次以获得稳定的测量
        import time
        start_time = time.time()

        for _ in range(100):
            result = fast_function()
            assert result == sum(range(1000))

        end_time = time.time()
        total_time = end_time - start_time

        # 验证函数仍然正常工作且性能在合理范围内
        assert total_time < 1.0  # 应该在1秒内完成

    def test_decorator_error_propagation(self):
        """测试装饰器正确传播异常"""
        registry = CollectorRegistry()

        @monitor_performance("error_test", registry=registry)
        @monitor_errors([ValueError], registry=registry)
        def error_function():
            raise ValueError("Propagated error")

        # 验证异常被正确传播
        with pytest.raises(ValueError, match="Propagated error"):
            error_function()

    def test_decorator_with_different_input_types(self):
        """测试装饰器处理不同输入类型"""
        registry = CollectorRegistry()

        @monitor_performance("flexible_function", registry=registry)
        def flexible_function(arg):
            if isinstance(arg, str):
                return len(arg)
            elif isinstance(arg, list):
                return sum(arg)
            elif isinstance(arg, dict):
                return len(arg.keys())
            else:
                return 0

        # 测试不同类型的输入
        assert flexible_function("hello") == 5
        assert flexible_function([1, 2, 3, 4, 5]) == 15
        assert flexible_function({"a": 1, "b": 2, "c": 3}) == 3
        assert flexible_function(42) == 0


class TestDecoratorEdgeCases:
    """测试装饰器边界情况"""

    def test_decorator_with_no_registry(self):
        """测试装饰器在没有提供注册表时的行为"""
        @monitor_performance("no_registry_test")
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

    def test_decorator_with_none_operation_name(self):
        """测试装饰器使用None作为操作名称"""
        registry = CollectorRegistry()

        @monitor_performance(None, registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

    def test_monitor_resource_with_none_labels(self):
        """测试资源监控装饰器使用None作为标签"""
        registry = CollectorRegistry()

        @monitor_resource("test_resource", labels=None, registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

    def test_decorator_with_special_characters_in_name(self):
        """测试装饰器处理名称中的特殊字符"""
        registry = CollectorRegistry()

        @monitor_performance("special_name_@#$%", registry=registry)
        def test_function():
            return "success"

        result = test_function()

        assert result == "success"

    def test_monitor_errors_empty_list(self):
        """测试错误监控装饰器使用空错误类型列表"""
        registry = CollectorRegistry()

        @monitor_errors([], registry=registry)
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            test_function()
