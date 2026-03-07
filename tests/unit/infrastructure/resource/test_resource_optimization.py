#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
资源优化器测试
测试ResourceOptimizer类的核心功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta

try:
    from src.infrastructure.resource.core.resource_optimization import ResourceOptimizer
    from src.infrastructure.resource.core.shared_interfaces import StandardLogger, BaseErrorHandler
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    # 创建mock类以避免导入错误
    class ResourceOptimizer:
        pass
    class StandardLogger:
        pass
    class BaseErrorHandler:
        pass
    print(f"Warning: 无法导入所需模块: {e}")


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Required imports not available")
class TestResourceOptimizer:
    """测试ResourceOptimizer类"""

    def setup_method(self):
        """测试前准备"""
        self.logger = Mock(spec=StandardLogger)
        self.error_handler = Mock(spec=BaseErrorHandler)
        self.optimizer = ResourceOptimizer(logger=self.logger, error_handler=self.error_handler)

    def test_resource_optimizer_initialization(self):
        """测试资源优化器初始化"""
        optimizer = ResourceOptimizer()

        # 验证基础组件已初始化
        assert hasattr(optimizer, 'logger')
        assert hasattr(optimizer, 'error_handler')
        assert hasattr(optimizer, 'system_analyzer')
        assert hasattr(optimizer, 'thread_analyzer')
        assert hasattr(optimizer, 'memory_detector')
        assert hasattr(optimizer, 'report_generator')
        assert hasattr(optimizer, 'optimization_engine')
        assert isinstance(optimizer.logger, StandardLogger)
        assert isinstance(optimizer.error_handler, BaseErrorHandler)

    def test_resource_optimizer_with_custom_logger(self):
        """测试带自定义日志器的资源优化器初始化"""
        optimizer = ResourceOptimizer(logger=self.logger, error_handler=self.error_handler)

        assert optimizer.logger == self.logger
        assert optimizer.error_handler == self.error_handler

    def test_get_system_resources_default_config(self):
        """测试获取系统资源使用情况（默认配置）"""
        with patch.object(self.optimizer.system_analyzer, 'get_system_resources') as mock_get_resources:
            mock_get_resources.return_value = {
                'cpu': {'usage': 45.0, 'cores': 8},
                'memory': {'usage': 60.0, 'total': 16 * 1024**3},
                'timestamp': '2024-01-01T00:00:00'
            }

            resources = self.optimizer.get_system_resources()

            assert isinstance(resources, dict)
            mock_get_resources.assert_called_once_with("basic")

    def test_get_system_resources_custom_config(self):
        """测试获取系统资源使用情况（自定义配置）"""
        with patch.object(self.optimizer.system_analyzer, 'get_system_resources') as mock_get_resources:
            mock_get_resources.return_value = {
                'cpu': {'usage': 45.0, 'cores': 8},
                'memory': {'usage': 60.0, 'total': 16 * 1024**3}
            }

            resources = self.optimizer.get_system_resources("detailed")

            assert isinstance(resources, dict)
            mock_get_resources.assert_called_once_with("detailed")

    def test_analyze_threads_basic(self):
        """测试分析线程使用情况（基本配置）"""
        with patch.object(self.optimizer.thread_analyzer, 'analyze_threads') as mock_analyze:
            mock_analyze.return_value = {
                'total_threads': 3,
                'daemon_threads': 2,
                'non_daemon_threads': 1,
                'alive_threads_count': 2
            }

            result = self.optimizer.analyze_threads()

            assert isinstance(result, dict)
            assert result['total_threads'] == 3
            mock_analyze.assert_called_once_with(False)

    def test_detect_memory_leaks_with_leaks(self):
        """测试内存泄漏检测（有泄漏）"""
        with patch.object(self.optimizer.memory_detector, 'detect_memory_leaks') as mock_detect:
            mock_detect.return_value = ['内存泄漏检测到: CacheManager实例过多']

            leaks = self.optimizer.detect_memory_leaks()

            assert isinstance(leaks, list)
            assert len(leaks) > 0
            mock_detect.assert_called_once()

    def test_optimize_resources_memory_optimization(self):
        """测试资源优化（内存优化）"""
        config = {
            'optimization_type': 'memory',
            'constraints': {'memory_usage': 90.0, 'memory_threshold': 80.0}
        }

        with patch.object(self.optimizer.optimization_engine, 'optimize_resources') as mock_optimize:
            mock_optimize.return_value = {
                'optimizations': {'memory_optimization': ['清理缓存', '释放未使用内存']},
                'recommendations': ['建议增加内存监控']
            }

            result = self.optimizer.optimize_resources(config)

            assert isinstance(result, dict)
            mock_optimize.assert_called_once_with(config)

    def test_optimize_resources_cpu_optimization(self):
        """测试资源优化（CPU优化）"""
        config = {
            'optimization_type': 'cpu',
            'constraints': {'cpu_usage': 95.0, 'cpu_threshold': 80.0}
        }

        with patch.object(self.optimizer.optimization_engine, 'optimize_resources') as mock_optimize:
            mock_optimize.return_value = {
                'optimizations': {'cpu_optimization': ['优化算法', '降低线程优先级']},
                'recommendations': ['建议升级CPU']
            }

            result = self.optimizer.optimize_resources(config)

            assert isinstance(result, dict)
            mock_optimize.assert_called_once_with(config)

    def test_get_recommendations(self):
        """测试获取优化建议"""
        with patch.object(self.optimizer.optimization_engine, 'get_optimization_recommendations') as mock_get_recommendations:
            mock_get_recommendations.return_value = ['建议1：升级内存', '建议2：优化线程池']

            recommendations = self.optimizer.get_recommendations()

            assert isinstance(recommendations, list)
            mock_get_recommendations.assert_called_once()

    def test_monitor_performance_decorator(self):
        """测试性能监控装饰器函数"""
        def test_function():
            return "test result"

        # 测试装饰器的使用
        decorator = self.optimizer.monitor_performance('test_operation')
        decorated_func = decorator(test_function)

        with patch.object(self.logger, 'log_info') as mock_log_info:
            result = decorated_func()

            assert result == "test result"
            # 验证日志记录器被调用（开始和结束）
            assert mock_log_info.call_count >= 2

    def test_generate_optimization_report(self):
        """测试生成优化报告"""
        with patch.object(self.optimizer.report_generator, 'generate_optimization_report') as mock_generate:
            mock_generate.return_value = {
                'report_type': 'summary',
                'recommendations': ['优化建议1', '优化建议2'],
                'metrics': {'cpu_usage': 75.5, 'memory_usage': 82.3}
            }

            result = self.optimizer.generate_optimization_report("detailed")

            assert isinstance(result, dict)
            mock_generate.assert_called_once_with("detailed")

    def test_monitor_performance_with_exception(self):
        """测试性能监控装饰器异常处理"""
        def failing_function():
            raise ValueError("Test error")

        decorator = self.optimizer.monitor_performance('failing_operation')
        decorated_func = decorator(failing_function)

        with patch.object(self.error_handler, 'handle_error') as mock_handle_error:
            with pytest.raises(ValueError, match="Test error"):
                decorated_func()

            # 验证错误处理器被调用
            mock_handle_error.assert_called_once()