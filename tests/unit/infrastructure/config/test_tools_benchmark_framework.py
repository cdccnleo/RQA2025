#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试基准测试框架
测试 src.infrastructure.config.tools.benchmark_framework 模块
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch
from typing import Any, Dict, Callable

from src.infrastructure.config.tools.benchmark_framework import BenchmarkFramework


class TestBenchmarkFramework:
    """测试基准测试框架"""

    def setup_method(self):
        """设置测试方法"""
        self.framework = BenchmarkFramework()

    def test_initialization(self):
        """测试初始化"""
        framework = BenchmarkFramework()

        assert framework.disable_background_collection is False
        assert framework.max_iterations == 100
        assert isinstance(framework.results, list)
        assert framework.results == []

    def test_attribute_assignment(self):
        """测试属性赋值"""
        self.framework.disable_background_collection = True
        self.framework.max_iterations = 50

        assert self.framework.disable_background_collection is True
        assert self.framework.max_iterations == 50

    def test_run_benchmark_method_exists(self):
        """测试运行基准测试方法存在"""
        # 验证BenchmarkFramework类有run_benchmark方法
        assert hasattr(self.framework, 'run_benchmark')
        assert callable(getattr(self.framework, 'run_benchmark', None))

    def test_get_results_method_exists(self):
        """测试获取结果方法存在"""
        # 验证BenchmarkFramework类有get_results方法
        assert hasattr(self.framework, 'get_results')
        assert callable(getattr(self.framework, 'get_results', None))

    def test_clear_results_method_exists(self):
        """测试清空结果方法存在"""
        # 验证BenchmarkFramework类有clear_results方法
        assert hasattr(self.framework, 'clear_results')
        assert callable(getattr(self.framework, 'clear_results', None))

    def test_framework_basic_operations(self):
        """测试框架基本操作"""
        # 测试可以创建实例
        framework = BenchmarkFramework()
        assert framework is not None

        # 测试可以设置属性
        framework.max_iterations = 200
        assert framework.max_iterations == 200

        # 测试results列表可以操作
        framework.results.append('test_value')
        assert framework.results[0] == 'test_value'

    def test_multiple_instances_independence(self):
        """测试多个实例的独立性"""
        framework1 = BenchmarkFramework()
        framework2 = BenchmarkFramework()

        # 修改第一个实例
        framework1.max_iterations = 50
        framework1.results.append('value1')

        # 第二个实例应该不受影响
        assert framework2.max_iterations == 100
        assert framework2.results == []

        # 修改第二个实例
        framework2.max_iterations = 75
        framework2.results.append('value2')

        # 第一个实例应该不受影响
        assert framework1.max_iterations == 50
        assert framework1.results[0] == 'value1'


class TestBenchmarkFrameworkIntegration:
    """测试基准测试框架集成"""

    def test_framework_creation_and_destruction(self):
        """测试框架创建和销毁"""
        framework = BenchmarkFramework()
        assert framework is not None

        # 验证所有必需属性都存在
        required_attrs = [
            'disable_background_collection',
            'max_iterations',
            'results'
        ]

        for attr in required_attrs:
            assert hasattr(framework, attr), f"BenchmarkFramework should have attribute '{attr}'"

    def test_framework_with_custom_settings(self):
        """测试使用自定义设置的框架"""
        framework = BenchmarkFramework()
        framework.disable_background_collection = True
        framework.max_iterations = 500

        # 验证设置生效
        assert framework.disable_background_collection is True
        assert framework.max_iterations == 500

        # 验证仍然可以正常操作
        framework.results.append({'iterations': framework.max_iterations})
        assert framework.results[0]['iterations'] == 500

    def test_framework_results_manipulation(self):
        """测试框架结果操作"""
        framework = BenchmarkFramework()

        # 测试添加结果
        framework.results.append({'time': 1.5, 'iterations': 100})
        framework.results.append({'time': 2.0, 'iterations': 200})

        assert len(framework.results) == 2
        assert framework.results[0]['time'] == 1.5
        assert framework.results[1]['iterations'] == 200

        # 测试修改结果
        framework.results[0]['time'] = 1.8
        assert framework.results[0]['time'] == 1.8

        # 测试删除结果
        framework.results.pop(1)
        assert len(framework.results) == 1

    def test_run_benchmark_functionality(self):
        """测试运行基准测试功能"""
        framework = BenchmarkFramework()

        def test_function(x, y):
            return x + y

        # 运行基准测试
        result = framework.run_benchmark("test_add", test_function, x=5, y=3)

        # 验证结果
        assert result['name'] == "test_add"
        assert result['result'] == 8  # 5 + 3
        assert 'execution_time' in result
        assert result['iterations'] == 100  # 默认值
        assert isinstance(result['execution_time'], float)
        assert result['execution_time'] >= 0

        # 验证结果已添加到框架
        assert len(framework.results) == 1
        assert framework.results[0] == result

    def test_get_results_and_clear(self):
        """测试获取结果和清空功能"""
        framework = BenchmarkFramework()

        # 添加一些结果
        framework.results.append({'test': 'result1'})
        framework.results.append({'test': 'result2'})

        # 获取结果
        results = framework.get_results()
        assert len(results) == 2
        assert results[0]['test'] == 'result1'
        assert results[1]['test'] == 'result2'

        # 验证是副本，不是引用
        results[0]['test'] = 'modified'
        assert framework.results[0]['test'] == 'result1'  # 原结果不变

        # 清空结果
        framework.clear_results()
        assert len(framework.results) == 0
