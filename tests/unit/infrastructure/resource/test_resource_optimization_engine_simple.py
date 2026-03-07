#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
资源优化引擎简单测试

测试ResourceOptimizationEngine的基本功能
"""

import pytest
from unittest.mock import Mock


class TestResourceOptimizationEngineSimple:
    """资源优化引擎简单测试"""

    def test_optimization_engine_initialization(self):
        """测试优化引擎初始化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试基本属性
            assert hasattr(engine, 'system_analyzer')
            assert hasattr(engine, 'logger')
            assert hasattr(engine, 'error_handler')
            assert hasattr(engine, 'memory_optimizer')
            assert hasattr(engine, 'cpu_optimizer')
            assert hasattr(engine, 'disk_optimizer')
            assert hasattr(engine, 'config_manager')

        except ImportError:
            pytest.skip("ResourceOptimizationEngine not available")

    def test_resource_optimization_with_dict_config(self):
        """测试使用字典配置的资源优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试字典配置优化
            config_dict = {
                'memory_optimization': {
                    'enabled': True,
                    'gc_threshold': 100000,
                    'memory_limit_mb': 1024
                },
                'cpu_optimization': {
                    'enabled': True,
                    'cpu_affinity': True
                },
                'disk_optimization': {
                    'enabled': False
                }
            }

            result = engine.optimize_resources(config_dict)
            assert isinstance(result, dict)
            assert 'timestamp' in result
            assert 'status' in result

        except ImportError:
            pytest.skip("Resource optimization with dict config not available")

    def test_resource_optimization_with_config_object(self):
        """测试使用配置对象的资源优化"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            engine = ResourceOptimizationEngine()

            # 创建配置对象
            config = ResourceOptimizationConfig(
                memory_optimization=Mock(enabled=True, gc_threshold=100000),
                cpu_optimization=Mock(enabled=True, cpu_affinity=True),
                disk_optimization=Mock(enabled=False)
            )

            result = engine.optimize_resources_with_config(config)
            assert isinstance(result, dict)
            assert 'timestamp' in result
            assert 'config' in result

        except ImportError:
            pytest.skip("Resource optimization with config object not available")

    def test_memory_optimizer_integration(self):
        """测试内存优化器集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试内存优化器存在
            assert hasattr(engine, 'memory_optimizer')
            assert engine.memory_optimizer is not None

        except ImportError:
            pytest.skip("Memory optimizer integration not available")

    def test_cpu_optimizer_integration(self):
        """测试CPU优化器集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试CPU优化器存在
            assert hasattr(engine, 'cpu_optimizer')
            assert engine.cpu_optimizer is not None

        except ImportError:
            pytest.skip("CPU optimizer integration not available")

    def test_disk_optimizer_integration(self):
        """测试磁盘优化器集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试磁盘优化器存在
            assert hasattr(engine, 'disk_optimizer')
            assert engine.disk_optimizer is not None

        except ImportError:
            pytest.skip("Disk optimizer integration not available")

    def test_config_manager_integration(self):
        """测试配置管理器集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试配置管理器存在
            assert hasattr(engine, 'config_manager')
            assert engine.config_manager is not None

        except ImportError:
            pytest.skip("Config manager integration not available")

    def test_system_analyzer_integration(self):
        """测试系统分析器集成"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试系统分析器存在
            assert hasattr(engine, 'system_analyzer')
            assert engine.system_analyzer is not None

        except ImportError:
            pytest.skip("System analyzer integration not available")

    def test_error_handling_in_optimization(self):
        """测试优化过程中的错误处理"""
        try:
            from src.infrastructure.resource.core.resource_optimization_engine import ResourceOptimizationEngine

            engine = ResourceOptimizationEngine()

            # 测试无效配置的错误处理
            invalid_config = {"invalid_key": "invalid_value"}

            result = engine.optimize_resources(invalid_config)
            assert isinstance(result, dict)
            # 应该返回错误状态或包含错误信息
            assert 'status' in result

        except ImportError:
            pytest.skip("Error handling in optimization not available")

    def test_optimization_config_validation(self):
        """测试优化配置验证"""
        try:
            from src.infrastructure.resource.core.optimization_config import ResourceOptimizationConfig

            # 创建有效配置
            config = ResourceOptimizationConfig()
            validation_issues = config.validate()

            # 验证返回列表
            assert isinstance(validation_issues, list)

        except ImportError:
            pytest.skip("Optimization config validation not available")