#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试可选组件加载工具
"""

import pytest
from unittest.mock import patch, MagicMock


def test_optional_import_success():
    """测试成功导入可选组件"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 测试导入一个存在的模块和类
    result = optional_components._optional_import(
        "src.infrastructure.monitoring.services.metrics_collector.MetricsCollector"
    )
    assert result is not None
    assert hasattr(result, '__init__')


def test_optional_import_failure():
    """测试导入失败返回None"""
    from src.infrastructure.monitoring.services import optional_components
    
    result = optional_components._optional_import("invalid.module.Class")
    assert result is None


def test_optional_import_module_error():
    """测试模块导入错误"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 模拟模块导入失败
    with patch('importlib.import_module', side_effect=ImportError("Module not found")):
        result = optional_components._optional_import("some.module.Class")
        assert result is None


def test_optional_import_attribute_error():
    """测试属性获取错误"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 模拟模块存在但属性不存在
    mock_module = MagicMock()
    del mock_module.NonExistentClass  # 确保属性不存在
    
    with patch('importlib.import_module', return_value=mock_module):
        result = optional_components._optional_import("some.module.NonExistentClass")
        assert result is None


def test_get_optional_component_success():
    """测试成功获取可选组件"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存以确保测试准确性
    optional_components.get_optional_component.cache_clear()
    
    component = optional_components.get_optional_component("MetricsCollector")
    assert component is not None
    assert hasattr(component, '__init__')


def test_get_optional_component_not_found():
    """测试组件不存在时返回None"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存
    optional_components.get_optional_component.cache_clear()
    
    component = optional_components.get_optional_component("NonExistentComponent")
    assert component is None


def test_get_optional_component_caching():
    """测试组件获取的缓存机制"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存
    optional_components.get_optional_component.cache_clear()
    
    # 第一次调用
    component1 = optional_components.get_optional_component("MetricsCollector")
    
    # 第二次调用应该使用缓存
    component2 = optional_components.get_optional_component("MetricsCollector")
    
    assert component1 is component2  # 应该是同一个对象（缓存）


def test_get_optional_component_multiple_candidates():
    """测试多个候选路径的情况"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存
    optional_components.get_optional_component.cache_clear()
    
    # 测试一个存在的组件（可能有多个候选路径）
    component = optional_components.get_optional_component("MetricsCollector")
    assert component is not None


def test_get_optional_component_all_candidates_fail():
    """测试所有候选路径都失败的情况"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存
    optional_components.get_optional_component.cache_clear()
    
    # 测试一个不存在的组件
    component = optional_components.get_optional_component("NonExistentComponent")
    assert component is None


def test_get_optional_component_empty_name():
    """测试空名称的情况"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 清除缓存
    optional_components.get_optional_component.cache_clear()
    
    component = optional_components.get_optional_component("")
    assert component is None


def test_module_imports():
    """测试模块导入"""
    # 直接导入模块，确保所有导入语句被执行
    from src.infrastructure.monitoring.services import optional_components
    
    # 验证模块属性存在
    assert hasattr(optional_components, '_COMPONENT_PATHS')
    assert hasattr(optional_components, '_optional_import')
    assert hasattr(optional_components, 'get_optional_component')
    
    # 验证常量字典包含预期的键
    assert 'MetricsCollector' in optional_components._COMPONENT_PATHS
    assert 'AlertManager' in optional_components._COMPONENT_PATHS
    assert 'DataPersistence' in optional_components._COMPONENT_PATHS
    assert 'OptimizationEngine' in optional_components._COMPONENT_PATHS
    assert 'HealthCheckInterface' in optional_components._COMPONENT_PATHS


def test_optional_import_with_getattr_error():
    """测试getattr抛出异常的情况"""
    from src.infrastructure.monitoring.services import optional_components
    from unittest.mock import patch, MagicMock
    
    # 模拟getattr抛出异常
    mock_module = MagicMock()
    with patch('importlib.import_module', return_value=mock_module):
        with patch('builtins.getattr', side_effect=AttributeError("No attribute")):
            result = optional_components._optional_import("some.module.Class")
            assert result is None


def test_optional_import_with_rsplit():
    """测试rsplit的正常使用"""
    from src.infrastructure.monitoring.services import optional_components
    
    # 测试rsplit正常工作
    result = optional_components._optional_import(
        "src.infrastructure.monitoring.services.metrics_collector.MetricsCollector"
    )
    # 应该成功导入
    assert result is not None

