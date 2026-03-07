#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控层异常工具函数测试
补充exceptions.py中工具函数的完整测试
"""

import sys
import importlib
from pathlib import Path
import pytest

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    exceptions_module = importlib.import_module('src.monitoring.core.exceptions')
    MonitoringException = getattr(exceptions_module, 'MonitoringException', None)
    MetricsCollectionError = getattr(exceptions_module, 'MetricsCollectionError', None)
    ConfigurationError = getattr(exceptions_module, 'ConfigurationError', None)
    handle_monitoring_exception = getattr(exceptions_module, 'handle_monitoring_exception', None)
    validate_metric_data = getattr(exceptions_module, 'validate_metric_data', None)
    validate_config_key = getattr(exceptions_module, 'validate_config_key', None)
    
    if MonitoringException is None:
        pytest.skip("监控异常模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控异常模块导入失败", allow_module_level=True)


class TestValidateMetricData:
    """测试validate_metric_data函数"""

    def test_validate_metric_data_valid(self):
        """测试验证有效的指标数据"""
        # 不抛出异常
        validate_metric_data('test_metric', 100)
        validate_metric_data('test_metric', 100.5)
        validate_metric_data('test_metric', 'value')
        validate_metric_data('test_metric', True)

    def test_validate_metric_data_none(self):
        """测试验证None值"""
        with pytest.raises(MetricsCollectionError) as exc_info:
            validate_metric_data('test_metric', None)
        
        assert exc_info.value.metric_name == 'test_metric'
        assert '不能为空' in str(exc_info.value)

    def test_validate_metric_data_with_type_check_int(self):
        """测试类型检查-整数"""
        validate_metric_data('test_metric', 100, expected_type=int)
        
        with pytest.raises(MetricsCollectionError) as exc_info:
            validate_metric_data('test_metric', 100.5, expected_type=int)
        
        assert exc_info.value.metric_name == 'test_metric'
        assert '类型错误' in str(exc_info.value)

    def test_validate_metric_data_with_type_check_float(self):
        """测试类型检查-浮点数"""
        validate_metric_data('test_metric', 100.5, expected_type=float)
        
        with pytest.raises(MetricsCollectionError) as exc_info:
            validate_metric_data('test_metric', 100, expected_type=float)
        
        assert exc_info.value.metric_name == 'test_metric'

    def test_validate_metric_data_with_type_check_str(self):
        """测试类型检查-字符串"""
        validate_metric_data('test_metric', 'value', expected_type=str)
        
        with pytest.raises(MetricsCollectionError) as exc_info:
            validate_metric_data('test_metric', 100, expected_type=str)
        
        assert exc_info.value.metric_name == 'test_metric'


class TestValidateConfigKey:
    """测试validate_config_key函数"""

    def test_validate_config_key_exists_not_required(self):
        """测试验证存在的配置键-非必需"""
        config = {'key1': 'value1'}
        # 不抛出异常
        validate_config_key(config, 'key1', required=False)

    def test_validate_config_key_missing_not_required(self):
        """测试验证缺失的配置键-非必需"""
        config = {}
        # 不抛出异常（非必需）
        validate_config_key(config, 'missing_key', required=False)

    def test_validate_config_key_required_exists(self):
        """测试验证必需的配置键-存在"""
        config = {'required_key': 'value'}
        # 不抛出异常
        validate_config_key(config, 'required_key', required=True)

    def test_validate_config_key_required_missing(self):
        """测试验证必需的配置键-缺失"""
        config = {}
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_key(config, 'required_key', required=True)
        
        assert exc_info.value.config_key == 'required_key'
        assert '必需配置项缺失' in str(exc_info.value)

    def test_validate_config_key_none_value(self):
        """测试配置键值为None"""
        config = {'key1': None}
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_key(config, 'key1', required=False)
        
        assert exc_info.value.config_key == 'key1'
        assert '不能为空' in str(exc_info.value)

    def test_validate_config_key_required_missing_none_value(self):
        """测试必需配置键值为None"""
        config = {'required_key': None}
        with pytest.raises(ConfigurationError) as exc_info:
            validate_config_key(config, 'required_key', required=True)
        
        assert exc_info.value.config_key == 'required_key'


class TestHandleMonitoringException:
    """测试handle_monitoring_exception装饰器"""

    def test_handle_monitoring_exception_normal_execution(self):
        """测试装饰器正常执行"""
        @handle_monitoring_exception
        def test_func():
            return "success"
        
        result = test_func()
        assert result == "success"

    def test_handle_monitoring_exception_preserves_monitoring_exception(self):
        """测试装饰器保留监控异常"""
        @handle_monitoring_exception
        def test_func():
            raise MetricsCollectionError("Test error", "test_metric")
        
        with pytest.raises(MetricsCollectionError) as exc_info:
            test_func()
        
        assert exc_info.value.metric_name == 'test_metric'

    def test_handle_monitoring_exception_wraps_general_exception(self):
        """测试装饰器包装一般异常"""
        @handle_monitoring_exception
        def test_func():
            raise ValueError("General error")
        
        with pytest.raises(MonitoringException) as exc_info:
            test_func()
        
        assert '意外错误' in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_handle_monitoring_exception_with_args(self):
        """测试装饰器处理带参数的函数"""
        @handle_monitoring_exception
        def test_func(a, b):
            return a + b
        
        result = test_func(1, 2)
        assert result == 3

    def test_handle_monitoring_exception_with_kwargs(self):
        """测试装饰器处理带关键字参数的函数"""
        @handle_monitoring_exception
        def test_func(a=1, b=2):
            return a + b
        
        result = test_func(a=10, b=20)
        assert result == 30

    def test_handle_monitoring_exception_exception_chaining(self):
        """测试装饰器异常链"""
        @handle_monitoring_exception
        def test_func():
            raise RuntimeError("Original error")
        
        with pytest.raises(MonitoringException) as exc_info:
            test_func()
        
        # 检查异常链
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, RuntimeError)



