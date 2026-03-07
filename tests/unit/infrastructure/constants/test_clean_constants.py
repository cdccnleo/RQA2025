"""
基础设施层常量测试 - 干净版本
"""

import pytest
import sys
from pathlib import Path

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 使用importlib动态导入
import importlib
constants_module = importlib.import_module('src.infrastructure.constants')


class TestConstantsModule:
    """常量模块测试"""

    def test_module_import(self):
        """测试模块导入"""
        assert constants_module is not None

    def test_config_constants_class(self):
        """测试ConfigConstants类"""
        assert hasattr(constants_module, 'ConfigConstants')
        config_cls = constants_module.ConfigConstants
        assert config_cls is not None
        assert hasattr(config_cls, '__name__')

    def test_format_constants_class(self):
        """测试FormatConstants类"""
        assert hasattr(constants_module, 'FormatConstants')
        format_cls = constants_module.FormatConstants
        assert format_cls is not None

    def test_http_constants_class(self):
        """测试HTTPConstants类"""
        assert hasattr(constants_module, 'HTTPConstants')
        http_cls = constants_module.HTTPConstants
        assert http_cls is not None

    def test_performance_constants_class(self):
        """测试PerformanceConstants类"""
        assert hasattr(constants_module, 'PerformanceConstants')
        perf_cls = constants_module.PerformanceConstants
        assert perf_cls is not None

    def test_size_constants_class(self):
        """测试SizeConstants类"""
        assert hasattr(constants_module, 'SizeConstants')
        size_cls = constants_module.SizeConstants
        assert size_cls is not None

    def test_threshold_constants_class(self):
        """测试ThresholdConstants类"""
        assert hasattr(constants_module, 'ThresholdConstants')
        threshold_cls = constants_module.ThresholdConstants
        assert threshold_cls is not None

    def test_time_constants_class(self):
        """测试TimeConstants类"""
        assert hasattr(constants_module, 'TimeConstants')
        time_cls = constants_module.TimeConstants
        assert time_cls is not None

    def test_constants_instances(self):
        """测试常量实例"""
        # 测试config_constants实例
        assert hasattr(constants_module, 'config_constants')
        assert constants_module.config_constants is not None

        # 测试其他常量实例
        assert hasattr(constants_module, 'format_constants')
        assert hasattr(constants_module, 'http_constants')
        assert hasattr(constants_module, 'performance_constants')
        assert hasattr(constants_module, 'size_constants')
        assert hasattr(constants_module, 'threshold_constants')
        assert hasattr(constants_module, 'time_constants')
