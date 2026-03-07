#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceAnalyzer瓶颈分析测试
补充瓶颈分析方法的测试覆盖率
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

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
    engine_performance_analyzer_module = importlib.import_module('src.monitoring.engine.performance_analyzer')
    BottleneckAnalysis = getattr(engine_performance_analyzer_module, 'BottleneckAnalysis', None)
    if BottleneckAnalysis is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestPerformanceAnalyzerBottleneck:
    """测试PerformanceAnalyzer瓶颈分析功能"""

    @pytest.fixture
    def analyzer(self):
        """创建analyzer实例"""
        try:
            engine_performance_analyzer_module = importlib.import_module('src.monitoring.engine.performance_analyzer')
            PerformanceAnalyzer = getattr(engine_performance_analyzer_module, 'PerformanceAnalyzer', None)
            if PerformanceAnalyzer:
                return PerformanceAnalyzer()
            return None
        except ImportError:
            return None

    def test_trigger_bottleneck_callbacks(self, analyzer):
        """测试触发瓶颈回调"""
        if analyzer is None or BottleneckAnalysis is None:
            pytest.skip("模块不可用")
        
        callback_called = []
        
        def callback(bottleneck):
            callback_called.append(bottleneck)
        
        analyzer.add_bottleneck_callback(callback)
        
        bottleneck = BottleneckAnalysis(
            component="CPU",
            severity="high",
            description="Test bottleneck",
            recommendations=["Recommendation 1"],
            impact_score=0.8,
            confidence=0.9
        )
        
        analyzer._trigger_bottleneck_callbacks(bottleneck)
        
        # 验证回调被调用
        assert len(callback_called) > 0

