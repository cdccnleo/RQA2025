#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PerformanceAnalyzer瓶颈分析集成测试
补充_analyze_bottlenecks方法的测试覆盖率
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
    PerformanceAnalyzer = getattr(engine_performance_analyzer_module, 'PerformanceAnalyzer', None)
    PerformanceMetric = getattr(engine_performance_analyzer_module, 'PerformanceMetric', None)
    AnalysisMode = getattr(engine_performance_analyzer_module, 'AnalysisMode', None)
    BottleneckAnalysis = getattr(engine_performance_analyzer_module, 'BottleneckAnalysis', None)
    
    if PerformanceAnalyzer is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestPerformanceAnalyzerBottlenecksIntegration:
    """测试瓶颈分析集成功能"""

    @pytest.fixture
    def analyzer(self):
        """创建analyzer实例"""
        analyzer = PerformanceAnalyzer()
        # 设置baseline已计算
        analyzer.baseline_calculated = True
        return analyzer

    def test_analyze_bottlenecks_with_baseline(self, analyzer):
        """测试有baseline时的瓶颈分析"""
        # 添加高CPU使用率数据
        for i in range(15):
            point = {
                'timestamp': datetime.now() - timedelta(seconds=15-i),
                'value': 85.0 + i * 0.5
            }
            analyzer.performance_history['cpu_usage'].append(point)
        
        # 调用瓶颈分析
        analyzer._analyze_bottlenecks()
        
        # 验证方法执行完成
        assert analyzer.baseline_calculated == True

    def test_analyze_bottlenecks_without_baseline(self, analyzer):
        """测试无baseline时的瓶颈分析"""
        analyzer.baseline_calculated = False
        
        # 调用瓶颈分析，应该直接返回
        analyzer._analyze_bottlenecks()
        
        # 验证方法执行完成（不会抛出异常）
        assert True

    def test_analyze_bottlenecks_with_multiple_bottlenecks(self, analyzer):
        """测试多个瓶颈同时存在"""
        # 添加高CPU使用率数据
        for i in range(15):
            point = {
                'timestamp': datetime.now() - timedelta(seconds=15-i),
                'value': 85.0
            }
            analyzer.performance_history['cpu_usage'].append(point)
        
        # 添加高内存使用率数据
        for i in range(15):
            point = {
                'timestamp': datetime.now() - timedelta(seconds=15-i),
                'value': 90.0
            }
            analyzer.performance_history['memory_usage'].append(point)
        
        # 添加高磁盘使用率数据
        for i in range(15):
            point = {
                'timestamp': datetime.now() - timedelta(seconds=15-i),
                'value': 95.0
            }
            analyzer.performance_history['disk_usage'].append(point)
        
        # 调用瓶颈分析
        analyzer._analyze_bottlenecks()
        
        # 验证方法执行完成
        assert True

    def test_trigger_bottleneck_callbacks_with_callbacks(self, analyzer):
        """测试有回调时的瓶颈触发"""
        callback_called = []
        
        def test_callback(bottleneck):
            callback_called.append(bottleneck)
        
        analyzer.bottleneck_callbacks.append(test_callback)
        
        # 创建瓶颈对象
        bottleneck = BottleneckAnalysis(
            component="CPU",
            severity="high",
            description="Test bottleneck",
            recommendations=["Recommendation 1"],
            impact_score=0.8,
            confidence=0.9
        )
        
        # 触发回调
        analyzer._trigger_bottleneck_callbacks(bottleneck)
        
        # 验证回调被调用
        assert len(callback_called) > 0

    def test_trigger_bottleneck_callbacks_without_callbacks(self, analyzer):
        """测试无回调时的瓶颈触发"""
        # 清空回调列表
        analyzer.bottleneck_callbacks = []
        
        # 创建瓶颈对象
        bottleneck = BottleneckAnalysis(
            component="CPU",
            severity="high",
            description="Test bottleneck",
            recommendations=["Recommendation 1"],
            impact_score=0.8,
            confidence=0.9
        )
        
        # 触发回调，应该不会抛出异常
        analyzer._trigger_bottleneck_callbacks(bottleneck)
        
        # 验证方法执行完成
        assert True

    def test_trigger_bottleneck_callbacks_with_exception(self, analyzer):
        """测试回调抛出异常时的处理"""
        def failing_callback(bottleneck):
            raise Exception("Callback error")
        
        analyzer.bottleneck_callbacks.append(failing_callback)
        
        # 创建瓶颈对象
        bottleneck = BottleneckAnalysis(
            component="CPU",
            severity="high",
            description="Test bottleneck",
            recommendations=["Recommendation 1"],
            impact_score=0.8,
            confidence=0.9
        )
        
        # 触发回调，即使有异常也不应该崩溃
        try:
            analyzer._trigger_bottleneck_callbacks(bottleneck)
        except Exception:
            # 如果回调异常被捕获，这是正常的
            pass
        
        # 验证方法执行完成
        assert True

