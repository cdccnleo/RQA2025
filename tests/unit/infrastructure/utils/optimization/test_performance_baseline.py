#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层性能基准组件测试

测试目标：提升utils/optimization/performance_baseline.py的真实覆盖率
实际导入和使用src.infrastructure.utils.optimization.performance_baseline模块
"""

import pytest
from datetime import datetime


class TestPerformanceBaseline:
    """测试性能基准数据类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline()
        assert baseline.test_name == "default_test"
        assert baseline.test_category == "default_category"
        assert baseline.baseline_execution_time == 1.0
        assert baseline.baseline_operations_per_second == 1.0
        assert baseline.threshold_percentage == 10.0
        assert baseline.created_at is not None
        assert baseline.updated_at is not None
    
    def test_init_with_params(self):
        """测试使用参数初始化"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            test_name="test_001",
            test_category="io_test",
            baseline_execution_time=0.5,
            threshold_percentage=5.0
        )
        assert baseline.test_name == "test_001"
        assert baseline.test_category == "io_test"
        assert baseline.baseline_execution_time == 0.5
        assert baseline.threshold_percentage == 5.0
    
    def test_post_init(self):
        """测试初始化后处理"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(baseline_execution_time=2.0)
        assert baseline.min_execution_time == 2.0
        assert baseline.max_execution_time == 2.0
        assert baseline.created_at is not None
        assert baseline.updated_at is not None
    
    def test_update_baseline(self):
        """测试更新基准值"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(baseline_execution_time=1.0)
        initial_count = baseline.sample_count
        
        baseline.update_baseline(
            execution_time=0.8,
            operations_per_second=2.0,
            memory_usage=100.0,
            cpu_usage=50.0
        )
        
        assert baseline.sample_count == initial_count + 1
        assert baseline.min_execution_time == 0.8
        assert baseline.max_execution_time == 1.0
    
    def test_is_within_threshold_execution_time(self):
        """测试检查执行时间是否在阈值内"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            baseline_execution_time=1.0,
            threshold_percentage=10.0
        )
        
        # 在阈值内（±10%）
        assert baseline.is_within_threshold(1.05, "execution_time") is True
        assert baseline.is_within_threshold(0.95, "execution_time") is True
        
        # 超出阈值
        assert baseline.is_within_threshold(1.15, "execution_time") is False
        assert baseline.is_within_threshold(0.85, "execution_time") is False
    
    def test_is_within_threshold_operations_per_second(self):
        """测试检查操作数是否在阈值内"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            baseline_operations_per_second=100.0,
            threshold_percentage=10.0
        )
        
        assert baseline.is_within_threshold(105.0, "operations_per_second") is True
        assert baseline.is_within_threshold(95.0, "operations_per_second") is True
        assert baseline.is_within_threshold(115.0, "operations_per_second") is False
    
    def test_is_within_threshold_invalid_metric(self):
        """测试无效指标"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline()
        assert baseline.is_within_threshold(100.0, "invalid_metric") is True
    
    def test_compare_performance(self):
        """测试比较性能"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(
            baseline_execution_time=1.0,
            threshold_percentage=10.0
        )
        
        result = baseline.compare_performance(1.05)
        assert isinstance(result, dict)
        assert "within_threshold" in result
        assert "change_percentage" in result
        assert "status" in result
    
    def test_compare_performance_invalid(self):
        """测试比较无效性能值"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline()
        result = baseline.compare_performance(0.0)
        
        assert result["status"] == "invalid"
        assert result["within_threshold"] is True
    
    def test_compare_performance_zero_baseline(self):
        """测试零基准值比较"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline
        
        baseline = PerformanceBaseline(baseline_execution_time=0.0)
        result = baseline.compare_performance(1.0)
        
        assert result["within_threshold"] is True
    
    def test_to_dict(self):
        """测试转换为字典"""
        from src.infrastructure.utils.optimization.performance_baseline import PerformanceBaseline, asdict
        
        baseline = PerformanceBaseline(test_name="test_001")
        result = asdict(baseline)
        
        assert isinstance(result, dict)
        assert result["test_name"] == "test_001"
        assert "baseline_execution_time" in result
        assert "created_at" in result

