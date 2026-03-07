#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig指标限制测试
测试record_metric方法的指标数量限制逻辑
"""

import pytest
from unittest.mock import patch

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
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(core_monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestMonitoringConfigMetricLimit:
    """测试指标数量限制逻辑"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_record_metric_limit_exactly_1000(self, monitoring_system):
        """测试记录恰好1000个指标"""
        # 记录恰好1000个指标
        for i in range(1000):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 应该保持1000个指标
        assert len(monitoring_system.metrics['limited_metric']) == 1000

    def test_record_metric_limit_exceeded_1001(self, monitoring_system):
        """测试记录1001个指标时触发限制"""
        # 记录1001个指标
        for i in range(1001):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 应该被限制在500个（最近500个）
        assert len(monitoring_system.metrics['limited_metric']) == 500
        # 验证保留的是最近的500个（值应该是500-1000）
        assert monitoring_system.metrics['limited_metric'][0]['value'] == 501.0
        assert monitoring_system.metrics['limited_metric'][-1]['value'] == 1000.0

    def test_record_metric_limit_exceeded_1500(self, monitoring_system):
        """测试记录1500个指标时触发限制"""
        # 记录1500个指标
        # 每次记录后都会检查，所以会在1001时触发限制，保留最后500个
        # 然后继续记录，直到1500
        for i in range(1500):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 验证最终数量不超过1000（因为每次超过1000都会裁剪）
        metrics = monitoring_system.metrics['limited_metric']
        assert len(metrics) <= 1000
        # 验证包含最新的指标（最后几个应该存在）
        assert metrics[-1]['value'] == 1499.0  # 最后一个是1499
        # 验证保留的是较新的指标（第一个值应该大于等于某个值）
        assert metrics[0]['value'] >= 500.0  # 至少保留了较新的指标

    def test_record_metric_limit_exceeded_multiple_times(self, monitoring_system):
        """测试多次超过限制时的行为"""
        # 第一次超过限制
        for i in range(1001):
            monitoring_system.record_metric('limited_metric', float(i))
        
        assert len(monitoring_system.metrics['limited_metric']) == 500
        
        # 继续记录，每次记录后都会检查限制
        for i in range(1001, 1501):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 验证最终数量不超过1000
        metrics = monitoring_system.metrics['limited_metric']
        assert len(metrics) <= 1000
        # 验证包含最新的指标
        assert metrics[-1]['value'] == 1500.0  # 最后一个是1500
        # 验证保留的是较新的指标
        assert metrics[0]['value'] >= 501.0  # 至少从501开始（第一次限制后保留的是501-1000）

    def test_record_metric_limit_multiple_metrics(self, monitoring_system):
        """测试多个指标各自的限制"""
        # 第一个指标超过限制
        for i in range(1001):
            monitoring_system.record_metric('metric1', float(i))
        
        # 第二个指标刚好1000个
        for i in range(1000):
            monitoring_system.record_metric('metric2', float(i))
        
        # 第三个指标小于1000个
        for i in range(500):
            monitoring_system.record_metric('metric3', float(i))
        
        # 验证各自的限制
        assert len(monitoring_system.metrics['metric1']) == 500
        assert len(monitoring_system.metrics['metric2']) == 1000
        assert len(monitoring_system.metrics['metric3']) == 500

    def test_record_metric_limit_preserves_latest(self, monitoring_system):
        """测试限制逻辑保留的是最新的指标"""
        # 记录1001个指标
        for i in range(1001):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 验证保留的是最后500个
        metrics = monitoring_system.metrics['limited_metric']
        # 第一个应该是第501个（索引500）
        assert metrics[0]['value'] == 501.0
        # 最后一个应该是第1000个（索引1000）
        assert metrics[-1]['value'] == 1000.0
        
        # 验证顺序正确
        for i in range(len(metrics) - 1):
            assert metrics[i]['value'] < metrics[i + 1]['value']

    def test_record_metric_limit_timestamp_order(self, monitoring_system):
        """测试限制后时间戳的顺序正确"""
        import time
        time.sleep(0.001)  # 确保时间戳不同
        
        # 记录1001个指标
        for i in range(1001):
            monitoring_system.record_metric('limited_metric', float(i))
            time.sleep(0.0001)  # 小延迟确保时间戳不同
        
        # 验证时间戳顺序正确（保留的500个中，时间戳应该是递增的）
        metrics = monitoring_system.metrics['limited_metric']
        for i in range(len(metrics) - 1):
            assert metrics[i]['timestamp'] <= metrics[i + 1]['timestamp']

