#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig告警检查边界情况测试
补充check_alerts方法的边界情况和未覆盖分支
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import patch
from datetime import datetime

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
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("监控配置模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestMonitoringSystemAlertEdgeCases:
    """测试MonitoringSystem告警检查边界情况"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_check_alerts_cpu_threshold_boundary(self, monitoring_system):
        """测试CPU告警阈值边界（正好等于80）"""
        # 正好等于80，不应该触发告警
        monitoring_system.record_metric('cpu_usage', 80.0)
        alerts = monitoring_system.check_alerts()
        
        cpu_alerts = [a for a in alerts if a.get('type') == 'cpu_high']
        assert len(cpu_alerts) == 0

    def test_check_alerts_cpu_threshold_above(self, monitoring_system):
        """测试CPU告警阈值之上（81）"""
        monitoring_system.record_metric('cpu_usage', 81.0)
        alerts = monitoring_system.check_alerts()
        
        cpu_alerts = [a for a in alerts if a.get('type') == 'cpu_high']
        assert len(cpu_alerts) > 0

    def test_check_alerts_memory_threshold_boundary(self, monitoring_system):
        """测试内存告警阈值边界（正好等于70）"""
        # 正好等于70，不应该触发告警
        monitoring_system.record_metric('memory_usage', 70.0)
        alerts = monitoring_system.check_alerts()
        
        memory_alerts = [a for a in alerts if a.get('type') == 'memory_high']
        assert len(memory_alerts) == 0

    def test_check_alerts_memory_threshold_above(self, monitoring_system):
        """测试内存告警阈值之上（71）"""
        monitoring_system.record_metric('memory_usage', 71.0)
        alerts = monitoring_system.check_alerts()
        
        memory_alerts = [a for a in alerts if a.get('type') == 'memory_high']
        assert len(memory_alerts) > 0

    def test_check_alerts_api_threshold_boundary(self, monitoring_system):
        """测试API告警阈值边界（正好等于1000）"""
        # 正好等于1000，不应该触发告警
        monitoring_system.record_metric('api_response_time', 1000.0)
        alerts = monitoring_system.check_alerts()
        
        api_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_alerts) == 0

    def test_check_alerts_api_threshold_above(self, monitoring_system):
        """测试API告警阈值之上（1001）"""
        monitoring_system.record_metric('api_response_time', 1001.0)
        alerts = monitoring_system.check_alerts()
        
        api_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_alerts) > 0

    def test_check_alerts_empty_metrics(self, monitoring_system):
        """测试空指标时不触发告警"""
        alerts = monitoring_system.check_alerts()
        
        assert isinstance(alerts, list)
        assert len(alerts) == 0

    def test_check_alerts_multiple_metrics_low(self, monitoring_system):
        """测试多个指标都在阈值以下"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        monitoring_system.record_metric('memory_usage', 60.0)
        monitoring_system.record_metric('api_response_time', 500.0)
        
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) == 0

    def test_check_alerts_multiple_metrics_high(self, monitoring_system):
        """测试多个指标都超过阈值"""
        monitoring_system.record_metric('cpu_usage', 90.0)
        monitoring_system.record_metric('memory_usage', 80.0)
        monitoring_system.record_metric('api_response_time', 2000.0)
        
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) == 3
        alert_types = [a.get('type') for a in alerts]
        assert 'cpu_high' in alert_types
        assert 'memory_high' in alert_types
        assert 'api_slow' in alert_types

    def test_check_alerts_metric_list_with_empty(self, monitoring_system):
        """测试指标列表为空的情况"""
        # 创建一个空的指标列表
        monitoring_system.metrics['cpu_usage'] = []
        
        alerts = monitoring_system.check_alerts()
        
        # 不应该触发告警
        assert len(alerts) == 0

    def test_check_alerts_uses_last_metric(self, monitoring_system):
        """测试使用最后一个指标值进行告警检查"""
        # 添加多个指标值，最后一个超过阈值
        monitoring_system.record_metric('cpu_usage', 50.0)
        monitoring_system.record_metric('cpu_usage', 60.0)
        monitoring_system.record_metric('cpu_usage', 85.0)  # 最后一个超过阈值
        
        alerts = monitoring_system.check_alerts()
        
        cpu_alerts = [a for a in alerts if a.get('type') == 'cpu_high']
        assert len(cpu_alerts) > 0

    def test_check_alerts_last_metric_below_threshold(self, monitoring_system):
        """测试最后一个指标值在阈值以下"""
        # 添加多个指标值，最后一个在阈值以下
        monitoring_system.record_metric('cpu_usage', 90.0)
        monitoring_system.record_metric('cpu_usage', 50.0)  # 最后一个在阈值以下
        
        alerts = monitoring_system.check_alerts()
        
        # 应该不触发告警（使用最后一个值）
        cpu_alerts = [a for a in alerts if a.get('type') == 'cpu_high']
        assert len(cpu_alerts) == 0



