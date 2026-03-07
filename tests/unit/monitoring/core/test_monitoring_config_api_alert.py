#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig API告警测试
补充check_alerts中API慢响应告警的测试（行115）
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
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("监控配置模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestMonitoringConfigApiAlert:
    """测试API慢响应告警"""

    def test_check_alerts_api_slow_threshold(self):
        """测试API响应时间超过1000ms触发告警"""
        monitoring_system = MonitoringSystem()
        
        # 记录API响应时间刚好超过1000ms
        monitoring_system.record_metric('api_response_time', 1001.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该产生API慢响应告警
        api_slow_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_slow_alerts) > 0
        assert api_slow_alerts[0]['severity'] == 'warning'
        assert 'API响应时间过慢' in api_slow_alerts[0]['message']

    def test_check_alerts_api_slow_exact_threshold(self):
        """测试API响应时间刚好等于阈值边界"""
        monitoring_system = MonitoringSystem()
        
        # 记录API响应时间刚好1000ms（应该不触发，因为是>1000）
        monitoring_system.record_metric('api_response_time', 1000.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该不产生告警（因为条件是>1000）
        api_slow_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_slow_alerts) == 0

    def test_check_alerts_api_slow_very_high(self):
        """测试API响应时间非常高的情况"""
        monitoring_system = MonitoringSystem()
        
        # 记录非常高的API响应时间
        monitoring_system.record_metric('api_response_time', 5000.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该产生告警
        api_slow_alerts = [a for a in alerts if a.get('type') == 'api_slow']
        assert len(api_slow_alerts) > 0
        assert api_slow_alerts[0]['severity'] == 'warning'

