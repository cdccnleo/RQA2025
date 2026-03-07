#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig告警阈值测试
补充check_alerts方法中具体告警规则和阈值判断的详细测试
"""

import pytest

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

class TestMonitoringSystemCheckAlertsThresholds:
    """测试MonitoringSystem告警阈值判断"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_check_alerts_cpu_exact_threshold_high(self, monitoring_system):
        """测试CPU使用率正好等于80%（不触发告警，因为>80才触发）"""
        monitoring_system.record_metric('cpu_usage', 80.0)
        alerts = monitoring_system.check_alerts()
        
        # 因为代码是 > 80，所以80.0不触发告警
        cpu_alerts = [a for a in alerts if 'cpu' in a.get('message', '').lower() or 'CPU' in a.get('message', '')]
        assert len(cpu_alerts) == 0

    def test_check_alerts_cpu_just_below_threshold_high(self, monitoring_system):
        """测试CPU使用率刚好低于80%"""
        monitoring_system.record_metric('cpu_usage', 79.9)
        alerts = monitoring_system.check_alerts()
        
        # 应该不触发告警（因为<80.0）
        cpu_alerts = [a for a in alerts if 'cpu' in a.get('message', '').lower()]
        assert len(cpu_alerts) == 0

    def test_check_alerts_cpu_just_above_threshold_high(self, monitoring_system):
        """测试CPU使用率刚好高于80%"""
        monitoring_system.record_metric('cpu_usage', 80.1)
        alerts = monitoring_system.check_alerts()
        
        # 应该触发告警
        cpu_alerts = [a for a in alerts if 'cpu' in a.get('message', '').lower()]
        assert len(cpu_alerts) > 0

    def test_check_alerts_cpu_exact_threshold_critical(self, monitoring_system):
        """测试CPU使用率正好等于95%（临界阈值）"""
        monitoring_system.record_metric('cpu_usage', 95.0)
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) > 0
        # 验证告警内容包含CPU相关信息
        alert_messages = [a['message'] for a in alerts]
        assert any('CPU' in msg or 'cpu' in msg.lower() for msg in alert_messages)

    def test_check_alerts_cpu_between_thresholds(self, monitoring_system):
        """测试CPU使用率在80%和95%之间"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        alerts = monitoring_system.check_alerts()
        
        # 应该触发告警（因为>80.0）
        cpu_alerts = [a for a in alerts if 'cpu' in a.get('message', '').lower()]
        assert len(cpu_alerts) > 0

    def test_check_alerts_memory_exact_threshold_high(self, monitoring_system):
        """测试内存使用率正好等于70%（不触发告警，因为>70才触发）"""
        monitoring_system.record_metric('memory_usage', 70.0)
        alerts = monitoring_system.check_alerts()
        
        # 因为代码是 > 70，所以70.0不触发告警
        memory_alerts = [a for a in alerts if '内存' in a.get('message', '') or 'memory' in a.get('message', '').lower()]
        assert len(memory_alerts) == 0

    def test_check_alerts_memory_just_below_threshold_high(self, monitoring_system):
        """测试内存使用率刚好低于70%"""
        monitoring_system.record_metric('memory_usage', 69.9)
        alerts = monitoring_system.check_alerts()
        
        # 应该不触发告警（因为<70.0）
        memory_alerts = [a for a in alerts if '内存' in a.get('message', '') or 'memory' in a.get('message', '').lower()]
        assert len(memory_alerts) == 0

    def test_check_alerts_memory_just_above_threshold_high(self, monitoring_system):
        """测试内存使用率刚好高于85%"""
        monitoring_system.record_metric('memory_usage', 85.1)
        alerts = monitoring_system.check_alerts()
        
        # 应该触发告警
        memory_alerts = [a for a in alerts if '内存' in a.get('message', '') or 'memory' in a.get('message', '').lower()]
        assert len(memory_alerts) > 0

    def test_check_alerts_memory_above_threshold(self, monitoring_system):
        """测试内存使用率高于70%（触发告警）"""
        monitoring_system.record_metric('memory_usage', 75.0)
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) > 0
        # 验证告警内容包含内存相关信息
        alert_messages = [a['message'] for a in alerts]
        assert any('内存' in msg or 'memory' in msg.lower() for msg in alert_messages)

    def test_check_alerts_api_response_time_exact_threshold(self, monitoring_system):
        """测试API响应时间正好等于1000ms（不触发告警，因为>1000才触发）"""
        monitoring_system.record_metric('api_response_time', 1000.0)
        alerts = monitoring_system.check_alerts()
        
        # 因为代码是 > 1000，所以1000.0不触发告警
        api_alerts = [a for a in alerts if 'API' in a.get('message', '') or '响应时间' in a.get('message', '')]
        assert len(api_alerts) == 0

    def test_check_alerts_api_response_time_just_below_threshold(self, monitoring_system):
        """测试API响应时间刚好低于1000ms"""
        monitoring_system.record_metric('api_response_time', 999.9)
        alerts = monitoring_system.check_alerts()
        
        # 应该不触发告警
        api_alerts = [a for a in alerts if 'API' in a.get('message', '') or '响应时间' in a.get('message', '')]
        assert len(api_alerts) == 0

    def test_check_alerts_api_response_time_just_above_threshold(self, monitoring_system):
        """测试API响应时间刚好高于1000ms"""
        monitoring_system.record_metric('api_response_time', 1000.1)
        alerts = monitoring_system.check_alerts()
        
        # 应该触发告警
        api_alerts = [a for a in alerts if 'API' in a.get('message', '') or '响应时间' in a.get('message', '')]
        assert len(api_alerts) > 0

    def test_check_alerts_zero_values(self, monitoring_system):
        """测试零值指标不触发告警"""
        monitoring_system.record_metric('cpu_usage', 0.0)
        monitoring_system.record_metric('memory_usage', 0.0)
        monitoring_system.record_metric('api_response_time', 0.0)
        alerts = monitoring_system.check_alerts()
        
        # 零值不应该触发告警
        assert len(alerts) == 0

    def test_check_alerts_negative_values(self, monitoring_system):
        """测试负值指标不触发告警"""
        monitoring_system.record_metric('cpu_usage', -10.0)
        monitoring_system.record_metric('memory_usage', -5.0)
        monitoring_system.record_metric('api_response_time', -100.0)
        alerts = monitoring_system.check_alerts()
        
        # 负值不应该触发告警
        assert len(alerts) == 0

    def test_check_alerts_multiple_metrics_all_trigger(self, monitoring_system):
        """测试多个指标都触发告警"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        monitoring_system.record_metric('memory_usage', 90.0)
        monitoring_system.record_metric('api_response_time', 2000.0)
        alerts = monitoring_system.check_alerts()
        
        # 应该有多个告警
        assert len(alerts) >= 3

    def test_check_alerts_multiple_metrics_partial_trigger(self, monitoring_system):
        """测试部分指标触发告警"""
        monitoring_system.record_metric('cpu_usage', 85.0)  # 触发
        monitoring_system.record_metric('memory_usage', 50.0)  # 不触发
        monitoring_system.record_metric('api_response_time', 500.0)  # 不触发
        alerts = monitoring_system.check_alerts()
        
        # 应该至少有一个告警
        assert len(alerts) >= 1

    def test_check_alerts_multiple_metrics_none_trigger(self, monitoring_system):
        """测试所有指标都不触发告警"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        monitoring_system.record_metric('memory_usage', 50.0)
        monitoring_system.record_metric('api_response_time', 100.0)
        alerts = monitoring_system.check_alerts()
        
        # 应该没有告警
        assert len(alerts) == 0

    def test_check_alerts_very_high_values(self, monitoring_system):
        """测试非常高的值触发告警"""
        monitoring_system.record_metric('cpu_usage', 99.9)
        monitoring_system.record_metric('memory_usage', 99.9)
        monitoring_system.record_metric('api_response_time', 10000.0)
        alerts = monitoring_system.check_alerts()
        
        # 应该有多个告警
        assert len(alerts) >= 3

    def test_check_alerts_alert_structure(self, monitoring_system):
        """测试告警结构完整性"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) > 0
        alert = alerts[0]
        
        # 验证告警结构（根据实际代码）
        assert 'message' in alert
        assert 'timestamp' in alert
        assert 'severity' in alert
        assert 'type' in alert
        # 验证type值
        assert alert['type'] == 'cpu_high'

    def test_check_alerts_alert_timestamp(self, monitoring_system):
        """测试告警时间戳"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) > 0
        alert = alerts[0]
        assert alert['timestamp'] is not None

    def test_check_alerts_alert_severity(self, monitoring_system):
        """测试告警严重程度"""
        # 测试高阈值告警
        monitoring_system.record_metric('cpu_usage', 85.0)
        alerts_high = monitoring_system.check_alerts()
        
        # 测试临界阈值告警
        monitoring_system.record_metric('cpu_usage', 96.0)
        alerts_critical = monitoring_system.check_alerts()
        
        # 验证告警严重程度被设置
        if alerts_high:
            assert alerts_high[0]['severity'] is not None
        if alerts_critical:
            assert alerts_critical[0]['severity'] is not None

    def test_check_alerts_alert_value(self, monitoring_system):
        """测试告警中的值"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        alerts = monitoring_system.check_alerts()
        
        assert len(alerts) > 0
        alert = alerts[0]
        
        # 验证告警中的值包含在message中
        assert '85.0' in alert['message'] or '85' in alert['message']
        assert alert['type'] == 'cpu_high'

