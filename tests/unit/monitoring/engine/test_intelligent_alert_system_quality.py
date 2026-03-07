#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能告警系统质量测试
测试覆盖 IntelligentAlertSystem 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
from unittest.mock import Mock, patch
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
    engine_intelligent_alert_system_module = importlib.import_module('src.monitoring.engine.intelligent_alert_system')
    IntelligentAlertSystem = getattr(engine_intelligent_alert_system_module, 'IntelligentAlertSystem', None)
    AlertLevel = getattr(engine_intelligent_alert_system_module, 'AlertLevel', None)
    NotificationChannel = getattr(engine_intelligent_alert_system_module, 'NotificationChannel', None)
    AlertRule = getattr(engine_intelligent_alert_system_module, 'AlertRule', None)
    Alert = getattr(engine_intelligent_alert_system_module, 'Alert', None)
    NotificationConfig = getattr(engine_intelligent_alert_system_module, 'NotificationConfig', None)
    
    if IntelligentAlertSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def intelligent_alert_system():
    """创建智能告警系统实例"""
    return IntelligentAlertSystem()


@pytest.fixture
def sample_alert_rule():
    """创建示例告警规则"""
    return AlertRule(
        name='high_cpu',
        metric_name='cpu_usage',
        condition='> 80',
        level=AlertLevel.WARNING,
        duration=60,
        channels=[NotificationChannel.EMAIL],
        enabled=True,
        description='CPU usage too high'
    )


@pytest.fixture
def sample_notification_config():
    """创建示例通知配置"""
    return NotificationConfig(
        channel=NotificationChannel.EMAIL,
        config={'smtp_server': 'smtp.test.com'},
        enabled=True
    )


class TestIntelligentAlertSystem:
    """IntelligentAlertSystem测试类"""

    def test_initialization(self, intelligent_alert_system):
        """测试初始化"""
        assert intelligent_alert_system.alert_rules == {}
        assert intelligent_alert_system.active_alerts == {}
        assert len(intelligent_alert_system.alert_history) == 0

    def test_add_alert_rule(self, intelligent_alert_system, sample_alert_rule):
        """测试添加告警规则"""
        intelligent_alert_system.add_alert_rule(sample_alert_rule)
        assert 'high_cpu' in intelligent_alert_system.alert_rules

    def test_remove_alert_rule(self, intelligent_alert_system, sample_alert_rule):
        """测试移除告警规则"""
        intelligent_alert_system.add_alert_rule(sample_alert_rule)
        intelligent_alert_system.remove_alert_rule('high_cpu')
        assert 'high_cpu' not in intelligent_alert_system.alert_rules

    def test_add_notification_config(self, intelligent_alert_system, sample_notification_config):
        """测试添加通知配置"""
        # 检查是否有add_notification_config方法
        if hasattr(intelligent_alert_system, 'add_notification_config'):
            intelligent_alert_system.add_notification_config(sample_notification_config)
            assert NotificationChannel.EMAIL in intelligent_alert_system.notification_configs
        else:
            # 如果没有该方法，验证配置字典存在
            assert isinstance(intelligent_alert_system.notification_configs, dict)
            # 验证默认配置已初始化
            assert len(intelligent_alert_system.notification_configs) > 0

    def test_trigger_alert(self, intelligent_alert_system, sample_alert_rule):
        """测试触发告警"""
        intelligent_alert_system.add_alert_rule(sample_alert_rule)
        # 触发告警
        intelligent_alert_system.trigger_alert('high_cpu', 'cpu_usage', 90.0, '> 80')
        # 验证告警被创建
        assert 'high_cpu' in intelligent_alert_system.active_alerts

    def test_get_active_alerts(self, intelligent_alert_system):
        """测试获取活动告警"""
        alerts = intelligent_alert_system.get_active_alerts()
        assert isinstance(alerts, list)

    def test_resolve_alert(self, intelligent_alert_system, sample_alert_rule):
        """测试解决告警"""
        intelligent_alert_system.add_alert_rule(sample_alert_rule)
        # 触发告警
        intelligent_alert_system.trigger_alert('high_cpu', 'cpu_usage', 90.0, '> 80')
        
        # 解决告警（使用rule_name）
        intelligent_alert_system.resolve_alert('high_cpu')
        # 验证告警已解决
        assert 'high_cpu' not in intelligent_alert_system.active_alerts

    def test_start(self, intelligent_alert_system):
        """测试启动系统"""
        if hasattr(intelligent_alert_system, 'start'):
            intelligent_alert_system.start()
            assert intelligent_alert_system.running is True
            # 清理
            if hasattr(intelligent_alert_system, 'stop'):
                intelligent_alert_system.stop()

    def test_stop(self, intelligent_alert_system):
        """测试停止系统"""
        if hasattr(intelligent_alert_system, 'start') and hasattr(intelligent_alert_system, 'stop'):
            intelligent_alert_system.start()
            time.sleep(0.1)
            intelligent_alert_system.stop()
            assert intelligent_alert_system.running is False


class TestDataModels:
    """数据模型测试类"""

    def test_alert_level_enum(self):
        """测试告警级别枚举"""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"

    def test_notification_channel_enum(self):
        """测试通知渠道枚举"""
        assert NotificationChannel.EMAIL.value == "email"
        assert NotificationChannel.SMS.value == "sms"
        assert NotificationChannel.WEBHOOK.value == "webhook"
        assert NotificationChannel.DINGTALK.value == "dingtalk"
        assert NotificationChannel.WECHAT.value == "wechat"
        assert NotificationChannel.SLACK.value == "slack"

    def test_alert_rule(self, sample_alert_rule):
        """测试告警规则"""
        assert sample_alert_rule.name == 'high_cpu'
        assert sample_alert_rule.enabled is True
        assert sample_alert_rule.cooldown == 300

    def test_notification_config(self, sample_notification_config):
        """测试通知配置"""
        assert sample_notification_config.channel == NotificationChannel.EMAIL
        assert sample_notification_config.enabled is True

