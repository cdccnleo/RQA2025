#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试生产环境告警管理器组件
"""

import importlib
import os
import sys
import time
from datetime import datetime
from unittest.mock import Mock, patch, mock_open, MagicMock

import pytest


@pytest.fixture
def production_alert_manager_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.production_alert_manager"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def manager(production_alert_manager_module):
    """创建ProductionAlertManager实例"""
    module = production_alert_manager_module
    return module.ProductionAlertManager()


@pytest.fixture
def manager_with_config(production_alert_manager_module):
    """创建带配置的ProductionAlertManager实例"""
    module = production_alert_manager_module
    config = {
        'alert_thresholds': {
            'cpu_percent': 75.0,
            'memory_percent': 80.0,
            'disk_percent': 85.0,
            'response_time': 3000,
            'error_rate': 3.0
        },
        'alert_cooldown': 60
    }
    return module.ProductionAlertManager(config)


def test_initialization_default_config(manager):
    """测试初始化（默认配置）"""
    assert manager.config is not None
    assert 'alert_thresholds' in manager.config
    assert 'alert_cooldown' in manager.config
    assert manager.config['alert_thresholds']['cpu_percent'] == 80.0
    assert manager.config['alert_cooldown'] == 300
    assert manager.last_alert_times == {}


def test_initialization_custom_config(manager_with_config):
    """测试初始化（自定义配置）"""
    assert manager_with_config.config['alert_thresholds']['cpu_percent'] == 75.0
    assert manager_with_config.config['alert_cooldown'] == 60


def test_check_alerts_no_alerts(manager):
    """测试检查告警条件（无告警）"""
    metrics = {
        'cpu': {'percent': 50.0},
        'memory': {'percent': 60.0},
        'disk': {'percent': 70.0}
    }
    
    alerts = manager.check_alerts(metrics)
    
    assert len(alerts) == 0


def test_check_alerts_cpu_alert(manager):
    """测试检查告警条件（CPU告警）"""
    metrics = {
        'cpu': {'percent': 90.0},
        'memory': {'percent': 60.0},
        'disk': {'percent': 70.0}
    }
    
    alerts = manager.check_alerts(metrics)
    
    assert len(alerts) == 1
    assert alerts[0]['type'] == 'cpu_high'
    assert alerts[0]['level'] == 'warning'
    assert alerts[0]['value'] == 90.0
    assert alerts[0]['threshold'] == 80.0


def test_check_alerts_memory_alert(manager):
    """测试检查告警条件（内存告警）"""
    metrics = {
        'cpu': {'percent': 50.0},
        'memory': {'percent': 90.0},
        'disk': {'percent': 70.0}
    }
    
    alerts = manager.check_alerts(metrics)
    
    assert len(alerts) == 1
    assert alerts[0]['type'] == 'memory_high'
    assert alerts[0]['level'] == 'warning'
    assert alerts[0]['value'] == 90.0


def test_check_alerts_disk_alert(manager):
    """测试检查告警条件（磁盘告警）"""
    metrics = {
        'cpu': {'percent': 50.0},
        'memory': {'percent': 60.0},
        'disk': {'percent': 95.0}
    }
    
    alerts = manager.check_alerts(metrics)
    
    assert len(alerts) == 1
    assert alerts[0]['type'] == 'disk_high'
    assert alerts[0]['level'] == 'error'
    assert alerts[0]['value'] == 95.0


def test_check_alerts_multiple_alerts(manager):
    """测试检查告警条件（多个告警）"""
    metrics = {
        'cpu': {'percent': 90.0},
        'memory': {'percent': 90.0},
        'disk': {'percent': 95.0}
    }
    
    alerts = manager.check_alerts(metrics)
    
    assert len(alerts) == 3
    alert_types = {alert['type'] for alert in alerts}
    assert 'cpu_high' in alert_types
    assert 'memory_high' in alert_types
    assert 'disk_high' in alert_types


def test_check_cpu_alert_threshold_exceeded(manager):
    """测试检查CPU告警（超过阈值）"""
    metrics = {'cpu': {'percent': 85.0}}
    thresholds = {'cpu_percent': 80.0}
    
    alert = manager._check_cpu_alert(metrics, thresholds)
    
    assert alert is not None
    assert alert['type'] == 'cpu_high'
    assert alert['value'] == 85.0
    assert alert['threshold'] == 80.0


def test_check_cpu_alert_threshold_not_exceeded(manager):
    """测试检查CPU告警（未超过阈值）"""
    metrics = {'cpu': {'percent': 70.0}}
    thresholds = {'cpu_percent': 80.0}
    
    alert = manager._check_cpu_alert(metrics, thresholds)
    
    assert alert is None


def test_check_cpu_alert_missing_key(manager):
    """测试检查CPU告警（缺少键）"""
    metrics = {}
    thresholds = {'cpu_percent': 80.0}
    
    alert = manager._check_cpu_alert(metrics, thresholds)
    
    assert alert is None


def test_check_memory_alert_threshold_exceeded(manager):
    """测试检查内存告警（超过阈值）"""
    metrics = {'memory': {'percent': 90.0}}
    thresholds = {'memory_percent': 85.0}
    
    alert = manager._check_memory_alert(metrics, thresholds)
    
    assert alert is not None
    assert alert['type'] == 'memory_high'
    assert alert['value'] == 90.0


def test_check_memory_alert_threshold_not_exceeded(manager):
    """测试检查内存告警（未超过阈值）"""
    metrics = {'memory': {'percent': 70.0}}
    thresholds = {'memory_percent': 85.0}
    
    alert = manager._check_memory_alert(metrics, thresholds)
    
    assert alert is None


def test_check_disk_alert_threshold_exceeded(manager):
    """测试检查磁盘告警（超过阈值）"""
    metrics = {'disk': {'percent': 95.0}}
    thresholds = {'disk_percent': 90.0}
    
    alert = manager._check_disk_alert(metrics, thresholds)
    
    assert alert is not None
    assert alert['type'] == 'disk_high'
    assert alert['level'] == 'error'
    assert alert['value'] == 95.0


def test_check_disk_alert_threshold_not_exceeded(manager):
    """测试检查磁盘告警（未超过阈值）"""
    metrics = {'disk': {'percent': 80.0}}
    thresholds = {'disk_percent': 90.0}
    
    alert = manager._check_disk_alert(metrics, thresholds)
    
    assert alert is None


def test_send_alerts_with_cooldown(manager, monkeypatch):
    """测试发送告警通知（有冷却时间）"""
    alerts = [
        {'type': 'cpu_high', 'level': 'warning', 'message': 'CPU high'}
    ]
    
    mock_print = MagicMock()
    monkeypatch.setattr('builtins.print', mock_print)
    
    with patch('builtins.open', mock_open()):
        sent = manager.send_alerts(alerts)
        
        assert len(sent) == 1
        assert sent[0] == 'cpu_high'
        assert mock_print.called


def test_send_alerts_cooldown_active(manager, monkeypatch):
    """测试发送告警通知（冷却时间未到）"""
    alerts = [
        {'type': 'cpu_high', 'level': 'warning', 'message': 'CPU high'}
    ]
    
    # 设置冷却时间，使告警在冷却期内
    alert_key = 'cpu_high_warning'
    manager.last_alert_times[alert_key] = time.time() - 10  # 10秒前
    
    mock_print = MagicMock()
    monkeypatch.setattr('builtins.print', mock_print)
    
    sent = manager.send_alerts(alerts)
    
    # 冷却时间未到，不应该发送告警
    assert len(sent) == 0


def test_should_send_alert_no_cooldown(manager):
    """测试判断是否应该发送告警（无冷却时间）"""
    alert = {'type': 'cpu_high', 'level': 'warning'}
    
    should_send = manager._should_send_alert(alert)
    
    assert should_send is True


def test_should_send_alert_cooldown_active(manager):
    """测试判断是否应该发送告警（冷却时间未到）"""
    alert = {'type': 'cpu_high', 'level': 'warning'}
    alert_key = 'cpu_high_warning'
    
    # 设置冷却时间，使告警在冷却期内
    manager.last_alert_times[alert_key] = time.time() - 10  # 10秒前
    
    should_send = manager._should_send_alert(alert)
    
    # 冷却时间300秒，10秒前发送的告警仍在冷却期内
    assert should_send is False


def test_should_send_alert_cooldown_expired(manager):
    """测试判断是否应该发送告警（冷却时间已过）"""
    alert = {'type': 'cpu_high', 'level': 'warning'}
    alert_key = 'cpu_high_warning'
    
    # 设置冷却时间，使告警冷却期已过
    manager.last_alert_times[alert_key] = time.time() - 400  # 400秒前
    
    should_send = manager._should_send_alert(alert)
    
    # 冷却时间300秒，400秒前发送的告警冷却期已过
    assert should_send is True


def test_update_alert_cooldown(manager):
    """测试更新告警冷却时间"""
    alert = {'type': 'cpu_high', 'level': 'warning'}
    
    initial_count = len(manager.last_alert_times)
    manager._update_alert_cooldown(alert)
    
    assert len(manager.last_alert_times) == initial_count + 1
    alert_key = 'cpu_high_warning'
    assert alert_key in manager.last_alert_times
    assert manager.last_alert_times[alert_key] > 0


def test_send_alert_notification_success(manager, monkeypatch, capsys):
    """测试发送告警通知（成功）"""
    alert = {
        'type': 'cpu_high',
        'level': 'warning',
        'message': 'CPU使用率过高: 90.0%'
    }
    
    with patch('builtins.open', mock_open()):
        manager._send_alert_notification(alert)
        
        captured = capsys.readouterr()
        assert '告警通知' in captured.out
        assert 'WARNING' in captured.out
        assert 'CPU使用率过高' in captured.out


def test_send_alert_notification_exception(manager, monkeypatch, capsys):
    """测试发送告警通知（异常）"""
    alert = {
        'type': 'cpu_high',
        'level': 'warning',
        'message': 'CPU使用率过高: 90.0%'
    }
    
    # 模拟_log_alert_to_file抛出异常
    def failing_log(*args, **kwargs):
        raise RuntimeError("Log error")
    
    monkeypatch.setattr(manager, '_log_alert_to_file', failing_log)
    
    manager._send_alert_notification(alert)
    
    # 异常应该被捕获，不会中断程序
    captured = capsys.readouterr()
    assert '发送告警通知失败' in captured.out or '告警通知' in captured.out


def test_log_alert_to_file_success(manager, monkeypatch):
    """测试记录告警到日志文件（成功）"""
    alert = {
        'type': 'cpu_high',
        'level': 'warning',
        'message': 'CPU使用率过高: 90.0%',
        'timestamp': '2025-01-01T10:00:00'
    }
    
    mock_file = mock_open()
    with patch('builtins.open', mock_file):
        manager._log_alert_to_file(alert)
        
        mock_file.assert_called_once_with('alerts.log', 'a', encoding='utf-8')
        handle = mock_file()
        handle.write.assert_called_once()
        written_content = handle.write.call_args[0][0]
        assert '2025-01-01T10:00:00' in written_content
        assert 'warning' in written_content
        assert 'CPU使用率过高' in written_content


def test_log_alert_to_file_exception(manager, monkeypatch, capsys):
    """测试记录告警到日志文件（异常）"""
    alert = {
        'type': 'cpu_high',
        'level': 'warning',
        'message': 'CPU使用率过高: 90.0%'
    }
    
    # 模拟文件操作抛出异常
    with patch('builtins.open', side_effect=IOError("File error")):
        manager._log_alert_to_file(alert)
        
        captured = capsys.readouterr()
        assert '写入告警日志失败' in captured.out


def test_update_threshold_success(manager):
    """测试更新告警阈值（成功）"""
    result = manager.update_threshold('cpu_percent', 75.0)
    
    assert result is True
    assert manager.config['alert_thresholds']['cpu_percent'] == 75.0


def test_update_threshold_invalid_metric(manager):
    """测试更新告警阈值（无效指标）"""
    result = manager.update_threshold('invalid_metric', 75.0)
    
    assert result is False
    assert 'invalid_metric' not in manager.config['alert_thresholds']


def test_get_alert_statistics(manager):
    """测试获取告警统计"""
    # 设置一些冷却时间
    manager.last_alert_times['cpu_high_warning'] = time.time()
    manager.last_alert_times['memory_high_warning'] = time.time()
    
    stats = manager.get_alert_statistics()
    
    assert 'thresholds' in stats
    assert 'cooldown_seconds' in stats
    assert 'active_cooldowns' in stats
    assert stats['active_cooldowns'] == 2
    assert stats['cooldown_seconds'] == 300


def test_clear_alert_cooldowns(manager):
    """测试清空告警冷却时间"""
    # 设置一些冷却时间
    manager.last_alert_times['cpu_high_warning'] = time.time()
    manager.last_alert_times['memory_high_warning'] = time.time()
    
    assert len(manager.last_alert_times) == 2
    
    manager.clear_alert_cooldowns()
    
    assert len(manager.last_alert_times) == 0


def test_send_alerts_multiple_alerts(manager, monkeypatch):
    """测试发送多个告警通知"""
    alerts = [
        {'type': 'cpu_high', 'level': 'warning', 'message': 'CPU high'},
        {'type': 'memory_high', 'level': 'warning', 'message': 'Memory high'}
    ]
    
    mock_print = MagicMock()
    monkeypatch.setattr('builtins.print', mock_print)
    
    with patch('builtins.open', mock_open()):
        sent = manager.send_alerts(alerts)
        
        assert len(sent) == 2
        assert 'cpu_high' in sent
        assert 'memory_high' in sent

