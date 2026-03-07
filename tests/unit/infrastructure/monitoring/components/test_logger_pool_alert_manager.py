#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Logger池告警管理器组件
"""

import importlib
import sys
import time
from datetime import datetime
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def alert_manager_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.logger_pool_alert_manager"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def alert_manager(alert_manager_module):
    """创建LoggerPoolAlertManager实例"""
    return alert_manager_module.LoggerPoolAlertManager(pool_name="test_pool")


@pytest.fixture
def alert_manager_custom_thresholds(alert_manager_module):
    """创建带自定义阈值的LoggerPoolAlertManager实例"""
    custom_thresholds = {
        'hit_rate_low': 0.7,
        'pool_usage_high': 0.8,
        'memory_high': 50.0,
    }
    return alert_manager_module.LoggerPoolAlertManager(
        pool_name="test_pool",
        alert_thresholds=custom_thresholds
    )


@pytest.fixture
def sample_stats(alert_manager_module):
    """创建示例统计数据"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    return LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )


def test_initialization_default_thresholds(alert_manager):
    """测试使用默认阈值初始化"""
    assert alert_manager.pool_name == "test_pool"
    assert alert_manager.alert_thresholds['hit_rate_low'] == 0.8
    assert alert_manager.alert_thresholds['pool_usage_high'] == 0.9
    assert alert_manager.alert_thresholds['memory_high'] == 100.0
    assert alert_manager.alert_callbacks == []
    assert alert_manager.alert_history == []
    assert alert_manager.max_alert_history == 100


def test_initialization_custom_thresholds(alert_manager_custom_thresholds):
    """测试使用自定义阈值初始化"""
    assert alert_manager_custom_thresholds.alert_thresholds['hit_rate_low'] == 0.7
    assert alert_manager_custom_thresholds.alert_thresholds['pool_usage_high'] == 0.8
    assert alert_manager_custom_thresholds.alert_thresholds['memory_high'] == 50.0


def test_check_alerts_empty_stats(alert_manager):
    """测试检查空统计信息"""
    alerts = alert_manager.check_alerts(None)
    assert alerts == []


def test_check_alerts_no_alerts(alert_manager, alert_manager_module):
    """测试检查告警（无告警）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,  # 高于0.8阈值，不会触发告警
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=50.0,  # 低于100MB阈值
        timestamp=time.time()
    )
    
    alerts = alert_manager.check_alerts(stats)
    assert alerts == []


def test_check_alerts_hit_rate_low(alert_manager, alert_manager_module):
    """测试命中率过低告警"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,  # 低于0.8阈值
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) == 1
    assert alerts[0]['alert_type'] == 'hit_rate_low'
    assert alerts[0]['severity'] == 'warning'
    assert alerts[0]['current_value'] == 0.5


def test_check_alerts_pool_usage_high(alert_manager, alert_manager_module):
    """测试池使用率过高告警"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=10,  # 10/10 = 1.0 > 0.9，触发告警
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=10,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) == 1
    assert alerts[0]['alert_type'] == 'pool_usage_high'
    assert alerts[0]['severity'] == 'warning'
    assert alerts[0]['current_value'] == pytest.approx(1.0)  # 10/10


def test_check_alerts_memory_high(alert_manager, alert_manager_module):
    """测试内存使用过高告警"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=150.0,  # 高于100MB阈值
        timestamp=time.time()
    )
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) == 1
    assert alerts[0]['alert_type'] == 'memory_high'
    assert alerts[0]['severity'] == 'error'
    assert alerts[0]['current_value'] == 150.0


def test_check_alerts_multiple_alerts(alert_manager, alert_manager_module):
    """测试多个告警同时触发"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=10,  # 10/10 = 1.0 > 0.9，触发使用率告警
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,  # 触发命中率告警
        logger_count=10,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=150.0,  # 触发内存告警
        timestamp=time.time()
    )
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) == 3  # 命中率、使用率、内存
    alert_types = [a['alert_type'] for a in alerts]
    assert 'hit_rate_low' in alert_types
    assert 'pool_usage_high' in alert_types
    assert 'memory_high' in alert_types


def test_check_hit_rate_alert_threshold_not_met(alert_manager, alert_manager_module):
    """测试命中率告警（未达到阈值）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,  # 高于0.8阈值，不会触发告警
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    assert alert is None


def test_check_hit_rate_alert_threshold_met(alert_manager, alert_manager_module):
    """测试命中率告警（达到阈值）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    assert alert is not None
    assert alert['alert_type'] == 'hit_rate_low'
    assert alert['threshold'] == 0.8
    assert alert['current_value'] == 0.5
    assert alert['pool_name'] == "test_pool"


def test_check_pool_usage_alert_zero_max_size(alert_manager, alert_manager_module):
    """测试池使用率告警（max_size为0）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=0,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_pool_usage_alert(stats)
    assert alert is None  # usage_rate = 0，不会触发告警


def test_check_pool_usage_alert_threshold_met(alert_manager, alert_manager_module):
    """测试池使用率告警（达到阈值）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=10,  # 使用10/10 = 1.0，高于0.9阈值
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=10,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_pool_usage_alert(stats)
    assert alert is not None
    assert alert['alert_type'] == 'pool_usage_high'
    assert alert['current_value'] == pytest.approx(1.0)  # 10/10


def test_check_memory_alert_threshold_not_met(alert_manager, sample_stats):
    """测试内存告警（未达到阈值）"""
    alert = alert_manager._check_memory_alert(sample_stats)
    assert alert is None


def test_check_memory_alert_threshold_met(alert_manager, alert_manager_module):
    """测试内存告警（达到阈值）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=150.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_memory_alert(stats)
    assert alert is not None
    assert alert['alert_type'] == 'memory_high'
    assert alert['severity'] == 'error'
    assert alert['current_value'] == 150.0


def test_trigger_alert_success(alert_manager, alert_manager_module):
    """测试触发告警（成功）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    assert alert is not None
    
    alert_manager._trigger_alert(alert)
    
    assert len(alert_manager.alert_history) == 1
    assert alert_manager.alert_history[0]['alert_type'] == 'hit_rate_low'


def test_trigger_alert_max_history_limit(alert_manager, alert_manager_module):
    """测试告警历史最大数量限制"""
    alert_manager.max_alert_history = 3
    
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    # 触发4次告警，应该只保留最后3条
    for i in range(4):
        alert = alert_manager._check_hit_rate_alert(stats)
        alert['message'] = f"Alert {i}"
        alert_manager._trigger_alert(alert)
    
    assert len(alert_manager.alert_history) == 3
    assert alert_manager.alert_history[0]['message'] == "Alert 1"
    assert alert_manager.alert_history[-1]['message'] == "Alert 3"


def test_trigger_alert_with_callback(alert_manager, alert_manager_module):
    """测试触发告警时调用回调函数"""
    callback_called = []
    
    def test_callback(alert_data):
        callback_called.append(alert_data)
    
    alert_manager.register_alert_callback(test_callback)
    
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    alert_manager._trigger_alert(alert)
    
    assert len(callback_called) == 1
    assert callback_called[0]['alert_type'] == 'hit_rate_low'


def test_trigger_alert_callback_exception(alert_manager, alert_manager_module, capsys):
    """测试告警回调函数抛出异常"""
    def failing_callback(alert_data):
        raise RuntimeError("Callback error")
    
    alert_manager.register_alert_callback(failing_callback)
    
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    alert_manager._trigger_alert(alert)
    
    # 应该记录告警历史，即使回调失败
    assert len(alert_manager.alert_history) == 1
    captured = capsys.readouterr()
    assert "告警回调执行失败" in captured.out


def test_trigger_alert_exception(alert_manager, monkeypatch, capsys):
    """测试触发告警时发生异常"""
    # 创建一个会抛出异常的Mock对象来替换alert_history
    class FailingList(list):
        def append(self, item):
            raise RuntimeError("Append error")
    
    alert_manager.alert_history = FailingList()
    
    alert_data = {'alert_type': 'test', 'message': 'Test alert', 'severity': 'warning'}
    alert_manager._trigger_alert(alert_data)
    
    captured = capsys.readouterr()
    assert "触发告警失败" in captured.out


def test_register_alert_callback(alert_manager):
    """测试注册告警回调函数"""
    def callback1(alert_data):
        pass
    
    def callback2(alert_data):
        pass
    
    alert_manager.register_alert_callback(callback1)
    alert_manager.register_alert_callback(callback2)
    
    assert len(alert_manager.alert_callbacks) == 2
    assert callback1 in alert_manager.alert_callbacks
    assert callback2 in alert_manager.alert_callbacks


def test_unregister_alert_callback_success(alert_manager):
    """测试注销告警回调函数（成功）"""
    def callback(alert_data):
        pass
    
    alert_manager.register_alert_callback(callback)
    assert len(alert_manager.alert_callbacks) == 1
    
    result = alert_manager.unregister_alert_callback(callback)
    assert result is True
    assert len(alert_manager.alert_callbacks) == 0


def test_unregister_alert_callback_not_found(alert_manager):
    """测试注销告警回调函数（未找到）"""
    def callback(alert_data):
        pass
    
    result = alert_manager.unregister_alert_callback(callback)
    assert result is False


def test_update_threshold_success(alert_manager):
    """测试更新告警阈值（成功）"""
    result = alert_manager.update_threshold('hit_rate_low', 0.7)
    assert result is True
    assert alert_manager.alert_thresholds['hit_rate_low'] == 0.7


def test_update_threshold_not_found(alert_manager):
    """测试更新告警阈值（未找到）"""
    result = alert_manager.update_threshold('unknown_threshold', 0.5)
    assert result is False


def test_get_alert_status_empty_stats(alert_manager):
    """测试获取告警状态（空统计信息）"""
    status = alert_manager.get_alert_status(None)
    assert status == {}


def test_get_alert_status_no_alerts(alert_manager, alert_manager_module):
    """测试获取告警状态（无告警）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,  # 高于0.8阈值，不会触发告警
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=50.0,  # 低于100MB阈值
        timestamp=time.time()
    )
    
    status = alert_manager.get_alert_status(stats)
    assert status['hit_rate_low'] is False
    assert status['pool_usage_high'] is False  # 5/10 = 0.5 < 0.9
    assert status['memory_high'] is False


def test_get_alert_status_with_alerts(alert_manager, alert_manager_module):
    """测试获取告警状态（有告警）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=10,  # 10/10 = 1.0 > 0.9，触发使用率告警
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,  # 触发命中率告警
        logger_count=10,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=150.0,  # 触发内存告警
        timestamp=time.time()
    )
    
    status = alert_manager.get_alert_status(stats)
    assert status['hit_rate_low'] is True
    assert status['pool_usage_high'] is True  # 10/10 = 1.0 > 0.9
    assert status['memory_high'] is True


def test_get_alert_status_zero_max_size(alert_manager, alert_manager_module):
    """测试获取告警状态（max_size为0）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=0,
        created_count=20,
        hit_count=15,
        hit_rate=0.85,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    status = alert_manager.get_alert_status(stats)
    assert status['pool_usage_high'] is False  # usage_rate = 0


def test_get_alert_history_no_limit(alert_manager, alert_manager_module):
    """测试获取告警历史（无限制）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    # 触发两次告警
    alert1 = alert_manager._check_hit_rate_alert(stats)
    alert_manager._trigger_alert(alert1)
    alert_manager._trigger_alert(alert1)
    
    history = alert_manager.get_alert_history()
    assert len(history) == 2
    # 验证返回的是副本
    assert history is not alert_manager.alert_history


def test_get_alert_history_with_limit(alert_manager, alert_manager_module):
    """测试获取告警历史（有限制）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    # 触发3次告警
    alert = alert_manager._check_hit_rate_alert(stats)
    for i in range(3):
        alert['message'] = f"Alert {i}"
        alert_manager._trigger_alert(alert)
    
    history = alert_manager.get_alert_history(limit=2)
    assert len(history) == 2
    assert history[0]['message'] == "Alert 1"
    assert history[1]['message'] == "Alert 2"


def test_get_alert_history_zero_limit(alert_manager, alert_manager_module):
    """测试获取告警历史（限制为0）"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    alert_manager._trigger_alert(alert)
    
    history = alert_manager.get_alert_history(limit=0)
    assert history == []


def test_clear_alert_history(alert_manager, alert_manager_module):
    """测试清空告警历史"""
    LoggerPoolStats = alert_manager_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    alert = alert_manager._check_hit_rate_alert(stats)
    alert_manager._trigger_alert(alert)
    
    assert len(alert_manager.alert_history) > 0
    
    alert_manager.clear_alert_history()
    assert len(alert_manager.alert_history) == 0


def test_fallback_logger_pool_stats_definition(alert_manager_module, monkeypatch):
    """测试LoggerPoolStats fallback定义"""
    import builtins
    original_import = builtins.__import__
    
    # 模拟导入失败
    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "logger_pool_monitor" in name:
            raise ImportError("Module not found")
        return original_import(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr(builtins, "__import__", failing_import)
    
    # 重新导入模块
    module_name = "src.infrastructure.monitoring.components.logger_pool_alert_manager"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    module = importlib.import_module(module_name)
    
    # 验证fallback LoggerPoolStats存在
    assert hasattr(module, "LoggerPoolStats")
    LoggerPoolStats = module.LoggerPoolStats
    
    # 验证可以创建实例
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    assert stats.pool_size == 5
    assert stats.hit_rate == 0.75

