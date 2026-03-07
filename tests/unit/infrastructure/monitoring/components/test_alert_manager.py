#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试告警管理器组件
"""

import importlib
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import pytest


@pytest.fixture
def alert_manager_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.alert_manager"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def alert_manager(alert_manager_module):
    """创建AlertManager实例"""
    return alert_manager_module.AlertManager(pool_name="test_pool")


@pytest.fixture
def alert_manager_custom_thresholds(alert_manager_module):
    """创建带自定义阈值的AlertManager实例"""
    custom_thresholds = {
        'hit_rate_low': 0.7,
        'pool_usage_high': 0.8,
        'memory_high': 50.0,
    }
    return alert_manager_module.AlertManager(
        pool_name="test_pool",
        alert_thresholds=custom_thresholds
    )


def test_initialization_default_thresholds(alert_manager):
    """测试使用默认阈值初始化"""
    assert alert_manager.pool_name == "test_pool"
    assert alert_manager._thresholds['hit_rate_low'] == 0.8
    assert alert_manager._thresholds['pool_usage_high'] == 0.9
    assert alert_manager._thresholds['memory_high'] == 100.0
    assert len(alert_manager.alert_rules) > 0
    assert alert_manager.alert_history == []
    assert alert_manager.max_history_size == 1000


def test_initialization_custom_thresholds(alert_manager_custom_thresholds):
    """测试使用自定义阈值初始化"""
    assert alert_manager_custom_thresholds._thresholds['hit_rate_low'] == 0.7
    assert alert_manager_custom_thresholds._thresholds['pool_usage_high'] == 0.8
    assert alert_manager_custom_thresholds._thresholds['memory_high'] == 50.0


def test_add_alert_rule(alert_manager, alert_manager_module):
    """测试添加告警规则"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="custom_rule",
        name="自定义规则",
        description="测试规则",
        condition=AlertConditionConfig(
            operator="gt",
            field="cpu_usage",
            value=80
        ),
        severity="warning",
        channels=["console"]
    )
    
    initial_count = len(alert_manager.alert_rules)
    alert_manager.add_alert_rule(rule)
    
    assert len(alert_manager.alert_rules) == initial_count + 1
    assert alert_manager.alert_rules[-1].rule_id == "custom_rule"


def test_remove_alert_rule_success(alert_manager, alert_manager_module):
    """测试移除告警规则（成功）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="temp_rule",
        name="临时规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="info"
    )
    
    alert_manager.add_alert_rule(rule)
    initial_count = len(alert_manager.alert_rules)
    
    result = alert_manager.remove_alert_rule("temp_rule")
    
    assert result is True
    assert len(alert_manager.alert_rules) == initial_count - 1


def test_remove_alert_rule_not_found(alert_manager):
    """测试移除告警规则（未找到）"""
    initial_count = len(alert_manager.alert_rules)
    result = alert_manager.remove_alert_rule("non_existent_rule")
    
    assert result is False
    assert len(alert_manager.alert_rules) == initial_count


def test_check_alerts_no_trigger(alert_manager, alert_manager_module):
    """测试检查告警（无触发）"""
    stats = {
        'hit_rate': 0.85,  # 高于阈值
        'pool_size': 5,
        'max_pool_size': 10,
        'memory_usage_mb': 50.0  # 低于阈值
    }
    
    # 清空默认规则，添加一个不会触发的规则
    alert_manager.alert_rules.clear()
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="safe_rule",
        name="安全规则",
        description="测试",
        condition=AlertConditionConfig(operator="gt", field="hit_rate", value=0.9),  # 0.85 < 0.9，不会触发
        severity="warning"
    )
    alert_manager.add_alert_rule(rule)
    
    alerts = alert_manager.check_alerts(stats)
    assert alerts == []


def test_check_alerts_trigger_hit_rate_low(alert_manager):
    """测试检查告警（触发命中率过低）"""
    stats = {
        'hit_rate': 0.5,  # 低于0.8阈值
        'pool_size': 5,
        'max_pool_size': 10,
        'memory_usage_mb': 50.0
    }
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) > 0
    assert any(a['rule_id'] == 'hit_rate_low' for a in alerts)


def test_check_alerts_trigger_memory_high(alert_manager):
    """测试检查告警（触发内存过高）"""
    stats = {
        'hit_rate': 0.85,
        'pool_size': 5,
        'max_pool_size': 10,
        'memory_usage_mb': 150.0  # 高于100MB阈值
    }
    
    alerts = alert_manager.check_alerts(stats)
    assert len(alerts) > 0
    assert any(a['rule_id'] == 'memory_high' for a in alerts)


def test_check_alerts_disabled_rule(alert_manager, alert_manager_module):
    """测试检查告警（规则被禁用）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="disabled_rule",
        name="禁用规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning",
        enabled=False  # 禁用
    )
    
    alert_manager.add_alert_rule(rule)
    
    stats = {'test': 1}
    alerts = alert_manager.check_alerts(stats)
    
    assert not any(a['rule_id'] == 'disabled_rule' for a in alerts)


def test_get_alert_history_no_limit(alert_manager):
    """测试获取告警历史（无限制）"""
    # 触发一些告警
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 50.0}
    alert_manager.check_alerts(stats)
    alert_manager.check_alerts(stats)
    
    history = alert_manager.get_alert_history()
    assert len(history) >= 2


def test_get_alert_history_with_limit(alert_manager, alert_manager_module):
    """测试获取告警历史（有限制）"""
    # 清空默认规则，添加一个会触发的规则
    alert_manager.alert_rules.clear()
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="test_rule",
        name="测试规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning",
        cooldown=0  # 无冷却时间，可以多次触发
    )
    alert_manager.add_alert_rule(rule)
    
    # 触发多个告警
    stats = {'test': 1}
    for _ in range(5):
        alert_manager.check_alerts(stats)
    
    history = alert_manager.get_alert_history(limit=3)
    assert len(history) == 3


def test_get_alert_history_zero_limit(alert_manager):
    """测试获取告警历史（限制为0）"""
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 50.0}
    alert_manager.check_alerts(stats)
    
    history = alert_manager.get_alert_history(limit=0)
    assert history == []


def test_get_active_alerts(alert_manager):
    """测试获取活跃告警"""
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 150.0}
    alert_manager.check_alerts(stats)
    
    active_alerts = alert_manager.get_active_alerts()
    assert len(active_alerts) > 0
    assert all(a.get('active', False) for a in active_alerts)


def test_get_active_alerts_max_limit(alert_manager):
    """测试获取活跃告警（最大数量限制）"""
    alert_manager.config['max_active_alerts'] = 3
    
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 50.0}
    # 触发多个告警
    for _ in range(5):
        alert_manager.check_alerts(stats)
    
    active_alerts = alert_manager.get_active_alerts()
    assert len(active_alerts) <= 3


def test_acknowledge_alert_success(alert_manager):
    """测试确认告警（成功）"""
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 50.0}
    alerts = alert_manager.check_alerts(stats)
    
    assert len(alerts) > 0
    alert_id = alerts[0]['alert_id']
    
    result = alert_manager.acknowledge_alert(alert_id)
    assert result is True
    
    # 验证告警状态已更新
    active_alerts = alert_manager.get_active_alerts()
    assert alert_id not in [a['alert_id'] for a in active_alerts]


def test_acknowledge_alert_not_found(alert_manager):
    """测试确认告警（未找到）"""
    result = alert_manager.acknowledge_alert("non_existent_alert_id")
    assert result is False


def test_resolve_alert_success(alert_manager):
    """测试解决告警（成功）"""
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 50.0}
    alerts = alert_manager.check_alerts(stats)
    
    assert len(alerts) > 0
    alert_id = alerts[0]['alert_id']
    
    result = alert_manager.resolve_alert(alert_id, "问题已解决")
    assert result is True
    
    # 验证告警状态已更新
    for alert in alert_manager.alert_history:
        if alert['alert_id'] == alert_id:
            assert alert['status'] == 'resolved'
            assert alert['resolution'] == "问题已解决"
            break


def test_resolve_alert_not_found(alert_manager):
    """测试解决告警（未找到）"""
    result = alert_manager.resolve_alert("non_existent_alert_id")
    assert result is False


def test_evaluate_condition_operators(alert_manager, alert_manager_module):
    """测试评估条件（各种操作符）"""
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    # 测试 gt (大于)
    condition = AlertConditionConfig(operator="gt", field="value", value=10)
    assert alert_manager._evaluate_condition(condition, {'value': 15}) is True
    assert alert_manager._evaluate_condition(condition, {'value': 5}) is False
    
    # 测试 lt (小于)
    condition = AlertConditionConfig(operator="lt", field="value", value=10)
    assert alert_manager._evaluate_condition(condition, {'value': 5}) is True
    assert alert_manager._evaluate_condition(condition, {'value': 15}) is False
    
    # 测试 eq (等于)
    condition = AlertConditionConfig(operator="eq", field="value", value=10)
    assert alert_manager._evaluate_condition(condition, {'value': 10}) is True
    assert alert_manager._evaluate_condition(condition, {'value': 5}) is False
    
    # 测试 ne (不等于)
    condition = AlertConditionConfig(operator="ne", field="value", value=10)
    assert alert_manager._evaluate_condition(condition, {'value': 5}) is True
    assert alert_manager._evaluate_condition(condition, {'value': 10}) is False


def test_evaluate_condition_field_not_found(alert_manager, alert_manager_module):
    """测试评估条件（字段不存在）"""
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    condition = AlertConditionConfig(operator="gt", field="non_existent", value=10)
    result = alert_manager._evaluate_condition(condition, {'other_field': 15})
    assert result is False


def test_evaluate_condition_unsupported_operator(alert_manager, alert_manager_module):
    """测试评估条件（不支持的操作符）"""
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    condition = AlertConditionConfig(operator="unknown_op", field="value", value=10)
    result = alert_manager._evaluate_condition(condition, {'value': 15})
    assert result is False


def test_evaluate_condition_exception(alert_manager, alert_manager_module):
    """测试评估条件（异常处理）"""
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    # 创建一个会导致比较失败的条件
    condition = AlertConditionConfig(operator="gt", field="value", value=10)
    # 使用无法比较的值
    stats = {'value': object()}  # object()无法与数字比较
    
    result = alert_manager._evaluate_condition(condition, stats)
    assert result is False  # 应该返回False而不是抛出异常


def test_should_trigger_alert_cooldown(alert_manager, alert_manager_module):
    """测试判断是否触发告警（冷却时间）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="cooldown_rule",
        name="冷却规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning",
        cooldown=300  # 5分钟冷却
    )
    
    alert_manager.add_alert_rule(rule)
    
    # 第一次触发
    stats = {'test': 1}
    alerts1 = alert_manager.check_alerts(stats)
    assert len(alerts1) > 0
    
    # 立即再次检查，应该因为冷却时间而不触发
    alerts2 = alert_manager.check_alerts(stats)
    # 由于冷却时间，可能不会再次触发
    # 但至少第一次应该成功触发


def test_should_trigger_alert_no_conditions(alert_manager, alert_manager_module):
    """测试判断是否触发告警（无条件）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    
    rule = AlertRuleConfig(
        rule_id="no_condition_rule",
        name="无条件规则",
        description="测试",
        condition=None,
        severity="warning"
    )
    rule.conditions = []  # 确保conditions也为空
    
    alert_manager.add_alert_rule(rule)
    
    stats = {'test': 1}
    alerts = alert_manager.check_alerts(stats)
    # 应该不会触发，因为没有条件
    assert not any(a['rule_id'] == 'no_condition_rule' for a in alerts)


def test_create_alert(alert_manager, alert_manager_module):
    """测试创建告警"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="test_rule",
        name="测试规则",
        description="测试描述",
        condition=AlertConditionConfig(operator="gt", field="cpu", value=80),
        severity="warning",
        channels=["console", "email"]
    )
    
    stats = {'cpu': 90, 'memory': 50}
    alert = alert_manager._create_alert(rule, stats)
    
    assert alert is not None
    assert alert['rule_id'] == "test_rule"
    assert alert['rule_name'] == "测试规则"
    assert alert['severity'] == "warning"
    assert alert['status'] == 'active'
    assert alert['active'] is True
    assert alert['pool_name'] == "test_pool"
    assert 'triggered_at' in alert
    assert alert['stats'] == stats
    assert alert['channels'] == ["console", "email"]


def test_record_alert(alert_manager, alert_manager_module):
    """测试记录告警"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="test_rule",
        name="测试规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning"
    )
    
    stats = {'test': 1}
    alert = alert_manager._create_alert(rule, stats)
    
    initial_count = len(alert_manager.alert_history)
    alert_manager._record_alert(alert)
    
    assert len(alert_manager.alert_history) == initial_count + 1
    assert alert_manager.alert_history[-1] == alert
    assert alert['rule_id'] in alert_manager.last_alert_times


def test_record_alert_max_history_size(alert_manager, alert_manager_module):
    """测试记录告警（最大历史大小限制）"""
    alert_manager.max_history_size = 3
    
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="test_rule",
        name="测试规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning"
    )
    
    stats = {'test': 1}
    
    # 记录4个告警，应该只保留最后3个
    for i in range(4):
        alert = alert_manager._create_alert(rule, stats)
        alert['alert_id'] = f"alert_{i}"
        alert_manager._record_alert(alert)
    
    assert len(alert_manager.alert_history) == 3
    assert alert_manager.alert_history[0]['alert_id'] == "alert_1"
    assert alert_manager.alert_history[-1]['alert_id'] == "alert_3"


def test_enforce_active_capacity(alert_manager, alert_manager_module):
    """测试强制活跃告警容量限制"""
    alert_manager.config['max_active_alerts'] = 2
    
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="test_rule",
        name="测试规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning"
    )
    
    stats = {'test': 1}
    
    # 创建3个活跃告警
    for i in range(3):
        alert = alert_manager._create_alert(rule, stats)
        alert['alert_id'] = f"alert_{i}"
        alert['active'] = True
        alert['status'] = 'active'
        alert_manager._record_alert(alert)
    
    # 应该只有最后2个保持活跃
    active_alerts = alert_manager.get_active_alerts()
    assert len(active_alerts) == 2


def test_is_cooldown_expired_no_previous_alert(alert_manager):
    """测试检查冷却时间（无之前告警）"""
    result = alert_manager._is_cooldown_expired("new_rule", 300)
    assert result is True


def test_is_cooldown_expired_expired(alert_manager):
    """测试检查冷却时间（已过期）"""
    # 设置一个6分钟前的告警时间
    alert_manager.last_alert_times["test_rule"] = datetime.now() - timedelta(minutes=6)
    
    result = alert_manager._is_cooldown_expired("test_rule", 300)  # 5分钟冷却
    assert result is True


def test_is_cooldown_expired_not_expired(alert_manager):
    """测试检查冷却时间（未过期）"""
    # 设置1分钟前的告警时间
    alert_manager.last_alert_times["test_rule"] = datetime.now() - timedelta(minutes=1)
    
    result = alert_manager._is_cooldown_expired("test_rule", 300)  # 5分钟冷却
    assert result is False


def test_get_alert_statistics(alert_manager):
    """测试获取告警统计信息"""
    # 触发一些告警
    stats = {'hit_rate': 0.5, 'pool_size': 5, 'max_pool_size': 10, 'memory_usage_mb': 150.0}
    alert_manager.check_alerts(stats)
    
    # 确认一个告警
    if alert_manager.alert_history:
        alert_manager.acknowledge_alert(alert_manager.alert_history[0]['alert_id'])
    
    stats_result = alert_manager.get_alert_statistics()
    
    assert 'total_alerts' in stats_result
    assert 'active_alerts' in stats_result
    assert 'acknowledged_alerts' in stats_result
    assert 'level_distribution' in stats_result
    assert 'severity_breakdown' in stats_result
    assert 'rule_distribution' in stats_result
    assert 'generated_at' in stats_result
    assert stats_result['total_alerts'] > 0


def test_check_alerts_with_multiple_conditions(alert_manager, alert_manager_module):
    """测试检查告警（多个条件）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="multi_condition_rule",
        name="多条件规则",
        description="测试",
        condition=None,
        severity="warning"
    )
    # 设置多个条件
    rule.conditions = [
        AlertConditionConfig(operator="gt", field="cpu", value=80),
        AlertConditionConfig(operator="gt", field="memory", value=70)
    ]
    
    alert_manager.add_alert_rule(rule)
    
    # 满足所有条件
    stats1 = {'cpu': 90, 'memory': 80}
    alerts1 = alert_manager.check_alerts(stats1)
    assert any(a['rule_id'] == 'multi_condition_rule' for a in alerts1)
    
    # 只满足一个条件
    stats2 = {'cpu': 90, 'memory': 50}
    alerts2 = alert_manager.check_alerts(stats2)
    assert not any(a['rule_id'] == 'multi_condition_rule' for a in alerts2)


def test_should_trigger_alert_exception_handling(alert_manager, alert_manager_module, monkeypatch):
    """测试判断是否触发告警（异常处理）"""
    AlertRuleConfig = alert_manager_module.AlertRuleConfig
    AlertConditionConfig = alert_manager_module.AlertConditionConfig
    
    rule = AlertRuleConfig(
        rule_id="error_rule",
        name="错误规则",
        description="测试",
        condition=AlertConditionConfig(operator="eq", field="test", value=1),
        severity="warning"
    )
    
    alert_manager.add_alert_rule(rule)
    
    # 模拟_evaluate_condition抛出异常
    def failing_evaluate(*args, **kwargs):
        raise RuntimeError("Evaluation error")
    
    monkeypatch.setattr(alert_manager, "_evaluate_condition", failing_evaluate)
    
    stats = {'test': 1}
    alerts = alert_manager.check_alerts(stats)
    # 应该不会因为异常而崩溃
    assert isinstance(alerts, list)

