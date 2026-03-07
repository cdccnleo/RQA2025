#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试告警规则管理器组件
"""

import importlib
import sys
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def alert_rule_manager_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.alert_rule_manager"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def manager(alert_rule_manager_module):
    """创建AlertRuleManager实例"""
    return alert_rule_manager_module.AlertRuleManager()


@pytest.fixture
def sample_rule(alert_rule_manager_module):
    """创建示例规则"""
    AlertRule = alert_rule_manager_module.AlertRule
    AlertLevel = alert_rule_manager_module.AlertLevel
    AlertChannel = alert_rule_manager_module.AlertChannel
    
    return AlertRule(
        rule_id="test_rule_1",
        name="Test Rule",
        description="Test Description",
        condition={"operator": "gt", "field": "cpu_usage", "value": 80},
        level=AlertLevel.WARNING,
        channels=[AlertChannel.CONSOLE, AlertChannel.EMAIL],
        enabled=True,
        cooldown=300
    )


def test_initialization(manager):
    """测试初始化"""
    assert manager.rules == {}
    assert manager.rule_last_triggered == {}


def test_add_alert_rule_success(manager, sample_rule):
    """测试成功添加告警规则"""
    result = manager.add_alert_rule(sample_rule)
    assert result is True
    assert "test_rule_1" in manager.rules
    assert manager.rules["test_rule_1"] == sample_rule


def test_add_alert_rule_exception(manager, sample_rule, monkeypatch):
    """测试添加告警规则时发生异常"""
    # 创建一个会抛出异常的规则对象
    class FailingRule:
        def __init__(self):
            self.rule_id = "failing_rule"
        
        def __hash__(self):
            raise RuntimeError("Storage error")
    
    failing_rule = FailingRule()
    # 由于rule_id属性访问可能失败，我们直接测试异常处理
    # 实际上，由于rule_id是字符串，不太可能失败，所以这个测试主要验证异常处理路径
    # 我们可以通过patch rules字典来模拟异常
    original_rules = manager.rules
    
    class FailingDict(dict):
        def __setitem__(self, key, value):
            raise RuntimeError("Storage error")
    
    manager.rules = FailingDict()
    result = manager.add_alert_rule(sample_rule)
    assert result is False
    manager.rules = original_rules


def test_remove_alert_rule_success(manager, sample_rule):
    """测试成功移除告警规则"""
    manager.add_alert_rule(sample_rule)
    manager.update_rule_last_triggered("test_rule_1")
    
    result = manager.remove_alert_rule("test_rule_1")
    assert result is True
    assert "test_rule_1" not in manager.rules
    assert "test_rule_1" not in manager.rule_last_triggered


def test_remove_alert_rule_not_found(manager):
    """测试移除不存在的规则"""
    result = manager.remove_alert_rule("non_existent")
    assert result is False


def test_remove_alert_rule_exception(manager, sample_rule, monkeypatch):
    """测试移除告警规则时发生异常"""
    manager.add_alert_rule(sample_rule)
    
    # 创建一个会抛出异常的字典
    original_rules = manager.rules
    original_triggered = manager.rule_last_triggered
    
    class FailingDict(dict):
        def __delitem__(self, key):
            raise RuntimeError("Delete error")
    
    manager.rules = FailingDict(original_rules)
    result = manager.remove_alert_rule("test_rule_1")
    assert result is False
    
    # 恢复原始字典
    manager.rules = original_rules
    manager.rule_last_triggered = original_triggered


def test_get_alert_rule_exists(manager, sample_rule):
    """测试获取存在的规则"""
    manager.add_alert_rule(sample_rule)
    rule = manager.get_alert_rule("test_rule_1")
    assert rule == sample_rule


def test_get_alert_rule_not_exists(manager):
    """测试获取不存在的规则"""
    rule = manager.get_alert_rule("non_existent")
    assert rule is None


def test_get_all_rules(manager, sample_rule, alert_rule_manager_module):
    """测试获取所有规则"""
    AlertRule = alert_rule_manager_module.AlertRule
    AlertLevel = alert_rule_manager_module.AlertLevel
    AlertChannel = alert_rule_manager_module.AlertChannel
    
    rule2 = AlertRule(
        rule_id="test_rule_2",
        name="Test Rule 2",
        description="Test Description 2",
        condition={"operator": "lt", "field": "memory", "value": 20},
        level=AlertLevel.ERROR,
        channels=[AlertChannel.CONSOLE]
    )
    
    manager.add_alert_rule(sample_rule)
    manager.add_alert_rule(rule2)
    
    all_rules = manager.get_all_rules()
    assert len(all_rules) == 2
    assert "test_rule_1" in all_rules
    assert "test_rule_2" in all_rules
    # 验证返回的是副本，不是原始字典
    assert all_rules is not manager.rules


def test_get_enabled_rules(manager, sample_rule, alert_rule_manager_module):
    """测试获取启用的规则"""
    AlertRule = alert_rule_manager_module.AlertRule
    AlertLevel = alert_rule_manager_module.AlertLevel
    AlertChannel = alert_rule_manager_module.AlertChannel
    
    disabled_rule = AlertRule(
        rule_id="disabled_rule",
        name="Disabled Rule",
        description="Disabled",
        condition={"operator": "eq", "field": "x", "value": 1},
        level=AlertLevel.INFO,
        channels=[AlertChannel.CONSOLE],
        enabled=False
    )
    
    manager.add_alert_rule(sample_rule)
    manager.add_alert_rule(disabled_rule)
    
    enabled_rules = manager.get_enabled_rules()
    assert len(enabled_rules) == 1
    assert "test_rule_1" in enabled_rules
    assert "disabled_rule" not in enabled_rules


def test_update_rule_last_triggered_with_time(manager):
    """测试使用指定时间更新规则触发时间"""
    trigger_time = datetime(2025, 1, 1, 10, 0, 0)
    manager.update_rule_last_triggered("test_rule", trigger_time)
    assert manager.rule_last_triggered["test_rule"] == trigger_time


def test_update_rule_last_triggered_without_time(manager):
    """测试不使用指定时间更新规则触发时间（使用当前时间）"""
    before = datetime.now()
    manager.update_rule_last_triggered("test_rule")
    after = datetime.now()
    
    triggered_time = manager.rule_last_triggered["test_rule"]
    assert before <= triggered_time <= after


def test_is_rule_in_cooldown_not_triggered(manager, sample_rule):
    """测试规则未触发过，不在冷却时间"""
    result = manager.is_rule_in_cooldown(sample_rule)
    assert result is False


def test_is_rule_in_cooldown_in_cooldown(manager, sample_rule):
    """测试规则在冷却时间内"""
    manager.update_rule_last_triggered("test_rule_1")
    # 规则冷却时间是300秒，刚触发，应该在冷却时间内
    result = manager.is_rule_in_cooldown(sample_rule)
    assert result is True


def test_is_rule_in_cooldown_cooldown_expired(manager, sample_rule):
    """测试规则冷却时间已过期"""
    # 设置一个很久以前的触发时间
    old_time = datetime.now() - timedelta(seconds=400)
    manager.update_rule_last_triggered("test_rule_1", old_time)
    
    result = manager.is_rule_in_cooldown(sample_rule)
    assert result is False


def test_is_rule_in_cooldown_with_current_time(manager, sample_rule):
    """测试使用指定的当前时间检查冷却"""
    trigger_time = datetime(2025, 1, 1, 10, 0, 0)
    manager.update_rule_last_triggered("test_rule_1", trigger_time)
    
    # 使用触发后100秒的时间（小于300秒冷却时间）
    current_time = trigger_time + timedelta(seconds=100)
    result = manager.is_rule_in_cooldown(sample_rule, current_time)
    assert result is True
    
    # 使用触发后400秒的时间（大于300秒冷却时间）
    current_time = trigger_time + timedelta(seconds=400)
    result = manager.is_rule_in_cooldown(sample_rule, current_time)
    assert result is False


def test_create_rule_from_template_performance(manager, alert_rule_manager_module):
    """测试从性能模板创建规则"""
    config = {
        "metric": "cpu_usage",
        "threshold": 85,
        "level": "error"
    }
    
    rule = manager.create_rule_from_template("performance_threshold", config)
    assert rule is not None
    assert rule.rule_id is not None
    assert "性能阈值告警" in rule.name or "performance" in rule.name.lower()


def test_create_rule_from_template_error_rate(manager):
    """测试从错误率模板创建规则"""
    config = {
        "threshold": 10
    }
    
    rule = manager.create_rule_from_template("error_rate_monitor", config)
    assert rule is not None
    assert rule.condition["field"] == "error_rate"
    assert rule.condition["value"] == 10


def test_create_rule_from_template_security(manager):
    """测试从安全模板创建规则"""
    config = {
        "event_type": "brute_force"
    }
    
    rule = manager.create_rule_from_template("security_alert", config)
    assert rule is not None
    assert rule.condition["field"] == "event_type"
    assert rule.condition["value"] == "brute_force"


def test_create_rule_from_template_invalid_template(manager):
    """测试使用无效模板名称"""
    rule = manager.create_rule_from_template("invalid_template", {})
    assert rule is None


def test_create_rule_from_template_with_custom_rule_id(manager):
    """测试使用自定义规则ID创建规则"""
    config = {
        "rule_id": "custom_rule_123",
        "threshold": 80
    }
    
    rule = manager.create_rule_from_template("performance_threshold", config)
    assert rule is not None
    assert rule.rule_id == "custom_rule_123"


def test_get_rules_count(manager, sample_rule):
    """测试获取规则数量"""
    assert manager.get_rules_count() == 0
    
    manager.add_alert_rule(sample_rule)
    assert manager.get_rules_count() == 1
    
    manager.add_alert_rule(sample_rule)  # 更新现有规则
    assert manager.get_rules_count() == 1


def test_clear_all_rules(manager, sample_rule):
    """测试清空所有规则"""
    manager.add_alert_rule(sample_rule)
    manager.update_rule_last_triggered("test_rule_1")
    
    assert len(manager.rules) == 1
    assert len(manager.rule_last_triggered) == 1
    
    manager.clear_all_rules()
    
    assert len(manager.rules) == 0
    assert len(manager.rule_last_triggered) == 0


def test_fallback_imports(monkeypatch):
    """测试导入失败时的fallback定义"""
    import builtins
    original_import = builtins.__import__
    
    # 保存原始导入函数
    import_count = {"count": 0}
    
    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        # 只对特定的模块抛出异常
        if "alert_service" in name or "alert_system" in name:
            raise ImportError("Module not found")
        # 对于其他模块，使用原始导入
        return original_import(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr(builtins, "__import__", failing_import)
    
    # 重新导入模块
    module_name = "src.infrastructure.monitoring.components.alert_rule_manager"
    if module_name in sys.modules:
        del sys.modules[module_name]
    module = importlib.import_module(module_name)
    
    # 验证fallback类型存在
    assert hasattr(module, "AlertLevel")
    assert hasattr(module, "AlertChannel")
    assert hasattr(module, "AlertRule")
    assert hasattr(module, "AlertRuleManager")
    
    # 验证可以创建实例
    manager = module.AlertRuleManager()
    assert manager is not None

