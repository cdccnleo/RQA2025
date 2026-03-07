#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试告警条件评估器组件
"""

import importlib
import sys
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def alert_condition_evaluator_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.alert_condition_evaluator"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def evaluator(alert_condition_evaluator_module):
    """创建AlertConditionEvaluator实例"""
    return alert_condition_evaluator_module.AlertConditionEvaluator()


def test_initialization(evaluator):
    """测试初始化"""
    assert evaluator.operator_handlers is not None
    assert 'eq' in evaluator.operator_handlers
    assert 'gt' in evaluator.operator_handlers
    assert 'lt' in evaluator.operator_handlers
    assert 'contains' in evaluator.operator_handlers
    assert 'regex' in evaluator.operator_handlers


def test_evaluate_condition_equals(evaluator):
    """测试评估条件（等于）"""
    condition = {'operator': 'eq', 'field': 'value', 'value': 10}
    data = {'value': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 20}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_not_equals(evaluator):
    """测试评估条件（不等于）"""
    condition = {'operator': 'ne', 'field': 'value', 'value': 10}
    data = {'value': 20}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 10}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_greater_than(evaluator):
    """测试评估条件（大于）"""
    condition = {'operator': 'gt', 'field': 'value', 'value': 10}
    data = {'value': 20}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 5}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_greater_than_or_equal(evaluator):
    """测试评估条件（大于等于）"""
    condition = {'operator': 'gte', 'field': 'value', 'value': 10}
    data = {'value': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 20}
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 5}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_less_than(evaluator):
    """测试评估条件（小于）"""
    condition = {'operator': 'lt', 'field': 'value', 'value': 10}
    data = {'value': 5}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 20}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_less_than_or_equal(evaluator):
    """测试评估条件（小于等于）"""
    condition = {'operator': 'lte', 'field': 'value', 'value': 10}
    data = {'value': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 5}
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'value': 20}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_contains(evaluator):
    """测试评估条件（包含）"""
    condition = {'operator': 'contains', 'field': 'message', 'value': 'error'}
    data = {'message': 'This is an error message'}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'message': 'This is a success message'}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_regex(evaluator):
    """测试评估条件（正则表达式）"""
    condition = {'operator': 'regex', 'field': 'email', 'value': r'^[a-z]+@[a-z]+\.[a-z]+$'}
    data = {'email': 'test@example.com'}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True
    
    data = {'email': 'invalid-email'}
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_field_not_found(evaluator):
    """测试评估条件（字段不存在）"""
    condition = {'operator': 'eq', 'field': 'missing_field', 'value': 10}
    data = {'other_field': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is False


def test_evaluate_condition_unsupported_operator(evaluator, capsys):
    """测试评估条件（不支持的操作符）"""
    condition = {'operator': 'unknown_op', 'field': 'value', 'value': 10}
    data = {'value': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'unknown_op' in captured.out or '不支持' in captured.out


def test_evaluate_condition_default_operator(evaluator):
    """测试评估条件（默认操作符）"""
    condition = {'field': 'value', 'value': 10}  # 没有operator，应该使用默认的'eq'
    data = {'value': 10}
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is True


def test_evaluate_condition_exception(evaluator, capsys):
    """测试评估条件（异常）"""
    condition = {'operator': 'eq', 'field': 'value', 'value': 10}
    
    # 创建一个在访问字段值时抛出异常的数据对象
    class FailingData:
        def __contains__(self, key):
            return True  # 让field检查通过
        
        def __getitem__(self, key):
            raise RuntimeError("Access error")
    
    data = FailingData()
    
    result = evaluator.evaluate_condition(condition, data)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'Access error' in captured.out or '条件评估失败' in captured.out


def test_evaluate_greater_than_with_strings(evaluator):
    """测试大于操作（字符串数字）"""
    result = evaluator._evaluate_greater_than("20", "10")
    assert result is True
    
    result = evaluator._evaluate_greater_than("5", "10")
    assert result is False


def test_evaluate_greater_than_with_invalid_types(evaluator):
    """测试大于操作（无效类型）"""
    result = evaluator._evaluate_greater_than("not_a_number", 10)
    assert result is False
    
    result = evaluator._evaluate_greater_than(None, 10)
    assert result is False


def test_evaluate_greater_than_or_equal_with_strings(evaluator):
    """测试大于等于操作（字符串数字）"""
    result = evaluator._evaluate_greater_than_or_equal("10", "10")
    assert result is True
    
    result = evaluator._evaluate_greater_than_or_equal("5", "10")
    assert result is False


def test_evaluate_less_than_with_strings(evaluator):
    """测试小于操作（字符串数字）"""
    result = evaluator._evaluate_less_than("5", "10")
    assert result is True
    
    result = evaluator._evaluate_less_than("20", "10")
    assert result is False


def test_evaluate_less_than_or_equal_with_strings(evaluator):
    """测试小于等于操作（字符串数字）"""
    result = evaluator._evaluate_less_than_or_equal("10", "10")
    assert result is True
    
    result = evaluator._evaluate_less_than_or_equal("20", "10")
    assert result is False


def test_evaluate_contains_with_different_types(evaluator):
    """测试包含操作（不同类型）"""
    result = evaluator._evaluate_contains("test string", "test")
    assert result is True
    
    result = evaluator._evaluate_contains(12345, "123")
    assert result is True
    
    result = evaluator._evaluate_contains("test", "missing")
    assert result is False


def test_evaluate_contains_exception(evaluator):
    """测试包含操作（异常）"""
    # 创建一个无法转换为字符串的对象
    class Unconvertible:
        def __str__(self):
            raise RuntimeError("Cannot convert")
    
    result = evaluator._evaluate_contains(Unconvertible(), "test")
    assert result is False


def test_evaluate_regex_success(evaluator):
    """测试正则表达式匹配（成功）"""
    result = evaluator._evaluate_regex("test@example.com", r'^[a-z]+@[a-z]+\.[a-z]+$')
    assert result is True
    
    result = evaluator._evaluate_regex("hello world", r'hello')
    assert result is True


def test_evaluate_regex_failure(evaluator):
    """测试正则表达式匹配（失败）"""
    result = evaluator._evaluate_regex("test", r'^[0-9]+$')
    assert result is False


def test_evaluate_regex_invalid_pattern(evaluator):
    """测试正则表达式匹配（无效模式）"""
    result = evaluator._evaluate_regex("test", r'[invalid(')
    assert result is False


def test_evaluate_regex_exception(evaluator):
    """测试正则表达式匹配（异常）"""
    class Unconvertible:
        def __str__(self):
            raise RuntimeError("Cannot convert")
    
    result = evaluator._evaluate_regex(Unconvertible(), r'test')
    assert result is False


def test_validate_condition_success(evaluator):
    """测试验证条件（成功）"""
    condition = {'operator': 'eq', 'field': 'value', 'value': 10}
    result = evaluator.validate_condition(condition)
    assert result is True


def test_validate_condition_missing_operator(evaluator, capsys):
    """测试验证条件（缺少操作符）"""
    condition = {'field': 'value', 'value': 10}
    
    result = evaluator.validate_condition(condition)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'operator' in captured.out.lower() or '必需字段' in captured.out


def test_validate_condition_missing_field(evaluator, capsys):
    """测试验证条件（缺少字段）"""
    condition = {'operator': 'eq', 'value': 10}
    
    result = evaluator.validate_condition(condition)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'field' in captured.out.lower() or '必需字段' in captured.out


def test_validate_condition_missing_value(evaluator, capsys):
    """测试验证条件（缺少值）"""
    condition = {'operator': 'eq', 'field': 'value'}
    
    result = evaluator.validate_condition(condition)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'value' in captured.out.lower() or '必需字段' in captured.out


def test_validate_condition_unsupported_operator(evaluator, capsys):
    """测试验证条件（不支持的操作符）"""
    condition = {'operator': 'unknown_op', 'field': 'value', 'value': 10}
    
    result = evaluator.validate_condition(condition)
    assert result is False
    
    # 验证打印了错误信息
    captured = capsys.readouterr()
    assert 'unknown_op' in captured.out or '不支持' in captured.out


def test_get_supported_operators(evaluator):
    """测试获取支持的操作符列表"""
    operators = evaluator.get_supported_operators()
    
    assert isinstance(operators, list)
    assert 'eq' in operators
    assert 'ne' in operators
    assert 'gt' in operators
    assert 'gte' in operators
    assert 'lt' in operators
    assert 'lte' in operators
    assert 'contains' in operators
    assert 'regex' in operators
    assert len(operators) == 8


def test_evaluate_equals(evaluator):
    """测试等于操作"""
    assert evaluator._evaluate_equals(10, 10) is True
    assert evaluator._evaluate_equals(10, 20) is False
    assert evaluator._evaluate_equals("test", "test") is True
    assert evaluator._evaluate_equals("test", "other") is False


def test_evaluate_not_equals(evaluator):
    """测试不等于操作"""
    assert evaluator._evaluate_not_equals(10, 20) is True
    assert evaluator._evaluate_not_equals(10, 10) is False
    assert evaluator._evaluate_not_equals("test", "other") is True
    assert evaluator._evaluate_not_equals("test", "test") is False

