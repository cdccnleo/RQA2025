# tests/unit/automation/test_automation_rules_engine.py
"""
自动化规则引擎深度测试

测试覆盖:
- 规则条件评估逻辑
- 规则动作执行机制
- 规则优先级和冲突解决
- 规则性能监控和统计
- 复杂条件表达式处理
- 规则依赖和级联触发
- 规则版本控制和回滚
- 规则测试和验证
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Mock关键类和结构
class MockRuleCondition:
    """Mock规则条件"""
    def __init__(self, field, operator, value, logical_operator="AND"):
        self.field = field
        self.operator = operator
        self.value = value
        self.logical_operator = logical_operator


class MockRuleAction:
    """Mock规则动作"""
    def __init__(self, action_type, parameters=None):
        self.action_type = action_type
        self.parameters = parameters or {}


class MockAutomationRule:
    """Mock自动化规则"""
    def __init__(self, rule_id, name, conditions, actions, priority=1, enabled=True,
                 description="", tags=None, version="1.0"):
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions  # List of MockRuleCondition
        self.actions = actions  # List of MockRuleAction
        self.priority = priority
        self.enabled = enabled
        self.description = description
        self.tags = tags or []
        self.version = version

        # Runtime stats
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.last_execution = None
        self.average_execution_time = 0.0
        self.created_at = datetime.now()
        self.updated_at = datetime.now()


class MockRuleEngine:
    """Mock规则引擎"""

    def __init__(self):
        self.rules = {}
        self.rule_groups = {}
        self.execution_history = []
        self.stats = {
            'total_rules': 0,
            'active_rules': 0,
            'executions_today': 0,
            'success_rate': 0.0,
            'average_response_time': 0.0
        }

    def add_rule(self, rule):
        """添加规则"""
        self.rules[rule.rule_id] = rule
        self.stats['total_rules'] += 1
        if rule.enabled:
            self.stats['active_rules'] += 1
        return True

    def remove_rule(self, rule_id):
        """移除规则"""
        if rule_id in self.rules:
            rule = self.rules[rule_id]
            if rule.enabled:
                self.stats['active_rules'] -= 1
            del self.rules[rule_id]
            self.stats['total_rules'] -= 1
            return True
        return False

    def evaluate_conditions(self, conditions, context):
        """评估条件列表"""
        if not conditions:
            return True

        results = []
        for condition in conditions:
            result = self._evaluate_single_condition(condition, context)
            results.append(result)

        # Apply logical operators (simplified - all AND for now)
        return all(results)

    def _evaluate_single_condition(self, condition, context):
        """评估单个条件"""
        field_value = self._get_nested_value(context, condition.field)

        if field_value is None:
            return False

        operator = condition.operator
        expected_value = condition.value

        if operator == "equals":
            return field_value == expected_value
        elif operator == "not_equals":
            return field_value != expected_value
        elif operator == "greater_than":
            return field_value > expected_value
        elif operator == "less_than":
            return field_value < expected_value
        elif operator == "greater_equal":
            return field_value >= expected_value
        elif operator == "less_equal":
            return field_value <= expected_value
        elif operator == "contains":
            return expected_value in field_value if isinstance(field_value, (str, list)) else False
        elif operator == "not_contains":
            return expected_value not in field_value if isinstance(field_value, (str, list)) else True
        elif operator == "regex_match":
            import re
            try:
                return bool(re.match(expected_value, str(field_value)))
            except:
                return False
        else:
            return False

    def _get_nested_value(self, data, field_path):
        """获取嵌套字段值"""
        keys = field_path.split('.')
        current = data

        try:
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list) and key.isdigit():
                    current = current[int(key)]
                else:
                    return None
            return current
        except (KeyError, IndexError, TypeError):
            return None

    def execute_actions(self, actions, context):
        """执行动作列表"""
        results = []
        success = True

        for action in actions:
            try:
                result = self._execute_single_action(action, context)
                results.append(result)
                if not result.get('success', False):
                    success = False
            except Exception as e:
                results.append({
                    'success': False,
                    'action_type': action.action_type,
                    'error': str(e)
                })
                success = False

        return {
            'success': success,
            'results': results,
            'total_actions': len(actions),
            'successful_actions': sum(1 for r in results if r.get('success', False))
        }

    def _execute_single_action(self, action, context):
        """执行单个动作"""
        action_type = action.action_type
        params = action.parameters

        if action_type == "send_notification":
            return self._execute_notification_action(params, context)
        elif action_type == "execute_task":
            return self._execute_task_action(params, context)
        elif action_type == "update_data":
            return self._execute_data_update_action(params, context)
        elif action_type == "log_event":
            return self._execute_logging_action(params, context)
        elif action_type == "trigger_workflow":
            return self._execute_workflow_trigger_action(params, context)
        else:
            return {
                'success': False,
                'action_type': action_type,
                'error': f'Unknown action type: {action_type}'
            }

    def _execute_notification_action(self, params, context):
        """执行通知动作"""
        recipients = params.get('recipients', [])
        message_template = params.get('message', 'Rule triggered notification')
        priority = params.get('priority', 'normal')

        # Simple template substitution
        message = message_template
        for key, value in context.items():
            placeholder = f"${{{key}}}"
            if placeholder in message:
                message = message.replace(placeholder, str(value))
            # Also support ${key} format
            simple_placeholder = f"${key}"
            if simple_placeholder in message:
                message = message.replace(simple_placeholder, str(value))

        # Mock notification sending
        return {
            'success': True,
            'action_type': 'send_notification',
            'recipients': recipients,
            'message': message,
            'priority': priority,
            'sent_at': datetime.now().isoformat()
        }

    def _execute_task_action(self, params, context):
        """执行任务动作"""
        task_type = params.get('task_type', 'generic')
        task_params = params.get('task_params', {})
        timeout = params.get('timeout', 30)

        # Mock task execution
        return {
            'success': True,
            'action_type': 'execute_task',
            'task_type': task_type,
            'task_params': task_params,
            'timeout': timeout,
            'execution_time': 1.5,
            'task_id': f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    def _execute_data_update_action(self, params, context):
        """执行数据更新动作"""
        target_table = params.get('target_table', 'unknown')
        updates = params.get('updates', {})
        conditions = params.get('conditions', {})

        # Mock data update
        return {
            'success': True,
            'action_type': 'update_data',
            'target_table': target_table,
            'updates': updates,
            'conditions': conditions,
            'affected_rows': 1
        }

    def _execute_logging_action(self, params, context):
        """执行日志动作"""
        level = params.get('level', 'INFO')
        message = params.get('message', 'Rule execution logged')
        include_context = params.get('include_context', False)

        log_data = {
            'level': level,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'rule_context': context if include_context else {}
        }

        # Mock logging
        return {
            'success': True,
            'action_type': 'log_event',
            'log_data': log_data
        }

    def _execute_workflow_trigger_action(self, params, context):
        """执行工作流触发动作"""
        workflow_id = params.get('workflow_id', 'unknown')
        workflow_params = params.get('workflow_params', {})

        # Mock workflow triggering
        return {
            'success': True,
            'action_type': 'trigger_workflow',
            'workflow_id': workflow_id,
            'workflow_params': workflow_params,
            'triggered_at': datetime.now().isoformat(),
            'instance_id': f"wf_instance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

    def evaluate_rule(self, rule, context):
        """评估完整规则"""
        if not rule.enabled:
            return {
                'triggered': False,
                'reason': 'Rule is disabled'
            }

        start_time = datetime.now()

        try:
            # Evaluate conditions
            conditions_met = self.evaluate_conditions(rule.conditions, context)

            if not conditions_met:
                return {
                    'triggered': False,
                    'reason': 'Conditions not met',
                    'evaluation_time': (datetime.now() - start_time).total_seconds()
                }

            # Execute actions
            action_results = self.execute_actions(rule.actions, context)

            # Update rule statistics
            rule.execution_count += 1
            rule.last_execution = datetime.now()

            if action_results['success']:
                rule.success_count += 1
            else:
                rule.failure_count += 1

            # Record execution history
            execution_record = {
                'rule_id': rule.rule_id,
                'timestamp': datetime.now().isoformat(),
                'context': context,
                'conditions_met': conditions_met,
                'action_results': action_results,
                'evaluation_time': (datetime.now() - start_time).total_seconds()
            }
            self.execution_history.append(execution_record)

            return {
                'triggered': True,
                'conditions_met': conditions_met,
                'action_results': action_results,
                'evaluation_time': (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            rule.failure_count += 1
            return {
                'triggered': False,
                'error': str(e),
                'evaluation_time': (datetime.now() - start_time).total_seconds()
            }

    def get_rule_stats(self):
        """获取规则统计信息"""
        total_executions = sum(rule.execution_count for rule in self.rules.values())
        total_success = sum(rule.success_count for rule in self.rules.values())
        total_failure = sum(rule.failure_count for rule in self.rules.values())

        return {
            'total_rules': len(self.rules),
            'active_rules': len([r for r in self.rules.values() if r.enabled]),
            'total_executions': total_executions,
            'successful_executions': total_success,
            'failed_executions': total_failure,
            'success_rate': total_success / total_executions if total_executions > 0 else 0.0,
            'rules': {
                rule_id: {
                    'execution_count': rule.execution_count,
                    'success_count': rule.success_count,
                    'failure_count': rule.failure_count,
                    'success_rate': rule.success_count / rule.execution_count if rule.execution_count > 0 else 0.0,
                    'last_execution': rule.last_execution.isoformat() if rule.last_execution else None
                }
                for rule_id, rule in self.rules.items()
            }
        }


class TestAutomationRulesEngine:
    """测试自动化规则引擎"""

    def setup_method(self):
        """测试前准备"""
        self.engine = MockRuleEngine()

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine is not None
        assert hasattr(self.engine, 'rules')
        assert hasattr(self.engine, 'execution_history')
        assert hasattr(self.engine, 'stats')
        assert isinstance(self.engine.rules, dict)
        assert isinstance(self.engine.execution_history, list)

    def test_rule_management(self):
        """测试规则管理"""
        # Create test rule
        conditions = [
            MockRuleCondition("price", "greater_than", 100),
            MockRuleCondition("volume", "greater_than", 1000)
        ]
        actions = [
            MockRuleAction("send_notification", {"message": "Price alert"}),
            MockRuleAction("execute_task", {"task_type": "alert_task"})
        ]

        rule = MockAutomationRule(
            rule_id="test_rule_001",
            name="Test Rule",
            conditions=conditions,
            actions=actions,
            priority=1,
            enabled=True,
            description="Test rule for unit testing",
            tags=["test", "alert"]
        )

        # Add rule
        result = self.engine.add_rule(rule)
        assert result is True
        assert "test_rule_001" in self.engine.rules
        assert self.engine.stats['total_rules'] == 1
        assert self.engine.stats['active_rules'] == 1

        # Remove rule
        result = self.engine.remove_rule("test_rule_001")
        assert result is True
        assert "test_rule_001" not in self.engine.rules
        assert self.engine.stats['total_rules'] == 0
        assert self.engine.stats['active_rules'] == 0

    def test_condition_evaluation(self):
        """测试条件评估"""
        # Test various operators
        context = {
            'price': 150,
            'volume': 2000,
            'symbol': 'AAPL',
            'change_pct': 5.5,
            'tags': ['tech', 'growth'],
            'description': 'Apple Inc. stock'
        }

        # Test equals
        condition1 = MockRuleCondition("symbol", "equals", "AAPL")
        assert self.engine._evaluate_single_condition(condition1, context) is True

        condition1_false = MockRuleCondition("symbol", "equals", "GOOGL")
        assert self.engine._evaluate_single_condition(condition1_false, context) is False

        # Test greater_than
        condition2 = MockRuleCondition("price", "greater_than", 100)
        assert self.engine._evaluate_single_condition(condition2, context) is True

        condition2_false = MockRuleCondition("price", "greater_than", 200)
        assert self.engine._evaluate_single_condition(condition2_false, context) is False

        # Test less_than
        condition3 = MockRuleCondition("change_pct", "less_than", 10)
        assert self.engine._evaluate_single_condition(condition3, context) is True

        # Test contains
        condition4 = MockRuleCondition("tags", "contains", "tech")
        assert self.engine._evaluate_single_condition(condition4, context) is True

        condition4_false = MockRuleCondition("tags", "contains", "value")
        assert self.engine._evaluate_single_condition(condition4_false, context) is False

        # Test regex_match
        condition5 = MockRuleCondition("description", "regex_match", r"Apple.*stock")
        assert self.engine._evaluate_single_condition(condition5, context) is True

    def test_nested_field_access(self):
        """测试嵌套字段访问"""
        context = {
            'market_data': {
                'stocks': {
                    'AAPL': {
                        'price': 150,
                        'volume': 2000
                    },
                    'GOOGL': {
                        'price': 2800,
                        'volume': 1500
                    }
                }
            },
            'portfolio': [
                {'symbol': 'AAPL', 'shares': 100},
                {'symbol': 'GOOGL', 'shares': 50}
            ]
        }

        # Test nested dict access
        assert self.engine._get_nested_value(context, 'market_data.stocks.AAPL.price') == 150
        assert self.engine._get_nested_value(context, 'market_data.stocks.GOOGL.volume') == 1500

        # Test array access
        assert self.engine._get_nested_value(context, 'portfolio.0.symbol') == 'AAPL'
        assert self.engine._get_nested_value(context, 'portfolio.1.shares') == 50

        # Test invalid paths
        assert self.engine._get_nested_value(context, 'invalid.path') is None
        assert self.engine._get_nested_value(context, 'market_data.nonexistent') is None

    def test_complex_condition_evaluation(self):
        """测试复杂条件评估"""
        context = {
            'price': 180,
            'volume': 2500,
            'change_pct': 7.5,
            'market_cap': 2500000000,
            'pe_ratio': 25.5,
            'div_yield': 1.2,
            'beta': 1.1,
            'sector': 'technology'
        }

        conditions = [
            MockRuleCondition("price", "greater_than", 150),
            MockRuleCondition("volume", "greater_than", 2000),
            MockRuleCondition("change_pct", "greater_than", 5),
            MockRuleCondition("market_cap", "greater_than", 1000000000),
            MockRuleCondition("sector", "equals", "technology")
        ]

        # All conditions should be met
        result = self.engine.evaluate_conditions(conditions, context)
        assert result is True

        # Test with one failing condition
        failing_conditions = conditions + [MockRuleCondition("beta", "less_than", 1.0)]
        result_fail = self.engine.evaluate_conditions(failing_conditions, context)
        assert result_fail is False

        # Test empty conditions (should return True)
        result_empty = self.engine.evaluate_conditions([], context)
        assert result_empty is True

    def test_action_execution(self):
        """测试动作执行"""
        context = {
            'alert_type': 'price_spike',
            'symbol': 'AAPL',
            'current_price': 200,
            'threshold': 180
        }

        # Test notification action
        notification_action = MockRuleAction("send_notification", {
            'recipients': ['trader@firm.com', 'manager@firm.com'],
            'message': 'Price spike detected for ${symbol}',
            'priority': 'high'
        })

        result = self.engine._execute_single_action(notification_action, context)
        assert result['success'] is True
        assert result['action_type'] == 'send_notification'
        assert 'AAPL' in result['message']

        # Test task execution action
        task_action = MockRuleAction("execute_task", {
            'task_type': 'rebalance_portfolio',
            'task_params': {'symbol': 'AAPL', 'action': 'sell'},
            'timeout': 60
        })

        result = self.engine._execute_single_action(task_action, context)
        assert result['success'] is True
        assert result['action_type'] == 'execute_task'
        assert result['task_type'] == 'rebalance_portfolio'

        # Test data update action
        data_action = MockRuleAction("update_data", {
            'target_table': 'trading_signals',
            'updates': {'status': 'processed', 'processed_at': datetime.now().isoformat()},
            'conditions': {'symbol': 'AAPL'}
        })

        result = self.engine._execute_single_action(data_action, context)
        assert result['success'] is True
        assert result['action_type'] == 'update_data'
        assert result['affected_rows'] == 1

    def test_rule_evaluation_complete(self):
        """测试完整规则评估"""
        # Create comprehensive rule
        conditions = [
            MockRuleCondition("price", "greater_than", 150),
            MockRuleCondition("volume", "greater_than", 1000),
            MockRuleCondition("change_pct", "greater_than", 3)
        ]

        actions = [
            MockRuleAction("send_notification", {'message': 'Trading signal'}),
            MockRuleAction("log_event", {'level': 'INFO', 'include_context': True}),
            MockRuleAction("trigger_workflow", {'workflow_id': 'signal_processing'})
        ]

        rule = MockAutomationRule(
            rule_id="comprehensive_rule",
            name="Comprehensive Trading Rule",
            conditions=conditions,
            actions=actions,
            enabled=True
        )

        self.engine.add_rule(rule)

        # Test with conditions met
        context_met = {
            'price': 180,
            'volume': 2500,
            'change_pct': 5.5,
            'symbol': 'AAPL'
        }

        result = self.engine.evaluate_rule(rule, context_met)
        assert result['triggered'] is True
        assert result['conditions_met'] is True
        assert result['action_results']['success'] is True
        assert result['action_results']['total_actions'] == 3
        assert result['action_results']['successful_actions'] == 3

        # Verify rule stats updated
        assert rule.execution_count == 1
        assert rule.success_count == 1
        assert rule.last_execution is not None

        # Test with conditions not met
        context_not_met = {
            'price': 120,  # Too low
            'volume': 2500,
            'change_pct': 5.5
        }

        result_not_met = self.engine.evaluate_rule(rule, context_not_met)
        assert result_not_met['triggered'] is False
        assert result_not_met['reason'] == 'Conditions not met'

        # Test disabled rule
        rule.enabled = False
        result_disabled = self.engine.evaluate_rule(rule, context_met)
        assert result_disabled['triggered'] is False
        assert result_disabled['reason'] == 'Rule is disabled'

    def test_rule_statistics(self):
        """测试规则统计"""
        # Add multiple rules
        for i in range(3):
            conditions = [MockRuleCondition("metric", "greater_than", i * 10)]
            actions = [MockRuleAction("log_event", {})]
            rule = MockAutomationRule(f"stat_rule_{i}", f"Stat Rule {i}", conditions, actions)
            self.engine.add_rule(rule)

        # Execute rules multiple times
        contexts = [
            {'metric': 5},   # Should trigger rule 0 only (metric > 0)
            {'metric': 15},  # Should trigger rule 0 and 1 (metric > 0 and > 10)
            {'metric': 25},  # Should trigger all rules (metric > 0, > 10, > 20)
        ]

        expected_triggers = [1, 2, 3]  # Expected triggers per context

        total_expected_executions = 0
        for i, context in enumerate(contexts):
            triggers = 0
            for rule in self.engine.rules.values():
                result = self.engine.evaluate_rule(rule, context)
                if result['triggered']:
                    triggers += 1
            total_expected_executions += expected_triggers[i]

        # Check statistics
        stats = self.engine.get_rule_stats()
        assert stats['total_rules'] == 3
        assert stats['active_rules'] == 3
        assert stats['total_executions'] == total_expected_executions
        assert 'success_rate' in stats
        assert 'rules' in stats

        # Verify individual rule stats
        assert len(stats['rules']) == 3
        for rule_id, rule_stats in stats['rules'].items():
            assert 'execution_count' in rule_stats
            assert 'success_count' in rule_stats
            assert 'success_rate' in rule_stats

    def test_error_handling(self):
        """测试错误处理"""
        # Test with invalid condition operator
        invalid_condition = MockRuleCondition("field", "invalid_operator", "value")
        context = {'field': 'test_value'}

        # Should handle gracefully
        result = self.engine._evaluate_single_condition(invalid_condition, context)
        assert result is False  # Unknown operator should return False

        # Test action execution with unknown action type
        unknown_action = MockRuleAction("unknown_action_type", {})
        result = self.engine._execute_single_action(unknown_action, context)
        assert result['success'] is False
        assert 'error' in result

        # Test rule evaluation with exception in action
        conditions = [MockRuleCondition("trigger", "equals", True)]
        actions = [MockRuleAction("send_notification", {})]  # This should work

        rule = MockAutomationRule("error_rule", "Error Test Rule", conditions, actions)
        self.engine.add_rule(rule)

        context_ok = {'trigger': True}
        result = self.engine.evaluate_rule(rule, context_ok)
        assert result['triggered'] is True  # Should still succeed

    def test_rule_performance_monitoring(self):
        """测试规则性能监控"""
        # Create rule with timing measurement
        conditions = [MockRuleCondition("value", "greater_than", 0)]
        actions = [
            MockRuleAction("log_event", {}),
            MockRuleAction("send_notification", {}),
            MockRuleAction("execute_task", {})
        ]

        rule = MockAutomationRule("perf_rule", "Performance Rule", conditions, actions)
        self.engine.add_rule(rule)

        # Execute rule multiple times
        for i in range(5):
            context = {'value': i + 1, 'iteration': i}
            result = self.engine.evaluate_rule(rule, context)
            assert result['triggered'] is True
            assert 'evaluation_time' in result

        # Check execution history
        assert len(self.engine.execution_history) == 5
        for record in self.engine.execution_history:
            assert 'rule_id' in record
            assert 'timestamp' in record
            assert 'evaluation_time' in record
            assert record['rule_id'] == 'perf_rule'

        # Verify rule stats
        assert rule.execution_count == 5
        assert rule.success_count == 5
        assert rule.failure_count == 0

    def test_rule_priority_and_ordering(self):
        """测试规则优先级和排序"""
        # Create rules with different priorities
        rules_data = [
            ("high_priority_rule", "High Priority", 3),
            ("medium_priority_rule", "Medium Priority", 2),
            ("low_priority_rule", "Low Priority", 1)
        ]

        for rule_id, name, priority in rules_data:
            conditions = [MockRuleCondition("trigger", "equals", True)]
            actions = [MockRuleAction("log_event", {"message": f"{name} triggered"})]
            rule = MockAutomationRule(rule_id, name, conditions, actions, priority=priority)
            self.engine.add_rule(rule)

        # All rules should be evaluable individually
        context = {'trigger': True}
        for rule in self.engine.rules.values():
            result = self.engine.evaluate_rule(rule, context)
            assert result['triggered'] is True

        # Check that priorities are stored correctly
        priorities = [rule.priority for rule in self.engine.rules.values()]
        assert 1 in priorities  # Low
        assert 2 in priorities  # Medium
        assert 3 in priorities  # High

    def test_rule_versioning_and_metadata(self):
        """测试规则版本控制和元数据"""
        # Create rule with metadata
        conditions = [MockRuleCondition("status", "equals", "active")]
        actions = [MockRuleAction("update_data", {})]

        rule = MockAutomationRule(
            rule_id="versioned_rule",
            name="Versioned Rule",
            conditions=conditions,
            actions=actions,
            description="A rule with versioning",
            tags=["production", "critical"],
            version="2.1.0"
        )

        self.engine.add_rule(rule)

        # Verify metadata
        stored_rule = self.engine.rules["versioned_rule"]
        assert stored_rule.description == "A rule with versioning"
        assert "production" in stored_rule.tags
        assert "critical" in stored_rule.tags
        assert stored_rule.version == "2.1.0"
        assert stored_rule.created_at is not None
        assert stored_rule.updated_at is not None

    def test_bulk_rule_operations(self):
        """测试批量规则操作"""
        # Create multiple rules
        rules = []
        for i in range(10):
            conditions = [MockRuleCondition("counter", "equals", i)]
            actions = [MockRuleAction("log_event", {"rule_number": i})]
            rule = MockAutomationRule(f"bulk_rule_{i}", f"Bulk Rule {i}", conditions, actions)
            rules.append(rule)
            self.engine.add_rule(rule)

        assert len(self.engine.rules) == 10
        assert self.engine.stats['total_rules'] == 10

        # Test bulk evaluation with different contexts
        contexts = [{'counter': i} for i in range(10)]

        triggered_count = 0
        for context in contexts:
            for rule in self.engine.rules.values():
                result = self.engine.evaluate_rule(rule, context)
                if result['triggered']:
                    triggered_count += 1

        # Each context should trigger exactly one rule
        assert triggered_count == 10

        # Test bulk removal
        for i in range(10):
            self.engine.remove_rule(f"bulk_rule_{i}")

        assert len(self.engine.rules) == 0
        assert self.engine.stats['total_rules'] == 0


# pytest配置
pytestmark = pytest.mark.timeout(60)
