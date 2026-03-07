"""
Automation Rule Module
自动化规则模块

This module provides automation rule functionality
此模块提供自动化规则功能

Extracted from automation_engine.py to improve code organization
从automation_engine.py中提取以改善代码组织

Author: RQA2025 Development Team
Date: 2025-11-01
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class AutomationRule:

    """
    Automation Rule Class
    自动化规则类

    Represents a single automation rule with conditions and actions
    表示具有条件和动作的单个自动化规则
    """

    def __init__(self,


                 rule_id: str,
                 name: str,
                 conditions: List[Dict[str, Any]],
                 actions: List[Dict[str, Any]],
                 priority: int = 1,
                 enabled: bool = True):
        """
        Initialize automation rule
        初始化自动化规则

        Args:
            rule_id: Unique rule identifier
                    唯一规则标识符
            name: Human - readable rule name
                 人类可读的规则名称
            conditions: List of condition dictionaries
                       条件字典列表
            actions: List of action dictionaries
                    动作字典列表
            priority: Rule priority (higher = executed first)
                     规则优先级（越高=越先执行）
            enabled: Whether the rule is enabled
                    规则是否启用
        """
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.enabled = enabled

        # Runtime state
        self.last_executed: Optional[datetime] = None
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

        # Performance metrics
        self.average_execution_time = 0.0

    def evaluate_conditions(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate rule conditions
        评估规则条件

        Args:
            context: Execution context with relevant data
                    包含相关数据的执行上下文

        Returns:
            bool: True if all conditions are met, False otherwise
                  如果满足所有条件则返回True，否则返回False
        """
        if not self.enabled:
            return False

        try:
            for condition in self.conditions:
                condition_type = condition.get('type', '')
                field = condition.get('field', '')
                operator = condition.get('operator', 'eq')
                value = condition.get('value')

                # Get field value from context
                field_value = self._get_nested_value(context, field.split('.'))

                # Evaluate condition
                if not self._evaluate_condition(field_value, operator, value):
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to evaluate conditions for rule {self.rule_id}: {str(e)}")
            return False

    def execute_actions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rule actions
        执行规则动作

        Args:
            context: Execution context
                    执行上下文

        Returns:
            dict: Action execution results
                  动作执行结果
        """
        self.last_executed = datetime.now()
        self.execution_count += 1

        results = {
            'rule_id': self.rule_id,
            'rule_name': self.name,
            'executed_at': self.last_executed,
            'actions_executed': 0,
            'actions_successful': 0,
            'actions_failed': 0,
            'action_results': []
        }

        start_time = time.time()

        try:
            for action in self.actions:
                action_result = self._execute_action(action, context)
                results['actions_executed'] += 1
                results['action_results'].append(action_result)

                if action_result.get('success', False):
                    results['actions_successful'] += 1
                else:
                    results['actions_failed'] += 1

            results['success'] = results['actions_failed'] == 0
            self.success_count += 1

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self.failure_count += 1
            logger.error(f"Failed to execute actions for rule {self.rule_id}: {str(e)}")

        execution_time = time.time() - start_time
        results['execution_time'] = execution_time

        # Update average execution time
        total_executions = self.success_count + self.failure_count
        if total_executions > 0:
            self.average_execution_time = (
                (self.average_execution_time * (total_executions - 1)) + execution_time
            ) / total_executions

        return results

    def _get_nested_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """
        Get nested value from dictionary
        从字典中获取嵌套值

        Args:
            data: Dictionary to search
                 要搜索的字典
            keys: List of keys for nested access
                 嵌套访问的键列表

        Returns:
            Nested value or None if not found
            嵌套值，如果未找到则返回None
        """
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _evaluate_condition(self, field_value: Any, operator: str, expected_value: Any) -> bool:
        """
        Evaluate a single condition
        评估单个条件

        Args:
            field_value: Actual field value
                        实际字段值
            operator: Comparison operator
                     比较运算符
            expected_value: Expected value
                           期望值

        Returns:
            bool: True if condition is met, False otherwise
                  如果满足条件则返回True，否则返回False
        """
        if field_value is None:
            return False

        try:
            if operator == 'eq':
                return field_value == expected_value
            elif operator == 'ne':
                return field_value != expected_value
            elif operator == 'gt':
                return field_value > expected_value
            elif operator == 'ge':
                return field_value >= expected_value
            elif operator == 'lt':
                return field_value < expected_value
            elif operator == 'le':
                return field_value <= expected_value
            elif operator == 'in':
                return field_value in expected_value
            elif operator == 'contains':
                return expected_value in field_value
            elif operator == 'regex':
                import re
                return bool(re.match(expected_value, str(field_value)))
            else:
                logger.warning(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"Condition evaluation error: {str(e)}")
            return False

    def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a single action
        执行单个动作

        Args:
            action: Action definition
                   动作定义
            context: Execution context
                    执行上下文

        Returns:
            dict: Action execution result
                  动作执行结果
        """
        action_type = action.get('type', '')
        result = {
            'action_type': action_type,
            'executed_at': datetime.now(),
            'success': False
        }

        try:
            if action_type == 'execute_function':
                func_name = action.get('function')
                args = action.get('args', [])
                kwargs = action.get('kwargs', {})

                # This would typically involve calling a registered function
                result['function_called'] = func_name
                result['args'] = args
                result['kwargs'] = kwargs
                result['success'] = True

            elif action_type == 'send_notification':
                message = action.get('message', '')
                level = action.get('level', 'info')

                logger.log(getattr(logging, level.upper(), logging.INFO), message)
                result['message'] = message
                result['level'] = level
                result['success'] = True

            elif action_type == 'update_configuration':
                config_path = action.get('config_path', '')
                updates = action.get('updates', {})

                # This would typically update configuration
                result['config_path'] = config_path
                result['updates'] = updates
                result['success'] = True

            else:
                result['error'] = f"Unknown action type: {action_type}"

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Action execution failed: {str(e)}")

        return result

    def get_rule_stats(self) -> Dict[str, Any]:
        """
        Get rule execution statistics
        获取规则执行统计信息

        Returns:
            dict: Rule statistics
                  规则统计信息
        """
        total_executions = self.success_count + self.failure_count
        return {
            'rule_id': self.rule_id,
            'rule_name': self.name,
            'enabled': self.enabled,
            'priority': self.priority,
            'total_executions': total_executions,
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'success_rate': self.success_count / max(total_executions, 1) * 100,
            'average_execution_time': self.average_execution_time,
            'last_executed': self.last_executed.isoformat() if self.last_executed else None
        }

