"""
Rule Engine Module
规则引擎模块

This module provides rule engine capabilities for automation decision making
此模块为自动化决策提供规则引擎能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
import time
from typing import Any, Dict, List, Optional, Callable, Union
from datetime import datetime
from abc import ABC, abstractmethod
import re

logger = logging.getLogger(__name__)


class RuleCondition(ABC):

    """
    Rule Condition Base Class
    规则条件基类

    Abstract base class for rule conditions
    规则条件的抽象基类
    """

    def __init__(self, condition_id: str, description: str = ""):
        """
        Initialize rule condition
        初始化规则条件

        Args:
            condition_id: Unique condition identifier
                         唯一条件标识符
            description: Condition description
                        条件描述
        """
        self.condition_id = condition_id
        self.description = description

    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate the condition
        评估条件

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            bool: True if condition is met, False otherwise
                  如果满足条件则返回True，否则返回False
        """


class ThresholdCondition(RuleCondition):

    """
    Threshold Condition Class
    阈值条件类

    Condition that checks if a value exceeds a threshold
    检查值是否超过阈值的条件
    """

    def __init__(self,


                 condition_id: str,
                 field: str,
                 operator: str,
                 threshold: Union[int, float],
                 description: str = ""):
        """
        Initialize threshold condition
        初始化阈值条件

        Args:
            condition_id: Condition identifier
                         条件标识符
            field: Field to check in context
                  要在上下文中检查的字段
            operator: Comparison operator ('>', '<', '>=', '<=', '==', '!=')
                     比较运算符 ('>', '<', '>=', '<=', '==', '!=')
            threshold: Threshold value
                      阈值
            description: Condition description
                        条件描述
        """
        super().__init__(condition_id, description)
        self.field = field
        self.operator = operator
        self.threshold = threshold

        # Map string operators to functions
        self.operator_map = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '==': operator.eq,
            '!=': operator.ne
        }

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate threshold condition
        评估阈值条件

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            bool: True if condition is met
                  如果满足条件则返回True
        """
        try:
            # Get field value
            field_value = self._get_nested_value(context, self.field.split('.'))

            if field_value is None:
                return False

            # Apply operator
            op_func = self.operator_map.get(self.operator)
            if op_func is None:
                logger.error(f"Unknown operator: {self.operator}")
                return False

            return op_func(field_value, self.threshold)

        except Exception as e:
            logger.error(f"Threshold condition evaluation failed: {str(e)}")
            return False

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


class PatternCondition(RuleCondition):

    """
    Pattern Condition Class
    模式条件类

    Condition that checks if a field matches a pattern
    检查字段是否匹配模式的条件
    """

    def __init__(self,


                 condition_id: str,
                 field: str,
                 pattern: str,
                 pattern_type: str = 'regex',
                 description: str = ""):
        """
        Initialize pattern condition
        初始化模式条件

        Args:
            condition_id: Condition identifier
                         条件标识符
            field: Field to check
                  要检查的字段
            pattern: Pattern to match
                    要匹配的模式
            pattern_type: Type of pattern ('regex', 'contains', 'startswith', 'endswith')
                         模式类型 ('regex', 'contains', 'startswith', 'endswith')
            description: Condition description
                        条件描述
        """
        super().__init__(condition_id, description)
        self.field = field
        self.pattern = pattern
        self.pattern_type = pattern_type

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate pattern condition
        评估模式条件

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            bool: True if condition is met
                  如果满足条件则返回True
        """
        try:
            # Get field value
            field_value = self._get_nested_value(context, self.field.split('.'))

            if field_value is None:
                return False

            field_str = str(field_value)

            # Apply pattern matching
            if self.pattern_type == 'regex':
                return bool(re.match(self.pattern, field_str))
            elif self.pattern_type == 'contains':
                return self.pattern in field_str
            elif self.pattern_type == 'startswith':
                return field_str.startswith(self.pattern)
            elif self.pattern_type == 'endswith':
                return field_str.endswith(self.pattern)
            else:
                logger.error(f"Unknown pattern type: {self.pattern_type}")
                return False

        except Exception as e:
            logger.error(f"Pattern condition evaluation failed: {str(e)}")
            return False

    def _get_nested_value(self, data: Dict[str, Any], keys: List[str]) -> Any:
        """Get nested value from dictionary"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current


class CompositeCondition(RuleCondition):

    """
    Composite Condition Class
    复合条件类

    Condition that combines multiple conditions with logical operators
    使用逻辑运算符组合多个条件的条件
    """

    def __init__(self,


                 condition_id: str,
                 conditions: List[RuleCondition],
                 operator: str = 'AND',
                 description: str = ""):
        """
        Initialize composite condition
        初始化复合条件

        Args:
            condition_id: Condition identifier
                         条件标识符
            conditions: List of conditions to combine
                       要组合的条件列表
            operator: Logical operator ('AND', 'OR', 'NOT')
                     逻辑运算符 ('AND', 'OR', 'NOT')
            description: Condition description
                        条件描述
        """
        super().__init__(condition_id, description)
        self.conditions = conditions
        self.operator = operator.upper()

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate composite condition
        评估复合条件

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            bool: True if condition is met
                  如果满足条件则返回True
        """
        try:
            if self.operator == 'AND':
                return all(condition.evaluate(context) for condition in self.conditions)
            elif self.operator == 'OR':
                return any(condition.evaluate(context) for condition in self.conditions)
            elif self.operator == 'NOT':
                if len(self.conditions) == 1:
                    return not self.conditions[0].evaluate(context)
                else:
                    return not all(condition.evaluate(context) for condition in self.conditions)
            else:
                logger.error(f"Unknown logical operator: {self.operator}")
                return False

        except Exception as e:
            logger.error(f"Composite condition evaluation failed: {str(e)}")
            return False


class RuleAction(ABC):

    """
    Rule Action Base Class
    规则动作基类

    Abstract base class for rule actions
    规则动作的抽象基类
    """

    def __init__(self, action_id: str, description: str = ""):
        """
        Initialize rule action
        初始化规则动作

        Args:
            action_id: Unique action identifier
                      唯一动作标识符
            description: Action description
                        动作描述
        """
        self.action_id = action_id
        self.description = description

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the action
        执行动作

        Args:
            context: Execution context
                    执行上下文

        Returns:
            dict: Action execution result
                  动作执行结果
        """


class FunctionAction(RuleAction):

    """
    Function Action Class
    函数动作类

    Action that executes a function
    执行函数的动作
    """

    def __init__(self,


                 action_id: str,
                 function: Callable,
                 args: Optional[List[Any]] = None,
                 kwargs: Optional[Dict[str, Any]] = None,
                 description: str = ""):
        """
        Initialize function action
        初始化函数动作

        Args:
            action_id: Action identifier
                      动作标识符
            function: Function to execute
                     要执行的函数
            args: Positional arguments for function
                 函数的位置参数
            kwargs: Keyword arguments for function
                   函数的关键字参数
            description: Action description
                        动作描述
        """
        super().__init__(action_id, description)
        self.function = function
        self.args = args or []
        self.kwargs = kwargs or {}

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute function action
        执行函数动作

        Args:
            context: Execution context
                    执行上下文

        Returns:
            dict: Action execution result
                  动作执行结果
        """
        try:
            # Execute function
            result = self.function(*self.args, **self.kwargs)

            return {
                'success': True,
                'action_id': self.action_id,
                'result': result,
                'executed_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"Function action execution failed: {str(e)}")
            return {
                'success': False,
                'action_id': self.action_id,
                'error': str(e),
                'executed_at': datetime.now()
            }


class NotificationAction(RuleAction):

    """
    Notification Action Class
    通知动作类

    Action that sends notifications
    发送通知的动作
    """

    def __init__(self,


                 action_id: str,
                 message: str,
                 notification_type: str = 'log',
                 level: str = 'info',
                 description: str = ""):
        """
        Initialize notification action
        初始化通知动作

        Args:
            action_id: Action identifier
                      动作标识符
            message: Notification message
                    通知消息
            notification_type: Type of notification ('log', 'email', 'webhook')
                             通知类型 ('log', 'email', 'webhook')
            level: Notification level ('info', 'warning', 'error')
                  通知级别 ('info', 'warning', 'error')
            description: Action description
                        动作描述
        """
        super().__init__(action_id, description)
        self.message = message
        self.notification_type = notification_type
        self.level = level

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute notification action
        执行通知动作

        Args:
            context: Execution context
                    执行上下文

        Returns:
            dict: Action execution result
                  动作执行结果
        """
        try:
            if self.notification_type == 'log':
                # Log notification
                log_level = getattr(logging, self.level.upper(), logging.INFO)
                logger.log(log_level, self.message)

            elif self.notification_type == 'email':
                # Email notification (placeholder)
                logger.info(f"Email notification: {self.message}")

            elif self.notification_type == 'webhook':
                # Webhook notification (placeholder)
                logger.info(f"Webhook notification: {self.message}")

            return {
                'success': True,
                'action_id': self.action_id,
                'notification_type': self.notification_type,
                'message': self.message,
                'executed_at': datetime.now()
            }

        except Exception as e:
            logger.error(f"Notification action execution failed: {str(e)}")
            return {
                'success': False,
                'action_id': self.action_id,
                'error': str(e),
                'executed_at': datetime.now()
            }


class BusinessRule:

    """
    Business Rule Class
    业务规则类

    Represents a complete business rule with conditions and actions
    表示具有条件和动作的完整业务规则
    """

    def __init__(self,


                 rule_id: str,
                 name: str,
                 conditions: List[RuleCondition],
                 actions: List[RuleAction],
                 priority: int = 1,
                 enabled: bool = True,
                 description: str = ""):
        """
        Initialize business rule
        初始化业务规则

        Args:
            rule_id: Unique rule identifier
                    唯一规则标识符
            name: Human - readable rule name
                 人类可读的规则名称
            conditions: List of conditions
                       条件列表
            actions: List of actions
                    动作列表
            priority: Rule priority (higher = executed first)
                     规则优先级（越高=越先执行）
            enabled: Whether the rule is enabled
                    规则是否启用
            description: Rule description
                        规则描述
        """
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.enabled = enabled
        self.description = description

        # Runtime statistics
        self.execution_count = 0
        self.success_count = 0
        self.last_executed: Optional[datetime] = None

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate rule conditions
        评估规则条件

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            bool: True if all conditions are met
                  如果满足所有条件则返回True
        """
        if not self.enabled:
            return False

        # Evaluate all conditions (AND logic)
        return all(condition.evaluate(context) for condition in self.conditions)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute rule actions
        执行规则动作

        Args:
            context: Execution context
                    执行上下文

        Returns:
            dict: Rule execution result
                  规则执行结果
        """
        self.execution_count += 1
        self.last_executed = datetime.now()

        results = {
            'rule_id': self.rule_id,
            'rule_name': self.name,
            'executed_at': self.last_executed,
            'conditions_met': True,
            'actions_executed': 0,
            'actions_successful': 0,
            'action_results': []
        }

        try:
            # Execute all actions
            for action in self.actions:
                action_result = action.execute(context)
                results['action_results'].append(action_result)

                results['actions_executed'] += 1
                if action_result.get('success', False):
                    results['actions_successful'] += 1

            results['success'] = results['actions_successful'] == results['actions_executed']
            if results['success']:
                self.success_count += 1

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            logger.error(f"Rule execution failed: {str(e)}")

        return results


class RuleEngine:

    """
    Rule Engine Class
    规则引擎类

    Core engine for evaluating and executing business rules
    用于评估和执行业务规则的核心引擎
    """

    def __init__(self, engine_name: str = "default_rule_engine"):
        """
        Initialize rule engine
        初始化规则引擎

        Args:
            engine_name: Name of the rule engine
                        规则引擎的名称
        """
        self.engine_name = engine_name
        self.rules: Dict[str, BusinessRule] = {}

        # Engine statistics
        self.stats = {
            'total_evaluations': 0,
            'rules_triggered': 0,
            'rules_executed': 0,
            'average_evaluation_time': 0.0
        }

        logger.info(f"Rule engine {engine_name} initialized")

    def add_rule(self, rule: BusinessRule) -> None:
        """
        Add a business rule to the engine
        将业务规则添加到引擎中

        Args:
            rule: Business rule to add
                 要添加的业务规则
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove a business rule from the engine
        从引擎中移除业务规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed rule: {rule_id}")
            return True
        return False

    def evaluate_rules(self, context: Dict[str, Any]) -> List[str]:
        """
        Evaluate all enabled rules against the given context
        根据给定上下文评估所有启用的规则

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            list: List of triggered rule IDs
                  触发的规则ID列表
        """
        triggered_rules = []
        start_time = time.time()

        try:
            # Sort rules by priority (highest first)
            sorted_rules = sorted(
                [rule for rule in self.rules.values() if rule.enabled],
                key=lambda r: r.priority,
                reverse=True
            )

            for rule in sorted_rules:
                try:
                    if rule.evaluate(context):
                        triggered_rules.append(rule.rule_id)
                        logger.debug(f"Rule triggered: {rule.name}")

                except Exception as e:
                    logger.error(f"Rule evaluation failed for {rule.rule_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Rule evaluation process failed: {str(e)}")

        # Update statistics
        evaluation_time = time.time() - start_time
        self.stats['total_evaluations'] += 1
        self.stats['rules_triggered'] += len(triggered_rules)

        # Update average evaluation time
        total_evals = self.stats['total_evaluations']
        current_avg = self.stats['average_evaluation_time']
        self.stats['average_evaluation_time'] = (
            (current_avg * (total_evals - 1)) + evaluation_time
        ) / total_evals

        return triggered_rules

    def execute_rules(self, rule_ids: List[str], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute specified rules
        执行指定的规则

        Args:
            rule_ids: List of rule IDs to execute
                     要执行的规则ID列表
            context: Execution context
                    执行上下文

        Returns:
            dict: Execution results
                  执行结果
        """
        results = {
            'executed_rules': len(rule_ids),
            'successful_executions': 0,
            'failed_executions': 0,
            'rule_results': []
        }

        for rule_id in rule_ids:
            if rule_id in self.rules:
                rule_result = self.rules[rule_id].execute(context)
                results['rule_results'].append(rule_result)

                if rule_result.get('success', False):
                    results['successful_executions'] += 1
                else:
                    results['failed_executions'] += 1

                self.stats['rules_executed'] += 1
            else:
                logger.warning(f"Rule {rule_id} not found")

        results['success'] = results['failed_executions'] == 0
        return results

    def process_event(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an event by evaluating rules and executing triggered rules
        通过评估规则和执行触发的规则来处理事件

        Args:
            context: Event context
                    事件上下文

        Returns:
            dict: Processing results
                  处理结果
        """
        # Evaluate rules
        triggered_rules = self.evaluate_rules(context)

        if not triggered_rules:
            return {
                'success': True,
                'rules_triggered': 0,
                'message': 'No rules triggered'
            }

        # Execute triggered rules
        execution_results = self.execute_rules(triggered_rules, context)

        return {
            'success': execution_results['success'],
            'rules_triggered': len(triggered_rules),
            'rules_executed': execution_results['executed_rules'],
            'successful_executions': execution_results['successful_executions'],
            'failed_executions': execution_results['failed_executions'],
            'triggered_rule_ids': triggered_rules,
            'execution_results': execution_results['rule_results']
        }

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get rule engine statistics
        获取规则引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        return {
            'engine_name': self.engine_name,
            'total_rules': len(self.rules),
            'enabled_rules': sum(1 for rule in self.rules.values() if rule.enabled),
            'disabled_rules': sum(1 for rule in self.rules.values() if not rule.enabled),
            'evaluation_stats': self.stats,
            'rules_summary': {
                rule_id: {
                    'name': rule.name,
                    'enabled': rule.enabled,
                    'priority': rule.priority,
                    'execution_count': rule.execution_count,
                    'success_rate': rule.success_count / max(rule.execution_count, 1) * 100
                }
                for rule_id, rule in self.rules.items()
            }
        }

    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable a rule
        启用规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if enabled successfully
                  启用成功返回True
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable a rule
        禁用规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if disabled successfully
                  禁用成功返回True
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            return True
        return False


# Global rule engine instance
# 全局规则引擎实例
rule_engine = RuleEngine()

__all__ = [
    'RuleCondition',
    'ThresholdCondition',
    'PatternCondition',
    'CompositeCondition',
    'RuleAction',
    'FunctionAction',
    'NotificationAction',
    'BusinessRule',
    'RuleEngine',
    'rule_engine'
]
