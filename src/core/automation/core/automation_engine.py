"""
Automation Engine Module
自动化引擎模块

This module provides the core automation engine for quantitative trading systems
此模块为量化交易系统提供核心自动化引擎

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import threading
import time
from collections import defaultdict, deque
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor

# 先定义所有需要的常量（确保可用）
MAX_CONCURRENT_TASKS = 10
DEFAULT_TASK_TIMEOUT = 30.0
DEPLOYMENT_TIMEOUT = 60.0
HEALTH_CHECK_INTERVAL = 60.0
NOTIFICATION_RETRY_LIMIT = 3
ALERT_COOLDOWN_SECONDS = 300
RULE_CACHE_SIZE = 1000
RULE_PROCESSING_TIMEOUT = 10.0
MAX_RULES_PER_WORKFLOW = 100
DEFAULT_RETRY_DELAY = 1.0
MAX_RETRIES = 3
DEFAULT_QUEUE_SIZE = 1000
MONITOR_UPDATE_INTERVAL = 60  # 默认监控更新间隔（秒）
DEFAULT_MAX_WORKERS = 4  # 默认最大工作线程数
CIRCUIT_BREAKER_TIMEOUT = 30.0  # 断路器超时时间（秒）

# 尝试导入constants覆盖（如果存在）
try:
    from src.constants import *
except ImportError:
    try:
        from src.core.constants import *
    except ImportError:
        pass  # 使用上面定义的默认值

# 导入或定义异常
try:
    from src.exceptions import *
except ImportError:
    try:
        from src.core.exceptions import *
    except ImportError:
        # 定义基础异常
        class AutomationException(Exception):
            pass

logger = logging.getLogger(__name__)


class TaskConcurrencyController:

    """
    任务并发控制器 - 防止资源竞争和死锁

    Task concurrency controller to prevent resource competition and deadlocks
    """

    def __init__(self, max_concurrent_tasks: int = MAX_CONCURRENT_TASKS, deadlock_timeout: float = DEFAULT_TASK_TIMEOUT):
        """
        初始化任务并发控制器

        Initialize task concurrency controller

        Args:
            max_concurrent_tasks: 最大并发任务数
                                Maximum number of concurrent tasks
            deadlock_timeout: 死锁检测超时时间（秒）
                           Deadlock detection timeout (seconds)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.deadlock_timeout = deadlock_timeout

        # 任务状态管理
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: deque = deque(maxlen=DEFAULT_QUEUE_SIZE)

        # 同步原语
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)

        # 资源依赖管理（用于死锁检测）
        self.resource_dependencies: Dict[str, set] = defaultdict(set)
        self.task_resources: Dict[str, set] = defaultdict(set)

        # 性能统计
        self.stats = {
            'total_tasks_processed': 0,
            'deadlock_detected': 0,
            'queue_full_rejections': 0,
            'timeout_rejections': 0,
            'average_wait_time': 0.0,
            'max_wait_time': 0.0
        }

        self.logger = logging.getLogger(self.__class__.__name__)

    def acquire_task_slot(self, task_id: str, required_resources: Optional[set] = None) -> bool:
        """
        获取任务执行槽位

        Acquire task execution slot

        Args:
            task_id: 任务ID
                    Task ID
            required_resources: 需要的资源集合
                              Required resources set

        Returns:
            bool: 是否成功获取
                 Whether acquisition was successful
        """
        start_time = time.time()

        with self.condition:
            # 检查是否已存在相同任务
            if task_id in self.active_tasks:
                self.logger.warning(f"Task {task_id} is already running")
                return False

            # 检查死锁风险
            if required_resources and self._would_cause_deadlock(task_id, required_resources):
                self.stats['deadlock_detected'] += 1
                self.logger.warning(f"Deadlock risk detected for task {task_id}")
                return False

            # 等待可用槽位
            while len(self.active_tasks) >= self.max_concurrent_tasks:
                if not self.condition.wait(timeout=self.deadlock_timeout):
                    self.stats['timeout_rejections'] += 1
                    self.logger.warning(f"Timeout waiting for task slot: {task_id}")
                    return False

                # 再次检查死锁风险（因为等待期间状态可能改变）
                if required_resources and self._would_cause_deadlock(task_id, required_resources):
                    self.stats['deadlock_detected'] += 1
                    self.logger.warning(f"Deadlock risk detected after wait for task {task_id}")
                    return False

            # 获取槽位
            self.active_tasks[task_id] = {
                'start_time': datetime.now(),
                'required_resources': required_resources or set(),
                'wait_time': time.time() - start_time
            }

            # 更新资源依赖
            if required_resources:
                self.task_resources[task_id] = required_resources
                for resource in required_resources:
                    self.resource_dependencies[resource].add(task_id)

            # 更新统计信息
            wait_time = time.time() - start_time
            self.stats['total_tasks_processed'] += 1
            self.stats['average_wait_time'] = (
                (self.stats['average_wait_time'] *
                 (self.stats['total_tasks_processed'] - 1)) + wait_time
            ) / self.stats['total_tasks_processed']
            self.stats['max_wait_time'] = max(self.stats['max_wait_time'], wait_time)

            self.logger.info(f"Task slot acquired: {task_id}")
            return True

    def release_task_slot(self, task_id: str):
        """
        释放任务执行槽位

        Release task execution slot

        Args:
            task_id: 任务ID
                    Task ID
        """
        with self.condition:
            if task_id in self.active_tasks:
                task_info = self.active_tasks[task_id]
                execution_time = (datetime.now() - task_info['start_time']).total_seconds()

                # 记录完成的任务
                self.completed_tasks.append({
                    'task_id': task_id,
                    'execution_time': execution_time,
                    'wait_time': task_info['wait_time'],
                    'completed_at': datetime.now()
                })

                # 清理资源依赖
                required_resources = task_info.get('required_resources', set())
                for resource in required_resources:
                    self.resource_dependencies[resource].discard(task_id)
                    if not self.resource_dependencies[resource]:
                        del self.resource_dependencies[resource]

                if task_id in self.task_resources:
                    del self.task_resources[task_id]

                # 释放槽位
                del self.active_tasks[task_id]

                # 通知等待的线程
                self.condition.notify()

                self.logger.info(
                    f"Task slot released: {task_id} (execution: {execution_time:.2f}s)")
            else:
                self.logger.warning(f"Attempt to release non - existent task: {task_id}")

    def _would_cause_deadlock(self, task_id: str, required_resources: set) -> bool:
        """
        检查是否会造成死锁

        Check if operation would cause deadlock

        Args:
            task_id: 任务ID
                    Task ID
            required_resources: 需要的资源
                              Required resources

        Returns:
            bool: 是否会造成死锁
                 Whether it would cause deadlock
        """
        # 简单的死锁检测：检查是否有循环等待
        # 这里实现一个简化的死锁检测算法

        # 获取所有当前活跃任务
        active_task_ids = set(self.active_tasks.keys())

        # 检查是否存在资源冲突
        for resource in required_resources:
            holders = self.resource_dependencies.get(resource, set())
            # 如果某个资源被其他任务持有，检查是否存在循环依赖
            for holder in holders:
                if holder in active_task_ids:
                    # 检查holder任务是否也在等待当前任务持有的资源
                    holder_resources = self.task_resources.get(holder, set())
                    if holder_resources.intersection(self._get_task_held_resources(task_id)):
                        return True

        return False

    def _get_task_held_resources(self, task_id: str) -> set:
        """
        获取任务持有的资源

        Get resources held by task

        Args:
            task_id: 任务ID

        Returns:
            set: 持有的资源集合
        """
        if task_id not in self.active_tasks:
            return set()

        return self.active_tasks[task_id].get('required_resources', set())

    async def execute_with_control(self, task_id: str, task_func: Callable, *args,
                                   required_resources: Optional[set] = None, **kwargs):
        """
        受控执行任务（异步版本）

        Controlled task execution (async version)

        Args:
            task_id: 任务ID
                    Task ID
            task_func: 任务函数
                      Task function
            required_resources: 需要的资源
                              Required resources
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务执行结果
            Task execution result
        """
        if not self.acquire_task_slot(task_id, required_resources):
            raise RuntimeError(f"Failed to acquire task slot for {task_id}")

        try:
            # 执行任务
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = await loop.run_in_executor(executor, task_func, *args, **kwargs)

            return result

        finally:
            self.release_task_slot(task_id)

    def execute_with_control_sync(self, task_id: str, task_func: Callable, *args,


                                  required_resources: Optional[set] = None, **kwargs):
        """
        受控执行任务（同步版本）

        Controlled task execution (sync version)

        Args:
            task_id: 任务ID
                    Task ID
            task_func: 任务函数
                      Task function
            required_resources: 需要的资源
                              Required resources
            *args: 位置参数
            **kwargs: 关键字参数

        Returns:
            任务执行结果
            Task execution result
        """
        if not self.acquire_task_slot(task_id, required_resources):
            raise RuntimeError(f"Failed to acquire task slot for {task_id}")

        try:
            result = task_func(*args, **kwargs)
            return result
        finally:
            self.release_task_slot(task_id)

    def get_controller_stats(self) -> Dict[str, Any]:
        """
        获取控制器统计信息

        Get controller statistics

        Returns:
            dict: 统计信息
                 Statistics
        """
        with self.lock:
            current_stats = self.stats.copy()
            current_stats.update({
                'active_tasks_count': len(self.active_tasks),
                'queued_tasks_count': len(self.task_queue),
                'resource_conflicts': len(self.resource_dependencies),
                'active_task_details': {
                    task_id: {
                        'start_time': info['start_time'].isoformat(),
                        'wait_time': info['wait_time'],
                        'resources': list(info.get('required_resources', []))
                    }
                    for task_id, info in self.active_tasks.items()
                }
            })

        return current_stats

    def force_release_stuck_tasks(self, max_age_seconds: float = DEPLOYMENT_TIMEOUT):
        """
        强制释放卡住的任务

        Force release stuck tasks

        Args:
            max_age_seconds: 最大任务年龄（秒）
                           Maximum task age (seconds)
        """
        with self.condition:
            current_time = datetime.now()
            stuck_tasks = []

            for task_id, task_info in self.active_tasks.items():
                age = (current_time - task_info['start_time']).total_seconds()
                if age > max_age_seconds:
                    stuck_tasks.append(task_id)

            for task_id in stuck_tasks:
                self.logger.warning(f"Forcing release of stuck task: {task_id}")
                self.release_task_slot(task_id)

            if stuck_tasks:
                self.condition.notify_all()

            return len(stuck_tasks)

    def set_max_concurrent_tasks(self, max_tasks: int):
        """
        设置最大并发任务数

        Set maximum concurrent tasks

        Args:
            max_tasks: 最大并发任务数
                      Maximum concurrent tasks
        """
        with self.condition:
            old_max = self.max_concurrent_tasks
            self.max_concurrent_tasks = max_tasks
            self.logger.info(f"Max concurrent tasks changed: {old_max} -> {max_tasks}")

            # 如果增加并发数，通知等待的线程
            if max_tasks > old_max:
                self.condition.notify_all()


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


class AutomationEngine:

    """
    Automation Engine Class
    自动化引擎类

    Core engine for managing and executing automation rules
    用于管理和执行自动化规则的核心引擎
    """

    def __init__(self, engine_name: str = "default_automation_engine"):
        """
        Initialize automation engine
        初始化自动化引擎

        Args:
            engine_name: Name of the automation engine
                        自动化引擎的名称
        """
        self.engine_name = engine_name
        self.rules: Dict[str, AutomationRule] = {}
        self.rule_execution_history = deque(maxlen=RULE_CACHE_SIZE)

        # Engine state
        self.is_running = False
        self.execution_thread: Optional[threading.Thread] = None

        # Configuration
        self.check_interval = MONITOR_UPDATE_INTERVAL  # seconds

        # 初始化任务并发控制器
        self.task_controller = TaskConcurrencyController(
            max_concurrent_tasks=DEFAULT_MAX_WORKERS,  # 默认最大并发任务数
            deadlock_timeout=CIRCUIT_BREAKER_TIMEOUT   # 死锁检测超时
        )

        logger.info(f"Automation engine {engine_name} initialized with concurrency control")

    def add_rule(self, rule: AutomationRule) -> None:
        """
        Add an automation rule
        添加自动化规则

        Args:
            rule: Automation rule to add
                 要添加的自动化规则
        """
        self.rules[rule.rule_id] = rule
        logger.info(f"Added automation rule: {rule.name} ({rule.rule_id})")

    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove an automation rule
        移除自动化规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if removed successfully, False otherwise
                  移除成功返回True，否则返回False
        """
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed automation rule: {rule_id}")
            return True
        return False

    def enable_rule(self, rule_id: str) -> bool:
        """
        Enable an automation rule
        启用自动化规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if enabled successfully, False otherwise
                  启用成功返回True，否则返回False
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            logger.info(f"Enabled automation rule: {rule_id}")
            return True
        return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable an automation rule
        禁用自动化规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if disabled successfully, False otherwise
                  禁用成功返回True，否则返回False
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled automation rule: {rule_id}")
            return True
        return False

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all enabled rules against the given context
        根据给定上下文评估所有启用的规则

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            list: List of triggered rules with their actions
                  触发的规则及其动作列表
        """
        triggered_rules = []

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [rule for rule in self.rules.values() if rule.enabled],
            key=lambda r: r.priority,
            reverse=True
        )

        for rule in sorted_rules:
            try:
                if rule.evaluate_conditions(context):
                    triggered_rules.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'priority': rule.priority,
                        'actions': rule.actions
                    })

                    logger.info(f"Rule triggered: {rule.name} ({rule.rule_id})")

            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.rule_id}: {str(e)}")

        return triggered_rules

    def execute_rule_actions(self, rule_id: str, context: Dict[str, Any],


                             required_resources: Optional[set] = None) -> Dict[str, Any]:
        """
        Execute actions for a specific rule with concurrency control
        使用并发控制为特定规则执行动作

        Args:
            rule_id: Rule identifier
                    规则标识符
            context: Execution context
                    执行上下文
            required_resources: Required resources for execution
                              执行所需的资源

        Returns:
            dict: Action execution results
                  动作执行结果
        """
        if rule_id not in self.rules:
            return {'error': f'Rule {rule_id} not found'}

        rule = self.rules[rule_id]

        # 使用任务并发控制器执行规则动作
        try:
            # 同步执行版本
            result = self.task_controller.execute_with_control_sync(
                task_id=f"rule_{rule_id}_{datetime.now().strftime('%Y % m % d % H % M % S % f')}",
                task_func=self._execute_rule_actions_internal,
                rule=rule,
                context=context,
                required_resources=required_resources
            )

            return result

        except RuntimeError as e:
            error_msg = str(e)
            logger.warning(f"Failed to execute rule {rule_id}: {error_msg}")
            return {'error': error_msg, 'success': False}

    def _execute_rule_actions_internal(self, rule: AutomationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        内部规则动作执行方法

        Internal rule actions execution method

        Args:
            rule: 自动化规则
                 Automation rule
            context: 执行上下文
                    Execution context

        Returns:
            dict: 执行结果
                 Execution result
        """
        result = rule.execute_actions(context)
        self.rule_execution_history.append(result)

        logger.info(f"Executed actions for rule: {rule.name}")

        return result

    def process_automation_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an automation event by evaluating rules and executing actions
        通过评估规则和执行动作来处理自动化事件

        Args:
            event_data: Event data containing context information
                       包含上下文信息的事件数据

        Returns:
            dict: Automation processing results
                  自动化处理结果
        """
        processing_result = {
            'event_processed_at': datetime.now(),
            'rules_evaluated': 0,
            'rules_triggered': 0,
            'actions_executed': 0,
            'processing_time': 0.0,
            'results': []
        }

        start_time = time.time()

        try:
            # Evaluate rules
            triggered_rules = self.evaluate_rules(event_data)
            processing_result['rules_evaluated'] = len(
                [r for r in self.rules.values() if r.enabled])
            processing_result['rules_triggered'] = len(triggered_rules)

            # Execute actions for triggered rules
            for triggered_rule in triggered_rules:
                rule_result = self.execute_rule_actions(triggered_rule['rule_id'], event_data)
                processing_result['results'].append({
                    'rule': triggered_rule,
                    'execution_result': rule_result
                })

                if rule_result.get('actions_executed', 0) > 0:
                    processing_result['actions_executed'] += rule_result['actions_executed']

            processing_result['success'] = True

        except Exception as e:
            processing_result['success'] = False
            processing_result['error'] = str(e)
            logger.error(f"Automation event processing failed: {str(e)}")

        processing_result['processing_time'] = time.time() - start_time

        return processing_result

    def start_automation_loop(self) -> bool:
        """
        Start the automation processing loop
        开始自动化处理循环

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Automation engine is already running")
            return False

        try:
            self.is_running = True
            self.execution_thread = threading.Thread(target=self._automation_loop, daemon=True)
            self.execution_thread.start()
            logger.info("Automation processing loop started")
            return True
        except Exception as e:
            logger.error(f"Failed to start automation loop: {str(e)}")
            self.is_running = False
            return False

    def stop_automation_loop(self) -> bool:
        """
        Stop the automation processing loop
        停止自动化处理循环

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Automation engine is not running")
            return False

        try:
            self.is_running = False
            if self.execution_thread and self.execution_thread.is_alive():
                self.execution_thread.join(timeout=5.0)
            logger.info("Automation processing loop stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop automation loop: {str(e)}")
            return False

    def _automation_loop(self) -> None:
        """
        Main automation processing loop
        主要的自动化处理循环
        """
        logger.info("Automation processing loop started")

        while self.is_running:
            try:
                # This would typically check for pending automation events
                # For now, just sleep and check periodically
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Automation loop error: {str(e)}")
                time.sleep(self.check_interval)

        logger.info("Automation processing loop stopped")

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get automation engine statistics
        获取自动化引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)

        # 获取任务控制器的统计信息
        controller_stats = self.task_controller.get_controller_stats()

        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'total_executions': len(self.rule_execution_history),
            'rules': {rule_id: rule.get_rule_stats() for rule_id, rule in self.rules.items()},
            'concurrency_control': {
                'max_concurrent_tasks': self.task_controller.max_concurrent_tasks,
                'deadlock_timeout': self.task_controller.deadlock_timeout,
                'active_tasks_count': controller_stats.get('active_tasks_count', 0),
                'total_tasks_processed': controller_stats.get('total_tasks_processed', 0),
                'deadlock_detected': controller_stats.get('deadlock_detected', 0),
                'average_wait_time': controller_stats.get('average_wait_time', 0.0),
                'max_wait_time': controller_stats.get('max_wait_time', 0.0),
                'resource_conflicts': controller_stats.get('resource_conflicts', 0)
            }
        }

    def export_rules(self, filepath: str) -> bool:
        """
        Export automation rules to file
        将自动化规则导出到文件

        Args:
            filepath: Path to export file
                     导出文件路径

        Returns:
            bool: True if export successful
                  导出成功返回True
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'engine_name': self.engine_name,
                'rules': {}
            }

            for rule_id, rule in self.rules.items():
                export_data['rules'][rule_id] = {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'conditions': rule.conditions,
                    'actions': rule.actions,
                    'priority': rule.priority,
                    'enabled': rule.enabled,
                    'stats': rule.get_rule_stats()
                }

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Automation rules exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export rules: {str(e)}")
            return False

    def import_rules(self, filepath: str) -> bool:
        """
        Import automation rules from file
        从文件导入自动化规则

        Args:
            filepath: Path to import file
                     导入文件路径

        Returns:
            bool: True if import successful
                  导入成功返回True
        """
        try:
            with open(filepath, 'r', encoding='utf - 8') as f:
                import_data = json.load(f)

            imported_count = 0
            for rule_id, rule_data in import_data.get('rules', {}).items():
                rule = AutomationRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    conditions=rule_data['conditions'],
                    actions=rule_data['actions'],
                    priority=rule_data.get('priority', 1),
                    enabled=rule_data.get('enabled', True)
                )

                self.add_rule(rule)
                imported_count += 1

            logger.info(f"Imported {imported_count} automation rules from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import rules: {str(e)}")
            return False

    def disable_rule(self, rule_id: str) -> bool:
        """
        Disable an automation rule
        禁用自动化规则

        Args:
            rule_id: Rule identifier
                    规则标识符

        Returns:
            bool: True if disabled successfully, False otherwise
                  禁用成功返回True，否则返回False
        """
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            logger.info(f"Disabled automation rule: {rule_id}")
            return True
        return False

    def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate all enabled rules against the given context
        根据给定上下文评估所有启用的规则

        Args:
            context: Evaluation context
                    评估上下文

        Returns:
            list: List of triggered rules with their actions
                  触发的规则及其动作列表
        """
        triggered_rules = []

        # Sort rules by priority (highest first)
        sorted_rules = sorted(
            [rule for rule in self.rules.values() if rule.enabled],
            key=lambda r: r.priority,
            reverse=True
        )

        for rule in sorted_rules:
            try:
                if rule.evaluate_conditions(context):
                    triggered_rules.append({
                        'rule_id': rule.rule_id,
                        'rule_name': rule.name,
                        'priority': rule.priority,
                        'actions': rule.actions
                    })

                    logger.info(f"Rule triggered: {rule.name} ({rule.rule_id})")

            except Exception as e:
                logger.error(f"Rule evaluation failed for {rule.rule_id}: {str(e)}")

        return triggered_rules

    def execute_rule_actions(self, rule_id: str, context: Dict[str, Any],


                             required_resources: Optional[set] = None) -> Dict[str, Any]:
        """
        Execute actions for a specific rule with concurrency control
        使用并发控制为特定规则执行动作

        Args:
            rule_id: Rule identifier
                    规则标识符
            context: Execution context
                    执行上下文
            required_resources: Required resources for execution
                              执行所需的资源

        Returns:
            dict: Action execution results
                  动作执行结果
        """
        if rule_id not in self.rules:
            return {'error': f'Rule {rule_id} not found'}

        rule = self.rules[rule_id]

        # 使用任务并发控制器执行规则动作
        try:
            # 同步执行版本
            result = self.task_controller.execute_with_control_sync(
                task_id=f"rule_{rule_id}_{datetime.now().strftime('%Y % m % d % H % M % S % f')}",
                task_func=self._execute_rule_actions_internal,
                rule=rule,
                context=context,
                required_resources=required_resources
            )

            return result

        except RuntimeError as e:
            error_msg = str(e)
            logger.warning(f"Failed to execute rule {rule_id}: {error_msg}")
            return {'error': error_msg, 'success': False}

    def _execute_rule_actions_internal(self, rule: AutomationRule, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        内部规则动作执行方法

        Internal rule actions execution method

        Args:
            rule: 自动化规则
                 Automation rule
            context: 执行上下文
                    Execution context

        Returns:
            dict: 执行结果
                 Execution result
        """
        result = rule.execute_actions(context)
        self.rule_execution_history.append(result)

        logger.info(f"Executed actions for rule: {rule.name}")

        return result

    def process_automation_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an automation event by evaluating rules and executing actions
        通过评估规则和执行动作来处理自动化事件

        Args:
            event_data: Event data containing context information
                       包含上下文信息的事件数据

        Returns:
            dict: Automation processing results
                  自动化处理结果
        """
        processing_result = {
            'event_processed_at': datetime.now(),
            'rules_evaluated': 0,
            'rules_triggered': 0,
            'actions_executed': 0,
            'processing_time': 0.0,
            'results': []
        }

        start_time = time.time()

        try:
            # Evaluate rules
            triggered_rules = self.evaluate_rules(event_data)
            processing_result['rules_evaluated'] = len(
                [r for r in self.rules.values() if r.enabled])
            processing_result['rules_triggered'] = len(triggered_rules)

            # Execute actions for triggered rules
            for triggered_rule in triggered_rules:
                rule_result = self.execute_rule_actions(triggered_rule['rule_id'], event_data)
                processing_result['results'].append({
                    'rule': triggered_rule,
                    'execution_result': rule_result
                })

                if rule_result.get('actions_executed', 0) > 0:
                    processing_result['actions_executed'] += rule_result['actions_executed']

            processing_result['success'] = True

        except Exception as e:
            processing_result['success'] = False
            processing_result['error'] = str(e)
            logger.error(f"Automation event processing failed: {str(e)}")

        processing_result['processing_time'] = time.time() - start_time

        return processing_result

    def start_automation_loop(self) -> bool:
        """
        Start the automation processing loop
        开始自动化处理循环

        Returns:
            bool: True if started successfully, False otherwise
                  启动成功返回True，否则返回False
        """
        if self.is_running:
            logger.warning("Automation engine is already running")
            return False

        try:
            self.is_running = True
            self.execution_thread = threading.Thread(target=self._automation_loop, daemon=True)
            self.execution_thread.start()
            logger.info("Automation processing loop started")
            return True
        except Exception as e:
            logger.error(f"Failed to start automation loop: {str(e)}")
            self.is_running = False
            return False

    def stop_automation_loop(self) -> bool:
        """
        Stop the automation processing loop
        停止自动化处理循环

        Returns:
            bool: True if stopped successfully, False otherwise
                  停止成功返回True，否则返回False
        """
        if not self.is_running:
            logger.warning("Automation engine is not running")
            return False

        try:
            self.is_running = False
            if self.execution_thread and self.execution_thread.is_alive():
                self.execution_thread.join(timeout=5.0)
            logger.info("Automation processing loop stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop automation loop: {str(e)}")
            return False

    def _automation_loop(self) -> None:
        """
        Main automation processing loop
        主要的自动化处理循环
        """
        logger.info("Automation processing loop started")

        while self.is_running:
            try:
                # This would typically check for pending automation events
                # For now, just sleep and check periodically
                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"Automation loop error: {str(e)}")
                time.sleep(self.check_interval)

        logger.info("Automation processing loop stopped")

    def get_engine_stats(self) -> Dict[str, Any]:
        """
        Get automation engine statistics
        获取自动化引擎统计信息

        Returns:
            dict: Engine statistics
                  引擎统计信息
        """
        total_rules = len(self.rules)
        enabled_rules = sum(1 for rule in self.rules.values() if rule.enabled)

        # 获取任务控制器的统计信息
        controller_stats = self.task_controller.get_controller_stats()

        return {
            'engine_name': self.engine_name,
            'is_running': self.is_running,
            'total_rules': total_rules,
            'enabled_rules': enabled_rules,
            'disabled_rules': total_rules - enabled_rules,
            'total_executions': len(self.rule_execution_history),
            'rules': {rule_id: rule.get_rule_stats() for rule_id, rule in self.rules.items()},
            'concurrency_control': {
                'max_concurrent_tasks': self.task_controller.max_concurrent_tasks,
                'deadlock_timeout': self.task_controller.deadlock_timeout,
                'active_tasks_count': controller_stats.get('active_tasks_count', 0),
                'total_tasks_processed': controller_stats.get('total_tasks_processed', 0),
                'deadlock_detected': controller_stats.get('deadlock_detected', 0),
                'average_wait_time': controller_stats.get('average_wait_time', 0.0),
                'max_wait_time': controller_stats.get('max_wait_time', 0.0),
                'resource_conflicts': controller_stats.get('resource_conflicts', 0)
            }
        }

    def export_rules(self, filepath: str) -> bool:
        """
        Export automation rules to file
        将自动化规则导出到文件

        Args:
            filepath: Path to export file
                     导出文件路径

        Returns:
            bool: True if export successful
                  导出成功返回True
        """
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'engine_name': self.engine_name,
                'rules': {}
            }

            for rule_id, rule in self.rules.items():
                export_data['rules'][rule_id] = {
                    'rule_id': rule.rule_id,
                    'name': rule.name,
                    'conditions': rule.conditions,
                    'actions': rule.actions,
                    'priority': rule.priority,
                    'enabled': rule.enabled,
                    'stats': rule.get_rule_stats()
                }

            with open(filepath, 'w', encoding='utf - 8') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Automation rules exported to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export rules: {str(e)}")
            return False

    def import_rules(self, filepath: str) -> bool:
        """
        Import automation rules from file
        从文件导入自动化规则

        Args:
            filepath: Path to import file
                     导入文件路径

        Returns:
            bool: True if import successful
                  导入成功返回True
        """
        try:
            with open(filepath, 'r', encoding='utf - 8') as f:
                import_data = json.load(f)

            imported_count = 0
            for rule_id, rule_data in import_data.get('rules', {}).items():
                rule = AutomationRule(
                    rule_id=rule_data['rule_id'],
                    name=rule_data['name'],
                    conditions=rule_data['conditions'],
                    actions=rule_data['actions'],
                    priority=rule_data.get('priority', 1),
                    enabled=rule_data.get('enabled', True)
                )

                self.add_rule(rule)
                imported_count += 1

            logger.info(f"Imported {imported_count} automation rules from {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to import rules: {str(e)}")
            return False


# Global automation engine instance
# 全局自动化引擎实例
automation_engine = AutomationEngine()

__all__ = [
    'TaskConcurrencyController',
    'AutomationRule',
    'AutomationEngine',
    'automation_engine'
]
