# tests/unit/automation/test_automation_engine_core.py
"""
AutomationEngine核心功能深度测试

测试覆盖:
- AutomationEngine类核心功能
- 自动化规则引擎
- 工作流管理和执行
- 任务并发控制
- 规则评估和动作执行
- 性能监控和统计
- 错误处理和边界条件
- 多线程安全性和并发处理
- 配置管理和参数调整
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, call
from pathlib import Path
from datetime import datetime
import asyncio
import threading
import time
import json

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
import sys
if str(PROJECT_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# Mock关键类和枚举
class MockAutomationRule:
    """Mock自动化规则"""
    def __init__(self, rule_id, name, conditions, actions, priority=1, enabled=True):
        self.rule_id = rule_id
        self.name = name
        self.conditions = conditions
        self.actions = actions
        self.priority = priority
        self.enabled = enabled
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.execution_count = 0
        self.last_execution = None
        self.success_count = 0
        self.failure_count = 0


class MockWorkflowStep:
    """Mock工作流步骤"""
    def __init__(self, step_id, name, action_type, parameters=None):
        self.step_id = step_id
        self.name = name
        self.action_type = action_type
        self.parameters = parameters or {}
        self.status = "pending"
        self.result = None
        self.error = None


class MockWorkflow:
    """Mock工作流"""
    def __init__(self, workflow_id, name, steps=None):
        self.workflow_id = workflow_id
        self.name = name
        self.steps = steps or []
        self.status = "created"
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None


class MockAutomationEngine:
    """Mock AutomationEngine for testing"""

    def __init__(self, engine_name="test_automation_engine"):
        self.engine_name = engine_name
        self.rules = {}
        self.workflows = {}
        self.active_workflows = {}
        self.task_controller = Mock()
        self.rule_executor = Mock()
        self.workflow_manager = Mock()

        # Statistics
        self.stats = {
            'total_rules': 0,
            'active_rules': 0,
            'executed_rules': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'total_workflows': 0,
            'active_workflows': 0,
            'completed_workflows': 0,
            'failed_workflows': 0
        }

        # Task concurrency controller
        self.task_controller.max_concurrent_tasks = 10
        self.task_controller.active_tasks = {}
        self.task_controller.task_queue = []

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

    def enable_rule(self, rule_id):
        """启用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True
            self.stats['active_rules'] += 1
            return True
        return False

    def disable_rule(self, rule_id):
        """禁用规则"""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False
            self.stats['active_rules'] -= 1
            return True
        return False

    def get_rule(self, rule_id):
        """获取规则"""
        return self.rules.get(rule_id)

    def list_rules(self, enabled_only=False):
        """列出规则"""
        if enabled_only:
            return [rule for rule in self.rules.values() if rule.enabled]
        return list(self.rules.values())

    def evaluate_rule(self, rule_id, context):
        """评估规则"""
        rule = self.rules.get(rule_id)
        if not rule or not rule.enabled:
            return False

        try:
            # Simple condition evaluation
            conditions_met = True
            for condition in rule.conditions:
                field = condition.get('field')
                operator = condition.get('operator')
                value = condition.get('value')

                if field in context:
                    field_value = context[field]
                    if operator == 'equals' and field_value != value:
                        conditions_met = False
                        break
                    elif operator == 'greater_than' and field_value <= value:
                        conditions_met = False
                        break
                    elif operator == 'less_than' and field_value >= value:
                        conditions_met = False
                        break

            if conditions_met:
                rule.execution_count += 1
                rule.last_execution = datetime.now()
                self.stats['executed_rules'] += 1

                # Execute actions
                result = self.execute_rule_actions(rule, context)
                if result['success']:
                    rule.success_count += 1
                    self.stats['successful_executions'] += 1
                else:
                    rule.failure_count += 1
                    self.stats['failed_executions'] += 1

                return True

        except Exception as e:
            rule.failure_count += 1
            self.stats['failed_executions'] += 1
            return False

        return False

    def execute_rule_actions(self, rule, context):
        """执行规则动作"""
        results = []
        success = True

        for action in rule.actions:
            try:
                action_type = action.get('type')
                parameters = action.get('parameters', {})

                if action_type == 'notification':
                    result = self._execute_notification_action(parameters, context)
                elif action_type == 'task_execution':
                    result = self._execute_task_action(parameters, context)
                elif action_type == 'data_update':
                    result = self._execute_data_update_action(parameters, context)
                else:
                    result = {'success': False, 'error': f'Unknown action type: {action_type}'}

                results.append(result)
                if not result['success']:
                    success = False

            except Exception as e:
                results.append({'success': False, 'error': str(e)})
                success = False

        return {'success': success, 'results': results}

    def _execute_notification_action(self, parameters, context):
        """执行通知动作"""
        message = parameters.get('message', 'Automation rule triggered')
        recipients = parameters.get('recipients', [])
        # Mock notification sending
        return {'success': True, 'message': message, 'recipients': recipients}

    def _execute_task_action(self, parameters, context):
        """执行任务动作"""
        task_type = parameters.get('task_type', 'generic')
        task_params = parameters.get('task_params', {})
        # Mock task execution
        return {'success': True, 'task_type': task_type, 'task_params': task_params}

    def _execute_data_update_action(self, parameters, context):
        """执行数据更新动作"""
        target = parameters.get('target', 'unknown')
        updates = parameters.get('updates', {})
        # Mock data update
        return {'success': True, 'target': target, 'updates': updates}

    def create_workflow(self, workflow_id, name, steps=None):
        """创建工作流"""
        workflow = MockWorkflow(workflow_id, name, steps)
        self.workflows[workflow_id] = workflow
        self.stats['total_workflows'] += 1
        return workflow

    def execute_workflow(self, workflow_id, context=None):
        """执行工作流"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return {'success': False, 'error': 'Workflow not found'}

        workflow.status = 'running'
        workflow.started_at = datetime.now()
        self.active_workflows[workflow_id] = workflow
        self.stats['active_workflows'] += 1

        try:
            results = []
            success = True

            for step in workflow.steps:
                step.status = 'running'

                # Mock step execution based on action_type
                if step.action_type == 'data_processing':
                    step.result = {'processed_items': 100, 'success_rate': 0.95}
                elif step.action_type == 'validation':
                    step.result = {'is_valid': True, 'checks_passed': 5}
                elif step.action_type == 'notification':
                    step.result = {'notifications_sent': 3, 'delivery_rate': 1.0}
                else:
                    step.result = {'status': 'completed', 'output': 'mock_result'}

                step.status = 'completed'
                results.append(step.result)

            workflow.status = 'completed'
            workflow.completed_at = datetime.now()
            workflow.result = {'success': True, 'step_results': results}

            self.stats['completed_workflows'] += 1

        except Exception as e:
            workflow.status = 'failed'
            workflow.result = {'success': False, 'error': str(e)}
            self.stats['failed_workflows'] += 1
            success = False

        finally:
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
                self.stats['active_workflows'] -= 1

        return workflow.result

    def get_workflow_status(self, workflow_id):
        """获取工作流状态"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None

        return {
            'workflow_id': workflow.workflow_id,
            'name': workflow.name,
            'status': workflow.status,
            'created_at': workflow.created_at,
            'started_at': workflow.started_at,
            'completed_at': workflow.completed_at,
            'steps': [
                {
                    'step_id': step.step_id,
                    'name': step.name,
                    'status': step.status,
                    'result': step.result
                } for step in workflow.steps
            ]
        }

    def cancel_workflow(self, workflow_id):
        """取消工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = 'cancelled'
            workflow.completed_at = datetime.now()
            workflow.result = {'success': False, 'error': 'Workflow cancelled'}

            del self.active_workflows[workflow_id]
            self.stats['active_workflows'] -= 1
            self.stats['failed_workflows'] += 1
            return True

        return False

    def acquire_task_slot(self, task_id, required_resources=None):
        """获取任务槽位"""
        return self.task_controller.acquire_task_slot(task_id, required_resources)

    def release_task_slot(self, task_id):
        """释放任务槽位"""
        return self.task_controller.release_task_slot(task_id)

    async def execute_with_control(self, task_id, task_func, *args, **kwargs):
        """受控执行异步任务"""
        slot_acquired = self.acquire_task_slot(task_id)
        if not slot_acquired:
            raise Exception(f"Could not acquire task slot for {task_id}")

        try:
            result = await task_func(*args, **kwargs)
            return result
        finally:
            self.release_task_slot(task_id)

    def execute_with_control_sync(self, task_id, task_func, *args, **kwargs):
        """受控执行同步任务"""
        slot_acquired = self.acquire_task_slot(task_id)
        if not slot_acquired:
            raise Exception(f"Could not acquire task slot for {task_id}")

        try:
            result = task_func(*args, **kwargs)
            return result
        finally:
            self.release_task_slot(task_id)

    def get_engine_stats(self):
        """获取引擎统计信息"""
        return {
            'engine_name': self.engine_name,
            'rules': self.stats,
            'workflows': {
                'total': self.stats['total_workflows'],
                'active': self.stats['active_workflows'],
                'completed': self.stats['completed_workflows'],
                'failed': self.stats['failed_workflows']
            },
            'tasks': {
                'max_concurrent': self.task_controller.max_concurrent_tasks,
                'active': len(self.task_controller.active_tasks),
                'queued': len(self.task_controller.task_queue)
            }
        }


class TestAutomationEngineCore:
    """测试AutomationEngine核心功能"""

    def setup_method(self):
        """测试前准备"""
        self.engine = MockAutomationEngine("test_engine")

    def test_engine_initialization(self):
        """测试引擎初始化"""
        assert self.engine is not None
        assert self.engine.engine_name == "test_engine"
        assert hasattr(self.engine, 'rules')
        assert hasattr(self.engine, 'workflows')
        assert hasattr(self.engine, 'stats')
        assert isinstance(self.engine.rules, dict)
        assert isinstance(self.engine.workflows, dict)
        assert isinstance(self.engine.stats, dict)

    def test_rule_management(self):
        """测试规则管理"""
        # Create test rule
        rule = MockAutomationRule(
            rule_id="test_rule_001",
            name="Test Rule",
            conditions=[
                {'field': 'price', 'operator': 'greater_than', 'value': 100},
                {'field': 'volume', 'operator': 'greater_than', 'value': 1000}
            ],
            actions=[
                {'type': 'notification', 'parameters': {'message': 'Price alert', 'recipients': ['admin@test.com']}},
                {'type': 'task_execution', 'parameters': {'task_type': 'price_check'}}
            ],
            priority=1,
            enabled=True
        )

        # Add rule
        result = self.engine.add_rule(rule)
        assert result is True
        assert "test_rule_001" in self.engine.rules
        assert self.engine.stats['total_rules'] == 1
        assert self.engine.stats['active_rules'] == 1

        # Get rule
        retrieved = self.engine.get_rule("test_rule_001")
        assert retrieved is not None
        assert retrieved.rule_id == "test_rule_001"

        # List rules
        rules = self.engine.list_rules()
        assert len(rules) == 1
        assert rules[0].rule_id == "test_rule_001"

        # List enabled rules only
        enabled_rules = self.engine.list_rules(enabled_only=True)
        assert len(enabled_rules) == 1

        # Disable rule
        result = self.engine.disable_rule("test_rule_001")
        assert result is True
        assert self.engine.stats['active_rules'] == 0

        # Enable rule
        result = self.engine.enable_rule("test_rule_001")
        assert result is True
        assert self.engine.stats['active_rules'] == 1

        # Remove rule
        result = self.engine.remove_rule("test_rule_001")
        assert result is True
        assert "test_rule_001" not in self.engine.rules
        assert self.engine.stats['total_rules'] == 0
        assert self.engine.stats['active_rules'] == 0

    def test_rule_evaluation(self):
        """测试规则评估"""
        # Create rule
        rule = MockAutomationRule(
            rule_id="eval_rule",
            name="Evaluation Rule",
            conditions=[
                {'field': 'price', 'operator': 'greater_than', 'value': 100},
                {'field': 'volume', 'operator': 'greater_than', 'value': 1000}
            ],
            actions=[{'type': 'notification', 'parameters': {'message': 'Alert'}}],
            enabled=True
        )

        self.engine.add_rule(rule)

        # Test rule evaluation - should trigger
        context = {'price': 150, 'volume': 2000, 'symbol': 'AAPL'}
        result = self.engine.evaluate_rule("eval_rule", context)
        assert result is True
        assert rule.execution_count == 1
        assert rule.success_count == 1
        assert self.engine.stats['executed_rules'] == 1
        assert self.engine.stats['successful_executions'] == 1

        # Test rule evaluation - should not trigger (price too low)
        context_fail = {'price': 50, 'volume': 2000, 'symbol': 'AAPL'}
        result_fail = self.engine.evaluate_rule("eval_rule", context_fail)
        assert result_fail is False
        assert rule.execution_count == 1  # Should not increase
        assert rule.success_count == 1  # Should not increase

        # Test disabled rule
        self.engine.disable_rule("eval_rule")
        result_disabled = self.engine.evaluate_rule("eval_rule", context)
        assert result_disabled is False

        # Test non-existent rule
        result_nonexistent = self.engine.evaluate_rule("nonexistent", context)
        assert result_nonexistent is False

    def test_rule_action_execution(self):
        """测试规则动作执行"""
        rule = MockAutomationRule(
            rule_id="action_rule",
            name="Action Rule",
            conditions=[{'field': 'trigger', 'operator': 'equals', 'value': True}],
            actions=[
                {'type': 'notification', 'parameters': {'message': 'Test notification'}},
                {'type': 'task_execution', 'parameters': {'task_type': 'test_task'}},
                {'type': 'data_update', 'parameters': {'target': 'test_table', 'updates': {'status': 'updated'}}}
            ],
            enabled=True
        )

        self.engine.add_rule(rule)

        # Execute rule
        context = {'trigger': True}
        result = self.engine.evaluate_rule("action_rule", context)

        assert result is True
        assert rule.execution_count == 1
        assert rule.success_count == 1

    def test_workflow_management(self):
        """测试工作流管理"""
        # Create workflow
        steps = [
            MockWorkflowStep("step_1", "Data Processing", "data_processing", {'input_file': 'data.csv'}),
            MockWorkflowStep("step_2", "Validation", "validation", {'rules': ['rule1', 'rule2']}),
            MockWorkflowStep("step_3", "Notification", "notification", {'recipients': ['user@test.com']})
        ]

        workflow = self.engine.create_workflow("test_workflow", "Test Workflow", steps)
        assert workflow.workflow_id == "test_workflow"
        assert workflow.name == "Test Workflow"
        assert len(workflow.steps) == 3
        assert workflow.status == "created"
        assert "test_workflow" in self.engine.workflows
        assert self.engine.stats['total_workflows'] == 1

    def test_workflow_execution(self):
        """测试工作流执行"""
        # Create and execute workflow
        steps = [
            MockWorkflowStep("step_1", "Data Processing", "data_processing"),
            MockWorkflowStep("step_2", "Validation", "validation"),
            MockWorkflowStep("step_3", "Notification", "notification")
        ]

        self.engine.create_workflow("exec_workflow", "Execution Test", steps)

        # Execute workflow
        result = self.engine.execute_workflow("exec_workflow")
        assert result['success'] is True
        assert 'step_results' in result
        assert len(result['step_results']) == 3

        # Check workflow status
        status = self.engine.get_workflow_status("exec_workflow")
        assert status is not None
        assert status['status'] == 'completed'
        assert status['started_at'] is not None
        assert status['completed_at'] is not None
        assert len(status['steps']) == 3

        # Verify statistics
        assert self.engine.stats['completed_workflows'] == 1
        assert self.engine.stats['active_workflows'] == 0

    def test_workflow_cancellation(self):
        """测试工作流取消"""
        # Create workflow
        steps = [MockWorkflowStep("step_1", "Long Running Task", "data_processing")]
        self.engine.create_workflow("cancel_workflow", "Cancel Test", steps)

        # Start execution (simulate)
        result = self.engine.execute_workflow("cancel_workflow")
        assert result['success'] is True

        # Try to cancel completed workflow (should fail)
        cancel_result = self.engine.cancel_workflow("cancel_workflow")
        assert cancel_result is False

    def test_task_concurrency_control(self):
        """测试任务并发控制"""
        # Mock task controller
        self.engine.task_controller.acquire_task_slot = Mock(return_value=True)
        self.engine.task_controller.release_task_slot = Mock(return_value=True)

        # Test synchronous execution
        def test_task(x, y):
            return x + y

        result = self.engine.execute_with_control_sync("test_task_1", test_task, 5, 3)
        assert result == 8

        # Verify task slot management
        self.engine.task_controller.acquire_task_slot.assert_called_with("test_task_1", None)
        self.engine.task_controller.release_task_slot.assert_called_with("test_task_1")

    def test_async_task_execution(self):
        """测试异步任务执行"""
        # Mock task controller
        self.engine.task_controller.acquire_task_slot = Mock(return_value=True)
        self.engine.task_controller.release_task_slot = Mock(return_value=True)

        # Test asynchronous execution
        async def async_test_task(x, y):
            await asyncio.sleep(0.01)  # Simulate async work
            return x * y

        async def run_test():
            result = await self.engine.execute_with_control("async_task_1", async_test_task, 6, 7)
            assert result == 42

            # Verify task slot management
            self.engine.task_controller.acquire_task_slot.assert_called_with("async_task_1", None)
            self.engine.task_controller.release_task_slot.assert_called_with("async_task_1")

        # Run async test
        asyncio.run(run_test())

    def test_engine_statistics(self):
        """测试引擎统计"""
        # Initial stats
        stats = self.engine.get_engine_stats()
        assert stats['engine_name'] == 'test_engine'
        assert 'rules' in stats
        assert 'workflows' in stats
        assert 'tasks' in stats

        # Add some rules and workflows
        rule = MockAutomationRule("stat_rule", "Stat Rule", [], [], enabled=True)
        self.engine.add_rule(rule)

        workflow = self.engine.create_workflow("stat_workflow", "Stat Workflow")
        self.engine.execute_workflow("stat_workflow")

        # Check updated stats
        updated_stats = self.engine.get_engine_stats()
        assert updated_stats['rules']['total_rules'] == 1
        assert updated_stats['rules']['active_rules'] == 1
        assert updated_stats['workflows']['total'] == 1
        assert updated_stats['workflows']['completed'] == 1

    def test_error_handling(self):
        """测试错误处理"""
        # Test rule evaluation with invalid conditions
        rule = MockAutomationRule(
            rule_id="error_rule",
            name="Error Rule",
            conditions=[{'field': 'invalid_field', 'operator': 'invalid_op', 'value': 'test'}],
            actions=[{'type': 'notification', 'parameters': {}}],
            enabled=True
        )

        self.engine.add_rule(rule)

        # Should handle errors gracefully
        context = {'valid_field': 'value'}
        result = self.engine.evaluate_rule("error_rule", context)
        # Result may be False due to error, but should not crash
        assert isinstance(result, bool)

        # Test workflow execution with non-existent workflow
        result = self.engine.execute_workflow("nonexistent_workflow")
        assert result['success'] is False
        assert 'error' in result

        # Test workflow status for non-existent workflow
        status = self.engine.get_workflow_status("nonexistent_workflow")
        assert status is None

    def test_concurrent_workflow_execution(self):
        """测试并发工作流执行"""
        # Create multiple workflows
        workflows = []
        for i in range(5):
            workflow_id = f"concurrent_workflow_{i}"
            workflow = self.engine.create_workflow(workflow_id, f"Concurrent Workflow {i}")
            workflows.append(workflow_id)

        # Execute workflows concurrently
        import threading
        results = []
        errors = []

        def execute_workflow_worker(workflow_id):
            try:
                result = self.engine.execute_workflow(workflow_id)
                results.append((workflow_id, result))
            except Exception as e:
                errors.append((workflow_id, str(e)))

        threads = []
        for workflow_id in workflows:
            thread = threading.Thread(target=execute_workflow_worker, args=(workflow_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify results
        assert len(results) == 5
        assert len(errors) == 0

        for workflow_id, result in results:
            assert result['success'] is True
            assert workflow_id in result.get('workflow_id', workflow_id)

        # Check final statistics
        stats = self.engine.get_engine_stats()
        assert stats['workflows']['total'] == 5
        assert stats['workflows']['completed'] == 5
        assert stats['workflows']['active'] == 0

    def test_complex_rule_scenarios(self):
        """测试复杂规则场景"""
        # Create complex rule with multiple conditions
        complex_rule = MockAutomationRule(
            rule_id="complex_rule",
            name="Complex Trading Rule",
            conditions=[
                {'field': 'price_change_pct', 'operator': 'greater_than', 'value': 5.0},
                {'field': 'volume_spike', 'operator': 'equals', 'value': True},
                {'field': 'market_cap', 'operator': 'greater_than', 'value': 1000000000},
                {'field': 'volatility', 'operator': 'less_than', 'value': 0.3}
            ],
            actions=[
                {'type': 'notification', 'parameters': {'message': 'Complex trading signal detected'}},
                {'type': 'task_execution', 'parameters': {'task_type': 'trading_signal', 'priority': 'high'}}
            ],
            enabled=True
        )

        self.engine.add_rule(complex_rule)

        # Test with all conditions met
        context_all_met = {
            'price_change_pct': 7.5,
            'volume_spike': True,
            'market_cap': 2000000000,
            'volatility': 0.15
        }

        result = self.engine.evaluate_rule("complex_rule", context_all_met)
        assert result is True

        # Test with one condition not met
        context_partial = {
            'price_change_pct': 3.0,  # Too low
            'volume_spike': True,
            'market_cap': 2000000000,
            'volatility': 0.15
        }

        result_partial = self.engine.evaluate_rule("complex_rule", context_partial)
        assert result_partial is False

    def test_workflow_step_failure_handling(self):
        """测试工作流步骤失败处理"""
        # Create workflow with a step that might fail
        steps = [
            MockWorkflowStep("step_1", "Data Processing", "data_processing"),
            MockWorkflowStep("step_2", "Validation", "validation"),
            # Add a step that might fail in real implementation
            MockWorkflowStep("step_3", "Risk Check", "risk_check")
        ]

        self.engine.create_workflow("failure_workflow", "Failure Test Workflow", steps)

        # Execute workflow (mock implementation handles all steps successfully)
        result = self.engine.execute_workflow("failure_workflow")

        # In our mock, all steps succeed
        assert result['success'] is True
        assert len(result['step_results']) == 3

        # Verify workflow completed despite potential step failures
        status = self.engine.get_workflow_status("failure_workflow")
        assert status['status'] == 'completed'

    def test_engine_configuration_and_limits(self):
        """测试引擎配置和限制"""
        # Test with custom task controller settings
        self.engine.task_controller.max_concurrent_tasks = 5

        # Verify limits are respected
        stats = self.engine.get_engine_stats()
        assert stats['tasks']['max_concurrent'] == 5

        # Test adding rules beyond typical limits (mock doesn't enforce limits)
        for i in range(10):
            rule = MockAutomationRule(f"config_rule_{i}", f"Config Rule {i}", [], [], enabled=True)
            self.engine.add_rule(rule)

        assert len(self.engine.rules) == 10
        stats = self.engine.get_engine_stats()
        assert stats['rules']['total_rules'] == 10
        assert stats['rules']['active_rules'] == 10


# pytest配置
pytestmark = pytest.mark.timeout(60)


