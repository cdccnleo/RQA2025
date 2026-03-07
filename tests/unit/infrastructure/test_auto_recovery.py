"""
测试自动恢复管理器

覆盖 auto_recovery.py 中的 AutoRecoveryManager 类
"""

import pytest
from unittest.mock import Mock, call
from src.infrastructure.auto_recovery import AutoRecoveryManager


class TestAutoRecoveryManager:
    """AutoRecoveryManager 类测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = AutoRecoveryManager()

        assert manager.recovery_actions == {}
        assert isinstance(manager.recovery_actions, dict)

    def test_register_recovery_action_function(self):
        """测试注册恢复动作（函数）"""
        manager = AutoRecoveryManager()

        def recovery_function():
            return "recovered"

        manager.register_recovery_action("test_action", recovery_function)

        assert "test_action" in manager.recovery_actions
        assert manager.recovery_actions["test_action"] == recovery_function

    def test_register_recovery_action_lambda(self):
        """测试注册恢复动作（lambda函数）"""
        manager = AutoRecoveryManager()

        recovery_lambda = lambda: "lambda_recovered"

        manager.register_recovery_action("lambda_action", recovery_lambda)

        assert "lambda_action" in manager.recovery_actions
        assert manager.recovery_actions["lambda_action"] == recovery_lambda

    def test_register_recovery_action_method(self):
        """测试注册恢复动作（对象方法）"""
        manager = AutoRecoveryManager()

        class RecoveryHandler:
            def recover(self):
                return "method_recovered"

        handler = RecoveryHandler()
        manager.register_recovery_action("method_action", handler.recover)

        assert "method_action" in manager.recovery_actions

    def test_register_recovery_action_callable_object(self):
        """测试注册恢复动作（可调用对象）"""
        manager = AutoRecoveryManager()

        class CallableRecovery:
            def __call__(self):
                return "callable_recovered"

        callable_obj = CallableRecovery()
        manager.register_recovery_action("callable_action", callable_obj)

        assert "callable_action" in manager.recovery_actions

    def test_register_multiple_actions(self):
        """测试注册多个恢复动作"""
        manager = AutoRecoveryManager()

        actions = {
            "action1": lambda: "result1",
            "action2": lambda: "result2",
            "action3": lambda: "result3"
        }

        for name, action in actions.items():
            manager.register_recovery_action(name, action)

        assert len(manager.recovery_actions) == 3
        for name in actions.keys():
            assert name in manager.recovery_actions

    def test_execute_recovery_existing_action(self):
        """测试执行存在的恢复动作"""
        manager = AutoRecoveryManager()

        def recovery_function():
            return "recovery_successful"

        manager.register_recovery_action("test_recovery", recovery_function)

        result = manager.execute_recovery("test_recovery")

        assert result == "recovery_successful"

    def test_execute_recovery_nonexistent_action(self):
        """测试执行不存在的恢复动作"""
        manager = AutoRecoveryManager()

        result = manager.execute_recovery("nonexistent_action")

        assert result == False

    def test_execute_recovery_with_exception(self):
        """测试执行恢复动作时发生异常"""
        manager = AutoRecoveryManager()

        def failing_recovery():
            raise Exception("Recovery failed")

        manager.register_recovery_action("failing_action", failing_recovery)

        with pytest.raises(Exception, match="Recovery failed"):
            manager.execute_recovery("failing_action")

    def test_execute_recovery_with_parameters(self):
        """测试执行带参数的恢复动作"""
        manager = AutoRecoveryManager()

        def recovery_with_params(service_name, error_code):
            return f"Recovered {service_name} from error {error_code}"

        # 注意：原始实现不支持参数，这里测试实际行为
        manager.register_recovery_action("param_action", recovery_with_params)

        # 由于原始实现不支持参数，这里会失败
        result = manager.execute_recovery("param_action")

        # 验证函数被调用但没有参数
        assert result == recovery_with_params()  # 应该使用默认参数或失败

    def test_recovery_action_overwrite(self):
        """测试恢复动作覆盖"""
        manager = AutoRecoveryManager()

        def old_action():
            return "old_result"

        def new_action():
            return "new_result"

        # 注册旧动作
        manager.register_recovery_action("test_action", old_action)
        result1 = manager.execute_recovery("test_action")
        assert result1 == "old_result"

        # 注册新动作（覆盖旧动作）
        manager.register_recovery_action("test_action", new_action)
        result2 = manager.execute_recovery("test_action")
        assert result2 == "new_result"

    def test_recovery_actions_isolation(self):
        """测试恢复动作隔离"""
        manager1 = AutoRecoveryManager()
        manager2 = AutoRecoveryManager()

        def action1():
            return "manager1_action"

        def action2():
            return "manager2_action"

        manager1.register_recovery_action("shared_name", action1)
        manager2.register_recovery_action("shared_name", action2)

        result1 = manager1.execute_recovery("shared_name")
        result2 = manager2.execute_recovery("shared_name")

        assert result1 == "manager1_action"
        assert result2 == "manager2_action"
        assert result1 != result2

    def test_recovery_action_return_values(self):
        """测试恢复动作的各种返回值"""
        manager = AutoRecoveryManager()

        test_cases = [
            ("true_action", lambda: True, True),
            ("false_action", lambda: False, False),
            ("none_action", lambda: None, None),
            ("string_action", lambda: "recovered", "recovered"),
            ("dict_action", lambda: {"status": "ok"}, {"status": "ok"}),
            ("list_action", lambda: [1, 2, 3], [1, 2, 3]),
        ]

        for action_name, action_func, expected_result in test_cases:
            manager.register_recovery_action(action_name, action_func)
            result = manager.execute_recovery(action_name)
            assert result == expected_result

    def test_empty_recovery_actions_dict(self):
        """测试空的恢复动作字典"""
        manager = AutoRecoveryManager()

        assert manager.recovery_actions == {}

        result = manager.execute_recovery("any_action")
        assert result == False

    def test_recovery_action_call_tracking(self):
        """测试恢复动作调用跟踪"""
        manager = AutoRecoveryManager()

        call_count = 0

        def counting_action():
            nonlocal call_count
            call_count += 1
            return f"called_{call_count}"

        manager.register_recovery_action("counting", counting_action)

        # 多次执行
        result1 = manager.execute_recovery("counting")
        result2 = manager.execute_recovery("counting")
        result3 = manager.execute_recovery("counting")

        assert result1 == "called_1"
        assert result2 == "called_2"
        assert result3 == "called_3"
        assert call_count == 3

    def test_recovery_action_side_effects(self):
        """测试恢复动作的副作用"""
        manager = AutoRecoveryManager()

        side_effects = []

        def action_with_side_effects():
            side_effects.append("action_executed")
            return "success"

        manager.register_recovery_action("side_effect_action", action_with_side_effects)

        result = manager.execute_recovery("side_effect_action")

        assert result == "success"
        assert len(side_effects) == 1
        assert side_effects[0] == "action_executed"

    def test_recovery_action_error_recovery(self):
        """测试恢复动作中的错误恢复"""
        manager = AutoRecoveryManager()

        def problematic_action():
            try:
                # 模拟一些可能失败的操作
                risky_operation()
                return "success"
            except:
                return "fallback_success"

        def risky_operation():
            raise ValueError("Simulated failure")

        manager.register_recovery_action("error_recovery", problematic_action)

        result = manager.execute_recovery("error_recovery")

        # 由于异常被捕获并返回fallback_success，函数应该正常执行
        assert result == "fallback_success"

    def test_multiple_recovery_actions_execution_order(self):
        """测试多个恢复动作的执行顺序"""
        manager = AutoRecoveryManager()

        execution_order = []

        def action1():
            execution_order.append(1)
            return "action1_result"

        def action2():
            execution_order.append(2)
            return "action2_result"

        def action3():
            execution_order.append(3)
            return "action3_result"

        manager.register_recovery_action("action1", action1)
        manager.register_recovery_action("action2", action2)
        manager.register_recovery_action("action3", action3)

        # 按不同顺序执行
        manager.execute_recovery("action2")
        manager.execute_recovery("action1")
        manager.execute_recovery("action3")
        manager.execute_recovery("action2")  # 再次执行action2

        assert execution_order == [2, 1, 3, 2]

    def test_recovery_manager_state_persistence(self):
        """测试恢复管理器状态持久性"""
        manager = AutoRecoveryManager()

        # 注册一些动作
        actions = {
            "restart_service": lambda: "service_restarted",
            "clear_cache": lambda: "cache_cleared",
            "reset_connections": lambda: "connections_reset"
        }

        for name, action in actions.items():
            manager.register_recovery_action(name, action)

        # 验证状态保持
        assert len(manager.recovery_actions) == 3
        for name in actions.keys():
            assert name in manager.recovery_actions

        # 执行一些动作
        results = []
        for name in actions.keys():
            result = manager.execute_recovery(name)
            results.append(result)

        # 验证结果
        expected_results = ["service_restarted", "cache_cleared", "connections_reset"]
        assert results == expected_results

        # 验证状态仍然保持
        assert len(manager.recovery_actions) == 3