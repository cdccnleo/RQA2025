#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试业务流程状态机

测试目标：提升business_process/state_machine/state_machine.py的覆盖率到100%
"""

import pytest

# 尝试导入所需模块
try:
    from src.core.business_process.state_machine.state_machine import BusinessProcessStateMachine
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.core.business_process.state_machine.state_machine import BusinessProcessStateMachine
from src.core.business_process.config.enums import BusinessProcessState
from src.core.business_process.models.models import ProcessConfig, ProcessInstance


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestBusinessProcessStateMachine:
    """测试业务流程状态机"""

    @pytest.fixture
    def process_config(self):
        """创建流程配置实例"""
        return ProcessConfig(
            process_id="test_process_001",
            process_name="Test Process",
            description="A test process configuration"
        )

    @pytest.fixture
    def state_machine(self, process_config):
        """创建状态机实例"""
        return BusinessProcessStateMachine(process_config)

    @pytest.fixture
    def process_instance(self):
        """创建流程实例"""
        return ProcessInstance(
            instance_id="instance_001",
            process_id="test_process_001",
            state=BusinessProcessState.CREATED
        )

    def test_state_machine_initialization(self, state_machine, process_config):
        """测试状态机初始化"""
        assert state_machine.process_config == process_config
        assert state_machine.state_enter_time is None
        assert isinstance(state_machine.state_timeouts, dict)
        assert isinstance(state_machine.state_listeners, dict)
        assert isinstance(state_machine.transition_hooks, dict)

    def test_initialize_state_timeouts(self, state_machine):
        """测试初始化状态超时"""
        state_machine._initialize_state_timeouts()

        # 检查是否有超时配置
        assert len(state_machine.state_timeouts) > 0

        # 检查一些关键状态的超时
        assert BusinessProcessState.DATA_COLLECTING in state_machine.state_timeouts
        assert BusinessProcessState.EXECUTING in state_machine.state_timeouts
        assert BusinessProcessState.MONITORING in state_machine.state_timeouts

    def test_initialize_state_timeouts_with_config(self, process_config, state_machine):
        """测试使用配置初始化状态超时"""
        # 设置配置中的超时
        process_config.state_timeouts = {
            BusinessProcessState.DATA_COLLECTING: 120,
            BusinessProcessState.EXECUTING: 180
        }

        state_machine._initialize_state_timeouts()

        assert state_machine.state_timeouts[BusinessProcessState.DATA_COLLECTING] == 120
        assert state_machine.state_timeouts[BusinessProcessState.EXECUTING] == 180

    def test_add_state_listener(self, state_machine):
        """测试添加状态监听器"""
        def test_listener(state, instance):
            pass

        state_machine.add_state_listener(BusinessProcessState.RUNNING, test_listener)

        assert test_listener in state_machine.state_listeners[BusinessProcessState.RUNNING]

    def test_remove_state_listener(self, state_machine):
        """测试移除状态监听器"""
        def test_listener(state, instance):
            pass

        state_machine.add_state_listener(BusinessProcessState.RUNNING, test_listener)
        assert test_listener in state_machine.state_listeners[BusinessProcessState.RUNNING]

        state_machine.remove_state_listener(BusinessProcessState.RUNNING, test_listener)
        assert test_listener not in state_machine.state_listeners[BusinessProcessState.RUNNING]

    def test_add_transition_hook(self, state_machine):
        """测试添加转换钩子"""
        def test_hook(from_state, to_state, instance):
            pass

        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )

        transition_key = (BusinessProcessState.CREATED, BusinessProcessState.RUNNING)
        assert test_hook in state_machine.transition_hooks[transition_key]

    def test_remove_transition_hook(self, state_machine):
        """测试移除转换钩子"""
        def test_hook(from_state, to_state, instance):
            pass

        transition_key = (BusinessProcessState.CREATED, BusinessProcessState.RUNNING)
        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )

        state_machine.remove_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )

        assert test_hook not in state_machine.transition_hooks[transition_key]

    def test_start_process(self, state_machine, process_instance):
        """测试开始流程"""
        result = state_machine.start_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.RUNNING
        assert state_machine.state_enter_time is not None

    def test_start_process_already_running(self, state_machine, process_instance):
        """测试开始已运行的流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.start_process(process_instance)

        assert result == False

    def test_stop_process(self, state_machine, process_instance):
        """测试停止流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.stop_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.STOPPED

    def test_stop_process_not_running(self, state_machine, process_instance):
        """测试停止未运行的流程"""
        process_instance.state = BusinessProcessState.CREATED

        result = state_machine.stop_process(process_instance)

        assert result == False

    def test_complete_process(self, state_machine, process_instance):
        """测试完成流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.complete_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.COMPLETED

    def test_complete_process_not_running(self, state_machine, process_instance):
        """测试完成未运行的流程"""
        process_instance.state = BusinessProcessState.CREATED

        result = state_machine.complete_process(process_instance)

        assert result == False

    def test_fail_process(self, state_machine, process_instance):
        """测试失败流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.fail_process(process_instance, "Test failure")

        assert result == True
        assert process_instance.state == BusinessProcessState.ERROR
        assert process_instance.error_message == "Test failure"

    def test_fail_process_not_running(self, state_machine, process_instance):
        """测试失败未运行的流程"""
        process_instance.state = BusinessProcessState.CREATED

        result = state_machine.fail_process(process_instance, "Test failure")

        assert result == False

    def test_pause_process(self, state_machine, process_instance):
        """测试暂停流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.pause_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.PAUSED

    def test_pause_process_not_running(self, state_machine, process_instance):
        """测试暂停未运行的流程"""
        process_instance.state = BusinessProcessState.CREATED

        result = state_machine.pause_process(process_instance)

        assert result == False

    def test_resume_process(self, state_machine, process_instance):
        """测试恢复流程"""
        process_instance.state = BusinessProcessState.PAUSED

        result = state_machine.resume_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.RUNNING

    def test_resume_process_not_paused(self, state_machine, process_instance):
        """测试恢复未暂停的流程"""
        process_instance.state = BusinessProcessState.CREATED

        result = state_machine.resume_process(process_instance)

        assert result == False

    def test_reset_process(self, state_machine, process_instance):
        """测试重置流程"""
        process_instance.state = BusinessProcessState.ERROR

        result = state_machine.reset_process(process_instance)

        assert result == True
        assert process_instance.state == BusinessProcessState.CREATED
        assert process_instance.error_message == ""

    def test_reset_process_running(self, state_machine, process_instance):
        """测试重置运行中的流程"""
        process_instance.state = BusinessProcessState.RUNNING

        result = state_machine.reset_process(process_instance)

        assert result == False

    def test_get_current_state(self, state_machine, process_instance):
        """测试获取当前状态"""
        state = state_machine.get_current_state(process_instance)

        assert state == process_instance.state

    def test_get_state_enter_time(self, state_machine):
        """测试获取状态进入时间"""
        enter_time = state_machine.get_state_enter_time()

        assert enter_time == state_machine.state_enter_time

    def test_get_state_timeouts(self, state_machine):
        """测试获取状态超时"""
        timeouts = state_machine.get_state_timeouts()

        assert timeouts == state_machine.state_timeouts

    def test_is_state_timeout(self, state_machine):
        """测试检查状态超时"""
        # 设置进入时间为过去
        past_time = datetime.now().replace(year=2020)
        state_machine.state_enter_time = past_time

        # 设置超时为1秒
        state_machine.state_timeouts[BusinessProcessState.RUNNING] = 1

        is_timeout = state_machine.is_state_timeout(BusinessProcessState.RUNNING)

        assert is_timeout == True

    def test_is_state_timeout_no_enter_time(self, state_machine):
        """测试检查状态超时 - 无进入时间"""
        state_machine.state_enter_time = None

        is_timeout = state_machine.is_state_timeout(BusinessProcessState.RUNNING)

        assert is_timeout == False

    def test_is_state_timeout_not_expired(self, state_machine):
        """测试检查状态超时 - 未过期"""
        # 设置进入时间为现在
        state_machine.state_enter_time = datetime.now()

        # 设置超时为很长
        state_machine.state_timeouts[BusinessProcessState.RUNNING] = 3600  # 1小时

        is_timeout = state_machine.is_state_timeout(BusinessProcessState.RUNNING)

        assert is_timeout == False

    def test_check_state_transitions(self, state_machine, process_instance):
        """测试检查状态转换"""
        # 测试有效转换
        valid_transitions = state_machine.check_state_transitions(
            process_instance.state,
            BusinessProcessState.RUNNING
        )

        assert valid_transitions == True

        # 测试无效转换
        invalid_transitions = state_machine.check_state_transitions(
            BusinessProcessState.CREATED,
            BusinessProcessState.COMPLETED  # 不能直接从CREATED到COMPLETED
        )

        assert invalid_transitions == False

    def test_get_allowed_transitions(self, state_machine):
        """测试获取允许的转换"""
        transitions = state_machine.get_allowed_transitions(BusinessProcessState.CREATED)

        assert isinstance(transitions, list)
        assert BusinessProcessState.RUNNING in transitions

    def test_get_transition_history(self, state_machine, process_instance):
        """测试获取转换历史"""
        # 先进行一些状态转换
        state_machine.start_process(process_instance)
        state_machine.pause_process(process_instance)
        state_machine.resume_process(process_instance)
        state_machine.complete_process(process_instance)

        history = state_machine.get_transition_history(process_instance)

        assert isinstance(history, list)
        assert len(history) >= 4  # 至少有4次转换

    def test_notify_state_listeners(self, state_machine, process_instance):
        """测试通知状态监听器"""
        listener_called = []

        def test_listener(state, instance):
            listener_called.append((state, instance.instance_id))

        state_machine.add_state_listener(BusinessProcessState.RUNNING, test_listener)

        state_machine.start_process(process_instance)

        assert len(listener_called) == 1
        assert listener_called[0][0] == BusinessProcessState.RUNNING
        assert listener_called[0][1] == process_instance.instance_id

    def test_notify_transition_hooks(self, state_machine, process_instance):
        """测试通知转换钩子"""
        hook_called = []

        def test_hook(from_state, to_state, instance):
            hook_called.append((from_state, to_state, instance.instance_id))

        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )

        state_machine.start_process(process_instance)

        assert len(hook_called) == 1
        assert hook_called[0][0] == BusinessProcessState.CREATED
        assert hook_called[0][1] == BusinessProcessState.RUNNING
        assert hook_called[0][2] == process_instance.instance_id

    def test_validate_process_instance(self, state_machine, process_instance):
        """测试验证流程实例"""
        is_valid = state_machine.validate_process_instance(process_instance)

        assert is_valid == True

    def test_validate_process_instance_invalid(self, state_machine):
        """测试验证无效的流程实例"""
        invalid_instance = ProcessInstance(
            instance_id="",
            process_id="test_process_001"
        )

        is_valid = state_machine.validate_process_instance(invalid_instance)

        assert is_valid == False

    def test_get_process_metrics(self, state_machine, process_instance):
        """测试获取流程指标"""
        # 先进行一些操作
        state_machine.start_process(process_instance)
        state_machine.pause_process(process_instance)
        state_machine.resume_process(process_instance)
        state_machine.complete_process(process_instance)

        metrics = state_machine.get_process_metrics(process_instance)

        assert isinstance(metrics, dict)
        assert "total_transitions" in metrics
        assert "total_time" in metrics
        assert metrics["total_transitions"] >= 4

    def test_get_state_machine_info(self, state_machine):
        """测试获取状态机信息"""
        info = state_machine.get_state_machine_info()

        assert isinstance(info, dict)
        assert "process_config" in info
        assert "state_timeouts_count" in info
        assert "listeners_count" in info
        assert "hooks_count" in info

    def test_clear_state_listeners(self, state_machine):
        """测试清空状态监听器"""
        def test_listener(state, instance):
            pass

        state_machine.add_state_listener(BusinessProcessState.RUNNING, test_listener)
        assert len(state_machine.state_listeners[BusinessProcessState.RUNNING]) == 1

        state_machine.clear_state_listeners(BusinessProcessState.RUNNING)
        assert len(state_machine.state_listeners[BusinessProcessState.RUNNING]) == 0

    def test_clear_transition_hooks(self, state_machine):
        """测试清空转换钩子"""
        def test_hook(from_state, to_state, instance):
            pass

        transition_key = (BusinessProcessState.CREATED, BusinessProcessState.RUNNING)
        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )
        assert len(state_machine.transition_hooks[transition_key]) == 1

        state_machine.clear_transition_hooks(BusinessProcessState.CREATED, BusinessProcessState.RUNNING)
        assert len(state_machine.transition_hooks[transition_key]) == 0


class TestBusinessProcessStateMachineIntegration:
    """测试业务流程状态机集成场景"""

    @pytest.fixture
    def process_config(self):
        """创建流程配置"""
        return ProcessConfig(
            process_id="integration_test",
            process_name="Integration Test Process"
        )

    @pytest.fixture
    def state_machine(self, process_config):
        """创建状态机"""
        return BusinessProcessStateMachine(process_config)

    @pytest.fixture
    def process_instance(self):
        """创建流程实例"""
        return ProcessInstance(
            instance_id="integration_instance",
            process_id="integration_test",
            state=BusinessProcessState.CREATED
        )

    def test_complete_process_lifecycle(self, state_machine, process_instance):
        """测试完整流程生命周期"""
        # 1. 开始流程
        assert state_machine.start_process(process_instance)
        assert process_instance.state == BusinessProcessState.RUNNING

        # 2. 暂停流程
        assert state_machine.pause_process(process_instance)
        assert process_instance.state == BusinessProcessState.PAUSED

        # 3. 恢复流程
        assert state_machine.resume_process(process_instance)
        assert process_instance.state == BusinessProcessState.RUNNING

        # 4. 完成流程
        assert state_machine.complete_process(process_instance)
        assert process_instance.state == BusinessProcessState.COMPLETED

    def test_error_handling_and_recovery(self, state_machine, process_instance):
        """测试错误处理和恢复"""
        # 1. 开始流程
        state_machine.start_process(process_instance)
        assert process_instance.state == BusinessProcessState.RUNNING

        # 2. 模拟失败
        assert state_machine.fail_process(process_instance, "Test error")
        assert process_instance.state == BusinessProcessState.ERROR
        assert process_instance.error_message == "Test error"

        # 3. 重置流程
        assert state_machine.reset_process(process_instance)
        assert process_instance.state == BusinessProcessState.CREATED
        assert process_instance.error_message == ""

        # 4. 重新开始
        assert state_machine.start_process(process_instance)
        assert process_instance.state == BusinessProcessState.RUNNING

    def test_state_timeout_handling(self, state_machine, process_instance):
        """测试状态超时处理"""
        # 设置短超时
        state_machine.state_timeouts[BusinessProcessState.RUNNING] = 1  # 1秒

        # 开始流程
        state_machine.start_process(process_instance)
        assert process_instance.state == BusinessProcessState.RUNNING

        # 检查是否超时（应该没有超时，因为刚开始）
        assert not state_machine.is_state_timeout(BusinessProcessState.RUNNING)

        # 手动设置旧的进入时间来模拟超时
        import time
        from datetime import timedelta
        state_machine.state_enter_time = datetime.now() - timedelta(seconds=2)

        # 现在应该超时
        assert state_machine.is_state_timeout(BusinessProcessState.RUNNING)

    def test_listener_and_hook_integration(self, state_machine, process_instance):
        """测试监听器和钩子集成"""
        events = []

        # 添加状态监听器
        def state_listener(state, instance):
            events.append(f"state_changed:{state.value}")

        state_machine.add_state_listener(BusinessProcessState.RUNNING, state_listener)
        state_machine.add_state_listener(BusinessProcessState.COMPLETED, state_listener)

        # 添加转换钩子
        def transition_hook(from_state, to_state, instance):
            events.append(f"transition:{from_state.value}->{to_state.value}")

        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            transition_hook
        )
        state_machine.add_transition_hook(
            BusinessProcessState.RUNNING,
            BusinessProcessState.COMPLETED,
            transition_hook
        )

        # 执行状态转换
        state_machine.start_process(process_instance)
        state_machine.complete_process(process_instance)

        # 检查事件顺序
        assert "transition:created->running" in events
        assert "state_changed:running" in events
        assert "transition:running->completed" in events
        assert "state_changed:completed" in events

    def test_concurrent_state_machine_operations(self, process_config):
        """测试并发状态机操作"""
        import threading
        import time

        results = []
        errors = []

        def operate_state_machine(instance_id):
            try:
                # 为每个线程创建独立的实例
                instance = ProcessInstance(
                    instance_id=f"concurrent_{instance_id}",
                    process_id="concurrent_test",
                    state=BusinessProcessState.CREATED
                )

                machine = BusinessProcessStateMachine(process_config)

                # 执行操作序列
                machine.start_process(instance)
                time.sleep(0.01)
                machine.pause_process(instance)
                time.sleep(0.01)
                machine.resume_process(instance)
                time.sleep(0.01)
                machine.complete_process(instance)

                results.append(f"instance_{instance_id}_completed")

            except Exception as e:
                errors.append(f"instance_{instance_id}_error: {str(e)}")

        # 创建多个线程并发操作
        threads = []
        for i in range(5):
            thread = threading.Thread(target=operate_state_machine, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有实例都成功完成
        assert len(results) == 5
        assert all("completed" in r for r in results)

    def test_state_machine_monitoring_and_metrics(self, state_machine, process_instance):
        """测试状态机监控和指标"""
        # 执行一系列操作
        operations = [
            lambda: state_machine.start_process(process_instance),
            lambda: state_machine.pause_process(process_instance),
            lambda: state_machine.resume_process(process_instance),
            lambda: state_machine.complete_process(process_instance)
        ]

        for operation in operations:
            operation()

        # 获取指标
        metrics = state_machine.get_process_metrics(process_instance)

        assert metrics["total_transitions"] >= 3
        assert isinstance(metrics["total_time"], (int, float))
        assert "average_transition_time" in metrics

        # 获取状态机信息
        info = state_machine.get_state_machine_info()

        assert info["process_config"].process_id == "integration_test"
        assert info["state_timeouts_count"] > 0
        assert "listeners_count" in info
        assert "hooks_count" in info

    def test_state_machine_configuration_and_customization(self, process_config, state_machine):
        """测试状态机配置和定制"""
        # 测试自定义超时配置
        custom_timeouts = {
            BusinessProcessState.DATA_COLLECTING: 120,
            BusinessProcessState.EXECUTING: 180,
            BusinessProcessState.MONITORING: 600
        }

        process_config.state_timeouts = custom_timeouts
        state_machine._initialize_state_timeouts()

        assert state_machine.state_timeouts[BusinessProcessState.DATA_COLLECTING] == 120
        assert state_machine.state_timeouts[BusinessProcessState.EXECUTING] == 180
        assert state_machine.state_timeouts[BusinessProcessState.MONITORING] == 600

        # 测试获取配置
        timeouts = state_machine.get_state_timeouts()
        assert timeouts == custom_timeouts

    def test_state_machine_validation_and_error_handling(self, state_machine):
        """测试状态机验证和错误处理"""
        # 测试有效实例
        valid_instance = ProcessInstance(
            instance_id="valid_instance",
            process_id="test_process",
            state=BusinessProcessState.CREATED
        )
        assert state_machine.validate_process_instance(valid_instance)

        # 测试无效实例
        invalid_instances = [
            ProcessInstance(instance_id="", process_id="test"),  # 空ID
            ProcessInstance(instance_id="test", process_id="", state=BusinessProcessState.CREATED),  # 空进程ID
        ]

        for invalid_instance in invalid_instances:
            assert not state_machine.validate_process_instance(invalid_instance)

    def test_state_machine_cleanup_and_resource_management(self, state_machine, process_instance):
        """测试状态机清理和资源管理"""
        # 添加一些监听器和钩子
        def test_listener(state, instance):
            pass

        def test_hook(from_state, to_state, instance):
            pass

        state_machine.add_state_listener(BusinessProcessState.RUNNING, test_listener)
        state_machine.add_transition_hook(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING,
            test_hook
        )

        # 执行一些操作
        state_machine.start_process(process_instance)

        # 清理监听器
        state_machine.clear_state_listeners(BusinessProcessState.RUNNING)
        assert len(state_machine.state_listeners[BusinessProcessState.RUNNING]) == 0

        # 清理钩子
        state_machine.clear_transition_hooks(
            BusinessProcessState.CREATED,
            BusinessProcessState.RUNNING
        )
        transition_key = (BusinessProcessState.CREATED, BusinessProcessState.RUNNING)
        assert len(state_machine.transition_hooks[transition_key]) == 0

    def test_state_machine_performance_under_load(self, process_config):
        """测试状态机负载下的性能"""
        import time

        start_time = time.time()

        # 创建多个状态机实例并执行操作
        for i in range(10):
            machine = BusinessProcessStateMachine(process_config)
            instance = ProcessInstance(
                instance_id=f"perf_test_{i}",
                process_id="perf_test",
                state=BusinessProcessState.CREATED
            )

            # 执行标准操作序列
            machine.start_process(instance)
            machine.pause_process(instance)
            machine.resume_process(instance)
            machine.complete_process(instance)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（应该在合理时间内完成）
        assert total_time < 5.0  # 10个状态机应该在5秒内完成

    def test_state_machine_state_persistence_and_recovery(self, state_machine, process_instance):
        """测试状态机状态持久化和恢复"""
        # 执行一些状态转换
        state_machine.start_process(process_instance)
        state_machine.pause_process(process_instance)

        # 模拟持久化当前状态
        persisted_state = {
            "instance_state": process_instance.state,
            "state_enter_time": state_machine.state_enter_time,
            "transition_history": state_machine.get_transition_history(process_instance)
        }

        # 模拟从持久化状态恢复
        recovered_instance = ProcessInstance(
            instance_id=process_instance.instance_id,
            process_id=process_instance.process_id,
            state=persisted_state["instance_state"]
        )

        recovered_machine = BusinessProcessStateMachine(state_machine.process_config)
        recovered_machine.state_enter_time = persisted_state["state_enter_time"]

        # 验证恢复的状态
        assert recovered_instance.state == BusinessProcessState.PAUSED
        assert recovered_machine.get_current_state(recovered_instance) == BusinessProcessState.PAUSED

        # 验证可以继续操作
        assert recovered_machine.resume_process(recovered_instance)
        assert recovered_instance.state == BusinessProcessState.RUNNING


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
