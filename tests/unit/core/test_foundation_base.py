#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心层基础组件

测试目标：提升foundation/base.py的覆盖率到100%
"""

import pytest
import time
import uuid
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime

from src.core.foundation.base import (
    ComponentStatus,
    ComponentHealth,
    ComponentInfo,
    BaseComponent,
    BaseService
)


class TestComponentStatus:
    """测试组件状态枚举"""

    def test_component_status_values(self):
        """测试组件状态枚举值"""
        assert ComponentStatus.UNKNOWN.value == "unknown"
        assert ComponentStatus.INITIALIZING.value == "initializing"
        assert ComponentStatus.INITIALIZED.value == "initialized"
        assert ComponentStatus.STARTING.value == "starting"
        assert ComponentStatus.RUNNING.value == "running"
        assert ComponentStatus.STOPPING.value == "stopping"
        assert ComponentStatus.STOPPED.value == "stopped"
        assert ComponentStatus.ERROR.value == "error"
        assert ComponentStatus.HEALTHY.value == "healthy"
        assert ComponentStatus.UNHEALTHY.value == "unhealthy"

    def test_component_status_all_values(self):
        """测试所有组件状态值"""
        expected_values = [
            "unknown", "initializing", "initialized", "starting",
            "running", "stopping", "stopped", "error", "healthy", "unhealthy"
        ]

        actual_values = [status.value for status in ComponentStatus]

        for expected in expected_values:
            assert expected in actual_values

        assert len(actual_values) == len(expected_values)

    def test_component_status_string_representation(self):
        """测试组件状态字符串表示"""
        assert str(ComponentStatus.RUNNING) == "ComponentStatus.RUNNING"
        assert repr(ComponentStatus.RUNNING) == "<ComponentStatus.RUNNING: 'running'>"


class TestComponentHealth:
    """测试组件健康状态枚举"""

    def test_component_health_values(self):
        """测试组件健康状态枚举值"""
        assert ComponentHealth.HEALTHY.value == "healthy"
        assert ComponentHealth.UNHEALTHY.value == "unhealthy"
        assert ComponentHealth.UNKNOWN.value == "unknown"

    def test_component_health_all_values(self):
        """测试所有组件健康状态值"""
        expected_values = ["healthy", "unhealthy", "unknown"]
        actual_values = [health.value for health in ComponentHealth]

        assert set(actual_values) == set(expected_values)
        assert len(actual_values) == 3


class TestComponentInfo:
    """测试组件信息数据类"""

    @pytest.fixture
    def component_info(self):
        """创建组件信息实例"""
        return ComponentInfo(
            name="TestComponent",
            version="2.1.0",
            description="A test component",
            status=ComponentStatus.RUNNING,
            health=ComponentHealth.HEALTHY
        )

    def test_component_info_creation(self, component_info):
        """测试组件信息创建"""
        assert component_info.name == "TestComponent"
        assert component_info.version == "2.1.0"
        assert component_info.description == "A test component"
        assert component_info.status == ComponentStatus.RUNNING
        assert component_info.health == ComponentHealth.HEALTHY

    def test_component_info_default_values(self):
        """测试组件信息默认值"""
        info = ComponentInfo(name="SimpleComponent")

        assert info.name == "SimpleComponent"
        assert info.version == "1.0.0"
        assert info.description == ""
        assert info.status == ComponentStatus.UNKNOWN
        assert info.health == ComponentHealth.UNKNOWN
        assert info.created_time is None

    def test_component_info_with_created_time(self):
        """测试组件信息带创建时间"""
        created_time = time.time()
        info = ComponentInfo(
            name="TimedComponent",
            created_time=created_time
        )

        assert info.created_time == created_time

    def test_component_info_immutability(self, component_info):
        """测试组件信息不可变性"""
        # ComponentInfo是dataclass，默认frozen=True，但这里没有设置frozen
        # 所以应该是可变的
        component_info.version = "3.0.0"
        assert component_info.version == "3.0.0"

    def test_component_info_equality(self):
        """测试组件信息相等性"""
        info1 = ComponentInfo(name="Test", version="1.0")
        info2 = ComponentInfo(name="Test", version="1.0")
        info3 = ComponentInfo(name="Test", version="2.0")

        assert info1 == info2
        assert info1 != info3

    def test_component_info_string_representation(self, component_info):
        """测试组件信息字符串表示"""
        str_repr = str(component_info)
        assert "TestComponent" in str_repr
        assert "2.1.0" in str_repr


class TestBaseComponent:
    """测试基础组件"""

    def test_base_component_is_abstract(self):
        """测试基础组件是抽象的"""
        with pytest.raises(TypeError):
            BaseComponent()

    @pytest.fixture
    def concrete_component(self):
        """创建具体的组件实现"""
        class ConcreteComponent(BaseComponent):
            def __init__(self, name="TestComponent", version="1.0.0", description="Test"):
                super().__init__(name, version, description)
                self.initialize_called = False
                self.start_called = False
                self.stop_called = False

            def initialize(self, config=None):
                self.initialize_called = True
                return True

            def start(self):
                self.start_called = True
                return True

            def stop(self):
                self.stop_called = True
                return True

        return ConcreteComponent()

    def test_concrete_component_creation(self, concrete_component):
        """测试具体组件创建"""
        assert concrete_component.name == "TestComponent"
        assert concrete_component.version == "1.0.0"
        assert concrete_component.description == "Test"
        assert concrete_component.status == ComponentStatus.UNKNOWN
        assert concrete_component.health == ComponentHealth.UNKNOWN

    def test_concrete_component_initialization(self, concrete_component):
        """测试具体组件初始化"""
        assert concrete_component.status == ComponentStatus.UNKNOWN

        result = concrete_component.initialize()

        assert result == True
        assert concrete_component.initialize_called == True
        assert concrete_component.status == ComponentStatus.INITIALIZED

    def test_concrete_component_start(self, concrete_component):
        """测试具体组件启动"""
        concrete_component.initialize()

        result = concrete_component.start()

        assert result == True
        assert concrete_component.start_called == True
        assert concrete_component.status == ComponentStatus.RUNNING

    def test_concrete_component_stop(self, concrete_component):
        """测试具体组件停止"""
        concrete_component.initialize()
        concrete_component.start()

        result = concrete_component.stop()

        assert result == True
        assert concrete_component.stop_called == True
        assert concrete_component.status == ComponentStatus.STOPPED

    def test_component_info_property(self, concrete_component):
        """测试组件信息属性"""
        info = concrete_component.info

        assert isinstance(info, ComponentInfo)
        assert info.name == "TestComponent"
        assert info.version == "1.0.0"
        assert info.description == "Test"
        assert info.status == ComponentStatus.UNKNOWN

    def test_component_info_update_after_operations(self, concrete_component):
        """测试操作后的组件信息更新"""
        concrete_component.initialize()
        info = concrete_component.info

        assert info.status == ComponentStatus.INITIALIZED

        concrete_component.start()
        info = concrete_component.info

        assert info.status == ComponentStatus.RUNNING

    def test_component_health_property(self, concrete_component):
        """测试组件健康属性"""
        # 默认应该是UNKNOWN
        assert concrete_component.health == ComponentHealth.UNKNOWN

        # 可以通过设置健康状态
        concrete_component._health = ComponentHealth.HEALTHY
        assert concrete_component.health == ComponentHealth.HEALTHY

    def test_component_unique_id(self, concrete_component):
        """测试组件唯一ID"""
        assert hasattr(concrete_component, '_component_id')
        assert isinstance(concrete_component._component_id, str)
        assert len(concrete_component._component_id) > 0

        # 不同实例应该有不同的ID
        another_component = type(concrete_component)("Another", "1.0.0", "Another")
        assert concrete_component._component_id != another_component._component_id

    def test_component_created_time(self, concrete_component):
        """测试组件创建时间"""
        assert hasattr(concrete_component, '_created_time')
        assert isinstance(concrete_component._created_time, float)

        # 创建时间应该在过去不久
        current_time = time.time()
        assert abs(current_time - concrete_component._created_time) < 1.0

    def test_component_error_handling(self, concrete_component):
        """测试组件错误处理"""
        # 设置错误状态
        concrete_component._status = ComponentStatus.ERROR

        assert concrete_component.status == ComponentStatus.ERROR

        # 测试健康状态
        concrete_component._health = ComponentHealth.UNHEALTHY
        assert concrete_component.health == ComponentHealth.UNHEALTHY


class TestBaseService:
    """测试基础服务"""

    def test_base_service_is_abstract(self):
        """测试基础服务是抽象的"""
        with pytest.raises(TypeError):
            BaseService("TestService", "1.0.0", "Test service")

    @pytest.fixture
    def concrete_service(self):
        """创建具体的服务实现"""
        class ConcreteService(BaseService):
            def __init__(self):
                super().__init__("TestService", "2.0.0", "A test service")
                self.process_called = False
                self.cleanup_called = False

            def process(self, data=None):
                self.process_called = True
                return {"result": "processed", "data": data}

            def cleanup(self):
                self.cleanup_called = True
                return True

        return ConcreteService()

    def test_concrete_service_creation(self, concrete_service):
        """测试具体服务创建"""
        assert concrete_service.name == "TestService"
        assert concrete_service.version == "2.0.0"
        assert concrete_service.description == "A test service"
        assert concrete_service.status == ComponentStatus.UNKNOWN

    def test_concrete_service_process(self, concrete_service):
        """测试具体服务处理"""
        test_data = {"input": "test"}

        result = concrete_service.process(test_data)

        assert concrete_service.process_called == True
        assert result == {"result": "processed", "data": test_data}

    def test_concrete_service_cleanup(self, concrete_service):
        """测试具体服务清理"""
        result = concrete_service.cleanup()

        assert concrete_service.cleanup_called == True
        assert result == True

    def test_service_inheritance_from_base_component(self, concrete_service):
        """测试服务继承自基础组件"""
        assert isinstance(concrete_service, BaseComponent)
        assert hasattr(concrete_service, 'initialize')
        assert hasattr(concrete_service, 'start')
        assert hasattr(concrete_service, 'stop')
        assert hasattr(concrete_service, 'info')

    def test_service_lifecycle(self, concrete_service):
        """测试服务生命周期"""
        # 初始化
        result = concrete_service.initialize()
        assert result == True
        assert concrete_service.status == ComponentStatus.INITIALIZED

        # 启动
        result = concrete_service.start()
        assert result == True
        assert concrete_service.status == ComponentStatus.RUNNING

        # 处理数据
        result = concrete_service.process({"action": "test"})
        assert result is not None

        # 清理
        result = concrete_service.cleanup()
        assert result == True

        # 停止
        result = concrete_service.stop()
        assert result == True
        assert concrete_service.status == ComponentStatus.STOPPED


class TestComponentIntegration:
    """测试组件集成场景"""

    def test_component_lifecycle_management(self):
        """测试组件生命周期管理"""
        class TestComponent(BaseComponent):
            def __init__(self):
                super().__init__("LifecycleTest", "1.0.0", "Lifecycle test component")
                self.operations = []

            def initialize(self, config=None):
                self.operations.append("initialize")
                return True

            def start(self):
                self.operations.append("start")
                return True

            def stop(self):
                self.operations.append("stop")
                return True

        component = TestComponent()

        # 测试完整生命周期
        assert component.initialize() == True
        assert component.status == ComponentStatus.INITIALIZED

        assert component.start() == True
        assert component.status == ComponentStatus.RUNNING

        assert component.stop() == True
        assert component.status == ComponentStatus.STOPPED

        # 验证操作顺序
        assert component.operations == ["initialize", "start", "stop"]

    def test_service_full_workflow(self):
        """测试服务完整工作流程"""
        class WorkflowService(BaseService):
            def __init__(self):
                super().__init__("WorkflowService", "1.0.0", "Workflow test service")
                self.workflow_data = []

            def process(self, data=None):
                self.workflow_data.append(f"processed_{data}")
                return {"status": "success", "processed": data}

            def cleanup(self):
                self.workflow_data.append("cleanup")
                return True

        service = WorkflowService()

        # 初始化并启动服务
        service.initialize()
        service.start()

        # 执行多个处理操作
        service.process("task1")
        service.process("task2")
        service.process("task3")

        # 验证处理结果
        assert len(service.workflow_data) == 3
        assert "processed_task1" in service.workflow_data
        assert "processed_task2" in service.workflow_data
        assert "processed_task3" in service.workflow_data

        # 清理服务
        service.cleanup()
        service.stop()

        # 验证清理操作
        assert service.workflow_data[-1] == "cleanup"
        assert service.status == ComponentStatus.STOPPED

    def test_component_error_recovery(self):
        """测试组件错误恢复"""
        class ErrorProneComponent(BaseComponent):
            def __init__(self):
                super().__init__("ErrorTest", "1.0.0", "Error recovery test")
                self.error_count = 0

            def initialize(self, config=None):
                if self.error_count < 2:  # 前两次初始化失败
                    self.error_count += 1
                    self._status = ComponentStatus.ERROR
                    return False
                return True

            def start(self):
                return True

            def stop(self):
                return True

        component = ErrorProneComponent()

        # 第一次初始化失败
        result = component.initialize()
        assert result == False
        assert component.status == ComponentStatus.ERROR

        # 第二次初始化失败
        result = component.initialize()
        assert result == False
        assert component.status == ComponentStatus.ERROR

        # 第三次初始化成功
        result = component.initialize()
        assert result == True
        assert component.status == ComponentStatus.INITIALIZED

    def test_multiple_components_coordination(self):
        """测试多组件协调"""
        class ProducerComponent(BaseComponent):
            def __init__(self):
                super().__init__("Producer", "1.0.0", "Data producer")
                self.data_produced = []

            def initialize(self, config=None):
                return True

            def start(self):
                return True

            def stop(self):
                return True

            def produce_data(self):
                data = f"data_{len(self.data_produced)}"
                self.data_produced.append(data)
                return data

        class ConsumerComponent(BaseComponent):
            def __init__(self):
                super().__init__("Consumer", "1.0.0", "Data consumer")
                self.data_consumed = []

            def initialize(self, config=None):
                return True

            def start(self):
                return True

            def stop(self):
                return True

            def consume_data(self, data):
                self.data_consumed.append(f"consumed_{data}")

        producer = ProducerComponent()
        consumer = ConsumerComponent()

        # 初始化组件
        producer.initialize()
        consumer.initialize()

        producer.start()
        consumer.start()

        # 生产和消费数据
        for i in range(5):
            data = producer.produce_data()
            consumer.consume_data(data)

        # 验证数据流
        assert len(producer.data_produced) == 5
        assert len(consumer.data_consumed) == 5

        for i in range(5):
            assert producer.data_produced[i] == f"data_{i}"
            assert consumer.data_consumed[i] == f"consumed_data_{i}"

        # 停止组件
        producer.stop()
        consumer.stop()

        assert producer.status == ComponentStatus.STOPPED
        assert consumer.status == ComponentStatus.STOPPED

    def test_component_performance_monitoring(self):
        """测试组件性能监控"""
        import time

        class MonitoredComponent(BaseComponent):
            def __init__(self):
                super().__init__("Monitored", "1.0.0", "Performance monitored component")
                self.operation_times = []

            def initialize(self, config=None):
                return True

            def start(self):
                return True

            def stop(self):
                return True

            def perform_operation(self, duration=0.01):
                start_time = time.time()
                time.sleep(duration)
                end_time = time.time()
                self.operation_times.append(end_time - start_time)
                return {"duration": end_time - start_time}

        component = MonitoredComponent()
        component.initialize()
        component.start()

        # 执行多个操作
        operations = [0.01, 0.02, 0.005, 0.015]
        for duration in operations:
            result = component.perform_operation(duration)
            assert result["duration"] >= duration

        # 验证性能数据
        assert len(component.operation_times) == len(operations)
        for recorded_time, expected_min in zip(component.operation_times, operations):
            assert recorded_time >= expected_min

        component.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
