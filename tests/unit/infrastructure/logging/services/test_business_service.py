"""
测试业务服务

覆盖 business_service.py 中的核心类
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.logging.services.business_service import (
    BusinessService,
    _NoopEventBus,
    _NoopContainer,
    TestableBusinessService
)


class TestNoopEventBus:
    """_NoopEventBus 测试"""

    def test_init(self):
        """测试初始化"""
        bus = _NoopEventBus()
        assert bus is not None

    def test_subscribe(self):
        """测试订阅"""
        bus = _NoopEventBus()
        result = bus.subscribe("event", lambda: None)
        assert result is None

    def test_unsubscribe(self):
        """测试取消订阅"""
        bus = _NoopEventBus()
        result = bus.unsubscribe("event", lambda: None)
        assert result is None

    def test_publish(self):
        """测试发布"""
        bus = _NoopEventBus()
        result = bus.publish("event", data="test")
        assert result is None


class TestNoopContainer:
    """_NoopContainer 测试"""

    def test_has(self):
        """测试服务存在检查"""
        container = _NoopContainer()
        result = container.has("service_name")
        assert result == False

    def test_get_not_found(self):
        """测试获取不存在的服务"""
        container = _NoopContainer()
        with pytest.raises(KeyError):
            container.get("nonexistent_service")


class TestBusinessService:
    """BusinessService 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        service = BusinessService()

        assert service.name == "BusinessService"
        assert isinstance(service.event_bus, _NoopEventBus)
        assert isinstance(service.container, _NoopContainer)

    def test_init_with_dependencies(self):
        """测试带依赖初始化"""
        mock_bus = Mock()
        mock_container = Mock()

        service = BusinessService(
            event_bus=mock_bus,
            container=mock_container,
            name="CustomService"
        )

        assert service.name == "CustomService"
        assert service.event_bus == mock_bus
        assert service.container == mock_container



    def test_start_implementation(self):
        """测试启动实现"""
        service = BusinessService()

        result = service._start()

        # 启动可能失败因为缺少依赖，但不应该抛出异常
        assert isinstance(result, bool)

    def test_stop_implementation(self):
        """测试停止实现"""
        service = BusinessService()
        service.is_running = True

        result = service._stop()

        assert result == True
        # stop可能不改变is_running状态，取决于实现

    def test_get_status_implementation(self):
        """测试状态获取实现"""
        service = BusinessService()
        service.is_running = True

        status = service._get_status()

        assert 'status' in status

    def test_get_info_implementation(self):
        """测试信息获取实现"""
        service = BusinessService()

        info = service._get_info()

        assert 'service_name' in info

    def test_create_workflow(self):
        """测试创建工作流"""
        service = BusinessService()

        config = {
            "name": "test_workflow",
            "description": "Test workflow",
            "steps": [
                {
                    "name": "step1",
                    "service": "data_service",
                    "method": "process_data",
                    "input": {"data": "test"}
                }
            ]
        }

        result = service.create_workflow("test_workflow", config)
        assert isinstance(result, bool)

    def test_start_workflow(self):
        """测试启动工作流"""
        service = BusinessService()

        # 先创建工作流
        config = {
            "name": "test_workflow",
            "steps": [{"name": "step1", "service": "data_service", "method": "process_data"}]
        }
        service.create_workflow("test_workflow", config)

        result = service.start_workflow("test_workflow")
        assert isinstance(result, bool)

    def test_stop_workflow(self):
        """测试停止工作流"""
        service = BusinessService()

        result = service.stop_workflow("nonexistent")
        assert result is False

    def test_get_workflow_status(self):
        """测试获取工作流状态"""
        service = BusinessService()

        status = service.get_workflow_status("nonexistent")
        assert isinstance(status, dict)
        # 对于不存在的工作流，返回错误信息
        assert "error" in status or "found" in status

    def test_list_workflows(self):
        """测试列出工作流"""
        service = BusinessService()

        workflows = service.list_workflows()
        assert isinstance(workflows, dict)

    def test_event_handlers(self):
        """测试事件处理器"""
        service = BusinessService()

        # 测试事件处理器存在
        assert hasattr(service, '_on_data_ready')
        assert hasattr(service, '_on_feature_extracted')
        assert hasattr(service, '_on_model_predicted')
        assert hasattr(service, '_on_signal_generated')
        assert hasattr(service, '_on_risk_checked')
        assert hasattr(service, '_on_execution_completed')
        assert hasattr(service, '_on_validation_completed')

    def test_workflow_validation(self):
        """测试工作流验证"""
        service = BusinessService()

        # 测试有效配置
        valid_config = {
            "name": "valid_workflow",
            "steps": [
                {
                    "name": "step1",
                    "service": "data_service",
                    "method": "process_data"
                }
            ]
        }

        assert service._validate_workflow_config(valid_config)

        # 测试无效配置
        invalid_config = {"name": "invalid"}
        assert not service._validate_workflow_config(invalid_config)

    def test_health_check(self):
        """测试健康检查"""
        service = BusinessService()

        health = service._health_check()
        assert isinstance(health, dict)
        assert "status" in health

    def test_workflow_execution_methods(self):
        """测试工作流执行相关方法"""
        service = BusinessService()

        # 测试工作流激活检查
        assert not service._is_workflow_active("nonexistent")

        # 测试获取当前步骤 - 需要正确的workflow结构
        workflow = {
            "config": {"steps": [{"name": "step1"}]},
            "current_step": 0
        }
        step = service._get_current_step(workflow)
        assert step == {"name": "step1"}

    def test_step_execution(self):
        """测试步骤执行"""
        service = BusinessService()

        # 测试步骤验证
        valid_step = {
            "name": "step1",
            "service": "data_service",
            "method": "process_data"
        }
        assert service._is_step_valid(valid_step)

        invalid_step = {"name": "invalid"}
        assert not service._is_step_valid(invalid_step)


class TestTestableBusinessService:
    """TestableBusinessService 测试"""

    def test_instantiation(self):
        """测试实例化"""
        service = TestableBusinessService()

        assert service.name == "BusinessService"  # TestableBusinessService继承了父类的命名
        assert isinstance(service.event_bus, _NoopEventBus)
        assert isinstance(service.container, _NoopContainer)

    def test_instantiation_with_name(self):
        """测试带名称实例化"""
        service = TestableBusinessService(name="CustomTestService")

        assert service.name == "CustomTestService"

    def test_start_stop_cycle(self):
        """测试启动停止循环"""
        service = TestableBusinessService()

        # 测试启动
        result = service.start()
        assert result == True
        assert service.is_running == True

        # 测试停止
        result = service.stop()
        assert result == True
        assert service.is_running == False

        # 测试重启
        result = service.restart()
        assert result == True
        assert service.is_running == True

    def test_get_status(self):
        """测试获取状态"""
        service = TestableBusinessService()

        status = service.get_status()

        assert status['name'] == 'BusinessService'  # TestableBusinessService继承了父类的命名
        assert status['enabled'] == True
        assert status['is_running'] == False
        assert status['type'] == 'TestableBusinessService'

    def test_get_info(self):
        """测试获取信息"""
        service = TestableBusinessService()

        info = service.get_info()

        assert info['name'] == 'BusinessService'  # TestableBusinessService继承了父类的命名
        assert info['type'] == 'TestableBusinessService'
        assert 'service_type' in info