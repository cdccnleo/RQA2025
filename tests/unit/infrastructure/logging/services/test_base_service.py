"""
测试基础日志服务

覆盖 base_service.py 中的 ILogService 接口和 BaseService 类
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from src.infrastructure.logging.services.base_service import ILogService, BaseService, TestableBaseService
from src.infrastructure.logging.core.exceptions import ResourceError


class TestILogService:
    """ILogService 接口测试"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # 不能直接实例化抽象类
        with pytest.raises(TypeError):
            ILogService()

    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        # 检查必需的方法是否存在
        required_methods = ['start', 'stop', 'restart', 'get_status', 'get_info']

        for method_name in required_methods:
            assert hasattr(ILogService, method_name), f"Missing method: {method_name}"

            # 检查方法是否是抽象的
            method = getattr(ILogService, method_name)
            assert hasattr(method, '__isabstractmethod__'), f"Method {method_name} should be abstract"


class TestBaseService:
    """BaseService 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        service = TestableBaseService()

        assert service.name == "test_service"
        assert service.config == {}
        assert service.enabled == True
        assert service.auto_start == False
        assert service.is_running == False
        assert service.start_time is None
        assert service.stop_time is None
        assert service.restart_count == 0
        assert service.total_requests == 0
        assert service.success_count == 0
        assert service.error_count == 0

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'enabled': False,
            'auto_start': True,
            'custom_param': 'value'
        }
        service = TestableBaseService("custom_service", config)

        assert service.name == "custom_service"
        assert service.config == config
        assert service.enabled == False
        assert service.auto_start == True

    def test_init_auto_start(self):
        """测试自动启动"""
        config = {'auto_start': True, 'enabled': True}
        with patch.object(TestableBaseService, '_start', return_value=True) as mock_start:
            service = TestableBaseService("auto_service", config)

            assert service.is_running == True
            assert service.start_time is not None
            mock_start.assert_called_once()

    def test_start_disabled_service(self):
        """测试启动禁用服务"""
        service = TestableBaseService()
        service.enabled = False

        result = service.start()
        assert result == False

    def test_start_already_running(self):
        """测试启动已运行的服务"""
        service = TestableBaseService()
        service.is_running = True

        result = service.start()
        assert result == True

    def test_start_success(self):
        """测试成功启动"""
        service = TestableBaseService()
        service.enabled = True
        service.is_running = False

        result = service.start()

        assert result == True
        assert service.is_running == True
        assert service.start_time is not None
        assert service.stop_time is None

    def test_start_failure(self):
        """测试启动失败"""
        service = TestableBaseService()
        service.enabled = True

        with patch.object(service, '_start', side_effect=Exception("Start failed")):
            with pytest.raises(ResourceError) as exc_info:
                service.start()

            assert "Failed to start service test_service: Start failed" in str(exc_info.value)
            assert service.is_running == False

    def test_stop_success(self):
        """测试成功停止"""
        service = TestableBaseService()
        service.is_running = True
        service.start_time = datetime.now()

        result = service.stop()

        assert result == True
        assert service.is_running == False
        assert service.stop_time is not None

    def test_stop_failure(self):
        """测试停止失败"""
        service = TestableBaseService()
        service.is_running = True

        with patch.object(service, '_stop', side_effect=Exception("Stop failed")):
            with pytest.raises(ResourceError) as exc_info:
                service.stop()

            assert "Failed to stop service test_service: Stop failed" in str(exc_info.value)

    def test_restart_disabled_service(self):
        """测试重启禁用服务"""
        service = TestableBaseService()
        service.enabled = False

        result = service.restart()
        assert result == False

    def test_restart_success(self):
        """测试成功重启"""
        service = TestableBaseService()
        service.enabled = True
        service.is_running = True
        service.start_time = datetime.now()
        initial_restart_count = service.restart_count

        result = service.restart()

        assert result == True
        assert service.is_running == True
        assert service.start_time is not None
        assert service.restart_count == initial_restart_count + 1

    def test_restart_failure(self):
        """测试重启失败"""
        service = TestableBaseService()
        service.enabled = True

        with patch.object(service, '_start', side_effect=Exception("Restart failed")):
            with pytest.raises(ResourceError) as exc_info:
                service.restart()

            assert "Failed to restart service test_service: Restart failed" in str(exc_info.value)

    def test_get_status_success(self):
        """测试获取状态成功"""
        service = TestableBaseService()
        service.is_running = True
        service.start_time = datetime.now()
        service.total_requests = 100
        service.success_count = 80
        service.error_count = 20
        service.restart_count = 2

        status = service.get_status()

        assert status['name'] == 'test_service'
        assert status['enabled'] == True
        assert status['is_running'] == True
        assert 'start_time' in status
        assert 'uptime' in status
        assert status['restart_count'] == 2
        assert status['total_requests'] == 100
        assert status['success_count'] == 80
        assert status['error_count'] == 20
        assert status['error_rate'] == 20.0  # 20/100 * 100
        assert status['type'] == 'TestableBaseService'

    def test_get_status_error_handling(self):
        """测试状态获取错误处理"""
        service = TestableBaseService()

        with patch.object(service, '_get_status', side_effect=Exception("Status error")):
            status = service.get_status()

            assert status['name'] == 'test_service'
            assert status['enabled'] == True
            assert status['is_running'] == False
            assert status['error'] == 'Status error'
            assert status['status'] == 'error'

    def test_get_info_success(self):
        """测试获取信息成功"""
        service = TestableBaseService()
        service.config = {'param': 'value'}

        info = service.get_info()

        assert info['name'] == 'test_service'
        assert info['type'] == 'TestableBaseService'
        assert info['version'] == '1.0.0'
        assert 'description' in info
        assert info['config'] == {'param': 'value'}

    def test_get_info_error_handling(self):
        """测试信息获取错误处理"""
        service = TestableBaseService()

        with patch.object(service, '_get_info', side_effect=Exception("Info error")):
            info = service.get_info()

            assert info['name'] == 'test_service'
            assert info['type'] == 'TestableBaseService'
            assert info['error'] == 'Info error'

    def test_record_request_success(self):
        """测试记录成功请求"""
        service = TestableBaseService()

        service._record_request(success=True)

        assert service.total_requests == 1
        assert service.success_count == 1
        assert service.error_count == 0

    def test_record_request_failure(self):
        """测试记录失败请求"""
        service = TestableBaseService()

        service._record_request(success=False)

        assert service.total_requests == 1
        assert service.success_count == 0
        assert service.error_count == 1

    def test_abstract_methods(self):
        """测试抽象方法"""
        service = TestableBaseService()

        # 这些方法在TestableBaseService中已经实现，不应该抛出NotImplementedError
        assert service._start() == True
        assert service._stop() == True
        assert isinstance(service._get_status(), dict)
        assert isinstance(service._get_info(), dict)


class TestTestableBaseService:
    """TestableBaseService 测试"""

    def test_instantiation(self):
        """测试实例化"""
        service = TestableBaseService("test_name", {"enabled": False})

        assert service.name == "test_name"
        assert service.config["enabled"] == False

    def test_start_implementation(self):
        """测试启动实现"""
        service = TestableBaseService()
        result = service._start()
        assert result == True

    def test_stop_implementation(self):
        """测试停止实现"""
        service = TestableBaseService()
        result = service._stop()
        assert result == True

    def test_get_status_implementation(self):
        """测试状态获取实现"""
        service = TestableBaseService()
        status = service._get_status()

        assert status["status"] == "stopped"
        assert status["name"] == "test_service"
        assert status["enabled"] == True
        assert "timestamp" in status

    def test_get_info_implementation(self):
        """测试信息获取实现"""
        service = TestableBaseService()
        info = service._get_info()

        assert info["service_name"] == "test_service"
        assert info["service_type"] == "TestableBaseService"
        assert info["config"] == {}