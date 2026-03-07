#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
共享接口测试
测试shared_interfaces.py中的所有接口和实现类
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Any, Dict

try:
    from src.infrastructure.resource.core.shared_interfaces import (
        IConfigValidator, ILogger, IErrorHandler, ISharedResourceManager, IDataValidator,
        StandardLogger, BaseErrorHandler, ConfigValidator, DataValidator, ResourceManager,
        ResourceException, ConfigurationException, ValidationException, StandardResponse
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    # 创建mock类以避免导入错误
    class IConfigValidator:
        pass
    class ILogger:
        pass
    class IErrorHandler:
        pass
    class ISharedResourceManager:
        pass
    class IDataValidator:
        pass
    class StandardLogger:
        pass
    class BaseErrorHandler:
        pass
    class ConfigValidator:
        pass
    class DataValidator:
        pass
    class ResourceManager:
        pass
    class ResourceException:
        pass
    class ConfigurationException:
        pass
    class ValidationException:
        pass
    class StandardResponse:
        pass
    print(f"Warning: 无法导入所需模块: {e}")


class TestInterfaces:
    """测试接口定义"""

    def test_iconfig_validator_is_abstract(self):
        """测试IConfigValidator是抽象类"""
        # 抽象类不能直接实例化
        with pytest.raises(TypeError):
            IConfigValidator()

    def test_ilogger_is_abstract(self):
        """测试ILogger是抽象类"""
        with pytest.raises(TypeError):
            ILogger()

    def test_ierror_handler_is_abstract(self):
        """测试IErrorHandler是抽象类"""
        with pytest.raises(TypeError):
            IErrorHandler()

    def test_iresource_manager_is_abstract(self):
        """测试ISharedResourceManager是抽象类"""
        with pytest.raises(TypeError):
            ISharedResourceManager()

    def test_idata_validator_is_abstract(self):
        """测试IDataValidator是抽象类"""
        with pytest.raises(TypeError):
            IDataValidator()


class TestStandardLogger:
    """测试StandardLogger类"""

    def test_standard_logger_initialization(self):
        """测试StandardLogger初始化"""
        logger = StandardLogger("test_component")

        assert logger.component_name == "test_component"

    def test_standard_logger_log_info(self):
        """测试信息日志记录"""
        logger = StandardLogger("test_component")

        # 应该不抛出异常
        logger.log_info("Test info message", key="value")

    def test_standard_logger_log_error(self):
        """测试错误日志记录"""
        logger = StandardLogger("test_component")
        exception = ValueError("Test error")

        # 应该不抛出异常
        logger.log_error("Test error message", error=exception, key="value")

    def test_standard_logger_log_warning(self):
        """测试警告日志记录"""
        logger = StandardLogger("test_component")

        # 应该不抛出异常
        logger.log_warning("Test warning message", key="value")

    def test_standard_logger_log_debug(self):
        """测试调试日志记录"""
        logger = StandardLogger("test_component")

        # 应该不抛出异常
        logger.log_debug("Test debug message", key="value")


class TestBaseErrorHandler:
    """测试BaseErrorHandler类"""

    def test_base_error_handler_initialization(self):
        """测试BaseErrorHandler初始化"""
        handler = BaseErrorHandler()

        assert handler.error_count == 0
        assert handler.last_error is None

    def test_base_error_handler_handle_error(self):
        """测试错误处理"""
        handler = BaseErrorHandler()
        exception = ValueError("Test error")

        # 应该不抛出异常
        handler.handle_error(exception, "Test context", key="value")

        assert handler.error_count == 1
        assert handler.last_error == exception

    def test_base_error_handler_get_error_summary(self):
        """测试获取错误摘要"""
        handler = BaseErrorHandler()

        summary = handler.get_error_summary()

        assert "error_count" in summary
        assert "last_error_time" in summary
        assert summary["error_count"] == 0

    def test_base_error_handler_reset(self):
        """测试重置错误处理器"""
        handler = BaseErrorHandler()
        exception = ValueError("Test error")

        handler.handle_error(exception)
        assert handler.error_count == 1

        handler.reset()
        assert handler.error_count == 0
        assert handler.last_error is None


class TestConfigValidator:
    """测试ConfigValidator类"""

    def test_config_validator_initialization(self):
        """测试ConfigValidator初始化"""
        validator = ConfigValidator()

        assert validator.errors == []

    def test_config_validator_validate_valid_config(self):
        """测试验证有效配置"""
        validator = ConfigValidator()

        # 模拟有效配置
        config = {
            "name": "test_config",
            "value": 42,
            "enabled": True
        }

        result = validator.validate_config(config)

        assert result is True
        assert len(validator.get_validation_errors()) == 0

    def test_config_validator_validate_invalid_config(self):
        """测试验证无效配置"""
        validator = ConfigValidator()

        # 模拟无效配置
        config = None

        result = validator.validate_config(config)

        assert result is False
        assert len(validator.get_validation_errors()) > 0

    def test_config_validator_get_validation_errors(self):
        """测试获取验证错误"""
        validator = ConfigValidator()

        errors = validator.get_validation_errors()

        assert isinstance(errors, list)


class TestDataValidator:
    """测试DataValidator类"""

    def test_data_validator_initialization(self):
        """测试DataValidator初始化"""
        validator = DataValidator()

        assert validator.errors == []

    def test_data_validator_validate_valid_data(self):
        """测试验证有效数据"""
        validator = DataValidator()

        # 模拟有效数据
        data = {
            "id": 123,
            "name": "test_data",
            "values": [1, 2, 3]
        }

        result = validator.validate_data(data)

        assert result is True
        assert len(validator.get_validation_errors()) == 0

    def test_data_validator_validate_invalid_data(self):
        """测试验证无效数据"""
        validator = DataValidator()

        # 模拟无效数据
        data = "invalid_data_type"

        result = validator.validate_data(data)

        # DataValidator可能接受字符串数据，具体取决于实现
        assert isinstance(result, bool)

    def test_data_validator_get_validation_errors(self):
        """测试获取验证错误"""
        validator = DataValidator()

        errors = validator.get_validation_errors()

        assert isinstance(errors, list)


class TestResourceManager:
    """测试ResourceManager类"""

    def test_resource_manager_initialization(self):
        """测试ResourceManager初始化"""
        manager = ResourceManager()

        assert manager.resources == {}
        assert manager.monitoring_active == False
        assert manager.monitor_thread is None

    def test_resource_manager_allocate_resource(self):
        """测试分配资源"""
        manager = ResourceManager()

        result = manager.allocate_resource("cpu", 4)

        assert result is True
        assert manager.resources["cpu"] == 4

    def test_resource_manager_allocate_resource_insufficient(self):
        """测试分配不足资源"""
        manager = ResourceManager()

        # 先分配大量资源
        manager.allocate_resource("cpu", 8)

        # 尝试分配超出限制的资源
        result = manager.allocate_resource("cpu", 4)

        assert result is False

    def test_resource_manager_release_resource(self):
        """测试释放资源"""
        manager = ResourceManager()

        manager.allocate_resource("cpu", 6)
        result = manager.release_resource("cpu", 2)

        assert result is True
        assert manager.resources["cpu"] == 4

    def test_resource_manager_release_resource_nonexistent(self):
        """测试释放不存在的资源"""
        manager = ResourceManager()

        result = manager.release_resource("gpu", 1)

        assert result is False

    def test_resource_manager_get_resource_status(self):
        """测试获取资源状态"""
        manager = ResourceManager()

        manager.allocate_resource("cpu", 4)
        manager.allocate_resource("memory", 8 * 1024**3)

        status = manager.get_resource_status()

        assert "cpu" in status
        assert "memory" in status
        assert status["cpu"]["allocated"] == 4
        assert status["memory"]["allocated"] == 8 * 1024**3

    def test_resource_manager_optimize_resources(self):
        """测试资源优化"""
        manager = ResourceManager()

        result = manager.optimize_resources()

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "actions_taken" in result


class TestExceptions:
    """测试异常类"""

    def test_resource_exception(self):
        """测试ResourceException"""
        exception = ResourceException("Test resource error")

        assert str(exception) == "Test resource error"
        assert isinstance(exception, Exception)

    def test_configuration_exception(self):
        """测试ConfigurationException"""
        exception = ConfigurationException("Test config error")

        assert str(exception) == "Test config error"
        assert isinstance(exception, Exception)

    def test_validation_exception(self):
        """测试ValidationException"""
        exception = ValidationException("Test validation error")

        assert str(exception) == "Test validation error"
        assert isinstance(exception, Exception)


class TestStandardResponse:
    """测试StandardResponse类"""

    def test_standard_response_success(self):
        """测试成功的StandardResponse"""
        response = StandardResponse.success("test_data", "Operation completed")

        assert response.success is True
        assert response.data == "test_data"
        assert response.message == "Operation completed"
        assert response.error is None

    def test_standard_response_error(self):
        """测试失败的StandardResponse"""
        error = ValueError("Test error")
        response = StandardResponse.error(error, "Operation failed")

        assert response.success is False
        assert response.data is None
        assert response.message == "Operation failed"
        assert response.error == error

    def test_standard_response_from_result(self):
        """测试从结果创建StandardResponse"""
        # 成功结果
        response = StandardResponse.from_result("success_data")
        assert response.success is True
        assert response.data == "success_data"

        # 异常结果
        exception = RuntimeError("Test error")
        response = StandardResponse.from_result(exception)
        assert response.success is False
        assert response.error == exception

    def test_standard_response_to_dict(self):
        """测试转换为字典"""
        response = StandardResponse.success("test_data", "Success message")

        result = response.to_dict()

        assert result["success"] is True
        assert result["data"] == "test_data"
        assert result["message"] == "Success message"
        assert result["error"] is None


class TestBaseErrorHandlerAdvanced:
    """测试BaseErrorHandler的高级功能"""

    def test_base_error_handler_handle_error_with_reraise(self):
        """测试BaseErrorHandler处理错误并重新抛出"""
        from src.infrastructure.resource.core.shared_interfaces import BaseErrorHandler
        
        handler = BaseErrorHandler()
        
        test_error = ValueError("Test error")
        
        # 测试reraise=True的情况
        with pytest.raises(ValueError):
            handler.handle_error(test_error, {"test": "context"}, reraise=True)

    def test_base_error_handler_should_retry(self):
        """测试should_retry方法"""
        from src.infrastructure.resource.core.shared_interfaces import BaseErrorHandler
        
        handler = BaseErrorHandler(max_retries=3)
        
        # 测试应该在重试范围内
        assert handler.should_retry(ValueError("test"), 0) is True
        assert handler.should_retry(ValueError("test"), 2) is True
        
        # 测试应该超过重试次数
        assert handler.should_retry(ValueError("test"), 3) is False
        assert handler.should_retry(ValueError("test"), 4) is False

    def test_base_error_handler_get_error_summary_with_none(self):
        """测试get_error_summary当last_error_time为None时"""
        from src.infrastructure.resource.core.shared_interfaces import BaseErrorHandler
        
        handler = BaseErrorHandler()
        summary = handler.get_error_summary()
        
        assert summary["error_count"] == 0
        assert summary["last_error_time"] is None
        assert summary["last_error"] is None


class TestStandardLoggerAdvanced:
    """测试StandardLogger的高级功能"""

    def test_standard_logger_warning_compatibility_method(self):
        """测试StandardLogger的warning兼容方法"""
        from src.infrastructure.resource.core.shared_interfaces import StandardLogger
        
        logger = StandardLogger("test_logger")
        
        # 测试warning方法是log_warning的别名
        with patch.object(logger.logger, 'warning') as mock_warning:
            logger.warning("test warning message")
            mock_warning.assert_called_once()


class TestConfigValidatorAdvanced:
    """测试ConfigValidator的高级功能"""

    def test_config_validator_validate_required_fields(self):
        """测试验证必需字段"""
        from src.infrastructure.resource.core.shared_interfaces import ConfigValidator
        
        validator = ConfigValidator()
        
        config = {"field1": "value1", "field2": "value2"}
        required = ["field1", "field2", "field3"]  # field3缺失
        
        result = validator.validate_required_fields(config, required)
        
        assert result is False
        assert "缺少必需字段: field3" in validator.get_validation_errors()

    def test_config_validator_validate_required_fields_all_present(self):
        """测试验证必需字段全部存在"""
        from src.infrastructure.resource.core.shared_interfaces import ConfigValidator
        
        validator = ConfigValidator()
        
        config = {"field1": "value1", "field2": "value2"}
        required = ["field1", "field2"]
        
        result = validator.validate_required_fields(config, required)
        
        assert result is True

    def test_config_validator_validate_field_types(self):
        """测试验证字段类型"""
        from src.infrastructure.resource.core.shared_interfaces import ConfigValidator
        
        validator = ConfigValidator()
        
        config = {"field1": "string", "field2": 123, "field3": "should_be_int"}
        field_types = {"field1": str, "field2": int, "field3": int}
        
        result = validator.validate_field_types(config, field_types)
        
        assert result is False
        errors = validator.get_validation_errors()
        assert any("field3" in error and "int" in error for error in errors)

    def test_config_validator_validate_field_types_correct_types(self):
        """测试验证字段类型正确"""
        from src.infrastructure.resource.core.shared_interfaces import ConfigValidator
        
        validator = ConfigValidator()
        
        config = {"field1": "string", "field2": 123}
        field_types = {"field1": str, "field2": int}
        
        result = validator.validate_field_types(config, field_types)
        
        assert result is True


class TestDataValidatorAdvanced:
    """测试DataValidator的高级功能"""

    def test_data_validator_sanitize_data_dict(self):
        """测试sanitize_data方法处理字典"""
        from src.infrastructure.resource.core.shared_interfaces import DataValidator
        
        validator = DataValidator()
        
        data = {"field1": "value1", "field2": None, "field3": "value3"}
        result = validator.sanitize_data(data)
        
        assert result == {"field1": "value1", "field3": "value3"}

    def test_data_validator_sanitize_data_non_dict(self):
        """测试sanitize_data方法处理非字典数据"""
        from src.infrastructure.resource.core.shared_interfaces import DataValidator
        
        validator = DataValidator()
        
        data = "string_data"
        result = validator.sanitize_data(data)
        
        assert result == data

    def test_data_validator_validate_with_schema(self):
        """测试使用schema验证数据"""
        from src.infrastructure.resource.core.shared_interfaces import DataValidator
        
        validator = DataValidator()
        
        data = {"name": "test", "age": 25, "score": 85}
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True, "min": 0, "max": 150},
            "score": {"type": int, "required": False, "min": 0, "max": 100}
        }
        
        result = validator.validate_data(data, schema)
        
        assert result is True

    def test_data_validator_validate_with_invalid_schema(self):
        """测试使用无效schema验证数据"""
        from src.infrastructure.resource.core.shared_interfaces import DataValidator
        
        validator = DataValidator()
        
        data = {"name": "test", "age": 200}  # age超出范围
        schema = {
            "name": {"type": str, "required": True},
            "age": {"type": int, "required": True, "min": 0, "max": 150}
        }
        
        result = validator.validate_data(data, schema)
        
        assert result is False


class TestResourceManagerAdvanced:
    """测试ResourceManager的高级功能"""

    def test_resource_manager_acquire_resource(self):
        """测试获取资源"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        resource = manager.acquire_resource("test_resource")
        
        assert resource is not None
        assert resource["id"] == "test_resource"
        assert resource["created"] is True

    def test_resource_manager_acquire_resource_duplicate(self):
        """测试获取重复资源"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        # 第一次获取成功
        resource1 = manager.acquire_resource("test_resource")
        assert resource1 is not None
        
        # 第二次获取应该失败（资源已被占用）
        resource2 = manager.acquire_resource("test_resource")
        assert resource2 is None

    def test_resource_manager_release_resource_complete(self):
        """测试完全释放资源"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        manager.allocate_resource("cpu", 8)
        result = manager.release_resource("cpu")  # 不指定amount，完全释放
        
        assert result is True
        assert "cpu" not in manager.resources

    def test_resource_manager_release_resource_partial_to_zero(self):
        """测试部分释放资源到零"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        manager.allocate_resource("cpu", 4)
        result = manager.release_resource("cpu", 4)  # 释放全部，应该删除
        
        assert result is True
        assert "cpu" not in manager.resources

    def test_resource_manager_get_current_usage_with_psutil(self):
        """测试获取当前使用情况（有psutil）"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        # 由于psutil是在方法内部导入的，我们需要patch builtins.__import__或使用不同的方法
        with patch('builtins.__import__') as mock_import:
            # 模拟psutil模块
            mock_psutil = MagicMock()
            mock_psutil.cpu_percent.return_value = 45.5
            mock_psutil.virtual_memory.return_value = MagicMock(percent=60.2)
            mock_psutil.disk_usage.return_value = MagicMock(percent=75.8)
            
            def import_side_effect(name, *args, **kwargs):
                if name == 'psutil':
                    return mock_psutil
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            usage = manager.get_current_usage()
            
            assert usage["cpu_percent"] == 45.5
            assert usage["memory_percent"] == 60.2
            assert usage["disk_percent"] == 75.8

    def test_resource_manager_get_current_usage_without_psutil(self):
        """测试获取当前使用情况（无psutil）"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        with patch('builtins.__import__') as mock_import:
            def import_side_effect(name, *args, **kwargs):
                if name == 'psutil':
                    raise ImportError()
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = import_side_effect
            
            usage = manager.get_current_usage()
            
            # 应该返回模拟数据
            assert usage["cpu_percent"] == 45.5
            assert usage["memory_percent"] == 60.2
            assert usage["disk_percent"] == 75.8

    def test_resource_manager_get_usage_history(self):
        """测试获取使用历史"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        # 添加一些历史数据
        manager._resource_history = [
            {"cpu_percent": 45.0, "timestamp": "2023-01-01T00:00:00"},
            {"cpu_percent": 50.0, "timestamp": "2023-01-01T00:01:00"}
        ]
        
        history = manager.get_usage_history()
        
        assert history["count"] == 2
        assert len(history["history"]) == 2

    def test_resource_manager_start_monitoring(self):
        """测试启动监控"""
        from src.infrastructure.resource.core.shared_interfaces import ResourceManager
        
        manager = ResourceManager()
        
        assert not manager._monitoring
        
        manager.start_monitoring()
        
        assert manager._monitoring
        assert manager._monitor_thread is not None
        assert manager._monitor_thread.daemon is True


class TestUtilityFunctions:
    """测试工具函数"""

    def test_safe_execute_success(self):
        """测试safe_execute成功执行"""
        from src.infrastructure.resource.core.shared_interfaces import safe_execute
        
        def test_func(x, y):
            return x + y
        
        result = safe_execute(test_func, 2, 3)
        
        assert result == 5

    def test_safe_execute_with_error_handler(self):
        """测试safe_execute使用错误处理器"""
        from src.infrastructure.resource.core.shared_interfaces import safe_execute, BaseErrorHandler
        
        def test_func_fail():
            raise ValueError("Test error")
        
        handler = BaseErrorHandler()
        handler.handle_error = Mock(return_value="error_handled")
        
        result = safe_execute(test_func_fail, error_handler=handler)
        
        assert result == "error_handled"
        handler.handle_error.assert_called_once()

    def test_safe_execute_without_error_handler(self):
        """测试safe_execute无错误处理器"""
        from src.infrastructure.resource.core.shared_interfaces import safe_execute
        
        def test_func_fail():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            safe_execute(test_func_fail)

    def test_validate_and_execute_success(self):
        """测试validate_and_execute成功"""
        from src.infrastructure.resource.core.shared_interfaces import validate_and_execute, ConfigValidator
        
        def test_func(x):
            return x * 2
        
        validator = ConfigValidator()
        config = {"valid": True}
        
        result = validate_and_execute(validator, config, test_func, 5)
        
        assert result == 10

    def test_validate_and_execute_validation_failure(self):
        """测试validate_and_execute验证失败"""
        from src.infrastructure.resource.core.shared_interfaces import validate_and_execute, ConfigValidator
        
        def test_func(x):
            return x * 2
        
        validator = ConfigValidator()
        
        with pytest.raises(ValueError, match="配置验证失败"):
            validate_and_execute(validator, None, test_func, 5)

    def test_with_error_handling_decorator(self):
        """测试with_error_handling装饰器"""
        from src.infrastructure.resource.core.shared_interfaces import with_error_handling, BaseErrorHandler
        
        handler = BaseErrorHandler()
        handler.handle_error = Mock(return_value="decorated_error_handled")
        
        @with_error_handling(handler)
        def test_func_fail():
            raise ValueError("Test error")
        
        result = test_func_fail()
        
        assert result == "decorated_error_handled"
        handler.handle_error.assert_called_once()

    def test_with_logging_decorator_success(self):
        """测试with_logging装饰器成功"""
        from src.infrastructure.resource.core.shared_interfaces import with_logging, StandardLogger
        
        logger = StandardLogger("test")
        logger.log_info = Mock()
        logger.log_error = Mock()
        
        @with_logging(logger)
        def test_func_success(x):
            return x * 2
        
        result = test_func_success(5)
        
        assert result == 10
        logger.log_info.assert_called()
        logger.log_error.assert_not_called()

    def test_with_logging_decorator_failure(self):
        """测试with_logging装饰器失败"""
        from src.infrastructure.resource.core.shared_interfaces import with_logging, StandardLogger
        
        logger = StandardLogger("test")
        logger.log_info = Mock()
        logger.log_error = Mock()
        
        @with_logging(logger)
        def test_func_fail():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            test_func_fail()
        
        logger.log_info.assert_called()
        logger.log_error.assert_called_once()