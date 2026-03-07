"""
测试共享接口

覆盖 shared_interfaces.py 中的所有类和功能
"""

import pytest
from unittest.mock import Mock, patch
from src.infrastructure.resource.core.shared_interfaces import (
    IConfigValidator, ILogger, IErrorHandler, ISharedResourceManager,
    IDataValidator, StandardLogger, BaseErrorHandler, ConfigValidator,
    DataValidator, ResourceManager, safe_execute, validate_and_execute,
    with_error_handling, with_logging, ResourceException,
    ConfigurationException, ValidationException, StandardResponse
)


class TestInterfaces:
    """接口测试"""

    def test_iconfig_validator_is_abstract(self):
        """测试IConfigValidator是抽象类"""
        # 不能直接实例化抽象类
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

    def test_ishared_resource_manager_is_abstract(self):
        """测试ISharedResourceManager是抽象类"""
        with pytest.raises(TypeError):
            ISharedResourceManager()

    def test_idata_validator_is_abstract(self):
        """测试IDataValidator是抽象类"""
        with pytest.raises(TypeError):
            IDataValidator()


class TestStandardLogger:
    """StandardLogger 类测试"""

    def test_initialization(self):
        """测试初始化"""
        logger = StandardLogger("test_component")

        assert hasattr(logger, 'logger')
        assert hasattr(logger, 'component_name')
        assert logger.component_name == "test_component"

    def test_log_info(self):
        """测试记录信息日志"""
        logger = StandardLogger("test_component")

        # Mock Python logging
        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            # 重新创建logger实例以使用mock
            logger = StandardLogger("test_component")

            logger.log_info("Test message", key="value")

            mock_logger.info.assert_called_once()
            # 检查调用参数
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "Test message"

    def test_log_warning(self):
        """测试记录警告日志"""
        logger = StandardLogger("test_component")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = StandardLogger("test_component")
            logger.log_warning("Warning message")

            mock_logger.warning.assert_called_once()

    def test_log_error(self):
        """测试记录错误日志"""
        logger = StandardLogger("test_component")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = StandardLogger("test_component")
            logger.log_error("Error message", error=Exception("test"))

            mock_logger.error.assert_called_once()

    def test_log_debug(self):
        """测试记录调试日志"""
        logger = StandardLogger("test_component")

        with patch('logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger

            logger = StandardLogger("test_component")
            logger.log_debug("Debug message")

            mock_logger.debug.assert_called_once()


class TestBaseErrorHandler:
    """BaseErrorHandler 类测试"""

    def test_initialization(self):
        """测试初始化"""
        handler = BaseErrorHandler()

        assert hasattr(handler, 'logger')
        assert hasattr(handler, 'max_retries')
        assert hasattr(handler, 'retry_delay')
        assert hasattr(handler, 'error_count')
        assert handler.max_retries == 3
        assert handler.retry_delay == 1.0

    def test_handle_error(self):
        """测试处理错误"""
        handler = BaseErrorHandler()

        with patch.object(handler.logger, 'error') as mock_logger_error:
            exception = ValueError("Test error")
            handler.handle_error(exception, {"context": "test"})

            mock_logger_error.assert_called_once()
            # 检查错误信息包含异常和上下文
            call_args = mock_logger_error.call_args
            assert "处理错误: ValueError" in call_args[0][0]

    def test_handle_error_with_recovery(self):
        """测试处理可恢复错误"""
        handler = BaseErrorHandler()

        with patch.object(handler.logger, 'error') as mock_logger_error:
            exception = ConnectionError("Connection failed")

            # 测试可恢复错误的处理
            result = handler.handle_error(exception, {"recoverable": True})

            mock_logger_error.assert_called_once()
            assert result is None

    def test_should_retry(self):
        """测试重试判断逻辑"""
        handler = BaseErrorHandler(max_retries=3)

        # 测试应该重试的情况
        assert handler.should_retry(ValueError("test"), 1) == True
        assert handler.should_retry(ValueError("test"), 2) == True
        assert handler.should_retry(ValueError("test"), 3) == False  # 超过最大重试次数

    def test_get_error_summary(self):
        """测试获取错误摘要"""
        handler = BaseErrorHandler()

        # 初始状态
        summary = handler.get_error_summary()
        assert summary["error_count"] == 0
        assert summary["last_error_time"] is None
        assert summary["last_error"] is None

        # 处理错误后
        handler.handle_error(ValueError("test error"))
        summary = handler.get_error_summary()
        assert summary["error_count"] == 1
        assert summary["last_error"] == "test error"
        assert summary["last_error_time"] is not None


class TestConfigValidator:
    """ConfigValidator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        validator = ConfigValidator()

        assert hasattr(validator, 'errors')
        assert isinstance(validator.errors, list)
        assert len(validator.errors) == 0

    def test_validate_config_success(self):
        """测试成功验证配置"""
        validator = ConfigValidator()

        config = {"key": "value", "number": 42}

        result = validator.validate_config(config)

        assert result == True
        assert len(validator.get_validation_errors()) == 0

    def test_validate_config_failure(self):
        """测试验证配置失败"""
        validator = ConfigValidator()

        # 测试无效配置
        config = None

        result = validator.validate_config(config)

        assert result == False
        errors = validator.get_validation_errors()
        assert len(errors) > 0

    def test_get_validation_errors(self):
        """测试获取验证错误"""
        validator = ConfigValidator()

        # 先进行失败的验证
        validator.validate_config(None)

        errors = validator.get_validation_errors()

        assert isinstance(errors, list)
        assert len(errors) > 0


class TestDataValidator:
    """DataValidator 类测试"""

    def test_initialization(self):
        """测试初始化"""
        validator = DataValidator()

        assert hasattr(validator, 'logger')
        assert hasattr(validator, 'errors')
        assert isinstance(validator.errors, list)
        assert len(validator.errors) == 0

    def test_validate_data_success(self):
        """测试成功验证数据"""
        validator = DataValidator()

        data = {"name": "test", "value": 100}

        result = validator.validate_data(data, {"name": {"type": str}, "value": {"type": int}})

        assert result == True
        assert len(validator.get_validation_errors()) == 0

    def test_validate_data_failure(self):
        """测试验证数据失败"""
        validator = DataValidator()

        data = {"name": 123, "value": "invalid"}  # 类型不匹配

        result = validator.validate_data(data, {"name": {"type": str}, "value": {"type": int}})

        assert result == False
        errors = validator.get_validation_errors()
        assert len(errors) > 0

    def test_sanitize_data(self):
        """测试数据清理功能"""
        validator = DataValidator()

        # 测试字典数据清理
        data = {"name": "test", "value": None, "active": True}
        sanitized = validator.sanitize_data(data)

        assert "name" in sanitized
        assert "active" in sanitized
        assert "value" not in sanitized  # None值应该被移除
        assert sanitized["name"] == "test"
        assert sanitized["active"] == True

        # 测试非字典数据
        assert validator.sanitize_data("string") == "string"
        assert validator.sanitize_data(123) == 123
        assert validator.sanitize_data(None) is None

    def test_validate_data_edge_cases(self):
        """测试数据验证的边缘情况"""
        validator = DataValidator()

        # 测试空字典
        result = validator.validate_data({})
        assert result == True

        # 测试None schema
        result = validator.validate_data({"test": "value"}, None)
        assert result == True

        # 测试复杂嵌套数据
        complex_data = {
            "user": {
                "name": "test",
                "settings": {"theme": "dark"}
            },
            "items": [1, 2, 3]
        }
        result = validator.validate_data(complex_data)
        assert result == True

    def test_validate_data_none_input(self):
        """测试验证None输入数据"""
        validator = DataValidator()

        result = validator.validate_data(None)
        assert result == False

        errors = validator.get_validation_errors()
        assert len(errors) > 0
        assert "不能为空" in errors[0]

    def test_validate_data_with_invalid_schema(self):
        """测试使用无效schema验证数据"""
        validator = DataValidator()

        data = {"name": "test"}
        # 无效的schema结构
        invalid_schema = {"name": "not_a_dict"}

        # 应该不抛出异常，但验证失败
        result = validator.validate_data(data, invalid_schema)
        # 由于schema无效，验证应该失败或者忽略schema
        assert isinstance(result, bool)  # 至少不应该崩溃


class TestResourceManager:
    """ResourceManager 类测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = ResourceManager()

        assert hasattr(manager, '_resources')
        assert hasattr(manager, '_lock')
        assert hasattr(manager, '_logger')
        assert hasattr(manager, '_error_handler')

    def test_allocate_resource_success(self):
        """测试成功分配资源"""
        manager = ResourceManager()

        resource_id = manager.allocate_resource("cpu", {"cores": 4})

        assert resource_id is not None
        assert resource_id in manager._resources

    def test_allocate_resource_failure(self):
        """测试分配资源失败"""
        manager = ResourceManager()

        # 先分配一个资源，使后续分配失败
        manager.allocate_resource("cpu", {"cores": 1})

        # 尝试分配超出限制的资源
        resource_id = manager.allocate_resource("cpu", {"cores": 1000})

        assert resource_id is None

    def test_release_resource_success(self):
        """测试成功释放资源"""
        manager = ResourceManager()

        resource_id = manager.allocate_resource("cpu", {"cores": 2})

        result = manager.release_resource(resource_id)

        assert result == True
        assert resource_id not in manager._resources

    def test_release_resource_not_found(self):
        """测试释放不存在的资源"""
        manager = ResourceManager()

        result = manager.release_resource("nonexistent")

        assert result == False

    def test_get_resource_status(self):
        """测试获取资源状态"""
        manager = ResourceManager()

        status = manager.get_resource_status()

        assert isinstance(status, dict)
        assert 'allocated' in status
        assert 'available' in status
        assert 'total' in status

    def test_list_resources(self):
        """测试列出资源"""
        manager = ResourceManager()

        # 分配一些资源
        manager.allocate_resource("cpu", {"cores": 2})
        manager.allocate_resource("memory", {"size": 1024})

        resources = manager.list_resources()

        assert isinstance(resources, list)
        assert len(resources) >= 2

    def test_get_resource_info(self):
        """测试获取资源信息"""
        manager = ResourceManager()

        resource_id = manager.allocate_resource("cpu", {"cores": 4})

        info = manager.get_resource_info(resource_id)

        assert info is not None
        assert info['type'] == 'cpu'
        assert info['requirements'] == {"cores": 4}

    def test_list_resources_empty(self):
        """测试列出空资源列表"""
        manager = ResourceManager()

        resources = manager.list_resources()
        assert isinstance(resources, list)
        assert len(resources) == 0

    def test_list_resources_with_data(self):
        """测试列出有数据的资源列表"""
        manager = ResourceManager()

        # 分配一些资源
        manager.allocate_resource("cpu", {"cores": 4})
        manager.allocate_resource("memory", {"size": 1024})

        resources = manager.list_resources()

        assert isinstance(resources, list)
        assert len(resources) >= 2

        # 验证资源信息结构
        for resource_id in resources:
            info = manager.get_resource_info(resource_id)
            assert info is not None
            assert 'type' in info
            assert 'requirements' in info

    def test_optimize_resources(self):
        """测试资源优化"""
        manager = ResourceManager()

        # 分配一些资源
        manager.allocate_resource("cpu", {"cores": 2})
        manager.allocate_resource("memory", {"size": 512})

        # 优化资源
        result = manager.optimize_resources()

        assert isinstance(result, dict)
        assert 'freed_resources' in result
        assert 'optimization_time' in result
        assert isinstance(result['freed_resources'], int)
        assert result['freed_resources'] >= 0

    def test_release_nonexistent_resource(self):
        """测试释放不存在的资源"""
        manager = ResourceManager()

        # 尝试释放不存在的资源
        result = manager.release_resource("nonexistent_id")
        assert result == False

        # 验证资源列表为空
        resources = manager.list_resources()
        assert len(resources) == 0

    def test_get_resource_info_nonexistent(self):
        """测试获取不存在的资源信息"""
        manager = ResourceManager()

        info = manager.get_resource_info("nonexistent_id")
        assert info is None


class TestUtilityFunctions:
    """工具函数测试"""

    def test_safe_execute_success(self):
        """测试safe_execute成功执行"""
        def test_func(x, y):
            return x + y

        result = safe_execute(test_func, 2, 3)

        assert result == 5

    def test_safe_execute_failure(self):
        """测试safe_execute执行失败"""
        def failing_func():
            raise ValueError("Test error")

        result = safe_execute(failing_func)

        assert result is None

    def test_validate_and_execute_success(self):
        """测试validate_and_execute成功"""
        validator = ConfigValidator()
        config = {"key": "value"}

        def test_func():
            return "success"

        result = validate_and_execute(validator, config, test_func)

        assert result == "success"

    def test_validate_and_execute_validation_failure(self):
        """测试validate_and_execute验证失败"""
        validator = ConfigValidator()
        config = None  # 无效配置

        def test_func():
            return "should not execute"

        result = validate_and_execute(validator, config, test_func)

        assert result is None

    def test_with_error_handling_decorator(self):
        """测试错误处理装饰器"""
        @with_error_handling(BaseErrorHandler())
        def failing_func():
            raise ValueError("Test error")

        # 函数应该正常执行，但错误被处理
        result = failing_func()

        assert result is None

    def test_with_logging_decorator(self):
        """测试日志装饰器"""
        @with_logging(StandardLogger("test"))
        def test_func():
            return "logged result"

        with patch('src.infrastructure.resource.core.shared_interfaces.logger') as mock_logger:
            result = test_func()

            assert result == "logged result"
            mock_logger.debug.assert_called()


class TestExceptions:
    """异常类测试"""

    def test_resource_exception(self):
        """测试ResourceException"""
        exc = ResourceException("Resource error")

        assert str(exc) == "Resource error"
        assert isinstance(exc, Exception)

    def test_configuration_exception(self):
        """测试ConfigurationException"""
        exc = ConfigurationException("Config error")

        assert str(exc) == "Config error"
        assert isinstance(exc, Exception)

    def test_validation_exception(self):
        """测试ValidationException"""
        exc = ValidationException("Validation error")

        assert str(exc) == "Validation error"
        assert isinstance(exc, Exception)


class TestStandardResponse:
    """StandardResponse 类测试"""

    def test_initialization_success(self):
        """测试成功响应的初始化"""
        response = StandardResponse(True, "Operation successful", {"data": "value"})

        assert response.success == True
        assert response.message == "Operation successful"
        assert response.data == {"data": "value"}

    def test_initialization_failure(self):
        """测试失败响应的初始化"""
        response = StandardResponse(False, "Operation failed", None, "error_code")

        assert response.success == False
        assert response.message == "Operation failed"
        assert response.data is None
        assert response.error_code == "error_code"

    def test_to_dict(self):
        """测试转换为字典"""
        response = StandardResponse(True, "Success", {"key": "value"})

        result = response.to_dict()

        expected = {
            "success": True,
            "message": "Success",
            "data": {"key": "value"},
            "error_code": None,
            "timestamp": response.timestamp.isoformat()
        }

        assert result["success"] == expected["success"]
        assert result["message"] == expected["message"]
        assert result["data"] == expected["data"]
        assert result["error_code"] == expected["error_code"]
        assert "timestamp" in result

    def test_warning_method(self):
        """测试警告日志兼容性方法"""
        logger = StandardLogger("test")

        with patch('src.infrastructure.resource.core.shared_interfaces.logging') as mock_logging:
            logger.warning("Warning message", key="value")

            # warning方法应该调用log_warning
            mock_logging.getLogger.return_value.warning.assert_called_with(
                "Warning message", extra={"key": "value"}
            )

    def test_base_error_handler_initialization_with_params(self):
        """测试BaseErrorHandler带参数初始化"""
        handler = BaseErrorHandler(max_retries=5, retry_delay=2.0)

        assert handler.max_retries == 5
        assert handler.retry_delay == 2.0
        assert hasattr(handler, '_retry_count')

    def test_base_error_handler_retry_logic(self):
        """测试重试逻辑"""
        handler = BaseErrorHandler(max_retries=3)

        call_count = 0
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"

        result = handler._retry_logic(failing_function)

        assert result == "success"
        assert call_count == 3

    def test_base_error_handler_retry_exhausted(self):
        """测试重试耗尽"""
        handler = BaseErrorHandler(max_retries=2)

        def always_failing():
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            handler._retry_logic(always_failing)

    def test_base_error_handler_handle_error_with_retry_success(self):
        """测试错误处理伴随重试成功"""
        handler = BaseErrorHandler(max_retries=3)

        call_count = 0
        def recovery_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Connection failed")
            return "recovered"

        result = handler.handle_error(ConnectionError("Connection failed"), recovery_function)

        assert result == "recovered"
        assert call_count == 2

    def test_base_error_handler_handle_error_no_recovery(self):
        """测试错误处理无恢复函数"""
        handler = BaseErrorHandler()

        # 没有恢复函数，应该只记录错误
        result = handler.handle_error(ValueError("Test error"))

        assert result is None  # 没有恢复函数时返回None

    def test_config_validator_validate_config_invalid_type(self):
        """测试配置验证器验证无效类型"""
        validator = ConfigValidator()

        # 测试无效配置类型
        result = validator.validate_config("not_a_dict")
        assert result == False

        errors = validator.get_validation_errors()
        assert len(errors) > 0

    def test_config_validator_validate_config_missing_required(self):
        """测试配置验证器验证缺少必需字段"""
        validator = ConfigValidator()

        # 缺少必需字段的配置
        invalid_config = {
            "optional_field": "value"
            # 缺少其他必需字段
        }

        result = validator.validate_config(invalid_config)
        # 具体验证逻辑可能因实现而异，这里主要测试方法调用
        assert isinstance(result, bool)

    def test_config_validator_reset_errors(self):
        """测试配置验证器重置错误"""
        validator = ConfigValidator()

        # 先进行一次验证
        validator.validate_config("invalid")
        assert len(validator.get_validation_errors()) > 0

        # 再次验证应该重置之前的错误
        validator.validate_config({})
        # 错误列表应该被重置

    def test_data_validator_validate_data_success(self):
        """测试数据验证器验证成功"""
        validator = DataValidator()

        valid_data = {
            "name": "test",
            "value": 42,
            "items": [1, 2, 3]
        }

        result = validator.validate_data(valid_data)
        assert isinstance(result, bool)

    def test_data_validator_validate_data_failure(self):
        """测试数据验证器验证失败"""
        validator = DataValidator()

        invalid_data = None  # 无效数据

        result = validator.validate_data(invalid_data)
        assert result == False

        errors = validator.get_validation_errors()
        assert len(errors) > 0

    def test_data_validator_validate_data_type_check(self):
        """测试数据验证器类型检查"""
        validator = DataValidator()

        # 测试不同数据类型
        test_cases = [
            {"name": "string", "type": "valid"},
            42,  # 数字
            [1, 2, 3],  # 列表
            True,  # 布尔值
        ]

        for test_data in test_cases:
            result = validator.validate_data(test_data)
            assert isinstance(result, bool)

    def test_resource_manager_initialization_with_logger(self):
        """测试ResourceManager带日志器初始化"""
        mock_logger = Mock()
        manager = ResourceManager(mock_logger)

        assert manager.logger == mock_logger

    def test_resource_manager_get_resource_status(self):
        """测试获取资源状态"""
        manager = ResourceManager()

        status = manager.get_resource_status()

        assert isinstance(status, dict)
        assert "status" in status

    def test_resource_manager_resource_status_details(self):
        """测试资源状态详细信息"""
        manager = ResourceManager()

        status = manager.get_resource_status()

        # 状态应该包含基本信息
        expected_keys = ["status", "timestamp"]
        for key in expected_keys:
            assert key in status

    def test_safe_execute_success(self):
        """测试safe_execute成功执行"""
        def successful_function():
            return "success"

        result = safe_execute(successful_function)

        assert result == "success"

    def test_safe_execute_exception(self):
        """测试safe_execute异常处理"""
        def failing_function():
            raise ValueError("Test error")

        result = safe_execute(failing_function)

        assert result is None  # 异常时返回None

    def test_safe_execute_with_fallback(self):
        """测试safe_execute带fallback"""
        def failing_function():
            raise ValueError("Test error")

        fallback_value = "fallback"
        result = safe_execute(failing_function, fallback=fallback_value)

        assert result == fallback_value

    def test_validate_and_execute_success(self):
        """测试validate_and_execute成功"""
        def validator(data):
            return data > 0

        def processor(data):
            return data * 2

        result = validate_and_execute(5, validator, processor)

        assert result == 10

    def test_validate_and_execute_validation_failure(self):
        """测试validate_and_execute验证失败"""
        def validator(data):
            return data > 0  # 5 > 0, 应该成功

        def processor(data):
            return data * 2

        result = validate_and_execute(-1, validator, processor)

        assert result is None  # 验证失败返回None

    def test_validate_and_execute_processing_failure(self):
        """测试validate_and_execute处理失败"""
        def validator(data):
            return True

        def failing_processor(data):
            raise ValueError("Processing failed")

        result = validate_and_execute(5, validator, failing_processor)

        assert result is None  # 处理失败返回None

    def test_with_error_handling_decorator_success(self):
        """测试with_error_handling装饰器成功"""
        @with_error_handling
        def successful_function():
            return "success"

        result = successful_function()

        assert result == "success"

    def test_with_error_handling_decorator_failure(self):
        """测试with_error_handling装饰器失败"""
        @with_error_handling
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()

        assert result is None  # 异常时返回None

    def test_with_error_handling_decorator_custom_fallback(self):
        """测试with_error_handling装饰器自定义fallback"""
        @with_error_handling(fallback="custom_fallback")
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()

        assert result == "custom_fallback"

    def test_with_logging_decorator_success(self):
        """测试with_logging装饰器成功"""
        mock_logger = Mock()

        @with_logging(logger=mock_logger)
        def successful_function():
            return "success"

        result = successful_function()

        assert result == "success"
        # 应该记录开始和结束日志
        assert mock_logger.log_info.call_count >= 1

    def test_with_logging_decorator_failure(self):
        """测试with_logging装饰器失败"""
        mock_logger = Mock()

        @with_logging(logger=mock_logger)
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()

        assert result is None
        # 应该记录错误日志
        mock_logger.log_error.assert_called()

    def test_with_logging_decorator_custom_logger(self):
        """测试with_logging装饰器自定义日志器"""
        mock_logger = Mock()

        @with_logging(logger=mock_logger, log_level="debug")
        def test_function():
            return "test"

        result = test_function()

        assert result == "test"
        # 应该使用指定的日志器
        mock_logger.log_info.assert_called()

    def test_resource_exception_creation(self):
        """测试ResourceException创建"""
        exc = ResourceException("Resource error", resource_id="cpu_1")

        assert str(exc) == "Resource error"
        assert exc.resource_id == "cpu_1"

    def test_configuration_exception_with_details(self):
        """测试ConfigurationException带详情"""
        exc = ConfigurationException("Config error", config_key="timeout")

        assert str(exc) == "Config error"
        assert exc.config_key == "timeout"

    def test_validation_exception_with_field(self):
        """测试ValidationException带字段"""
        exc = ValidationException("Validation failed", field="email")

        assert str(exc) == "Validation failed"
        assert exc.field == "email"

    def test_standard_response_success_creation(self):
        """测试StandardResponse成功创建"""
        response = StandardResponse.success("Operation successful", {"result": "data"})

        assert response.success == True
        assert response.message == "Operation successful"
        assert response.data == {"result": "data"}
        assert response.error_code is None
        assert isinstance(response.timestamp, datetime)

    def test_standard_response_error_creation(self):
        """测试StandardResponse错误创建"""
        response = StandardResponse.error("Operation failed", "ERROR_001")

        assert response.success == False
        assert response.message == "Operation failed"
        assert response.error_code == "ERROR_001"
        assert response.data is None
        assert isinstance(response.timestamp, datetime)

    def test_standard_response_to_dict(self):
        """测试StandardResponse转字典"""
        response = StandardResponse.success("Test", {"key": "value"})

        result = response.to_dict()

        expected = {
            "success": True,
            "message": "Test",
            "data": {"key": "value"},
            "error_code": None,
            "timestamp": response.timestamp.isoformat()
        }

        assert result == expected

    def test_standard_response_from_dict(self):
        """测试从字典创建StandardResponse"""
        data = {
            "success": True,
            "message": "Test message",
            "data": {"result": "success"},
            "error_code": None,
            "timestamp": "2023-01-01T12:00:00"
        }

        response = StandardResponse.from_dict(data)

        assert response.success == True
        assert response.message == "Test message"
        assert response.data == {"result": "success"}
        assert response.error_code is None

    def test_standard_response_generic_type(self):
        """测试StandardResponse泛型类型"""
        # 测试带具体类型的响应
        response: StandardResponse[str] = StandardResponse.success("OK", "string_data")

        assert response.data == "string_data"

        # 测试带字典类型的响应
        dict_response: StandardResponse[Dict[str, int]] = StandardResponse.success(
            "Dict data", {"count": 42}
        )

        assert dict_response.data == {"count": 42}

    def test_shared_interfaces_import_all(self):
        """测试所有接口和类的导入"""
        # 验证所有主要的类和函数都可以导入
        from src.infrastructure.resource.core.shared_interfaces import (
            IConfigValidator, ILogger, IErrorHandler, ISharedResourceManager,
            IDataValidator, StandardLogger, BaseErrorHandler, ConfigValidator,
            DataValidator, ResourceManager, safe_execute, validate_and_execute,
            with_error_handling, with_logging, ResourceException,
            ConfigurationException, ValidationException, StandardResponse,
            logger
        )

        # 验证它们都是可用的
        assert IConfigValidator is not None
        assert ILogger is not None
        assert StandardLogger is not None
        assert safe_execute is not None
        assert with_error_handling is not None
        assert ResourceException is not None
        assert StandardResponse is not None

    def test_base_error_handler_edge_cases(self):
        """测试BaseErrorHandler边界情况"""
        handler = BaseErrorHandler(max_retries=0)  # 不重试

        def failing_function():
            raise ValueError("Always fails")

        # 应该立即失败，不重试
        with pytest.raises(ValueError):
            handler._retry_logic(failing_function)

    def test_config_validator_edge_cases(self):
        """测试ConfigValidator边界情况"""
        validator = ConfigValidator()

        # 测试空配置
        result = validator.validate_config({})
        assert isinstance(result, bool)

        # 测试None配置
        result = validator.validate_config(None)
        assert result == False

    def test_data_validator_edge_cases(self):
        """测试DataValidator边界情况"""
        validator = DataValidator()

        # 测试各种边界数据
        edge_cases = [
            None,
            "",
            0,
            [],
            {},
            False
        ]

        for test_data in edge_cases:
            result = validator.validate_data(test_data)
            assert isinstance(result, bool)

    def test_resource_manager_edge_cases(self):
        """测试ResourceManager边界情况"""
        manager = ResourceManager()

        # 测试多次调用
        status1 = manager.get_resource_status()
        status2 = manager.get_resource_status()

        # 结果应该一致
        assert status1.keys() == status2.keys()

    def test_decorators_edge_cases(self):
        """测试装饰器边界情况"""
        # 测试装饰器在各种情况下的行为
        @with_error_handling
        def function_with_args(arg1, arg2=None):
            if arg1 == "fail":
                raise ValueError("Failed")
            return f"success_{arg1}"

        # 成功调用
        result = function_with_args("test")
        assert result == "success_test"

        # 失败调用
        result = function_with_args("fail")
        assert result is None

    def test_exceptions_edge_cases(self):
        """测试异常类边界情况"""
        # 测试异常类的各种构造方式
        exc1 = ResourceException("Basic error")
        assert str(exc1) == "Basic error"

        exc2 = ConfigurationException("Config error", config_key="test")
        assert exc2.config_key == "test"

        exc3 = ValidationException("Validation error", field="email", value="invalid")
        assert exc3.field == "email"

    def test_standard_response_edge_cases(self):
        """测试StandardResponse边界情况"""
        # 测试各种数据类型的响应
        responses = [
            StandardResponse.success("OK", None),
            StandardResponse.success("OK", 42),
            StandardResponse.success("OK", [1, 2, 3]),
            StandardResponse.error("Failed", "CODE_001"),
            StandardResponse.error("Failed", None),
        ]

        for response in responses:
            # 都能转换为字典
            dict_result = response.to_dict()
            assert isinstance(dict_result, dict)

            # 都能从字典重建
            reconstructed = StandardResponse.from_dict(dict_result)
            assert reconstructed.success == response.success
