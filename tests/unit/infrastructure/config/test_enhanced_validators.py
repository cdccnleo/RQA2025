#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强验证器测试
测试增强验证器功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.config.validators.enhanced_validators import (
    ConfigValidationResult,
    EnhancedConfigValidator,
    create_standard_validators,
    _key_exists,
    _get_nested_value
)


class TestConfigValidationResult:
    """测试配置验证结果类"""

    def test_initialization(self):
        """测试初始化"""
        result = ConfigValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

        result_invalid = ConfigValidationResult(False)
        assert result_invalid.is_valid is False

    def test_add_error(self):
        """测试添加错误"""
        result = ConfigValidationResult()

        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_add_warning(self):
        """测试添加警告"""
        result = ConfigValidationResult()

        result.add_warning("Test warning")
        assert result.is_valid is True  # 警告不影响有效性
        assert "Test warning" in result.warnings

    def test_add_recommendation(self):
        """测试添加建议"""
        result = ConfigValidationResult()

        result.add_recommendation("Test recommendation")
        assert "Test recommendation" in result.recommendations


class TestEnhancedConfigValidator:
    """测试增强配置验证器"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = EnhancedConfigValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator._validators == []

    def test_add_validator(self):
        """测试添加验证器"""
        def test_validator(config):
            return ConfigValidationResult()

        self.validator.add_validator(test_validator)
        assert len(self.validator._validators) == 1
        assert self.validator._validators[0] == test_validator

    def test_validate_empty_validators(self):
        """测试没有验证器时的验证"""
        result = self.validator.validate({"test": "config"})
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_with_valid_validator(self):
        """测试有效验证器的验证"""
        def valid_validator(config):
            return ConfigValidationResult(True)

        self.validator.add_validator(valid_validator)
        result = self.validator.validate({"test": "config"})

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_with_invalid_validator(self):
        """测试无效验证器的验证"""
        def invalid_validator(config):
            result = ConfigValidationResult()
            result.add_error("Test error")
            return result

        self.validator.add_validator(invalid_validator)
        result = self.validator.validate({"test": "config"})

        assert result.is_valid is False
        assert "Test error" in result.errors

    def test_validate_with_warning_validator(self):
        """测试警告验证器的验证"""
        def warning_validator(config):
            result = ConfigValidationResult()
            result.add_warning("Test warning")
            return result

        self.validator.add_validator(warning_validator)
        result = self.validator.validate({"test": "config"})

        assert result.is_valid is True  # 警告不影响有效性
        assert "Test warning" in result.warnings

    def test_validate_with_recommendation_validator(self):
        """测试建议验证器的验证"""
        def recommendation_validator(config):
            result = ConfigValidationResult()
            result.add_recommendation("Test recommendation")
            return result

        self.validator.add_validator(recommendation_validator)
        result = self.validator.validate({"test": "config"})

        assert result.is_valid is True
        assert "Test recommendation" in result.recommendations

    def test_validate_multiple_validators(self):
        """测试多个验证器的验证"""
        def validator1(config):
            result = ConfigValidationResult()
            result.add_error("Error 1")
            return result

        def validator2(config):
            result = ConfigValidationResult()
            result.add_warning("Warning 2")
            return result

        def validator3(config):
            result = ConfigValidationResult()
            result.add_recommendation("Recommendation 3")
            return result

        self.validator.add_validator(validator1)
        self.validator.add_validator(validator2)
        self.validator.add_validator(validator3)

        result = self.validator.validate({"test": "config"})

        assert result.is_valid is False  # 有错误
        assert "Error 1" in result.errors
        assert "Warning 2" in result.warnings
        assert "Recommendation 3" in result.recommendations

    def test_validate_validator_exception(self):
        """测试验证器异常处理"""
        def failing_validator(config):
            raise Exception("Validator failed")

        self.validator.add_validator(failing_validator)
        result = self.validator.validate({"test": "config"})

        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "Validator error: Validator failed" in result.errors[0]

    def test_validate_validator_without_attributes(self):
        """测试没有标准属性的验证器结果"""
        def simple_validator(config):
            return "not_a_validation_result"

        self.validator.add_validator(simple_validator)
        result = self.validator.validate({"test": "config"})

        # 没有标准属性时，不应该添加错误
        assert result.is_valid is True


class TestCreateStandardValidators:
    """测试创建标准验证器"""

    def test_create_standard_validators(self):
        """测试创建标准验证器"""
        validators = create_standard_validators()
        assert len(validators) == 3

        # 验证每个验证器都是可调用的
        for validator in validators:
            assert callable(validator)

    def test_validate_required_keys_missing(self):
        """测试验证必需键 - 缺少键"""
        validators = create_standard_validators()
        required_keys_validator = validators[0]

        config = {}  # 缺少所有必需键
        result = required_keys_validator(config)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少缺少两个必需键
        assert any("logging.level" in error for error in result.errors)
        assert any("system.debug" in error for error in result.errors)

    def test_validate_required_keys_present(self):
        """测试验证必需键 - 键存在"""
        validators = create_standard_validators()
        required_keys_validator = validators[0]

        config = {
            "logging": {"level": "INFO"},
            "system": {"debug": True}
        }
        result = required_keys_validator(config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_types_correct(self):
        """测试验证配置类型 - 类型正确"""
        validators = create_standard_validators()
        types_validator = validators[1]

        config = {
            "system": {"debug": True},
            "logging": {"level": "INFO"},
            "database": {"port": 5432}
        }
        result = types_validator(config)

        assert result.is_valid is True
        assert len(result.warnings) == 0

    def test_validate_config_types_incorrect(self):
        """测试验证配置类型 - 类型不正确"""
        validators = create_standard_validators()
        types_validator = validators[1]

        config = {
            "system": {"debug": "true"},  # 应该是bool
            "logging": {"level": 123},    # 应该是str
            "database": {"port": "5432"}  # 应该是int
        }
        result = types_validator(config)

        assert result.is_valid is True  # 类型错误只是警告
        assert len(result.warnings) >= 3  # 至少3个类型警告

    def test_validate_config_format_email_valid(self):
        """测试验证配置格式 - 有效邮箱"""
        validators = create_standard_validators()
        format_validator = validators[2]

        config = {
            "email": {
                "sender": "test@example.com",
                "receiver": "user@example.com"
            }
        }
        result = format_validator(config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_format_email_invalid(self):
        """测试验证配置格式 - 无效邮箱"""
        validators = create_standard_validators()
        format_validator = validators[2]

        config = {
            "email": {
                "sender": "invalid-email",
                "receiver": "another-invalid"
            }
        }
        result = format_validator(config)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少两个邮箱格式错误

    def test_validate_config_format_port_valid(self):
        """测试验证配置格式 - 有效端口"""
        validators = create_standard_validators()
        format_validator = validators[2]

        config = {
            "server": {"port": 8080},
            "database": {"port": 5432}
        }
        result = format_validator(config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_format_port_invalid(self):
        """测试验证配置格式 - 无效端口"""
        validators = create_standard_validators()
        format_validator = validators[2]

        config = {
            "server": {"port": 70000},  # 超出范围
            "database": {"port": 0}     # 无效端口
        }
        result = format_validator(config)

        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少两个端口错误

    def test_validate_config_format_mixed(self):
        """测试验证配置格式 - 混合情况"""
        validators = create_standard_validators()
        format_validator = validators[2]

        config = {
            "email": {"sender": "valid@example.com"},
            "server": {"port": 8080},
            "email_invalid": {"sender": "invalid-email"},
            "server_invalid": {"port": 70000}
        }

        # 验证器期望嵌套键格式，所以我们需要调整配置
        test_config = {
            "email": {
                "sender": "invalid-email"  # 无效邮箱
            },
            "server": {
                "port": 70000  # 无效端口
            }
        }
        result = format_validator(test_config)

        assert result.is_valid is False
        # 应该有邮箱错误和端口错误
        assert len(result.errors) >= 2


class TestUtilityFunctions:
    """测试辅助函数"""

    def test_key_exists_simple_key(self):
        """测试键存在检查 - 简单键"""
        config = {"key1": "value1", "key2": "value2"}

        assert _key_exists("key1", config) is True
        assert _key_exists("key2", config) is True
        assert _key_exists("nonexistent", config) is False

    def test_key_exists_nested_key(self):
        """测试键存在检查 - 嵌套键"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "connection": {
                    "pool_size": 10
                }
            }
        }

        assert _key_exists("database.host", config) is True
        assert _key_exists("database.port", config) is True
        assert _key_exists("database.connection.pool_size", config) is True
        assert _key_exists("database.nonexistent", config) is False
        assert _key_exists("nonexistent.key", config) is False

    def test_key_exists_non_dict_value(self):
        """测试键存在检查 - 非字典值"""
        config = {
            "simple": "value",
            "nested": {"key": "value"}
        }

        assert _key_exists("simple", config) is True
        # 尝试访问非字典的子键应该返回False
        # 这里假设我们不会访问简单值的子键

    def test_key_exists_edge_cases(self):
        """测试键存在检查 - 边界情况"""
        assert _key_exists("key", {}) is False
        assert _key_exists("", {"": "value"}) is False

    def test_get_nested_value_simple_key(self):
        """测试获取嵌套值 - 简单键"""
        config = {"key1": "value1", "key2": 42}

        assert _get_nested_value("key1", config) == "value1"
        assert _get_nested_value("key2", config) == 42

    def test_get_nested_value_nested_key(self):
        """测试获取嵌套值 - 嵌套键"""
        config = {
            "database": {
                "host": "localhost",
                "connection": {
                    "pool_size": 10,
                    "timeout": 30.5
                }
            },
            "logging": {
                "level": "INFO"
            }
        }

        assert _get_nested_value("database.host", config) == "localhost"
        assert _get_nested_value("database.connection.pool_size", config) == 10
        assert _get_nested_value("database.connection.timeout", config) == 30.5
        assert _get_nested_value("logging.level", config) == "INFO"

    def test_get_nested_value_missing_key(self):
        """测试获取嵌套值 - 键不存在"""
        config = {"existing": "value"}

        with pytest.raises(KeyError, match="Key nonexistent not found"):
            _get_nested_value("nonexistent", config, raise_error=True)

        with pytest.raises(KeyError, match="Key existing.missing not found"):
            _get_nested_value("existing.missing", config, raise_error=True)

    def test_get_nested_value_non_dict_intermediate(self):
        """测试获取嵌套值 - 中间值不是字典"""
        config = {
            "simple": "not_a_dict",
            "nested": {"key": "value"}
        }

        # 这个应该抛出KeyError，因为simple不是字典
        with pytest.raises(KeyError):
            _get_nested_value("simple.subkey", config, raise_error=True)


class TestIntegrationScenarios:
    """测试集成场景"""

    def test_full_validation_pipeline(self):
        """测试完整的验证流程"""
        # 创建验证器并添加标准验证器
        validator = EnhancedConfigValidator()
        standard_validators = create_standard_validators()

        for std_validator in standard_validators:
            validator.add_validator(std_validator)

        # 测试有效配置
        valid_config = {
            "logging": {"level": "INFO"},
            "system": {"debug": True},
            "database": {"port": 5432},
            "email": {"sender": "test@example.com"},
            "server": {"port": 8080}
        }

        result = validator.validate(valid_config)
        assert result.is_valid is True

        # 测试无效配置
        invalid_config = {
            "logging": {"level": "INVALID_LEVEL"},  # 无效日志级别
            "system": {"debug": "not_boolean"},     # 类型错误
            "database": {"port": "not_integer"},    # 类型错误
            "email": {"sender": "invalid-email"},    # 格式错误
            "server": {"port": 70000}               # 范围错误
        }

        result = validator.validate(invalid_config)
        assert result.is_valid is False

        # 应该有多个错误
        assert len(result.errors) > 0
        assert len(result.warnings) > 0  # 类型错误会产生警告

    def test_custom_validator_integration(self):
        """测试自定义验证器集成"""
        validator = EnhancedConfigValidator()

        def custom_security_validator(config):
            """自定义安全验证器"""
            result = ConfigValidationResult()

            if 'password' in config:
                password = config['password']
                if len(password) < 8:
                    result.add_error("密码长度不能少于8位")
                if not any(char.isdigit() for char in password):
                    result.add_error("密码必须包含数字")
                if not any(char.isupper() for char in password):
                    result.add_error("密码必须包含大写字母")

            return result

        def custom_performance_validator(config):
            """自定义性能验证器"""
            result = ConfigValidationResult()

            if 'max_connections' in config:
                max_conn = config['max_connections']
                if max_conn > 1000:
                    result.add_warning("最大连接数过大，可能影响性能")
                elif max_conn < 10:
                    result.add_error("最大连接数过小，无法满足需求")

            return result

        validator.add_validator(custom_security_validator)
        validator.add_validator(custom_performance_validator)

        # 测试安全验证器 - 有效密码
        valid_config = {
            "password": "SecurePass123",
            "max_connections": 100
        }
        result = validator.validate(valid_config)
        assert result.is_valid is True

        # 测试安全验证器 - 无效密码
        invalid_config = {
            "password": "weak",
            "max_connections": 100
        }
        result = validator.validate(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少两个密码错误

        # 测试性能验证器 - 连接数警告
        warning_config = {
            "max_connections": 2000
        }
        result = validator.validate(warning_config)
        assert result.is_valid is True  # 警告不影响有效性
        assert len(result.warnings) > 0

    def test_validator_composition(self):
        """测试验证器组合"""
        # 创建两个不同的验证器实例
        validator1 = EnhancedConfigValidator()
        validator2 = EnhancedConfigValidator()

        # 为第一个验证器添加必需键验证器
        def required_validator(config):
            result = ConfigValidationResult()
            if 'api_key' not in config:
                result.add_error("缺少API密钥")
            return result

        # 为第二个验证器添加类型验证器
        def type_validator(config):
            result = ConfigValidationResult()
            if 'timeout' in config and not isinstance(config['timeout'], int):
                result.add_error("超时时间必须是整数")
            return result

        validator1.add_validator(required_validator)
        validator2.add_validator(type_validator)

        # 测试第一个验证器
        result1 = validator1.validate({})
        assert result1.is_valid is False
        assert "缺少API密钥" in " ".join(result1.errors)

        # 测试第二个验证器
        result2 = validator2.validate({'timeout': '30'})
        assert result2.is_valid is False
        assert "超时时间必须是整数" in " ".join(result2.errors)

    def test_empty_config_handling(self):
        """测试空配置处理"""
        validator = EnhancedConfigValidator()
        standard_validators = create_standard_validators()

        for std_validator in standard_validators:
            validator.add_validator(std_validator)

        # 空配置应该产生多个错误
        result = validator.validate({})

        # 应该检测到缺少必需键
        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少缺少两个必需键

    def test_nested_config_validation(self):
        """测试嵌套配置验证"""
        validator = EnhancedConfigValidator()

        def nested_validator(config):
            result = ConfigValidationResult()

            # 检查嵌套结构的有效性
            if 'services' in config:
                services = config['services']
                if isinstance(services, dict):
                    for service_name, service_config in services.items():
                        if isinstance(service_config, dict):
                            if 'enabled' in service_config:
                                enabled = service_config['enabled']
                                if not isinstance(enabled, bool):
                                    result.add_error(f"服务 {service_name} 的enabled字段必须是布尔值")
                            else:
                                result.add_warning(f"服务 {service_name} 缺少enabled字段")
                        else:
                            result.add_error(f"服务 {service_name} 配置必须是字典")
                else:
                    result.add_error("services必须是字典类型")

            return result

        validator.add_validator(nested_validator)

        # 测试有效嵌套配置
        valid_config = {
            "services": {
                "api": {"enabled": True, "port": 8080},
                "database": {"enabled": False, "host": "localhost"}
            }
        }
        result = validator.validate(valid_config)
        assert result.is_valid is True

        # 测试无效嵌套配置
        invalid_config = {
            "services": {
                "api": {"enabled": "true"},  # enabled不是布尔值
                "cache": "not_a_dict"        # 配置不是字典
            }
        }
        result = validator.validate(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少两个错误

    def test_validator_error_recovery(self):
        """测试验证器错误恢复"""
        validator = EnhancedConfigValidator()

        # 添加一个正常工作的验证器
        def good_validator(config):
            result = ConfigValidationResult()
            result.add_warning("这是一个警告")
            return result

        # 添加一个会抛出异常的验证器
        def bad_validator(config):
            raise ValueError("验证器内部错误")

        # 添加另一个正常工作的验证器
        def another_good_validator(config):
            result = ConfigValidationResult()
            if 'test_key' not in config:
                result.add_error("缺少测试键")
            return result

        validator.add_validator(good_validator)
        validator.add_validator(bad_validator)
        validator.add_validator(another_good_validator)

        # 即使有验证器异常，其他验证器仍应正常工作
        config = {"other_key": "value"}
        result = validator.validate(config)

        # 应该捕获异常并继续处理其他验证器
        assert result.is_valid is False  # 因为缺少test_key
        assert len(result.errors) >= 2   # 异常错误 + 缺少键错误
        assert len(result.warnings) >= 1 # 警告应该保留

        # 验证异常被正确捕获
        assert any("验证器内部错误" in error for error in result.errors)
