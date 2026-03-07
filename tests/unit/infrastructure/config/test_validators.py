#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证器测试
测试验证器功能完整性
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from infrastructure.config.validators import (
    ValidationSeverity,
    ValidationType,
    ValidationResult,
    IConfigValidator,
    BaseConfigValidator,
    TradingHoursValidator,
    DatabaseConfigValidator,
    LoggingConfigValidator,
    NetworkConfigValidator,
    ConfigValidators,
    UnifiedValidatorFactory,
    ConfigValidator,
    create_validator,
    create_validator_suite,
    validate_config_with_suite,
    get_validator_factory,
    reset_validator_factory,
    validate_trading_hours,
    validate_database_config,
    validate_logging_config,
    validate_network_config
)


class TestValidationSeverity:
    """测试验证严重程度枚举"""

    def test_validation_severity_values(self):
        """测试验证严重程度值"""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_validation_severity_ordering(self):
        """测试验证严重程度排序"""
        severities = [ValidationSeverity.INFO, ValidationSeverity.WARNING,
                     ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]

        # 验证值递增
        values = [s.value for s in severities]
        assert values == ["info", "warning", "error", "critical"]


class TestValidationType:
    """测试验证类型枚举"""

    def test_validation_type_values(self):
        """测试验证类型值"""
        assert ValidationType.REQUIRED.value == "required"
        assert ValidationType.TYPE.value == "type"
        assert ValidationType.RANGE.value == "range"
        assert ValidationType.PATTERN.value == "pattern"
        assert ValidationType.CUSTOM.value == "custom"
        assert ValidationType.DEPENDENCY.value == "dependency"


class TestValidationResult:
    """测试验证结果类"""

    def test_initialization(self):
        """测试初始化"""
        result = ValidationResult(True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.severity == ValidationSeverity.INFO

        result_with_errors = ValidationResult(False, ["error1", "error2"])
        assert result_with_errors.is_valid is False
        assert result_with_errors.errors == ["error1", "error2"]

    def test_add_error(self):
        """测试添加错误"""
        result = ValidationResult(True)

        result.add_error("Test error")
        assert result.is_valid is False
        assert "Test error" in result.errors
        assert result.severity == ValidationSeverity.ERROR

        # 测试严重程度升级
        result.add_error("Critical error", ValidationSeverity.CRITICAL)
        assert result.severity == ValidationSeverity.CRITICAL

    def test_add_warning(self):
        """测试添加警告"""
        result = ValidationResult(True)

        result.add_warning("Test warning")
        assert result.is_valid is True  # 警告不影响有效性
        assert "Test warning" in result.warnings

    def test_to_dict(self):
        """测试转换为字典"""
        result = ValidationResult(False, ["error1"])
        result.add_warning("warning1")
        result.add_error("error2", ValidationSeverity.WARNING)

        data = result.to_dict()
        assert data['is_valid'] is False
        assert data['errors'] == ["error1", "error2"]
        assert data['warnings'] == ["warning1"]
        assert data['severity'] == "warning"
        assert data['error_count'] == 2
        assert data['warning_count'] == 1

    def test_merge(self):
        """测试合并验证结果"""
        result1 = ValidationResult(True)
        result1.add_warning("warning1")

        result2 = ValidationResult(False, ["error1"])
        result2.add_error("error2", ValidationSeverity.CRITICAL)

        result1.merge(result2)

        assert result1.is_valid is False
        assert "error1" in result1.errors
        assert "error2" in result1.errors
        assert "warning1" in result1.warnings
        assert result1.severity == ValidationSeverity.CRITICAL


class TestBaseConfigValidator:
    """测试基础配置验证器"""

    def test_initialization(self):
        """测试初始化"""
        # 使用具体的验证器类来测试
        from unittest.mock import Mock
        validator = TradingHoursValidator()
        validator.validate = Mock(return_value=ValidationResult(True))

        # 测试属性
        assert validator.name == "TradingHoursValidator"
        assert validator.description == "验证交易时段配置"

        # 测试BaseConfigValidator的属性访问
        assert hasattr(validator, 'name')
        assert hasattr(validator, 'description')


class TestTradingHoursValidator:
    """测试交易时段验证器"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = TradingHoursValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.name == "TradingHoursValidator"
        assert self.validator.description == "验证交易时段配置"

    def test_validate_missing_trading_hours(self):
        """测试缺少trading_hours字段"""
        result = self.validator.validate({})
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "缺少trading_hours字段" in result.errors[0]

    def test_validate_invalid_trading_hours_type(self):
        """测试trading_hours字段类型无效"""
        config = {"trading_hours": "invalid"}
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "trading_hours必须是字典类型" in result.errors[0]

    def test_validate_valid_trading_hours(self):
        """测试有效的交易时段配置"""
        config = {
            "trading_hours": {
                "morning": ["09:30", "11:30"],
                "afternoon": ["13:00", "15:00"]
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_invalid_time_format(self):
        """测试无效的时间格式"""
        config = {
            "trading_hours": {
                "morning": ["invalid", "11:30"]
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "时间格式不正确" in " ".join(result.errors)

    def test_validate_overlapping_hours(self):
        """测试重叠的交易时段"""
        config = {
            "trading_hours": {
                "morning": ["09:30", "12:00"],
                "afternoon": ["11:00", "15:00"]  # 与morning重叠
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True  # 重叠只是警告，不影响有效性
        assert len(result.warnings) > 0
        assert "重叠" in " ".join(result.warnings)


class TestDatabaseConfigValidator:
    """测试数据库配置验证器"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = DatabaseConfigValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.name == "DatabaseConfigValidator"
        assert self.validator.description == "验证数据库配置"

    def test_validate_missing_database_config(self):
        """测试缺少database配置"""
        result = self.validator.validate({})
        assert result.is_valid is False  # 缺少配置是错误
        assert len(result.errors) > 0

    def test_validate_missing_required_fields(self):
        """测试缺少必需字段"""
        config = {"database": {"host": "localhost"}}  # 缺少port和database
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert len(result.errors) >= 2  # 至少缺少两个字段

    def test_validate_invalid_port(self):
        """测试无效端口"""
        config = {
            "database": {
                "host": "localhost",
                "port": 70000,  # 无效端口
                "name": "test",
                "username": "testuser"
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "端口必须是1024-65535之间的整数" in " ".join(result.errors)

    def test_validate_invalid_port_type(self):
        """测试端口类型无效"""
        config = {
            "database": {
                "host": "localhost",
                "port": "5432",  # 应该是整数
                "database": "test"
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "端口必须是1024-65535之间的整数" in " ".join(result.errors)

    def test_validate_invalid_pool_config(self):
        """测试无效连接池配置"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "database": "test",
                "pool": {
                    "min_size": 10,
                    "max_size": 5  # min > max
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "最小大小不能大于最大大小" in " ".join(result.errors)

    def test_validate_valid_database_config(self):
        """测试有效的数据库配置"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test",
                "username": "testuser",
                "pool": {
                    "min_size": 1,
                    "max_size": 10
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_database_config_compatibility(self):
        """测试数据库配置验证兼容性方法"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test",
                "username": "testuser"
            }
        }
        result = self.validator.validate_database_config(config)
        assert result.is_valid is True


class TestLoggingConfigValidator:
    """测试日志配置验证器"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = LoggingConfigValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.name == "LoggingConfigValidator"
        assert self.validator.description == "验证日志配置"

    def test_validate_missing_logging_config(self):
        """测试缺少logging配置"""
        result = self.validator.validate({})
        assert result.is_valid is True  # 缺少配置只是警告
        assert len(result.warnings) > 0

    def test_validate_invalid_log_level(self):
        """测试无效日志级别"""
        config = {
            "logging": {
                "level": "INVALID"
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "无效的日志级别" in " ".join(result.errors)

    def test_validate_valid_log_level(self):
        """测试有效日志级别"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            config = {"logging": {"level": level}}
            result = self.validator.validate(config)
            assert result.is_valid is True, f"日志级别 {level} 应该有效"

        # 测试小写
        config = {"logging": {"level": "info"}}
        result = self.validator.validate(config)
        assert result.is_valid is True

    def test_validate_logging_file_config(self):
        """测试日志文件配置"""
        config = {
            "logging": {
                "file": {
                    "path": "/var/log/app.log",
                    "max_size": "10MB"
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True

    def test_validate_logging_file_missing_path(self):
        """测试日志文件缺少路径"""
        config = {
            "logging": {
                "file": {
                    "max_size": "10MB"
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "缺少path字段" in " ".join(result.errors)

    def test_validate_invalid_max_size(self):
        """测试无效的最大文件大小"""
        config = {
            "logging": {
                "file": {
                    "path": "/var/log/app.log",
                    "max_size": "INVALID"
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert "无效的日志文件大小格式" in " ".join(result.errors)

    def test_validate_valid_max_size_formats(self):
        """测试有效的大小格式"""
        valid_formats = ["10MB", "1GB", "500KB", "100B", "2TB"]
        for size_format in valid_formats:
            config = {
                "logging": {
                    "file": {
                        "path": "/var/log/app.log",
                        "max_size": size_format
                    }
                }
            }
            result = self.validator.validate(config)
            assert result.is_valid is True, f"大小格式 {size_format} 应该有效"


class TestNetworkConfigValidator:
    """测试网络配置验证器"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = NetworkConfigValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.name == "NetworkConfigValidator"
        assert self.validator.description == "验证网络配置"

    def test_validate_missing_network_config(self):
        """测试缺少network配置"""
        result = self.validator.validate({})
        assert result.is_valid is True  # 缺少配置只是警告
        assert len(result.warnings) > 0

    def test_validate_valid_host(self):
        """测试有效主机配置"""
        valid_hosts = ["localhost", "127.0.0.1", "0.0.0.0", "192.168.1.1"]
        for host in valid_hosts:
            config = {"network": {"host": host}}
            result = self.validator.validate(config)
            assert result.is_valid is True, f"主机 {host} 应该有效"

    def test_validate_invalid_host(self):
        """测试无效主机配置"""
        config = {"network": {"host": "invalid.host.name"}}
        result = self.validator.validate(config)
        assert result.is_valid is True  # 无效主机只是警告
        assert len(result.warnings) > 0

    def test_validate_invalid_port(self):
        """测试无效端口"""
        invalid_ports = [0, -1, 70000, "5432"]  # 移除None，因为None被认为是缺少值
        for port in invalid_ports:
            config = {"network": {"port": port}}
            result = self.validator.validate(config)
            assert result.is_valid is False, f"端口 {port} 应该无效"

    def test_validate_valid_port(self):
        """测试有效端口"""
        config = {"network": {"port": 8080}}
        result = self.validator.validate(config)
        assert result.is_valid is True

    def test_validate_ssl_config_missing_fields(self):
        """测试SSL配置缺少字段"""
        config = {
            "network": {
                "ssl": {
                    "enabled": True
                    # 缺少cert_file和key_file
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is False
        assert len([e for e in result.errors if "必须配置" in e]) >= 2

    def test_validate_ssl_config_complete(self):
        """测试完整的SSL配置"""
        config = {
            "network": {
                "ssl": {
                    "enabled": True,
                    "cert_file": "/path/to/cert.pem",
                    "key_file": "/path/to/key.pem"
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True

    def test_validate_ssl_disabled(self):
        """测试SSL禁用"""
        config = {
            "network": {
                "ssl": {
                    "enabled": False
                    # 不需要cert_file和key_file
                }
            }
        }
        result = self.validator.validate(config)
        assert result.is_valid is True

    def test_is_valid_ip(self):
        """测试IP地址验证"""
        assert self.validator._is_valid_ip("192.168.1.1") is True
        assert self.validator._is_valid_ip("2001:db8::1") is True
        assert self.validator._is_valid_ip("invalid.ip") is False
        assert self.validator._is_valid_ip("256.1.1.1") is False


class TestConfigValidators:
    """测试配置验证器组合"""

    def test_initialization(self):
        """测试初始化"""
        validators = [TradingHoursValidator(), DatabaseConfigValidator()]
        config_validators = ConfigValidators(validators)
        assert len(config_validators.validators) == 2

    def test_validate_all_valid(self):
        """测试所有验证都通过"""
        validators = [TradingHoursValidator(), DatabaseConfigValidator()]
        config_validators = ConfigValidators(validators)

        config = {
            "trading_hours": {
                "morning": ["09:30", "11:30"]
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test",
                "username": "testuser"
            }
        }

        is_valid, result = config_validators.validate(config)
        assert is_valid is True
        assert result.is_valid is True

    def test_validate_with_errors(self):
        """测试有验证错误"""
        validators = [TradingHoursValidator()]
        config_validators = ConfigValidators(validators)

        config = {}  # 缺少trading_hours

        is_valid, result = config_validators.validate(config)
        assert is_valid is False
        assert result is not None
        assert hasattr(result, 'is_valid')
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_with_warnings(self):
        """测试有验证警告"""
        validators = [DatabaseConfigValidator()]
        config_validators = ConfigValidators(validators)

        config = {}  # 缺少database配置

        is_valid, result = config_validators.validate(config)
        assert is_valid is False  # 缺少database配置是错误
        assert result is not None
        assert len(result.errors) > 0

    def test_validate_exception_handling(self):
        """测试异常处理"""
        # 创建一个会抛出异常的验证器
        class FailingValidator(BaseConfigValidator):
            def validate(self, config):
                raise Exception("Test exception")

        validators = [FailingValidator()]
        config_validators = ConfigValidators(validators)

        is_valid, result = config_validators.validate({})
        assert is_valid is False
        assert result is not None
        assert len(result.errors) > 0


class TestUnifiedValidatorFactory:
    """测试统一验证器工厂"""

    def setup_method(self):
        """设置测试方法"""
        self.factory = UnifiedValidatorFactory()

    def test_initialization(self):
        """测试初始化"""
        assert len(self.factory._validator_classes) == 4  # 默认注册的4个验证器
        assert 'trading_hours' in self.factory._validator_classes
        assert 'database' in self.factory._validator_classes
        assert 'logging' in self.factory._validator_classes
        assert 'network' in self.factory._validator_classes

    def test_register_validator(self):
        """测试注册验证器"""
        class TestValidator(BaseConfigValidator):
            pass

        self.factory.register_validator("test", TestValidator)
        assert "test" in self.factory._validator_classes
        assert self.factory._validator_classes["test"] == TestValidator

    def test_register_invalid_validator(self):
        """测试注册无效验证器"""
        class InvalidValidator:
            pass

        with pytest.raises(ValueError, match="必须实现IConfigValidator接口"):
            self.factory.register_validator("invalid", InvalidValidator)

    def test_create_validator(self):
        """测试创建验证器"""
        validator = self.factory.create_validator("database")
        assert isinstance(validator, DatabaseConfigValidator)

    def test_create_unknown_validator(self):
        """测试创建未知验证器"""
        with pytest.raises(ValueError, match="未知的验证器类型"):
            self.factory.create_validator("unknown")

    def test_get_available_validators(self):
        """测试获取可用验证器"""
        available = self.factory.get_available_validators()
        assert len(available) == 4
        assert "database" in available

    def test_create_validator_suite(self):
        """测试创建验证器套件"""
        suite = self.factory.create_validator_suite(["database", "logging"])
        assert len(suite.validators) == 2
        assert isinstance(suite.validators[0], DatabaseConfigValidator)
        assert isinstance(suite.validators[1], LoggingConfigValidator)

    def test_create_validator_suite_with_unknown(self):
        """测试创建包含未知验证器的套件"""
        suite = self.factory.create_validator_suite(["database", "unknown"])
        assert len(suite.validators) == 1  # 只有有效的验证器被创建
        assert isinstance(suite.validators[0], DatabaseConfigValidator)


class TestGlobalValidatorFactory:
    """测试全局验证器工厂"""

    def test_get_validator_factory(self):
        """测试获取验证器工厂"""
        factory = get_validator_factory()
        assert isinstance(factory, UnifiedValidatorFactory)

        # 测试单例模式
        factory2 = get_validator_factory()
        assert factory is factory2

    def test_reset_validator_factory(self):
        """测试重置验证器工厂"""
        factory1 = get_validator_factory()
        reset_validator_factory()
        factory2 = get_validator_factory()

        assert factory1 is not factory2


class TestConfigValidator:
    """测试配置验证器（向后兼容）"""

    def setup_method(self):
        """设置测试方法"""
        self.validator = ConfigValidator()

    def test_initialization(self):
        """测试初始化"""
        assert self.validator.name == "ConfigValidator"
        assert self.validator.description == "简单配置验证器"

    def test_validate_without_rules(self):
        """测试没有验证规则的验证"""
        result = self.validator.validate({"any": "config"})
        assert result is True

    def test_validate_with_rules(self):
        """测试带验证规则的验证"""
        def port_rule(value):
            return isinstance(value, int) and 1024 <= value <= 65535

        self.validator.add_validation_rule("database.port", port_rule)

        # 有效配置
        valid_config = {"database": {"port": 5432}}
        assert self.validator.validate(valid_config) is True

        # 无效配置
        invalid_config = {"database": {"port": 70000}}
        assert self.validator.validate(invalid_config) is False

    def test_add_validation_rule(self):
        """测试添加验证规则"""
        def test_rule(value):
            return value == "test"

        self.validator.add_validation_rule("field", test_rule)
        assert "field" in self.validator.rules
        assert self.validator.rules["field"] == test_rule

    def test_get_nested_value(self):
        """测试获取嵌套值"""
        config = {
            "database": {
                "connection": {
                    "host": "localhost",
                    "port": 5432
                }
            }
        }

        assert self.validator._get_nested_value(config, "database.connection.host") == "localhost"
        assert self.validator._get_nested_value(config, "database.connection.port") == 5432

    def test_get_nested_value_missing_key(self):
        """测试获取不存在的嵌套值"""
        config = {"database": {}}

        with pytest.raises(KeyError):
            self.validator._get_nested_value(config, "database.missing.key")


class TestConvenienceFunctions:
    """测试便捷函数"""

    def test_create_validator(self):
        """测试创建验证器便捷函数"""
        validator = create_validator("database")
        assert isinstance(validator, DatabaseConfigValidator)

    def test_create_validator_suite(self):
        """测试创建验证器套件便捷函数"""
        suite = create_validator_suite(["database", "logging"])
        assert len(suite.validators) == 2

    def test_validate_config_with_suite(self):
        """测试使用套件验证配置的便捷函数"""
        config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test",
                "username": "testuser"
            }
        }

        is_valid, result = validate_config_with_suite(config, ["database"])
        assert is_valid is True

    def test_legacy_validation_functions(self):
        """测试遗留验证函数"""
        # 有效的交易时段配置
        trading_config = {
            "trading_hours": {
                "morning": ["09:30", "11:30"]
            }
        }
        assert validate_trading_hours(trading_config) is True

        # 有效的数据库配置
        db_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "username": "testuser",
                "password": "testpass"
            }
        }
        assert validate_database_config(db_config) is True

        # 有效的日志配置
        logging_config = {
            "logging": {
                "level": "INFO"
            }
        }
        assert validate_logging_config(logging_config) is True

        # 有效的网络配置
        network_config = {
            "network": {
                "host": "localhost",
                "port": 8080
            }
        }
        assert validate_network_config(network_config) is True

        # 无效配置测试
        assert validate_trading_hours({}) is False
        assert validate_database_config({}) is False
        assert validate_logging_config({"logging": {"level": "INVALID"}}) is False
        assert validate_network_config({"network": {"port": 70000}}) is False
