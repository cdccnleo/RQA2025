#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 增强配置验证器深度覆盖测试
测试enhanced_validators.py模块的所有功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch

from infrastructure.config.validators.enhanced_validators import (
    ConfigValidationResult, EnhancedConfigValidator, create_standard_validators,
    _key_exists, _get_nested_value
)


class TestConfigValidationResult:
    """测试配置验证结果类"""

    def test_initialization_valid(self):
        """测试有效结果初始化"""
        result = ConfigValidationResult(True)
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_initialization_invalid(self):
        """测试无效结果初始化"""
        result = ConfigValidationResult(False)
        assert result.is_valid is False
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_add_error(self):
        """测试添加错误"""
        result = ConfigValidationResult()
        result.add_error("Test error")
        assert result.errors == ["Test error"]
        assert result.is_valid is False

    def test_add_warning(self):
        """测试添加警告"""
        result = ConfigValidationResult()
        result.add_warning("Test warning")
        assert result.warnings == ["Test warning"]
        assert result.is_valid is True  # 警告不影响有效性

    def test_add_recommendation(self):
        """测试添加建议"""
        result = ConfigValidationResult()
        result.add_recommendation("Test recommendation")
        assert result.recommendations == ["Test recommendation"]
        assert result.is_valid is True

    def test_error_makes_invalid(self):
        """测试错误会使结果无效"""
        result = ConfigValidationResult(True)
        result.add_error("Test error")
        assert result.is_valid is False

    def test_multiple_messages(self):
        """测试多个消息"""
        result = ConfigValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_recommendation("Rec 1")
        result.add_recommendation("Rec 2")

        assert result.errors == ["Error 1", "Error 2"]
        assert result.warnings == ["Warning 1"]
        assert result.recommendations == ["Rec 1", "Rec 2"]
        assert result.is_valid is False


class TestEnhancedConfigValidator:
    """测试增强配置验证器"""

    def test_initialization(self):
        """测试初始化"""
        validator = EnhancedConfigValidator()
        assert validator._validators == []

    def test_add_validator(self):
        """测试添加验证器"""
        validator = EnhancedConfigValidator()

        def test_validator(config):
            return ConfigValidationResult(True)

        validator.add_validator(test_validator)
        assert len(validator._validators) == 1
        assert validator._validators[0] == test_validator

    def test_validate_empty_validators(self):
        """测试空验证器列表验证"""
        validator = EnhancedConfigValidator()
        result = validator.validate({})
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.recommendations == []

    def test_validate_single_validator_success(self):
        """测试单个验证器成功"""
        validator = EnhancedConfigValidator()

        def success_validator(config):
            result = ConfigValidationResult(True)
            result.add_warning("Test warning")
            return result

        validator.add_validator(success_validator)
        result = validator.validate({"test": "config"})

        assert result.is_valid is True
        assert result.warnings == ["Test warning"]
        assert result.errors == []

    def test_validate_single_validator_failure(self):
        """测试单个验证器失败"""
        validator = EnhancedConfigValidator()

        def failure_validator(config):
            result = ConfigValidationResult(False)
            result.add_error("Test error")
            result.add_recommendation("Fix it")
            return result

        validator.add_validator(failure_validator)
        result = validator.validate({"test": "config"})

        assert result.is_valid is False
        assert result.errors == ["Test error"]
        assert result.recommendations == ["Fix it"]

    def test_validate_multiple_validators_mixed(self):
        """测试多个验证器混合结果"""
        validator = EnhancedConfigValidator()

        def success_validator(config):
            result = ConfigValidationResult(True)
            result.add_warning("Warning from success")
            return result

        def failure_validator(config):
            result = ConfigValidationResult(False)
            result.add_error("Error from failure")
            result.add_recommendation("Fix failure")
            return result

        validator.add_validator(success_validator)
        validator.add_validator(failure_validator)

        result = validator.validate({"test": "config"})

        assert result.is_valid is False
        assert result.errors == ["Error from failure"]
        assert result.warnings == ["Warning from success"]
        assert result.recommendations == ["Fix failure"]

    def test_validate_exception_handling(self):
        """测试异常处理"""
        validator = EnhancedConfigValidator()

        def exception_validator(config):
            raise ValueError("Test exception")

        validator.add_validator(exception_validator)

        # 应该不会抛出异常，而是返回无效结果
        result = validator.validate({"test": "config"})
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert "异常" in result.errors[0] or "exception" in result.errors[0].lower()

    def test_validate_with_config_passing(self):
        """测试配置传递给验证器"""
        validator = EnhancedConfigValidator()
        received_config = None

        def config_capturing_validator(config):
            nonlocal received_config
            received_config = config
            return ConfigValidationResult(True)

        validator.add_validator(config_capturing_validator)
        test_config = {"test": "value", "nested": {"key": "value"}}

        validator.validate(test_config)
        assert received_config == test_config


class TestCreateStandardValidators:
    """测试标准验证器创建函数"""

    def test_create_standard_validators_returns_list(self):
        """测试返回列表"""
        validators = create_standard_validators()
        assert isinstance(validators, list)
        assert len(validators) > 0

    def test_create_standard_validators_each_is_callable(self):
        """测试每个验证器都是可调用的"""
        validators = create_standard_validators()
        for validator in validators:
            assert callable(validator)

    def test_create_standard_validators_execution(self):
        """测试验证器执行"""
        validators = create_standard_validators()

        # 测试第一个验证器
        if validators:
            result = validators[0]({"test": "config"})
            assert isinstance(result, ConfigValidationResult)

    def test_create_standard_validators_with_various_configs(self):
        """测试不同配置的验证器"""
        validators = create_standard_validators()
        test_configs = [
            {},
            {"test": "value"},
            {"database": {"host": "localhost"}},
            {"logging": {"level": "INFO"}},
            {"network": {"port": 8080}}
        ]

        for config in test_configs:
            for validator in validators:
                result = validator(config)
                assert isinstance(result, ConfigValidationResult)
                assert hasattr(result, 'is_valid')


class TestKeyExists:
    """测试_key_exists函数"""

    def test_key_exists_simple(self):
        """测试简单键存在"""
        config = {"key1": "value1", "key2": "value2"}
        assert _key_exists("key1", config) is True
        assert _key_exists("key2", config) is True
        assert _key_exists("key3", config) is False

    def test_key_exists_nested(self):
        """测试嵌套键存在"""
        config = {
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        }
        assert _key_exists("level1", config) is True
        assert _key_exists("level1.level2", config) is True
        assert _key_exists("level1.level2.key", config) is True
        assert _key_exists("level1.level3", config) is False
        assert _key_exists("level1.level2.missing", config) is False

    def test_key_exists_empty_config(self):
        """测试空配置"""
        assert _key_exists("any_key", {}) is False

    def test_key_exists_none_config(self):
        """测试None配置"""
        assert _key_exists("any_key", None) is False

    def test_key_exists_complex_path(self):
        """测试复杂路径"""
        config = {
            "database": {
                "connection": {
                    "pool": {
                        "max_size": 10
                    }
                }
            }
        }
        assert _key_exists("database.connection.pool.max_size", config) is True
        assert _key_exists("database.connection.pool.min_size", config) is False


class TestGetNestedValue:
    """测试_get_nested_value函数"""

    def test_get_nested_value_simple(self):
        """测试简单嵌套值获取"""
        config = {"key": "value"}
        assert _get_nested_value("key", config) == "value"

    def test_get_nested_value_nested(self):
        """测试嵌套值获取"""
        config = {
            "level1": {
                "level2": {
                    "key": "nested_value"
                }
            }
        }
        assert _get_nested_value("level1.level2.key", config) == "nested_value"

    def test_get_nested_value_default(self):
        """测试默认值"""
        config = {"key1": "value1"}
        assert _get_nested_value("missing_key", config, "default") == "default"
        assert _get_nested_value("missing.nested.key", config, "default") == "default"

    def test_get_nested_value_no_default(self):
        """测试无默认值"""
        config = {"key1": "value1"}
        assert _get_nested_value("missing_key", config) is None

    def test_get_nested_value_complex_types(self):
        """测试复杂类型值"""
        config = {
            "list": [1, 2, 3],
            "dict": {"nested": "value"},
            "number": 42,
            "boolean": True
        }
        assert _get_nested_value("list", config) == [1, 2, 3]
        assert _get_nested_value("dict", config) == {"nested": "value"}
        assert _get_nested_value("number", config) == 42
        assert _get_nested_value("boolean", config) is True

    def test_get_nested_value_empty_path(self):
        """测试空路径"""
        config = {"key": "value"}
        assert _get_nested_value("", config) is None


class TestEnhancedValidatorsIntegration:
    """测试增强验证器集成场景"""

    def test_complete_validation_workflow(self):
        """测试完整验证工作流"""
        # 创建验证器
        validator = EnhancedConfigValidator()

        # 添加多个验证器
        def database_validator(config):
            result = ConfigValidationResult()
            if _key_exists("database.host", config):
                host = _get_nested_value("database.host", config)
                if not isinstance(host, str) or not host:
                    result.add_error("数据库主机无效")
            else:
                result.add_warning("建议配置数据库主机")
            return result

        def logging_validator(config):
            result = ConfigValidationResult()
            if _key_exists("logging.level", config):
                level = _get_nested_value("logging.level", config)
                valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
                if level not in valid_levels:
                    result.add_error(f"无效的日志级别: {level}")
            return result

        validator.add_validator(database_validator)
        validator.add_validator(logging_validator)

        # 测试有效配置
        valid_config = {
            "database": {"host": "localhost"},
            "logging": {"level": "INFO"}
        }
        result = validator.validate(valid_config)
        assert result.is_valid is True

        # 测试无效配置
        invalid_config = {
            "database": {"host": ""},
            "logging": {"level": "INVALID"}
        }
        result = validator.validate(invalid_config)
        assert result.is_valid is False
        assert len(result.errors) >= 1

    def test_validator_exception_isolation(self):
        """测试验证器异常隔离"""
        validator = EnhancedConfigValidator()

        def good_validator(config):
            result = ConfigValidationResult(True)
            result.add_warning("Good warning")
            return result

        def bad_validator(config):
            raise RuntimeError("Bad validator error")

        def another_good_validator(config):
            result = ConfigValidationResult(True)
            result.add_recommendation("Good recommendation")
            return result

        validator.add_validator(good_validator)
        validator.add_validator(bad_validator)
        validator.add_validator(another_good_validator)

        result = validator.validate({"test": "config"})

        # 应该捕获异常，但不影响其他验证器
        assert result.is_valid is False  # 因为有一个验证器失败
        assert len(result.errors) >= 1  # 应该有异常错误
        assert "Good warning" in result.warnings
        assert "Good recommendation" in result.recommendations

    def test_empty_config_handling(self):
        """测试空配置处理"""
        validator = EnhancedConfigValidator()

        def strict_validator(config):
            result = ConfigValidationResult()
            if not config:
                result.add_error("配置不能为空")
            return result

        validator.add_validator(strict_validator)

        # 空配置应该失败
        result = validator.validate({})
        assert result.is_valid is False
        assert "配置不能为空" in " ".join(result.errors)

    def test_config_modification_safety(self):
        """测试配置修改安全性"""
        validator = EnhancedConfigValidator()

        def modifying_validator(config):
            # 尝试修改配置（应该避免）
            if isinstance(config, dict):
                config["modified"] = True
            return ConfigValidationResult(True)

        validator.add_validator(modifying_validator)

        original_config = {"original": True}
        config_copy = original_config.copy()

        validator.validate(original_config)

        # 验证器不应该修改原始配置
        assert original_config == config_copy
        assert "modified" not in original_config
