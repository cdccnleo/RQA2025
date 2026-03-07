#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Config Validators 测试

测试 src/infrastructure/config/core/config_validators.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock

# 尝试导入模块
try:
    from src.infrastructure.config.core.config_validators import (
        ConfigKeyValidator,
        ConfigValueValidator
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigKeyValidator:
    """测试ConfigKeyValidator功能"""

    def test_validate_key_valid_strings(self):
        """测试有效键的验证"""
        valid_keys = [
            "valid_key",
            "key123",
            "nested.key",
            "deeply.nested.key",
            "key_with_underscores",
            "key-with-dashes",
            "key.with.multiple.dots"
        ]
        
        for key in valid_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is True, f"Key '{key}' should be valid but got error: {error}"
            assert error is None

    def test_validate_key_none(self):
        """测试None键验证"""
        is_valid, error = ConfigKeyValidator.validate_key(None)
        assert is_valid is False
        assert error == "Key cannot be None"

    def test_validate_key_empty_string(self):
        """测试空字符串键验证"""
        is_valid, error = ConfigKeyValidator.validate_key("")
        assert is_valid is False
        assert error == "Key must be a non-empty string"

    def test_validate_key_non_string(self):
        """测试非字符串键验证"""
        non_string_keys = [123, [], {}, True, False]
        
        for key in non_string_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert error == "Key must be a non-empty string"

    def test_validate_key_length_exceeded(self):
        """测试键长度超限"""
        long_key = "a" * (ConfigKeyValidator.MAX_KEY_LENGTH + 1)
        
        is_valid, error = ConfigKeyValidator.validate_key(long_key)
        assert is_valid is False
        assert f"Key length exceeds maximum {ConfigKeyValidator.MAX_KEY_LENGTH} characters" in error

    def test_validate_key_max_length(self):
        """测试最大长度键（边界值）"""
        max_length_key = "a" * ConfigKeyValidator.MAX_KEY_LENGTH
        
        is_valid, error = ConfigKeyValidator.validate_key(max_length_key)
        assert is_valid is True
        assert error is None

    def test_validate_key_invalid_format_dot_only(self):
        """测试只有点的键"""
        is_valid, error = ConfigKeyValidator.validate_key(".")
        assert is_valid is False
        assert "Invalid key format: cannot start/end with '.' or contain '..'" in error

    def test_validate_key_invalid_format_start_dot(self):
        """测试以点开头的键"""
        invalid_keys = [".key", ".nested.key"]
        
        for key in invalid_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert "Invalid key format: cannot start/end with '.' or contain '..'" in error

    def test_validate_key_invalid_format_end_dot(self):
        """测试以点结尾的键"""
        invalid_keys = ["key.", "nested.key."]
        
        for key in invalid_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert "Invalid key format: cannot start/end with '.' or contain '..'" in error

    def test_validate_key_invalid_format_double_dots(self):
        """测试包含双点的键"""
        invalid_keys = ["key..test", "..key", "key..", "test..key.nested"]
        
        for key in invalid_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert "Invalid key format: cannot start/end with '.' or contain '..'" in error

    def test_validate_key_dangerous_characters(self):
        """测试危险字符"""
        dangerous_chars = ConfigKeyValidator.DANGEROUS_CHARS
        
        for char in dangerous_chars:
            key = f"test{char}key"
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert f"Key contains dangerous characters: {dangerous_chars}" in error

    def test_validate_key_multiple_dangerous_characters(self):
        """测试多个危险字符"""
        key = "test<key>with;dangerous/chars\\"
        
        is_valid, error = ConfigKeyValidator.validate_key(key)
        assert is_valid is False
        assert "Key contains dangerous characters" in error

    def test_validate_key_whitespace(self):
        """测试包含空格的键"""
        key = "test key"
        
        is_valid, error = ConfigKeyValidator.validate_key(key)
        assert is_valid is False
        assert "Key contains dangerous characters" in error

    def test_validate_key_valid_complex(self):
        """测试复杂的有效键"""
        valid_complex_keys = [
            "app_configuration",
            "database.connection.pool",
            "cache.redis.host",
            "monitoring.metrics.collection.interval",
            "auth.jwt.secret_key"
        ]
        
        for key in valid_complex_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is True, f"Complex key '{key}' should be valid but got error: {error}"

    def test_parse_key_structure_valid(self):
        """测试解析有效键结构"""
        test_cases = [
            ("simple", ["simple"]),
            ("nested.key", ["nested", "key"]),
            ("deeply.nested.structure", ["deeply", "nested", "structure"]),
            ("single", ["single"])
        ]
        
        for key, expected_parts in test_cases:
            is_valid, parts, error = ConfigKeyValidator.parse_key_structure(key)
            assert is_valid is True
            assert parts == expected_parts
            assert error is None

    def test_parse_key_structure_empty(self):
        """测试解析空键"""
        is_valid, parts, error = ConfigKeyValidator.parse_key_structure("")
        # 空字符串split('.')返回['']，len为1，所以实际实现认为这是有效的
        assert is_valid is True
        assert parts == ['']
        assert error is None

    def test_parse_key_structure_complex(self):
        """测试解析复杂键结构"""
        complex_key = "database.connection.pool.settings.max_connections"
        expected_parts = ["database", "connection", "pool", "settings", "max_connections"]
        
        is_valid, parts, error = ConfigKeyValidator.parse_key_structure(complex_key)
        assert is_valid is True
        assert parts == expected_parts
        assert error is None

    def test_constants(self):
        """测试常量定义"""
        assert hasattr(ConfigKeyValidator, 'DANGEROUS_CHARS')
        assert isinstance(ConfigKeyValidator.DANGEROUS_CHARS, set)
        assert len(ConfigKeyValidator.DANGEROUS_CHARS) > 0
        
        assert hasattr(ConfigKeyValidator, 'MAX_KEY_LENGTH')
        assert isinstance(ConfigKeyValidator.MAX_KEY_LENGTH, int)
        assert ConfigKeyValidator.MAX_KEY_LENGTH > 0

    def test_dangerous_chars_completeness(self):
        """测试危险字符集合的完整性"""
        dangerous_chars = ConfigKeyValidator.DANGEROUS_CHARS
        
        # 验证包含预期的危险字符
        expected_chars = {'<', '>', ';', ' ', '/', '\\', ':', '@', '#'}
        assert dangerous_chars == expected_chars

    def test_multiple_dots_validation(self):
        """测试多点键的验证（应该在validate_key中被拒绝）"""
        multi_dot_keys = [
            "key.with..double.dots",
            "test...triple",
            "...multiple.dots"
        ]
        
        for key in multi_dot_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert "Invalid key format" in error


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigValueValidator:
    """测试ConfigValueValidator功能"""

    def test_validate_value_valid_values(self):
        """测试有效值的验证"""
        valid_values = [
            "string_value",
            123,
            45.67,
            True,
            False,
            [],
            {},
            {"nested": "object"},
            [1, 2, 3],
            None  # 根据当前实现，None可能被认为是无效的
        ]
        
        for value in valid_values:
            is_valid = ConfigValueValidator.validate_value(value)
            # 根据当前实现，只有None应该返回False
            if value is None:
                assert is_valid is False
            else:
                assert is_valid is True

    def test_validate_value_none(self):
        """测试None值验证"""
        is_valid = ConfigValueValidator.validate_value(None)
        assert is_valid is False

    def test_validate_value_string(self):
        """测试字符串值验证"""
        string_values = ["", "hello", "世界", "   spaced   ", "special-chars!@#"]
        
        for value in string_values:
            is_valid = ConfigValueValidator.validate_value(value)
            assert is_valid is True

    def test_validate_value_numbers(self):
        """测试数值验证"""
        numeric_values = [0, 1, -1, 100, 3.14, -2.5, float('inf'), 0.0]
        
        for value in numeric_values:
            is_valid = ConfigValueValidator.validate_value(value)
            assert is_valid is True

    def test_validate_value_boolean(self):
        """测试布尔值验证"""
        boolean_values = [True, False]
        
        for value in boolean_values:
            is_valid = ConfigValueValidator.validate_value(value)
            assert is_valid is True

    def test_validate_value_collections(self):
        """测试集合类型值验证"""
        collection_values = [
            [],
            [1, 2, 3],
            ["a", "b", "c"],
            {},
            {"key": "value"},
            {"nested": {"deep": "value"}}
        ]
        
        for value in collection_values:
            is_valid = ConfigValueValidator.validate_value(value)
            assert is_valid is True

    def test_validate_value_edge_cases(self):
        """测试边界情况"""
        edge_cases = [
            0,
            "",
            [],
            {},
            False,  # False是有效的非None值
            object(),
            lambda x: x  # 函数对象
        ]
        
        for value in edge_cases:
            is_valid = ConfigValueValidator.validate_value(value)
            # 根据当前实现，只有None返回False
            assert is_valid is True


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigValidatorsIntegration:
    """测试配置验证器集成功能"""

    def test_validator_interaction(self):
        """测试两个验证器的交互"""
        # 有效键和值
        key = "test.config.key"
        value = {"setting": "value"}
        
        # 验证键
        key_valid, key_error = ConfigKeyValidator.validate_key(key)
        assert key_valid is True
        
        # 验证值
        value_valid = ConfigValueValidator.validate_value(value)
        assert value_valid is True

    def test_complete_validation_workflow(self):
        """测试完整的验证工作流"""
        test_cases = [
            ("valid.key", "valid_value", True),
            ("invalid.key.", "valid_value", False),  # 键无效
            ("valid.key", None, False),  # 值无效
            (None, "valid_value", False),  # 键无效
            ("", "valid_value", False),  # 键无效
        ]
        
        for key, value, should_be_valid in test_cases:
            # 验证键
            key_valid, key_error = ConfigKeyValidator.validate_key(key)
            
            # 验证值
            value_valid = ConfigValueValidator.validate_value(value)
            
            # 整体有效性的逻辑与
            overall_valid = key_valid and value_valid
            assert overall_valid is should_be_valid, f"Expected {should_be_valid} for key='{key}', value={value}"

    def test_error_message_quality(self):
        """测试错误消息的质量"""
        # 测试各种无效键的错误消息
        invalid_keys = [
            (None, "Key cannot be None"),
            ("", "Key must be a non-empty string"),
            (".invalid", "Invalid key format"),
            ("test key", "Key contains dangerous characters"),
            ("a" * 101, "Key length exceeds maximum")  # 假设MAX_KEY_LENGTH是100
        ]
        
        for key, expected_error_content in invalid_keys:
            is_valid, error = ConfigKeyValidator.validate_key(key)
            assert is_valid is False
            assert expected_error_content in error, f"Expected error message containing '{expected_error_content}' for key {key}, got: {error}"

    def test_static_methods(self):
        """测试静态方法特性"""
        # 验证方法是静态的（可以通过类直接调用，不需要实例）
        validator = ConfigKeyValidator()
        
        # 直接通过类调用静态方法
        result1 = ConfigKeyValidator.validate_key("test.key")
        
        # 通过实例调用静态方法（Python允许这样做）
        result2 = validator.validate_key("test.key")
        
        assert result1 == result2

    def test_validator_performance_large_input(self):
        """测试验证器对大型输入的性能"""
        # 测试最大长度的键
        max_key = "a" * ConfigKeyValidator.MAX_KEY_LENGTH
        
        # 这应该很快完成
        is_valid, error = ConfigKeyValidator.validate_key(max_key)
        assert is_valid is True
        
        # 测试长但有效的嵌套键
        long_nested_key = "level1.level2.level3.level4.level5.level6.level7.level8.level9.level10"
        is_valid, error = ConfigKeyValidator.validate_key(long_nested_key)
        assert is_valid is True
