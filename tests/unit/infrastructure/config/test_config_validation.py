"""
配置验证功能测试
测试配置数据验证逻辑
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ConfigScope(Enum):
    """配置作用域枚举"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    APPLICATION = "application"


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]

    def __init__(self, is_valid: bool = True, errors: Optional[List[str]] = None, warnings: Optional[List[str]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []


class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_string(value: Any, min_length: int = 0, max_length: int = 1000) -> ValidationResult:
        """验证字符串类型"""
        errors = []
        warnings = []

        if not isinstance(value, str):
            errors.append(f"Expected string, got {type(value).__name__}")
            return ValidationResult(False, errors)

        if len(value) < min_length:
            errors.append(f"String too short: {len(value)} < {min_length}")
        elif len(value) > max_length:
            errors.append(f"String too long: {len(value)} > {max_length}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    @staticmethod
    def validate_number(value: Any, min_value: Optional[float] = None, max_value: Optional[float] = None) -> ValidationResult:
        """验证数字类型"""
        errors = []
        warnings = []

        try:
            num_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Cannot convert to number: {value}")
            return ValidationResult(False, errors)

        if min_value is not None and num_value < min_value:
            errors.append(f"Value too small: {num_value} < {min_value}")
        if max_value is not None and num_value > max_value:
            errors.append(f"Value too large: {num_value} > {max_value}")

        return ValidationResult(len(errors) == 0, errors, warnings)

    @staticmethod
    def validate_port(value: Any) -> ValidationResult:
        """验证端口号"""
        result = ConfigValidator.validate_number(value, 1, 65535)
        if not result.is_valid:
            return result

        num_value = int(float(value))
        if num_value < 1024:
            result.warnings.append("Using privileged port (< 1024)")

        return result

    @staticmethod
    def validate_config_key(key: str) -> ValidationResult:
        """验证配置键格式"""
        errors = []
        warnings = []

        if not key or not isinstance(key, str):
            errors.append("Config key must be non-empty string")
            return ValidationResult(False, errors)

        # Check for valid characters
        import re
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', key):
            errors.append("Config key contains invalid characters")

        # Check for consecutive dots
        if '..' in key:
            errors.append("Config key cannot contain consecutive dots")

        # Check length
        if len(key) > 255:
            errors.append("Config key too long (>255 characters)")

        return ValidationResult(len(errors) == 0, errors, warnings)


class TestConfigValidation:
    """配置验证测试"""

    def test_validate_string_valid(self):
        """测试字符串验证 - 有效值"""
        result = ConfigValidator.validate_string("valid_string")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_string_invalid_type(self):
        """测试字符串验证 - 无效类型"""
        result = ConfigValidator.validate_string(123)
        assert result.is_valid is False
        assert "Expected string" in result.errors[0]

    def test_validate_string_length_constraints(self):
        """测试字符串验证 - 长度约束"""
        # Too short
        result = ConfigValidator.validate_string("a", min_length=2)
        assert result.is_valid is False
        assert "too short" in result.errors[0]

        # Too long
        result = ConfigValidator.validate_string("a" * 100, max_length=50)
        assert result.is_valid is False
        assert "too long" in result.errors[0]

    def test_validate_number_valid(self):
        """测试数字验证 - 有效值"""
        result = ConfigValidator.validate_number(42)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_number_invalid_type(self):
        """测试数字验证 - 无效类型"""
        result = ConfigValidator.validate_number("not_a_number")
        assert result.is_valid is False
        assert "Cannot convert to number" in result.errors[0]

    def test_validate_number_range_constraints(self):
        """测试数字验证 - 范围约束"""
        # Too small
        result = ConfigValidator.validate_number(5, min_value=10)
        assert result.is_valid is False
        assert "too small" in result.errors[0]

        # Too large
        result = ConfigValidator.validate_number(100, max_value=50)
        assert result.is_valid is False
        assert "too large" in result.errors[0]

    def test_validate_port_valid(self):
        """测试端口验证 - 有效值"""
        result = ConfigValidator.validate_port(8080)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_port_privileged_warning(self):
        """测试端口验证 - 特权端口警告"""
        result = ConfigValidator.validate_port(80)
        assert result.is_valid is True
        assert "privileged port" in result.warnings[0]

    def test_validate_port_invalid_range(self):
        """测试端口验证 - 无效范围"""
        # Too small
        result = ConfigValidator.validate_port(0)
        assert result.is_valid is False
        assert "too small" in result.errors[0]

        # Too large
        result = ConfigValidator.validate_port(70000)
        assert result.is_valid is False
        assert "too large" in result.errors[0]

    def test_validate_config_key_valid(self):
        """测试配置键验证 - 有效值"""
        result = ConfigValidator.validate_config_key("database.host")
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_config_key_invalid(self):
        """测试配置键验证 - 无效值"""
        # Empty key
        result = ConfigValidator.validate_config_key("")
        assert result.is_valid is False
        assert "non-empty string" in result.errors[0]

        # Invalid characters
        result = ConfigValidator.validate_config_key("invalid-key!")
        assert result.is_valid is False
        assert "invalid characters" in result.errors[0]

        # Consecutive dots
        result = ConfigValidator.validate_config_key("db..host")
        assert result.is_valid is False
        assert "consecutive dots" in result.errors[0]

        # Too long
        long_key = "a" * 256
        result = ConfigValidator.validate_config_key(long_key)
        assert result.is_valid is False
        assert "too long" in result.errors[0]

    def test_validation_result_creation(self):
        """测试验证结果创建"""
        # Default valid result
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

        # Invalid result with errors and warnings
        result = ValidationResult(False, ["error1", "error2"], ["warning1"])
        assert result.is_valid is False
        assert result.errors == ["error1", "error2"]
        assert result.warnings == ["warning1"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
