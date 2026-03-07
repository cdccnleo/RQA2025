"""
直接测试配置验证器的测试文件
测试src/infrastructure/config/validators目录下的实际代码
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.config.validators.validator_base import ValidationResult


class TestConfigValidatorsDirect:
    """直接测试配置验证器"""

    def test_validation_result_creation(self):
        """测试ValidationResult创建"""
        result = ValidationResult(True, errors=[], warnings=[], field="test", value="Valid")
        assert result.is_valid is True
        assert result.value == "Valid"
        assert result.errors == []

    def test_validation_result_with_errors(self):
        """测试ValidationResult带错误信息"""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(False, errors=errors, warnings=[], field="test", value="Invalid")
        assert result.is_valid is False
        assert result.value == "Invalid"
        assert result.errors == errors

    def test_validation_result_str(self):
        """测试ValidationResult字符串表示"""
        result = ValidationResult(True, errors=[], warnings=[], field="test", value="Success")
        assert "✓ test" in str(result)

    def test_validation_result_bool(self):
        """测试ValidationResult布尔值"""
        valid_result = ValidationResult(True, errors=[], warnings=[], field="test")
        invalid_result = ValidationResult(False, errors=["error"], warnings=[], field="test")

        assert bool(valid_result) is True
        assert bool(invalid_result) is False

