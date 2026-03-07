"""
验证器基础边界测试模块

测试验证器基础类的边界条件和错误处理，包括：
- 组合验证失败测试
- 自定义验证器边界测试
- 验证结果边界测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from typing import Dict, Any

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../../'))

from src.infrastructure.config.validators.validator_base import (
    ValidationSeverity, ValidationType, ValidationResult, 
    BaseConfigValidator
)


class TestValidationSeverity:
    """验证严重程度测试类"""
    
    def test_severity_ordering(self):
        """测试严重程度排序"""
        assert ValidationSeverity.INFO < ValidationSeverity.WARNING
        assert ValidationSeverity.WARNING < ValidationSeverity.ERROR
        assert ValidationSeverity.ERROR < ValidationSeverity.CRITICAL
        
    def test_severity_comparison_edge_cases(self):
        """测试严重程度比较边界情况"""
        # 相同严重程度不应该小于自己
        assert not (ValidationSeverity.WARNING < ValidationSeverity.WARNING)
        # 由于只定义了__lt__，我们测试相等的情况
        assert ValidationSeverity.WARNING == ValidationSeverity.WARNING


class TestValidationResult:
    """验证结果测试类"""
    
    def test_validation_result_creation(self):
        """测试验证结果创建"""
        result = ValidationResult(
            field="test_field",
            value="test_value",
            is_valid=False,
            message="Test message",
            severity=ValidationSeverity.ERROR
        )
        
        assert result.field == "test_field"
        assert result.value == "test_value"
        assert result.is_valid is False
        assert result.message == "Test message"
        assert result.severity == ValidationSeverity.ERROR
        
    def test_validation_result_default_values(self):
        """测试验证结果默认值"""
        result = ValidationResult(
            field="test_field",
            value="test_value"
        )
        
        assert result.is_valid is True
        assert result.message == ""
        assert result.severity == ValidationSeverity.INFO


class TestBaseConfigValidator:
    """基础验证器测试类"""
    
    def test_base_validator_implementation(self):
        """测试基础验证器实现"""
        class TestValidator(BaseConfigValidator):
            def validate(self, config: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(is_valid=True)
            
            def validate_field(self, field: str, value: Any) -> ValidationResult:
                return ValidationResult(
                    field=field,
                    value=value,
                    is_valid=isinstance(value, str)
                )
            
            @property
            def name(self) -> str:
                return "TestValidator"
        
        validator = TestValidator()
        result = validator.validate_field("test_field", "test")
        assert result.is_valid is True
        
        result = validator.validate_field("test_field", 123)
        assert result.is_valid is False


class TestBaseConfigValidatorExtended:
    """基础配置验证器扩展测试类"""
    
    def test_base_config_validator_with_rules(self):
        """测试带规则的基础配置验证器"""
        class TestValidator(BaseConfigValidator):
            def validate(self, config: Dict[str, Any]) -> ValidationResult:
                return ValidationResult(is_valid=True)
            
            def validate_field(self, field: str, value: Any) -> ValidationResult:
                return ValidationResult(
                    field=field,
                    value=value,
                    is_valid=value is not None
                )
            
            # 移除重写的属性，使用基类的实现
        
        validator = TestValidator("TestName", "Test Description")
        
        assert validator.name == "TestName"  # 基类会使用传入的name参数
        assert validator.description == "Test Description"
        
        # 测试字段验证
        result = validator.validate_field("test_field", "value")
        assert result.is_valid is True
        
        result = validator.validate_field("test_field", None)
        assert result.is_valid is False

    def test_validation_result_edge_cases(self):
        """测试验证结果边界情况"""
        # 测试空消息的错误结果
        result = ValidationResult(
            is_valid=False,
            errors=[],
            warnings=[],
            severity=ValidationSeverity.ERROR
        )
        assert result.is_valid is False
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        # 测试带多个错误的结果
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            severity=ValidationSeverity.CRITICAL
        )
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 1
        assert result.severity == ValidationSeverity.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
