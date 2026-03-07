"""
测试API配置基础类

覆盖 BaseConfig 和相关类的功能
"""

import pytest
from unittest.mock import Mock, patch
from abc import ABC
from src.infrastructure.api.configs.base_config import (
    ValidationResult,
    BaseConfig,
    Priority,
    ExportFormat
)


class TestValidationResult:
    """ValidationResult 类测试"""

    def test_initialization(self):
        """测试初始化"""
        result = ValidationResult()

        assert result.is_valid == True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        """测试添加错误"""
        result = ValidationResult()

        result.add_error("Field is required")
        result.add_error("Invalid format")

        assert result.is_valid == False
        assert result.errors == ["Field is required", "Invalid format"]
        assert result.warnings == []

    def test_add_warning(self):
        """测试添加警告"""
        result = ValidationResult()

        result.add_warning("Deprecated field")
        result.add_warning("Performance issue")

        assert result.is_valid == True
        assert result.errors == []
        assert result.warnings == ["Deprecated field", "Performance issue"]

    def test_mixed_errors_and_warnings(self):
        """测试错误和警告混合"""
        result = ValidationResult()

        result.add_error("Critical error")
        result.add_warning("Minor issue")
        result.add_error("Another error")
        result.add_warning("Another warning")

        assert result.is_valid == False
        assert result.errors == ["Critical error", "Another error"]
        assert result.warnings == ["Minor issue", "Another warning"]

    def test_empty_result_is_valid(self):
        """测试空结果是有效的"""
        result = ValidationResult()

        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0


class TestBaseConfig:
    """BaseConfig 抽象基类测试"""

    def test_is_abstract(self):
        """测试BaseConfig是抽象类"""
        assert issubclass(BaseConfig, ABC)

        # 不能直接实例化
        with pytest.raises(TypeError):
            BaseConfig()

    def test_required_methods_not_implemented(self):
        """测试必需方法未实现"""

        class ConcreteConfig(BaseConfig):
            """实现所有必需方法的配置类"""
            def __init__(self):
                super().__init__()
                self.name = "test"
                self.value = 42

            def validate(self) -> ValidationResult:
                return ValidationResult()

            def to_dict(self) -> dict:
                return {"name": self.name, "value": self.value}

        # 应该可以实例化
        config = ConcreteConfig()
        assert config.name == "test"
        assert config.value == 42

    def test_concrete_implementation(self):
        """测试具体实现"""

        class TestConfig(BaseConfig):
            def __init__(self):
                super().__init__()
                self.title = "Test Config"
                self.enabled = True
                self.max_items = 100

            def validate(self) -> ValidationResult:
                result = ValidationResult()
                if not self.title:
                    result.add_error("Title is required")
                if self.max_items < 0:
                    result.add_error("Max items must be non-negative")
                if self.max_items > 1000:
                    result.add_warning("Max items is very high")
                return result

            def to_dict(self) -> dict:
                return {
                    "title": self.title,
                    "enabled": self.enabled,
                    "max_items": self.max_items
                }

        config = TestConfig()

        # 测试验证 - 有效配置
        result = config.validate()
        assert result.is_valid == True
        assert result.errors == []
        assert result.warnings == []

        # 测试验证 - 无效配置
        config.title = ""
        config.max_items = -1
        result = config.validate()
        assert result.is_valid == False
        assert "Title is required" in result.errors
        assert "Max items must be non-negative" in result.errors

        # 测试验证 - 警告
        config.title = "Valid Title"
        config.max_items = 1500
        result = config.validate()
        assert result.is_valid == True
        assert "Max items is very high" in result.warnings

        # 测试to_dict
        config.max_items = 100
        data = config.to_dict()
        assert data == {
            "title": "Valid Title",
            "enabled": True,
            "max_items": 100
        }


class TestPriority:
    """Priority 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"

    def test_enum_membership(self):
        """测试枚举成员"""
        assert Priority.LOW in Priority
        assert Priority.MEDIUM in Priority
        assert Priority.HIGH in Priority
        assert Priority.CRITICAL in Priority

    def test_enum_iteration(self):
        """测试枚举迭代"""
        values = [member.value for member in Priority]
        assert "low" in values
        assert "medium" in values
        assert "high" in values
        assert "critical" in values

    def test_string_conversion(self):
        """测试字符串转换"""
        assert str(Priority.HIGH) == "Priority.HIGH"
        assert str(Priority.CRITICAL) == "Priority.CRITICAL"


class TestExportFormat:
    """ExportFormat 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ExportFormat.JSON.value == "json"
        assert ExportFormat.YAML.value == "yaml"
        assert ExportFormat.HTML.value == "html"
        assert ExportFormat.MARKDOWN.value == "markdown"
        assert ExportFormat.PYTHON.value == "python"

    def test_enum_membership(self):
        """测试枚举成员"""
        assert ExportFormat.JSON in ExportFormat
        assert ExportFormat.YAML in ExportFormat
        assert ExportFormat.HTML in ExportFormat
        assert ExportFormat.MARKDOWN in ExportFormat
        assert ExportFormat.PYTHON in ExportFormat

    def test_enum_iteration(self):
        """测试枚举迭代"""
        values = [member.value for member in ExportFormat]
        assert "json" in values
        assert "yaml" in values
        assert "html" in values
        assert "markdown" in values
        assert "python" in values

    def test_string_conversion(self):
        """测试字符串转换"""
        assert str(ExportFormat.JSON) == "ExportFormat.JSON"
        assert str(ExportFormat.HTML) == "ExportFormat.HTML"


class TestBaseConfigIntegration:
    """BaseConfig 集成测试"""

    def test_inheritance_hierarchy(self):
        """测试继承层次结构"""

        class SimpleConfig(BaseConfig):
            def __init__(self):
                super().__init__()
                self.name = "simple"

            def validate(self) -> ValidationResult:
                result = ValidationResult()
                if not self.name:
                    result.add_error("Name is required")
                return result

            def to_dict(self) -> dict:
                return {"name": self.name}

        class ExtendedConfig(SimpleConfig):
            def __init__(self):
                super().__init__()
                self.version = "1.0"
                self.enabled = True

            def validate(self) -> ValidationResult:
                result = super().validate()
                if not self.version:
                    result.add_error("Version is required")
                return result

            def to_dict(self) -> dict:
                data = super().to_dict()
                data.update({
                    "version": self.version,
                    "enabled": self.enabled
                })
                return data

        # 测试简单配置
        simple = SimpleConfig()
        assert simple.validate().is_valid == True

        simple.name = ""
        assert simple.validate().is_valid == False

        # 测试扩展配置
        extended = ExtendedConfig()
        assert extended.validate().is_valid == True
        assert extended.to_dict() == {
            "name": "simple",
            "version": "1.0",
            "enabled": True
        }

        extended.version = ""
        assert extended.validate().is_valid == False

    def test_validation_workflow(self):
        """测试验证工作流"""

        class WorkflowConfig(BaseConfig):
            def __init__(self):
                super().__init__()
                self.required_field = "value"
                self.optional_field = None
                self.numeric_field = 100

            def validate(self) -> ValidationResult:
                result = ValidationResult()

                # 必需字段验证
                if not self.required_field:
                    result.add_error("Required field cannot be empty")

                # 数值范围验证
                if not isinstance(self.numeric_field, (int, float)):
                    result.add_error("Numeric field must be a number")
                elif self.numeric_field < 0:
                    result.add_error("Numeric field must be non-negative")
                elif self.numeric_field > 1000:
                    result.add_warning("Numeric field is very high")

                # 可选字段警告
                if self.optional_field is None:
                    result.add_warning("Optional field is not set")

                return result

            def to_dict(self) -> dict:
                return {
                    "required_field": self.required_field,
                    "optional_field": self.optional_field,
                    "numeric_field": self.numeric_field
                }

        config = WorkflowConfig()

        # 测试有效配置
        result = config.validate()
        assert result.is_valid == True
        assert len(result.warnings) == 1  # optional_field 警告
        assert "Optional field is not set" in result.warnings

        # 测试无效配置
        config.required_field = ""
        config.numeric_field = -10
        config.optional_field = "set"

        result = config.validate()
        assert result.is_valid == False
        assert len(result.errors) == 2
        assert "Required field cannot be empty" in result.errors
        assert "Numeric field must be non-negative" in result.errors
        assert len(result.warnings) == 0  # 警告被清除

        # 测试边界情况
        config.required_field = "valid"
        config.numeric_field = 1500

        result = config.validate()
        assert result.is_valid == True
        assert len(result.warnings) == 1
        assert "Numeric field is very high" in result.warnings
