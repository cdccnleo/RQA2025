"""
API配置基础类测试
"""
from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.api.configs.base_config import ValidationResult, BaseConfig


class TestValidationResult:
    """测试验证结果类"""

    def test_init(self):
        """测试初始化"""
        result = ValidationResult()
        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []

    def test_add_error(self):
        """测试添加错误"""
        result = ValidationResult()
        result.add_error("Test error")

        assert result.is_valid is False
        assert result.errors == ["Test error"]
        assert result.warnings == []

    def test_add_warning(self):
        """测试添加警告"""
        result = ValidationResult()
        result.add_warning("Test warning")

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == ["Test warning"]

    def test_merge_valid(self):
        """测试合并有效结果"""
        result1 = ValidationResult()
        result2 = ValidationResult()

        result1.merge(result2)

        assert result1.is_valid is True
        assert result1.errors == []
        assert result1.warnings == []

    def test_merge_with_errors(self):
        """测试合并包含错误的结果"""
        result1 = ValidationResult()
        result2 = ValidationResult()
        result2.add_error("Merge error")
        result2.add_warning("Merge warning")

        result1.merge(result2)

        assert result1.is_valid is False
        assert result1.errors == ["Merge error"]
        assert result1.warnings == ["Merge warning"]

    def test_bool_conversion(self):
        """测试布尔转换"""
        valid_result = ValidationResult()
        assert bool(valid_result) is True

        invalid_result = ValidationResult()
        invalid_result.add_error("Error")
        assert bool(invalid_result) is False

    def test_str_conversion_valid(self):
        """测试字符串转换（有效）"""
        result = ValidationResult()
        assert str(result) == "Validation passed"

    def test_str_conversion_invalid(self):
        """测试字符串转换（无效）"""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")

        assert str(result) == "Validation failed: Error 1, Error 2"


class TestBaseConfig:
    """测试配置基类"""

    def test_abstract_methods(self):
        """测试抽象方法定义"""
        # BaseConfig是抽象类，不能直接实例化
        with pytest.raises(TypeError):
            BaseConfig()

    def test_validate_method(self):
        """测试验证方法存在"""
        assert hasattr(BaseConfig, 'validate')
        assert callable(getattr(BaseConfig, 'validate'))

    def test_abstract_validate_impl(self):
        """测试抽象验证实现方法"""
        # 检查是否有抽象方法
        import inspect
        methods = [name for name, method in inspect.getmembers(BaseConfig, predicate=inspect.isfunction)]
        assert '_validate_impl' in methods

        # 尝试调用抽象方法应该失败
        config = BaseConfig.__new__(BaseConfig)  # 创建实例而不调用__init__
        result = ValidationResult()
        with pytest.raises(NotImplementedError):
            config._validate_impl(result)
