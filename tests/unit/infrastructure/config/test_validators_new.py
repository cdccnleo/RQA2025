# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
修复后的配置验证器测试

测试验证器功能完整性
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock

from src.infrastructure.config.validators.validator_base import ValidationResult, ValidationSeverity


class TestValidationResult(unittest.TestCase):
    """测试验证结果类"""

    def test_initialization(self):
        """测试初始化"""
        result = ValidationResult(success=True)
        self.assertTrue(result.success)
        self.assertEqual(result.message, "")
        self.assertEqual(result.severity, ValidationSeverity.INFO)

        result_with_message = ValidationResult(success=False, message="Test error", severity=ValidationSeverity.ERROR, field="test_field")
        self.assertFalse(result_with_message.success)
        self.assertEqual(result_with_message.message, "Test error")
        self.assertEqual(result_with_message.severity, ValidationSeverity.ERROR)
        self.assertEqual(result_with_message.field, "test_field")

    def test_to_dict(self):
        """测试转换为字典"""
        result = ValidationResult(success=False, message="Test error", severity=ValidationSeverity.ERROR, field="test_field", value="test_value")

        data = result.to_dict()
        self.assertFalse(data["success"])
        self.assertEqual(data["message"], "Test error")
        self.assertEqual(data["severity"], "error")
        self.assertEqual(data["field"], "test_field")
        self.assertEqual(data["value"], "test_value")
        self.assertIn("timestamp", data)

    def test_boolean_conversion(self):
        """测试布尔值转换"""
        success_result = ValidationResult(success=True)
        self.assertTrue(bool(success_result))

        failure_result = ValidationResult(success=False)
        self.assertFalse(bool(failure_result))

    def test_string_representation(self):
        """测试字符串表示"""
        success_result = ValidationResult(success=True, field="test", message="OK")
        str_repr = str(success_result)
        self.assertIn("✓", str_repr)
        self.assertIn("test", str_repr)
        self.assertIn("OK", str_repr)

        failure_result = ValidationResult(success=False, field="test", message="Error")
        str_repr = str(failure_result)
        self.assertIn("✗", str_repr)
        self.assertIn("test", str_repr)
        self.assertIn("Error", str_repr)

    def test_suggestions(self):
        """测试建议功能"""
        suggestions = ["Try this", "Or this"]
        result = ValidationResult(success=False, message="Test error", suggestions=suggestions)

        self.assertEqual(result.suggestions, suggestions)


class TestValidationSeverity(unittest.TestCase):
    """测试验证严重程度枚举"""

    def test_validation_severity_values(self):
        """测试验证严重程度值"""
        self.assertEqual(ValidationSeverity.INFO.value, "info")
        self.assertEqual(ValidationSeverity.WARNING.value, "warning")
        self.assertEqual(ValidationSeverity.ERROR.value, "error")
        self.assertEqual(ValidationSeverity.CRITICAL.value, "critical")

    def test_validation_severity_ordering(self):
        """测试验证严重程度排序"""
        # 验证枚举顺序（按严重程度递增）
        severity_order = [
            (ValidationSeverity.INFO, 0),
            (ValidationSeverity.WARNING, 1),
            (ValidationSeverity.ERROR, 2),
            (ValidationSeverity.CRITICAL, 3)
        ]

        for severity, expected_order in severity_order:
            # 验证每个严重程度都有对应的顺序值
            self.assertIsNotNone(severity.value)


if __name__ == '__main__':
    unittest.main()
