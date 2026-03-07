#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对核心 EnhancedConfigValidator 的行为覆盖测试
"""

import pytest

from src.infrastructure.config.core.enhanced_validators.enhanced_config_validator import (
    EnhancedConfigValidator,
)
from src.infrastructure.config.validators.enhanced_validators import ConfigValidationResult


@pytest.fixture()
def validator():
    return EnhancedConfigValidator()


def test_validate_merges_config_validation_result(validator):
    """当子验证器返回 ConfigValidationResult 时应正确聚合"""

    def rich_validator(config):
        sub_result = ConfigValidationResult(is_valid=False)
        sub_result.add_error("missing key")
        sub_result.add_warning("format warning")
        sub_result.add_recommendation("use default value")
        return sub_result

    validator.add_validator("rich", rich_validator)

    aggregated = validator.validate({"key": "value"})
    assert aggregated.is_valid is False
    assert "missing key" in aggregated.errors
    assert "format warning" in aggregated.warnings
    assert "use default value" in aggregated.recommendations


def test_validate_flags_boolean_failure(validator):
    """返回 False 的验证器应被视为失败并记录错误"""
    validator.add_validator("bool_fail", lambda cfg: False)

    aggregated = validator.validate({})
    assert aggregated.is_valid is False
    assert any("bool_fail" in msg for msg in aggregated.errors)


def test_validate_collects_warning_string(validator):
    """返回字符串的验证器应记录为警告"""
    validator.add_validator("warn", lambda cfg: "needs review")

    aggregated = validator.validate({})
    assert aggregated.is_valid is True
    assert "needs review" in aggregated.warnings


def test_validate_captures_exceptions(validator):
    """抛出异常的验证器应被捕获并转换为错误"""

    def raising_validator(config):
        raise ValueError("boom")

    validator.add_validator("explode", raising_validator)

    aggregated = validator.validate({})
    assert aggregated.is_valid is False
    assert any("explode" in msg and "boom" in msg for msg in aggregated.errors)

