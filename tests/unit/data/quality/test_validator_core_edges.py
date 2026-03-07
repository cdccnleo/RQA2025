import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pandas as pd
import pytest

from src.data.quality.validator import DataValidator


def test_validate_dataframe_missing_values_and_ranges():
    validator = DataValidator()
    df = pd.DataFrame(
        {
            "price": [10, -5, 200000],  # 一个小于0，一个超上限
            "volume": [100, "x", 300],  # 非数值类型将被判定为类型不匹配
            "amount": [1.0, 2.0, None],
            "open": [1, 2, 3],
            "close": [1, 2, 3],
        }
    )
    result = validator.validate(df, data_type="generic")
    assert result["is_valid"] is False
    # 缺失值/类型/范围至少有一项命中
    assert any("missing values" in e or "type mismatch" in e or "value out of expected range" in e for e in result["errors"])
    # 附带检查基础指标字段存在
    assert "error_count" in result["metrics"]
    assert "checks" in result


def test_validate_dict_payload_missing_fields():
    validator = DataValidator()
    payload = {"a": 1, "b": None}
    result = validator.validate(payload, data_type="generic")
    assert result["is_valid"] is False
    assert any("missing value for field 'b'" in e for e in result["errors"])
    assert result["metrics"]["field_count"] == float(len(payload))


def test_validate_custom_rule_and_exception_path(monkeypatch):
    # 自定义规则命中与异常兜底
    def rule_ok(data):
        return ["custom error 1"]

    def rule_raise(data):
        raise RuntimeError("boom")

    validator = DataValidator(config={"custom_rules": [rule_ok, rule_raise]})
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = validator.validate(df, data_type="generic")
    # 命中自定义规则的错误
    assert any("custom error 1" in e for e in result["errors"])
    # 触发自定义规则异常的兜底信息
    assert any("custom rule execution failed" in e for e in result["errors"])


def test_is_probably_numeric_column_heuristics():
    validator = DataValidator()
    # 数值列名称提示+内容特征
    df = pd.DataFrame(
        {
            "price": ["1", "2", "bad", None, "3.3"],
            "misc": ["1", "2", "3", "x", "y"],
        }
    )
    # 触发类型不匹配（字符串无法全部转为数值）
    result = validator.validate(df, data_type="generic")
    assert any("type mismatch detected in column 'price'" in e for e in result["errors"])


def test_validate_duplicates_detected():
    validator = DataValidator()
    df = pd.DataFrame({"a": [1, 1], "b": [2, 2]})
    result = validator.validate(df, data_type="generic")
    assert any("duplicate rows detected" in e for e in result["errors"])


def test_validate_unsupported_type_returns_error():
    validator = DataValidator()
    data = [1, 2, 3]  # 非 DataFrame/字典
    result = validator.validate(data, data_type="generic")
    assert result["is_valid"] is False
    assert any("unsupported data type" in e for e in result["errors"])


def test_validate_empty_dict_payload():
    validator = DataValidator()
    payload = {}
    result = validator.validate(payload, data_type="generic")
    assert result["is_valid"] is False
    assert any("missing data payload" in e for e in result["errors"])


def test_validation_error_with_result():
    """测试ValidationError异常类（34-35行）"""
    from src.data.quality.validator import ValidationError, ValidationResult
    
    result = ValidationResult(is_valid=False, errors=["test error"])
    error = ValidationError("Test error message", result=result)
    
    assert str(error) == "Test error message"
    assert error.result is not None
    assert error.result.errors == ["test error"]


def test_validation_error_without_result():
    """测试ValidationError异常类（无result参数）"""
    from src.data.quality.validator import ValidationError
    
    error = ValidationError("Test error message")
    
    assert str(error) == "Test error message"
    assert error.result is None


def test_looks_like_number_non_numeric_non_string():
    """测试_looks_like_number处理非数值非字符串类型（228行）"""
    validator = DataValidator()
    
    # 测试非int/float/str类型（如list、dict等），应该返回False（228行）
    assert validator._looks_like_number([1, 2, 3]) is False
    assert validator._looks_like_number({"a": 1}) is False
    assert validator._looks_like_number(None) is False

