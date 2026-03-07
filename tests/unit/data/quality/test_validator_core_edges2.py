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

from src.data.quality.validator import DataValidator, ValidationError


def test_calculate_completeness_metrics_empty_dataframe():
    """测试空 DataFrame 的完整性指标计算"""
    validator = DataValidator()
    df = pd.DataFrame()
    
    # 空 DataFrame 应该返回默认值
    metrics = validator._calculate_completeness_metrics(df)
    assert "total_cells" in metrics
    assert "missing_cells" in metrics
    assert "missing_rate" in metrics
    assert "completeness" in metrics
    # 空 DataFrame 的 total_cells 应该为 1（避免除零）
    assert metrics["total_cells"] == 1.0


def test_calculate_completeness_metrics_single_cell():
    """测试单单元格 DataFrame 的完整性指标"""
    validator = DataValidator()
    df = pd.DataFrame({"a": [None]})
    
    metrics = validator._calculate_completeness_metrics(df)
    assert metrics["total_cells"] == 1.0
    assert metrics["missing_cells"] == 1.0
    assert metrics["missing_rate"] == 1.0
    assert metrics["completeness"] == 0.0


def test_add_rule_non_callable_raises():
    """测试添加非可调用规则时抛出异常"""
    validator = DataValidator()
    
    with pytest.raises(TypeError, match="rule must be callable"):
        validator.add_rule("not a function")


def test_clear_rules_removes_all_custom_rules():
    """测试清空规则后自定义规则被移除"""
    validator = DataValidator()
    
    def rule1(data):
        return ["error1"]
    
    def rule2(data):
        return ["error2"]
    
    validator.add_rule(rule1)
    validator.add_rule(rule2)
    assert len(validator.rules) == 2
    
    validator.clear_rules()
    assert len(validator.rules) == 0
    
    # 清空后验证应该不包含自定义规则错误
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = validator.validate(df, data_type="generic")
    assert "error1" not in result["errors"]
    assert "error2" not in result["errors"]


def test_validation_history_accumulates_results():
    """测试验证历史记录的累积"""
    validator = DataValidator()
    
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    result1 = validator.validate(df1, data_type="type1")
    
    df2 = pd.DataFrame({"b": [4, 5, 6]})
    result2 = validator.validate(df2, data_type="type2")
    
    assert len(validator.validation_history) == 2
    assert validator.validation_history[0].data_type == "type1"
    assert validator.validation_history[1].data_type == "type2"


def test_strict_mode_affects_validation():
    """测试严格模式对验证的影响"""
    # 创建严格模式验证器
    validator_strict = DataValidator(config={"strict_mode": True})
    validator_normal = DataValidator(config={"strict_mode": False})
    
    df = pd.DataFrame({"price": [10, -5, 200000]})
    
    result_strict = validator_strict.validate(df, data_type="generic")
    result_normal = validator_normal.validate(df, data_type="generic")
    
    # 两者都应该检测到错误（因为范围检查不依赖严格模式）
    assert result_strict["is_valid"] is False
    assert result_normal["is_valid"] is False


def test_is_probably_numeric_column_empty_sample():
    """测试空样本的数值列判断"""
    validator = DataValidator()
    df = pd.DataFrame({"col": [None, None, None]})
    
    # 空样本应该返回 False
    result = validator._is_probably_numeric_column("col", df["col"])
    assert result is False


def test_is_probably_numeric_column_mixed_types():
    """测试混合类型的数值列判断"""
    validator = DataValidator()
    df = pd.DataFrame({"col": ["1", "2", "bad", "3", "4"]})
    
    # 60% 以上是数值，应该返回 True
    result = validator._is_probably_numeric_column("col", df["col"])
    assert result is True


def test_validate_value_ranges_boundary_values():
    """测试边界值的范围验证"""
    validator = DataValidator()
    
    # 测试正好在边界上的值
    df_boundary = pd.DataFrame({
        "price": [0.0, 1e5],  # 正好在边界上
        "volume": [0.0, 1e9],
    })
    
    result = validator.validate(df_boundary, data_type="generic")
    # 边界值应该通过验证
    assert "value out of expected range" not in str(result["errors"])


def test_validate_value_ranges_outside_bounds():
    """测试超出范围的值"""
    validator = DataValidator()
    
    df_outside = pd.DataFrame({
        "price": [-1, 1e5 + 1],  # 超出边界
        "volume": [-1, 1e9 + 1],
    })
    
    result = validator.validate(df_outside, data_type="generic")
    assert result["is_valid"] is False
    assert any("value out of expected range" in e for e in result["errors"])


def test_validate_duplicates_empty_dataframe():
    """测试空 DataFrame 的重复检测"""
    validator = DataValidator()
    df = pd.DataFrame()
    
    result = validator._validate_duplicates(df)
    assert result is None  # 空 DataFrame 不应该有重复


def test_validate_duplicates_single_row():
    """测试单行 DataFrame 的重复检测"""
    validator = DataValidator()
    df = pd.DataFrame({"a": [1]})
    
    result = validator._validate_duplicates(df)
    assert result is None  # 单行不应该有重复


def test_validate_dict_payload_none_values():
    """测试字典负载中 None 值的验证"""
    validator = DataValidator()
    
    payload = {
        "field1": "value1",
        "field2": None,
        "field3": 123,
        "field4": None,
    }
    
    errors = validator._validate_dict_payload(payload)
    assert len(errors) == 2  # field2 和 field4 是 None
    assert any("field2" in e for e in errors)
    assert any("field4" in e for e in errors)


def test_validate_dict_payload_all_none():
    """测试所有字段都是 None 的字典负载"""
    validator = DataValidator()
    
    payload = {
        "field1": None,
        "field2": None,
    }
    
    errors = validator._validate_dict_payload(payload)
    assert len(errors) == 2


def test_validate_numeric_types_with_nan():
    """测试包含 NaN 的数值类型验证"""
    validator = DataValidator()
    
    df = pd.DataFrame({
        "price": [1.0, 2.0, float("nan"), 3.0],
        "volume": ["1", "2", "bad", "3"],
    })
    
    result = validator.validate(df, data_type="generic")
    # volume 列应该检测到类型不匹配
    assert any("type mismatch" in e for e in result["errors"])


def test_validate_missing_values_all_missing():
    """测试所有值都缺失的情况"""
    validator = DataValidator()
    
    df = pd.DataFrame({
        "col1": [None, None, None],
        "col2": [None, None, None],
    })
    
    errors = validator._validate_missing_values(df)
    assert len(errors) > 0
    assert any("missing values detected" in e for e in errors)


def test_validate_missing_values_no_missing():
    """测试没有缺失值的情况"""
    validator = DataValidator()
    
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": [4, 5, 6],
    })
    
    errors = validator._validate_missing_values(df)
    assert len(errors) == 0


def test_validate_custom_rule_returns_non_list():
    """测试自定义规则返回非列表的情况"""
    validator = DataValidator()
    
    def rule_returns_string(data):
        return "error string"  # 返回字符串而不是列表
    
    validator.add_rule(rule_returns_string)
    
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = validator.validate(df, data_type="generic")
    # 应该能处理非列表返回值（通过 str(err) 转换）
    assert isinstance(result["errors"], list)


def test_validate_custom_rule_returns_empty_list():
    """测试自定义规则返回空列表的情况"""
    validator = DataValidator()
    
    def rule_no_errors(data):
        return []  # 返回空列表
    
    validator.add_rule(rule_no_errors)
    
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = validator.validate(df, data_type="generic")
    # 应该不影响验证结果
    assert result["is_valid"] is True


def test_validate_empty_dataframe():
    """测试空 DataFrame 的完整验证流程"""
    validator = DataValidator()
    df = pd.DataFrame()
    
    result = validator.validate(df, data_type="generic")
    assert "checks" in result
    assert result["checks"]["row_count"] == 0
    assert result["checks"]["column_count"] == 0


def test_validate_dataframe_with_all_numeric_columns():
    """测试所有列都是数值类型的 DataFrame"""
    validator = DataValidator()
    
    df = pd.DataFrame({
        "price": [1.0, 2.0, 3.0],
        "volume": [100, 200, 300],
        "amount": [1000, 2000, 3000],
    })
    
    result = validator.validate(df, data_type="generic")
    # 所有列都是有效的数值，应该通过验证
    assert result["is_valid"] is True
    assert len([e for e in result["errors"] if "type mismatch" in e]) == 0


def test_validate_dataframe_with_mixed_numeric_and_string():
    """测试混合数值和字符串列的 DataFrame"""
    validator = DataValidator()
    
    df = pd.DataFrame({
        "price": [1.0, 2.0, 3.0],
        "name": ["A", "B", "C"],
        "volume": [100, 200, 300],
    })
    
    result = validator.validate(df, data_type="generic")
    # name 列不是数值列，不应该触发类型错误
    assert "name" not in str(result["errors"])


def test_validate_duration_tracking():
    """测试验证持续时间的跟踪"""
    validator = DataValidator()
    
    df = pd.DataFrame({"x": [1, 2, 3]})
    result = validator.validate(df, data_type="generic")
    
    assert "duration_seconds" in result
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] >= 0

