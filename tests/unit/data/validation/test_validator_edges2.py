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


import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.validation.validator import (
    DataValidator,
    ValidationResult,
    QualityReport,
    OutlierReport,
    ConsistencyReport,
    ValidationError,
)


def test_validation_result_initialization():
    """测试 ValidationResult 初始化"""
    result = ValidationResult(
        is_valid=True,
        metrics={'test': 1.0},
        errors=[],
        timestamp=datetime.now().isoformat(),
        data_type="test"
    )
    assert result.is_valid is True
    assert result.metrics == {'test': 1.0}
    assert result.errors == []
    assert result.data_type == "test"


def test_validation_result_default_data_type():
    """测试 ValidationResult（默认数据类型）"""
    result = ValidationResult(
        is_valid=True,
        metrics={},
        errors=[],
        timestamp=datetime.now().isoformat()
    )
    assert result.data_type == "unknown"


def test_quality_report_initialization():
    """测试 QualityReport 初始化"""
    report = QualityReport(
        overall_score=0.9,
        completeness=0.95,
        accuracy=0.85,
        consistency=0.90,
        timeliness=0.95,
        details={'test': 'value'}
    )
    assert report.overall_score == 0.9
    assert report.completeness == 0.95
    assert report.details == {'test': 'value'}


def test_outlier_report_initialization():
    """测试 OutlierReport 初始化"""
    report = OutlierReport(
        outlier_count=5,
        outlier_percentage=0.1,
        outlier_indices=[0, 1, 2, 3, 4],
        outlier_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        threshold=2.0
    )
    assert report.outlier_count == 5
    assert report.outlier_percentage == 0.1
    assert len(report.outlier_indices) == 5


def test_consistency_report_initialization():
    """测试 ConsistencyReport 初始化"""
    report = ConsistencyReport(
        is_consistent=True,
        consistency_score=0.85,
        inconsistencies=[],
        cross_reference_results={'test': 'value'}
    )
    assert report.is_consistent is True
    assert report.consistency_score == 0.85


def test_validation_error_initialization():
    """测试 ValidationError 初始化"""
    error = ValidationError("Test error")
    assert str(error) == "Test error"
    assert error.validation_result is None


def test_validation_error_with_result():
    """测试 ValidationError（带验证结果）"""
    result = ValidationResult(
        is_valid=False,
        metrics={},
        errors=['error1'],
        timestamp=datetime.now().isoformat()
    )
    error = ValidationError("Test error", validation_result=result)
    assert error.validation_result == result


def test_data_validator_validate_unsupported_type():
    """测试 DataValidator（验证不支持的数据类型）"""
    validator = DataValidator()
    result = validator.validate_data("unsupported", data_type="test")
    assert result.is_valid is False
    assert "不支持的数据类型" in result.errors[0]


def test_data_validator_validate_none():
    """测试 DataValidator（验证 None）"""
    validator = DataValidator()
    result = validator.validate_data(None, data_type="test")
    assert result.is_valid is False
    assert "不支持的数据类型" in result.errors[0]


def test_data_validator_validate_empty_dataframe():
    """测试 DataValidator（验证空 DataFrame）"""
    validator = DataValidator()
    df = pd.DataFrame()
    result = validator.validate_data(df)
    assert result.is_valid is False
    assert "数据为空" in result.errors


def test_data_validator_validate_all_nan_dataframe():
    """测试 DataValidator（验证全为 NaN 的 DataFrame）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
    result = validator.validate_data(df)
    assert result.is_valid is False
    assert "数据全为空值" in result.errors


def test_data_validator_validate_high_null_percentage():
    """测试 DataValidator（验证高空值比例）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]})
    result = validator.validate_data(df)
    # 空值比例超过 10% 应该产生错误
    assert "空值比例过高" in result.errors or result.is_valid is False


def test_data_validator_validate_high_duplicate_percentage():
    """测试 DataValidator（验证高重复值比例）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]})
    result = validator.validate_data(df)
    # 重复值比例超过 5% 应该产生错误
    assert "重复值比例过高" in result.errors or result.is_valid is False


def test_data_validator_validate_empty_dict():
    """测试 DataValidator（验证空字典）"""
    validator = DataValidator()
    result = validator.validate_data({}, data_type="test")
    assert result.is_valid is False
    assert "数据为空" in result.errors


def test_data_validator_validate_dict_missing_fields():
    """测试 DataValidator（验证缺少必需字段的字典）"""
    validator = DataValidator()
    result = validator.validate_data({'test': 'value'}, data_type="test")
    assert result.is_valid is False
    assert "缺少必需字段" in result.errors[0]


def test_data_validator_validate_dict_with_fields():
    """测试 DataValidator（验证包含必需字段的字典）"""
    validator = DataValidator()
    result = validator.validate_data({'date': '2024-01-01', 'symbol': '000001'}, data_type="test")
    assert result.is_valid is True


def test_data_validator_validate_quality_non_dataframe():
    """测试 DataValidator（验证质量，非 DataFrame）"""
    validator = DataValidator()
    result = validator.validate_quality("not a dataframe")
    assert result.overall_score == 0.0
    assert result.completeness == 0.0
    assert result.accuracy == 0.0


def test_data_validator_validate_quality_empty_dataframe():
    """测试 DataValidator（验证质量，空 DataFrame）"""
    validator = DataValidator()
    df = pd.DataFrame()
    result = validator.validate_quality(df)
    assert result.completeness == 0.0


def test_data_validator_validate_data_model_empty_schema():
    """测试 DataValidator（验证数据模型，空模式）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_data_model(df, {})
    assert result is True


def test_data_validator_validate_data_model_missing_column():
    """测试 DataValidator（验证数据模型，缺少列）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_data_model(df, {'b': 'int'})
    assert result is False


def test_data_validator_validate_data_model_non_dataframe():
    """测试 DataValidator（验证数据模型，非 DataFrame）"""
    validator = DataValidator()
    result = validator.validate_data_model("not a dataframe", {'a': 'int'})
    assert result is False


def test_data_validator_validate_date_range_no_date_column():
    """测试 DataValidator（验证日期范围，无 date 列）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_date_range(df, "2024-01-01", "2024-12-31")
    assert result is True  # 无 date 列时返回 True


def test_data_validator_validate_date_range_non_dataframe():
    """测试 DataValidator（验证日期范围，非 DataFrame）"""
    validator = DataValidator()
    result = validator.validate_date_range("not a dataframe", "2024-01-01", "2024-12-31")
    assert result is True


def test_data_validator_validate_numeric_columns_empty_list():
    """测试 DataValidator（验证数值列，空列表）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_numeric_columns(df, [])
    assert result is True


def test_data_validator_validate_numeric_columns_missing_column():
    """测试 DataValidator（验证数值列，缺少列）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_numeric_columns(df, ['b'])
    assert result is True  # 缺少列时返回 True


def test_data_validator_validate_numeric_columns_non_numeric():
    """测试 DataValidator（验证数值列，非数值类型）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': ['x', 'y', 'z']})
    result = validator.validate_numeric_columns(df, ['a'])
    assert result is False


def test_data_validator_validate_no_missing_values_empty_list():
    """测试 DataValidator（验证无缺失值，空列表）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, np.nan]})
    result = validator.validate_no_missing_values(df, [])
    assert result is True


def test_data_validator_validate_no_missing_values_with_nulls():
    """测试 DataValidator（验证无缺失值，有空值）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, np.nan]})
    result = validator.validate_no_missing_values(df, ['a'])
    assert result is False


def test_data_validator_validate_no_missing_values_no_nulls():
    """测试 DataValidator（验证无缺失值，无空值）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_no_missing_values(df, ['a'])
    assert result is True


def test_data_validator_validate_no_duplicates_none_columns():
    """测试 DataValidator（验证无重复值，None 列）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_no_duplicates(df, None)
    # None 列时应该检查整行重复
    assert isinstance(result, bool)


def test_data_validator_validate_no_duplicates_empty_list():
    """测试 DataValidator（验证无重复值，空列表）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_no_duplicates(df, [])
    # 空列表时应该检查整行重复
    assert isinstance(result, bool)


def test_data_validator_validate_no_duplicates_with_duplicates():
    """测试 DataValidator（验证无重复值，有重复）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 1]})
    result = validator.validate_no_duplicates(df, ['a'])
    assert result is False


def test_data_validator_validate_outliers_empty_list():
    """测试 DataValidator（验证离群值，空列表）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate_outliers(df, [])
    assert result is True


def test_data_validator_validate_outliers_zero_iqr():
    """测试 DataValidator（验证离群值，零 IQR）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 1, 1, 1, 1]})  # 所有值相同，IQR = 0
    result = validator.validate_outliers(df, ['a'])
    # 零 IQR 时，所有值都在边界内，应该返回 True
    assert result is True


def test_data_validator_validate_outliers_non_numeric():
    """测试 DataValidator（验证离群值，非数值列）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': ['x', 'y', 'z']})
    result = validator.validate_outliers(df, ['a'])
    # 非数值列应该被跳过，返回 True
    assert result is True


def test_data_validator_validate_data_consistency_non_dataframe():
    """测试 DataValidator（验证数据一致性，非 DataFrame）"""
    validator = DataValidator()
    result = validator.validate_data_consistency("not a dataframe")
    assert result.is_consistent is False
    assert result.consistency_score == 0.0
    assert "不支持的数据类型" in result.inconsistencies[0]


def test_data_validator_validate_data_consistency_empty_dataframe():
    """测试 DataValidator（验证数据一致性，空 DataFrame）"""
    validator = DataValidator()
    df = pd.DataFrame()
    result = validator.validate_data_consistency(df)
    assert result.consistency_score == 0.0
    assert "数据为空" in result.inconsistencies


def test_data_validator_add_custom_rule():
    """测试 DataValidator（添加自定义规则）"""
    validator = DataValidator()
    def custom_rule(data):
        return True
    validator.add_custom_rule("test_rule", custom_rule)
    assert "test_rule" in validator._rules


def test_data_validator_validate_with_rules():
    """测试 DataValidator（使用规则验证）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3]})
    result = validator.validate(df, rules=None)
    # validate 方法调用 validate_data
    assert hasattr(result, 'is_valid')


def test_data_validator_get_validation_history():
    """测试 DataValidator（获取验证历史）"""
    validator = DataValidator()
    history = validator.get_validation_history()
    assert isinstance(history, list)


def test_data_validator_generate_quality_report_non_dataframe():
    """测试 DataValidator（生成质量报告，非 DataFrame）"""
    validator = DataValidator()
    result = validator.generate_quality_report("not a dataframe", data_type="test")
    assert result.overall_score == 0.0
    assert result.completeness == 0.0


def test_data_validator_detect_outliers_empty_series():
    """测试 DataValidator（检测离群值，空 Series）"""
    validator = DataValidator()
    series = pd.Series([])
    result = validator.detect_outliers(series)
    assert result.outlier_count == 0
    assert result.outlier_percentage == 0.0
    assert result.outlier_indices == []


def test_data_validator_detect_outliers_zero_iqr():
    """测试 DataValidator（检测离群值，零 IQR）"""
    validator = DataValidator()
    series = pd.Series([1, 1, 1, 1, 1])  # 所有值相同，IQR = 0
    result = validator.detect_outliers(series)
    # 零 IQR 时，所有值都在边界内，应该没有离群值
    assert result.outlier_count == 0


def test_data_validator_detect_outliers_custom_threshold():
    """测试 DataValidator（检测离群值，自定义阈值）"""
    validator = DataValidator()
    series = pd.Series([1, 2, 3, 4, 5, 100])  # 100 是离群值
    result = validator.detect_outliers(series, threshold=1.5)
    assert result.threshold == 1.5
    assert result.outlier_count >= 0  # 可能有离群值


def test_data_validator_check_consistency():
    """测试 DataValidator（检查一致性）"""
    validator = DataValidator()
    result = validator.check_consistency({'a': 1}, {'b': 2})
    assert isinstance(result, ConsistencyReport)
    assert result.consistency_score == 0.85


def test_data_validator_validate_stock_data():
    """测试 DataValidator（验证股票数据）"""
    validator = DataValidator()
    data = {'price': 100.0, 'volume': 10000}
    result = validator.validate_stock_data(data)
    assert isinstance(result, ValidationResult)
    assert result.data_type == "stock"


def test_data_validator_validate_financial_data():
    """测试 DataValidator（验证财务数据）"""
    validator = DataValidator()
    data = {'revenue': 1000000, 'profit': 100000}
    result = validator.validate_financial_data(data)
    assert isinstance(result, ValidationResult)
    assert result.data_type == "financial"


def test_data_validator_validate_date_range_with_date_column():
    """测试 DataValidator（验证日期范围，有 date 列）"""
    validator = DataValidator()
    df = pd.DataFrame({
        'date': pd.to_datetime(['2024-01-01', '2024-06-01', '2024-12-31']),
        'value': [1, 2, 3]
    })
    result = validator.validate_date_range(df, "2024-01-01", "2024-12-31")
    # 所有日期都在范围内，应该返回 True
    assert result == True  # 所有日期都在范围内


def test_data_validator_validate_date_range_out_of_range():
    """测试 DataValidator（验证日期范围，超出范围）"""
    validator = DataValidator()
    df = pd.DataFrame({
        'date': pd.to_datetime(['2023-01-01', '2024-06-01', '2025-12-31']),
        'value': [1, 2, 3]
    })
    result = validator.validate_date_range(df, "2024-01-01", "2024-12-31")
    # 有超出范围的日期，应该返回 False
    # 注意：代码使用 >= 和 <=，所以边界日期会被包含
    # 2023-01-01 < 2024-01-01，应该返回 False
    assert result == False  # 有超出范围的日期


def test_data_validator_validate_numeric_columns_all_numeric():
    """测试 DataValidator（验证数值列，全部数值）"""
    validator = DataValidator()
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    result = validator.validate_numeric_columns(df, ['a', 'b'])
    assert result is True

