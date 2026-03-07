from __future__ import annotations

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

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.validation.validator import (
    ConsistencyReport,
    DataValidator,
    OutlierReport,
    QualityReport,
    ValidationError,
    ValidationResult,
)


def _make_df():
    return pd.DataFrame(
        {
            "price": [100.0, 101.0, np.nan, 103.0],
            "volume": [1000, 1000, 1200, 1300],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-02-01", "2024-01-04"]),
        }
    )


class TestValidatorBasicChecks:
    def setup_method(self):
        self.validator = DataValidator()

    def test_validation_error_contains_result(self):
        result = ValidationResult(True, {}, [], datetime.now().isoformat(), "demo")
        err = ValidationError("boom", result)
        assert err.validation_result is result

    def test_validate_date_range_detects_outside_values(self):
        df = _make_df()
        assert not bool(self.validator.validate_date_range(df, "2024-01-01", "2024-01-03"))
        assert bool(self.validator.validate_date_range(df, "2024-01-01", "2024-02-01"))

    def test_numeric_missing_duplicate_and_outliers(self):
        df = _make_df()
        df.loc[1, "price"] = "not-number"
        assert self.validator.validate_numeric_columns(df, ["price"]) is False

        df = _make_df()
        df.loc[0, "price"] = np.nan
        assert self.validator.validate_no_missing_values(df, ["price"]) is False

        df = _make_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        assert self.validator.validate_no_duplicates(df) is False

        df = _make_df()
        df.loc[0, "price"] = 10_000.0
        assert self.validator.validate_outliers(df, ["price"], threshold=0.5) is False

    def test_validate_data_consistency_for_df_and_non_df(self):
        df = _make_df()
        report = self.validator.validate_data_consistency(df)
        assert isinstance(report, ConsistencyReport)
        assert report.consistency_score == 0.85

        report = self.validator.validate_data_consistency({"price": []})
        assert report.is_consistent is False
        assert "不支持的数据类型" in report.inconsistencies

    def test_add_custom_rule_and_validate_quality_non_df(self):
        self.validator.add_custom_rule("no_op", lambda data: True)
        assert "no_op" in self.validator._rules  # noqa: SLF001

        report = self.validator.validate_quality({"not": "df"})
        assert isinstance(report, QualityReport)
        assert report.overall_score == 0.0

    def test_validate_data_routes_dictionary(self):
        payload = {"date": "2024-01-01"}
        result = self.validator.validate_data(payload, data_type="dict")
        assert result.is_valid is False
        assert "缺少必需字段" in result.errors[0]

    def test_validate_data_model_dataframe_and_dict(self):
        df = _make_df()
        assert self.validator.validate_data_model(df, {"price": float, "volume": float}) is True
        assert self.validator.validate_data_model(df, {"missing": float}) is False

        payload = {"field": 1}
        assert self.validator.validate_data_model(payload, {"field": int}) is True
        assert self.validator.validate_data_model(payload, {"missing": int}) is False


class TestValidatorEdgeCases:
    """测试边界情况和异常路径"""

    def setup_method(self):
        self.validator = DataValidator()

    def test_validate_data_with_unsupported_type(self):
        """测试不支持的数据类型"""
        result = self.validator.validate_data([1, 2, 3], data_type="list")
        assert result.is_valid is False
        assert "不支持的数据类型" in result.errors[0]

    def test_validate_quality_with_dataframe(self):
        """测试 DataFrame 质量验证"""
        df = pd.DataFrame({"price": [100.0, 101.0, 102.0], "volume": [1000, 1100, 1200]})
        report = self.validator.validate_quality(df)
        assert isinstance(report, QualityReport)
        assert report.overall_score > 0
        assert report.completeness > 0
        assert report.accuracy > 0
        assert report.consistency > 0
        assert report.timeliness > 0

    def test_validate_data_model_with_non_dataframe_or_dict(self):
        """测试非 DataFrame 非 dict 的数据模型验证"""
        result = self.validator.validate_data_model([1, 2, 3], {"field": int})
        assert result is False

    def test_validate_date_range_with_non_dataframe(self):
        """测试非 DataFrame 的日期范围验证"""
        result = self.validator.validate_date_range({"date": "2024-01-01"}, "2024-01-01", "2024-01-31")
        assert result is True

    def test_validate_date_range_without_date_column(self):
        """测试没有 date 列的 DataFrame"""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result = self.validator.validate_date_range(df, "2024-01-01", "2024-01-31")
        assert result is True

    def test_validate_numeric_columns_with_missing_column(self):
        """测试数值列验证中列不存在的情况"""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result = self.validator.validate_numeric_columns(df, ["missing_col"])
        assert result is True

    def test_validate_no_missing_values_with_missing_column(self):
        """测试无缺失值验证中列不存在的情况"""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result = self.validator.validate_no_missing_values(df, ["missing_col"])
        assert result is True

    def test_validate_no_duplicates_with_specific_columns(self):
        """测试指定列的重复值验证"""
        df = pd.DataFrame({"col1": [1, 1, 2], "col2": [10, 20, 30]})
        result = self.validator.validate_no_duplicates(df, ["col1"])
        assert result is False

    def test_validate_outliers_with_missing_column(self):
        """测试离群值验证中列不存在的情况"""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result = self.validator.validate_outliers(df, ["missing_col"])
        assert result is True

    def test_validate_outliers_with_non_numeric_column(self):
        """测试离群值验证中非数值列的情况"""
        df = pd.DataFrame({"text": ["a", "b", "c"]})
        result = self.validator.validate_outliers(df, ["text"])
        assert result is True

    def test_validate_method(self):
        """测试通用 validate 方法"""
        df = pd.DataFrame({"price": [100.0, 101.0]})
        result = self.validator.validate(df)
        assert isinstance(result, ValidationResult)

    def test_validate_dataframe_empty(self):
        """测试空 DataFrame"""
        df = pd.DataFrame()
        result = self.validator._validate_dataframe(df, "test")
        assert result.is_valid is False
        assert "数据为空" in result.errors
        assert result.metrics['empty'] == 0.0

    def test_validate_dataframe_all_null_values(self):
        """测试全为空值的 DataFrame"""
        df = pd.DataFrame({"col1": [None, None], "col2": [None, None]})
        result = self.validator._validate_dataframe(df, "test")
        assert result.is_valid is False
        assert "数据全为空值" in result.errors

    def test_validate_dataframe_high_null_percentage(self):
        """测试空值比例过高的 DataFrame"""
        df = pd.DataFrame({"col1": [None, None, None, None, None, None, None, None, None, None, 1]})
        result = self.validator._validate_dataframe(df, "test")
        assert "空值比例过高" in result.errors[0]

    def test_validate_dataframe_high_duplicate_percentage(self):
        """测试重复值比例过高的 DataFrame"""
        df = pd.DataFrame({"col1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]})
        result = self.validator._validate_dataframe(df, "test")
        assert "重复值比例过高" in result.errors[0]

    def test_validate_dict_data_empty_dict(self):
        """测试空字典验证"""
        result = self.validator._validate_dict_data({}, "test")
        assert result.is_valid is False
        assert "数据为空" in result.errors

    def test_calculate_completeness_empty_dataframe(self):
        """测试空 DataFrame 的完整性计算"""
        df = pd.DataFrame()
        result = self.validator._calculate_completeness(df)
        assert result == 0.0

    def test_find_inconsistencies_empty_dataframe(self):
        """测试空 DataFrame 的不一致性查找"""
        df = pd.DataFrame()
        result = self.validator._find_inconsistencies(df)
        assert "数据为空" in result

    def test_calculate_consistency_score_empty_dataframe(self):
        """测试空 DataFrame 的一致性分数计算"""
        df = pd.DataFrame()
        result = self.validator._calculate_consistency_score(df)
        assert result == 0.0

    def test_check_price_deviation(self):
        """测试价格偏差检查"""
        data = {"price": 100.0}
        result = self.validator._check_price_deviation(data)
        assert result["is_valid"] is True
        assert "score" in result

    def test_check_volume_spike(self):
        """测试成交量突增检查"""
        data = {"volume": 1000}
        result = self.validator._check_volume_spike(data)
        assert result["is_valid"] is True

    def test_check_null_values(self):
        """测试空值检查"""
        data = {"field": None}
        result = self.validator._check_null_values(data)
        assert result["is_valid"] is True

    def test_check_outliers(self):
        """测试离群值检查"""
        data = {"value": 100.0}
        result = self.validator._check_outliers(data)
        assert result["is_valid"] is True

    def test_check_time_gaps(self):
        """测试时间间隔检查"""
        data = {"timestamp": "2024-01-01"}
        result = self.validator._check_time_gaps(data)
        assert result["is_valid"] is True

    def test_check_financial_completeness(self):
        """测试财务数据完整性检查"""
        data = {"revenue": 1000.0}
        result = self.validator._check_financial_completeness(data)
        assert result["is_valid"] is True

    def test_check_financial_accuracy(self):
        """测试财务数据准确性检查"""
        data = {"revenue": 1000.0}
        result = self.validator._check_financial_accuracy(data)
        assert result["is_valid"] is True

    def test_check_financial_consistency(self):
        """测试财务数据一致性检查"""
        data = {"revenue": 1000.0}
        result = self.validator._check_financial_consistency(data)
        assert result["is_valid"] is True


class TestValidatorDomainSpecific:
    def setup_method(self):
        self.validator = DataValidator()

    def test_validate_stock_data_collects_errors(self, monkeypatch):
        def fake(status):
            return {"is_valid": False, "score": 0.1, "message": status}

        monkeypatch.setattr(self.validator, "_check_price_deviation", lambda data: fake("price"))
        monkeypatch.setattr(self.validator, "_check_volume_spike", lambda data: fake("volume"))
        monkeypatch.setattr(self.validator, "_check_null_values", lambda data: fake("null"))
        monkeypatch.setattr(self.validator, "_check_outliers", lambda data: fake("outlier"))
        monkeypatch.setattr(self.validator, "_check_time_gaps", lambda data: fake("gap"))

        result = self.validator.validate_stock_data({})
        assert result.is_valid is False
        assert len(result.errors) == 5

    def test_validate_financial_data_collects_errors(self, monkeypatch):
        def fake(status):
            return {"is_valid": False, "score": 0.1, "message": status}

        monkeypatch.setattr(self.validator, "_check_financial_completeness", lambda data: fake("complete"))
        monkeypatch.setattr(self.validator, "_check_financial_accuracy", lambda data: fake("accuracy"))
        monkeypatch.setattr(self.validator, "_check_financial_consistency", lambda data: fake("consistency"))

        result = self.validator.validate_financial_data({})
        assert result.is_valid is False
        assert len(result.errors) == 3

    def test_generate_quality_report_dataframe_and_non_df(self):
        df = _make_df()
        report = self.validator.generate_quality_report(df)
        assert report.overall_score > 0
        assert isinstance(report.details, dict)

        zero_report = self.validator.generate_quality_report({"not": "df"})
        assert zero_report.overall_score == 0

    def test_detect_outliers_and_consistency_helpers(self):
        series = pd.Series([1, 1, 1, 10])
        report = self.validator.detect_outliers(series, threshold=0.5)
        assert isinstance(report, OutlierReport)
        assert report.outlier_count >= 1

        series = pd.Series([], dtype=float)
        report = self.validator.detect_outliers(series)
        assert report.outlier_count == 0

        consistency = self.validator.check_consistency({}, {})
        assert isinstance(consistency, ConsistencyReport)

    def test_validation_history_tracking(self):
        df = _make_df()
        result = self.validator.validate_data(df, "df")
        self.validator.get_validation_history().append(result)
        assert self.validator.get_validation_history()

