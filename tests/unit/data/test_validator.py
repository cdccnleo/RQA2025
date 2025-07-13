"""
数据验证器单元测试
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.data.validator import (
    DataValidator, ValidationResult, QualityReport, 
    OutlierReport, ConsistencyReport, ValidationError
)

class TestDataValidator:
    """DataValidator单元测试"""
    
    @pytest.fixture
    def validator(self):
        """创建DataValidator实例"""
        return DataValidator()
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'price': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    @pytest.fixture
    def data_with_issues(self):
        """创建包含问题的数据"""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10),
            'price': [100, 101, None, 103, 104, 105, 106, 107, 108, 109],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        })
    
    @pytest.fixture
    def duplicate_data(self):
        """创建包含重复行的数据"""
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=5),
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        return pd.concat([df, df.iloc[0:2]])  # 添加重复行
    
    def test_validator_init(self, validator):
        """测试验证器初始化"""
        assert validator is not None
        assert hasattr(validator, '_schema_registry')
        assert hasattr(validator, '_custom_rules')
    
    def test_validate_data_none(self, validator):
        """测试验证空数据"""
        result = validator.validate_data(None)
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert "数据为空" in result.errors
    
    def test_validate_data_not_dataframe(self, validator):
        """测试验证非DataFrame数据"""
        result = validator.validate_data("not a dataframe")
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert "数据不是DataFrame类型" in result.errors
    
    def test_validate_data_empty(self, validator):
        """测试验证空DataFrame"""
        empty_df = pd.DataFrame()
        result = validator.validate_data(empty_df)
        assert isinstance(result, ValidationResult)
        assert result.is_valid  # 空DataFrame是有效的
        assert "数据框为空" in result.warnings
    
    def test_validate_data_valid(self, validator, sample_data):
        """测试验证有效数据"""
        result = validator.validate_data(sample_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_data_with_null_values(self, validator, data_with_issues):
        """测试验证包含空值的数据"""
        result = validator.validate_data(data_with_issues)
        assert isinstance(result, ValidationResult)
        assert result.is_valid  # 空值只是警告，不是错误
        assert len(result.warnings) > 0
        assert any("空值" in warning for warning in result.warnings)
    
    def test_validate_data_with_duplicates(self, validator, duplicate_data):
        """测试验证包含重复行的数据"""
        result = validator.validate_data(duplicate_data)
        assert isinstance(result, ValidationResult)
        assert result.is_valid  # 重复行只是警告，不是错误
        assert len(result.warnings) > 0
        assert any("重复行" in warning for warning in result.warnings)
    
    def test_validate_quality_empty_data(self, validator):
        """测试质量验证 - 空数据"""
        result = validator.validate_quality(None)
        assert isinstance(result, QualityReport)
        assert result.score == 0.0
        assert "数据为空" in result.issues
    
    def test_validate_quality_valid_data(self, validator, sample_data):
        """测试质量验证 - 有效数据"""
        result = validator.validate_quality(sample_data)
        assert isinstance(result, QualityReport)
        assert 0.0 <= result.score <= 1.0
        assert 'null_rate' in result.metrics
        assert 'duplicate_rate' in result.metrics
        assert result.metrics['null_rate'] == 0.0
        assert result.metrics['duplicate_rate'] == 0.0
    
    def test_validate_quality_data_with_issues(self, validator, data_with_issues):
        """测试质量验证 - 有问题的数据"""
        result = validator.validate_quality(data_with_issues)
        assert isinstance(result, QualityReport)
        assert result.score < 1.0  # 质量分数应该降低
        assert result.metrics['null_rate'] > 0.0
    
    def test_validate_data_model_none(self, validator):
        """测试数据模型验证 - 空模型"""
        result = validator.validate_data_model(None)
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert "数据模型为空" in result.errors
    
    def test_validate_data_model_missing_attributes(self, validator):
        """测试数据模型验证 - 缺少属性"""
        class Empty:
            pass
        empty_model = Empty()
        result = validator.validate_data_model(empty_model)
        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert "数据模型缺少data属性" in result.errors
        assert "数据模型缺少frequency属性" in result.errors
    
    def test_validate_data_model_valid(self, validator):
        """测试数据模型验证 - 有效模型"""
        mock_model = Mock()
        mock_model.data = pd.DataFrame({'col': [1, 2, 3]})
        mock_model.frequency = '1D'
        
        result = validator.validate_data_model(mock_model)
        assert isinstance(result, ValidationResult)
        assert result.is_valid
    
    def test_validate_date_range_valid(self, validator, sample_data):
        """测试日期范围验证 - 有效范围"""
        result = validator.validate_date_range(
            sample_data, 'date', '2023-01-01', '2023-01-10'
        )
        assert result is True
    
    def test_validate_date_range_invalid_column(self, validator, sample_data):
        """测试日期范围验证 - 无效列名"""
        with pytest.raises(ValidationError):
            validator.validate_date_range(
                sample_data, 'invalid_col', '2023-01-01', '2023-01-10'
            )
    
    def test_validate_date_range_out_of_range(self, validator, sample_data):
        """测试日期范围验证 - 超出范围"""
        result = validator.validate_date_range(
            sample_data, 'date', '2023-01-05', '2023-01-07'
        )
        assert result is False
    
    def test_validate_date_range_invalid_dates(self, validator, sample_data):
        """测试日期范围验证 - 无效日期格式"""
        with pytest.raises(ValidationError):
            validator.validate_date_range(
                sample_data, 'date', 'invalid-date', '2023-01-10'
            )
    
    def test_validate_no_missing_values_valid(self, validator, sample_data):
        """测试无缺失值验证 - 有效数据"""
        result = validator.validate_no_missing_values(sample_data)
        assert result is True
    
    def test_validate_no_missing_values_with_nulls(self, validator, data_with_issues):
        """测试无缺失值验证 - 包含空值"""
        result = validator.validate_no_missing_values(data_with_issues)
        assert result is False
    
    def test_validate_no_duplicates_valid(self, validator, sample_data):
        """测试无重复验证 - 有效数据"""
        result = validator.validate_no_duplicates(sample_data)
        assert result is True
    
    def test_validate_no_duplicates_with_duplicates(self, validator, duplicate_data):
        """测试无重复验证 - 包含重复"""
        result = validator.validate_no_duplicates(duplicate_data)
        assert result is False
    
    def test_validate_outliers_iqr_method(self, validator, sample_data):
        """测试异常值验证 - IQR方法"""
        result = validator.validate_outliers(sample_data, 'price', 'iqr')
        assert isinstance(result, OutlierReport)
        assert hasattr(result, 'outlier_count')
        assert hasattr(result, 'outlier_indices')
        assert hasattr(result, 'threshold')
    
    def test_validate_outliers_zscore_method(self, validator, sample_data):
        """测试异常值验证 - Z-score方法"""
        result = validator.validate_outliers(sample_data, 'price', 'zscore')
        assert isinstance(result, OutlierReport)
    
    def test_validate_outliers_invalid_column(self, validator, sample_data):
        """测试异常值验证 - 无效列名"""
        with pytest.raises(ValidationError):
            validator.validate_outliers(sample_data, 'invalid_col', 'iqr')
    
    def test_validate_outliers_invalid_method(self, validator, sample_data):
        """测试异常值验证 - 无效方法"""
        with pytest.raises(ValidationError):
            validator.validate_outliers(sample_data, 'price', 'invalid_method')
    
    def test_validate_data_consistency_valid(self, validator, sample_data):
        """测试数据一致性验证 - 有效数据"""
        result = validator.validate_data_consistency(sample_data)
        assert isinstance(result, ConsistencyReport)
        assert result.is_consistent
        assert len(result.inconsistencies) == 0
    
    def test_validate_data_consistency_with_issues(self, validator, data_with_issues):
        """测试数据一致性验证 - 有问题的数据"""
        result = validator.validate_data_consistency(data_with_issues)
        assert isinstance(result, ConsistencyReport)
        # 可能不是一致的，取决于具体实现
    
    def test_add_custom_rule(self, validator):
        """测试添加自定义规则"""
        def custom_rule(data):
            return len(data) > 0
        
        validator.add_custom_rule(custom_rule)
        assert len(validator._custom_rules) == 1
        assert custom_rule in validator._custom_rules
    
    def test_validation_result(self):
        """测试ValidationResult类"""
        result = ValidationResult(True, ["error1"], ["warning1"])
        assert result.is_valid is True
        assert "error1" in result.errors
        assert "warning1" in result.warnings
    
    def test_quality_report(self):
        """测试QualityReport类"""
        report = QualityReport(0.8, ["issue1"], {"metric1": 0.5})
        assert report.score == 0.8
        assert "issue1" in report.issues
        assert report.metrics["metric1"] == 0.5
    
    def test_outlier_report(self):
        """测试OutlierReport类"""
        report = OutlierReport(5, [1, 2, 3], 2.0)
        assert report.outlier_count == 5
        assert report.outlier_indices == [1, 2, 3]
        assert report.threshold == 2.0
    
    def test_consistency_report(self):
        """测试ConsistencyReport类"""
        report = ConsistencyReport(False, ["inconsistency1"])
        assert report.is_consistent is False
        assert "inconsistency1" in report.inconsistencies 