# -*- coding: utf-8 -*-
"""
数据预处理器真实实现测试
测试 DataPreprocessor 的核心功能
"""

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
from datetime import datetime, timedelta

from src.data.preprocessing.data_preprocessor import DataPreprocessor, DataQualityMonitor


@pytest.fixture
def preprocessor():
    """创建数据预处理器实例"""
    return DataPreprocessor()


@pytest.fixture
def sample_ohlcv_data():
    """创建示例OHLCV数据"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    return pd.DataFrame({
        'timestamp': dates,
        'open': [100 + i * 0.5 + np.random.randn() for i in range(20)],
        'high': [105 + i * 0.5 + np.random.randn() for i in range(20)],
        'low': [95 + i * 0.5 + np.random.randn() for i in range(20)],
        'close': [102 + i * 0.5 + np.random.randn() for i in range(20)],
        'volume': [1000 + i * 50 + abs(np.random.randn() * 100) for i in range(20)]
    })


@pytest.fixture
def data_with_missing_values(sample_ohlcv_data):
    """创建包含缺失值的数据"""
    data = sample_ohlcv_data.copy()
    data.loc[5, 'close'] = np.nan
    data.loc[10, 'volume'] = np.nan
    data.loc[15, 'open'] = np.nan
    return data


@pytest.fixture
def data_with_duplicates(sample_ohlcv_data):
    """创建包含重复时间戳的数据"""
    data = sample_ohlcv_data.copy()
    data = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)
    return data


@pytest.fixture
def data_with_outliers(sample_ohlcv_data):
    """创建包含异常值的数据"""
    data = sample_ohlcv_data.copy()
    data.loc[5, 'close'] = 10000  # 异常高值
    data.loc[10, 'volume'] = -1000  # 异常负值
    return data


def test_preprocessor_initialization(preprocessor):
    """测试预处理器初始化"""
    assert preprocessor.config == {}
    assert preprocessor.preprocessing_stats['total_processed'] == 0


def test_preprocess_empty_data(preprocessor):
    """测试预处理空数据"""
    empty_data = pd.DataFrame()
    
    result = preprocessor.preprocess(empty_data)
    
    assert result.empty


def test_preprocess_none_data(preprocessor):
    """测试预处理None数据"""
    result = preprocessor.preprocess(None)
    
    assert result is None


def test_preprocess_with_default_steps(preprocessor, sample_ohlcv_data):
    """测试使用默认步骤预处理"""
    result = preprocessor.preprocess(sample_ohlcv_data)
    
    assert result is not None
    assert len(result) == len(sample_ohlcv_data)
    assert preprocessor.preprocessing_stats['total_processed'] == 1


def test_preprocess_with_custom_steps(preprocessor, sample_ohlcv_data):
    """测试使用自定义步骤预处理"""
    result = preprocessor.preprocess(sample_ohlcv_data, steps=['validate', 'clean'])
    
    assert result is not None
    assert len(result) <= len(sample_ohlcv_data)


def test_validate_data_checks_required_columns(preprocessor, sample_ohlcv_data):
    """测试验证必需列"""
    # 删除必需列
    invalid_data = sample_ohlcv_data.drop(columns=['close'])
    
    with pytest.raises(ValueError, match="缺少必需列"):
        preprocessor._validate_data(invalid_data)


def test_validate_data_converts_types(preprocessor, sample_ohlcv_data):
    """测试验证数据类型转换"""
    # 将数值列转换为字符串
    data = sample_ohlcv_data.copy()
    data['close'] = data['close'].astype(str)
    
    result = preprocessor._validate_data(data)
    
    assert pd.api.types.is_numeric_dtype(result['close'])


def test_clean_data_removes_duplicates(preprocessor, data_with_duplicates):
    """测试清理重复数据"""
    original_count = len(data_with_duplicates)
    
    result = preprocessor._clean_data(data_with_duplicates)
    
    assert len(result) < original_count
    assert preprocessor.preprocessing_stats['duplicates_removed'] > 0


def test_clean_data_handles_missing_values(preprocessor, data_with_missing_values):
    """测试处理缺失值"""
    missing_before = data_with_missing_values.isnull().sum().sum()
    
    result = preprocessor._clean_data(data_with_missing_values)
    
    missing_after = result.isnull().sum().sum()
    assert missing_after < missing_before
    assert preprocessor.preprocessing_stats['missing_values_handled'] > 0


def test_handle_missing_values_forward_fill_prices(preprocessor, data_with_missing_values):
    """测试价格数据前向填充"""
    result = preprocessor._handle_missing_values(data_with_missing_values)
    
    # 价格列应该被前向填充
    assert not result['open'].isnull().any()
    assert not result['close'].isnull().any()


def test_handle_missing_values_zero_fill_volume(preprocessor, data_with_missing_values):
    """测试成交量零值填充"""
    result = preprocessor._handle_missing_values(data_with_missing_values)
    
    # 成交量应该被填充为0
    if 'volume' in result.columns:
        assert not result['volume'].isnull().any()


def test_handle_outliers_with_iqr(preprocessor, data_with_outliers):
    """测试IQR方法处理异常值"""
    result = preprocessor._handle_outliers(data_with_outliers)
    
    assert preprocessor.preprocessing_stats['outliers_handled'] > 0
    # 异常值应该被替换为中位数
    assert result.loc[5, 'close'] != 10000


def test_normalize_data_with_prices(preprocessor, sample_ohlcv_data):
    """测试价格标准化"""
    config = {'normalize_prices': True}
    preprocessor.config = config
    
    result = preprocessor._normalize_data(sample_ohlcv_data)
    
    # 价格应该相对于close标准化
    assert 'open' in result.columns
    assert 'close' in result.columns


def test_normalize_data_with_volume(preprocessor, sample_ohlcv_data):
    """测试成交量标准化"""
    config = {'normalize_volume': True}
    preprocessor.config = config
    
    result = preprocessor._normalize_data(sample_ohlcv_data)
    
    # 成交量应该被log变换
    assert 'volume' in result.columns
    assert result['volume'].min() >= 0


def test_quality_monitor_initialization():
    """测试质量监控器初始化"""
    monitor = DataQualityMonitor()
    
    assert monitor.quality_metrics['total_records'] == 0
    assert monitor.quality_metrics['data_quality_score'] == 0.0


def test_assess_quality_calculates_metrics(sample_ohlcv_data):
    """测试质量评估计算指标"""
    monitor = DataQualityMonitor()
    
    metrics = monitor.assess_quality(sample_ohlcv_data)
    
    assert 'total_records' in metrics
    assert 'missing_values' in metrics
    assert 'duplicates' in metrics
    assert 'completeness' in metrics
    assert 'data_quality_score' in metrics
    assert 0 <= metrics['data_quality_score'] <= 100


def test_generate_quality_report(sample_ohlcv_data):
    """测试生成质量报告"""
    monitor = DataQualityMonitor()
    
    report = monitor.generate_quality_report(sample_ohlcv_data)
    
    assert isinstance(report, str)
    assert '数据质量评估报告' in report
    assert '总记录数' in report
    assert '数据质量评分' in report

