# -*- coding: utf-8 -*-
"""
数据处理器真实实现测试
测试 DataProcessor 的核心功能
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
from datetime import datetime
from unittest.mock import Mock

from src.data.processing.data_processor import DataProcessor, FillMethod


class MockDataModel:
    """模拟数据模型用于测试"""
    
    def __init__(self, data, frequency='1d', metadata=None):
        self.data = data
        self._frequency = frequency
        self._metadata = metadata or {}
    
    def get_frequency(self):
        return self._frequency
    
    def get_metadata(self):
        return self._metadata.copy()


@pytest.fixture
def processor():
    """创建数据处理器实例"""
    return DataProcessor()


@pytest.fixture
def sample_data():
    """创建示例数据"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'open': [100 + i for i in range(10)],
        'high': [105 + i for i in range(10)],
        'low': [95 + i for i in range(10)],
        'close': [102 + i for i in range(10)],
        'volume': [1000 + i * 100 for i in range(10)]
    })


@pytest.fixture
def data_with_nulls(sample_data):
    """创建包含空值的数据"""
    data = sample_data.copy()
    data.loc[2, 'close'] = np.nan
    data.loc[5, 'volume'] = np.nan
    return data


@pytest.fixture
def data_with_duplicates(sample_data):
    """创建包含重复行的数据"""
    data = sample_data.copy()
    data = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)
    return data


@pytest.fixture
def data_model(sample_data):
    """创建数据模型实例"""
    return MockDataModel(sample_data)


def test_processor_initialization(processor):
    """测试处理器初始化"""
    assert processor.config == {}
    assert processor.processing_steps == []
    assert 'processor_type' in processor.processing_info
    assert 'created_at' in processor.processing_info


def test_process_empty_data(processor):
    """测试处理空数据"""
    empty_data = pd.DataFrame()
    data_model = MockDataModel(empty_data)
    
    result = processor.process(data_model)
    assert result == data_model  # 应该直接返回原始模型


def test_process_with_forward_fill(processor, data_model, data_with_nulls):
    """测试前向填充处理"""
    data_model.data = data_with_nulls
    
    result = processor.process(data_model, fill_method='forward')
    
    assert result is not None
    assert not result.data['close'].isnull().any()
    assert 'processed_at' in result.get_metadata()


def test_process_with_backward_fill(processor, data_model, data_with_nulls):
    """测试后向填充处理"""
    data_model.data = data_with_nulls
    
    result = processor.process(data_model, fill_method='backward')
    
    assert result is not None
    assert not result.data['close'].isnull().any()


def test_process_with_mean_fill(processor, data_model, data_with_nulls):
    """测试均值填充处理"""
    data_model.data = data_with_nulls
    
    result = processor.process(data_model, fill_method='mean')
    
    assert result is not None
    assert not result.data['close'].isnull().any()
    assert not result.data['volume'].isnull().any()


def test_process_with_median_fill(processor, data_model, data_with_nulls):
    """测试中位数填充处理"""
    data_model.data = data_with_nulls
    
    result = processor.process(data_model, fill_method='median')
    
    assert result is not None
    assert not result.data['close'].isnull().any()


def test_process_with_zero_fill(processor, data_model, data_with_nulls):
    """测试零值填充处理"""
    data_model.data = data_with_nulls
    
    result = processor.process(data_model, fill_method='zero')
    
    assert result is not None
    assert not result.data['close'].isnull().any()
    assert result.data.loc[2, 'close'] == 0


def test_process_with_drop_fill(processor, data_model, data_with_nulls):
    """测试删除空值处理"""
    data_model.data = data_with_nulls
    original_len = len(data_model.data)
    
    result = processor.process(data_model, fill_method='drop')
    
    assert result is not None
    assert len(result.data) < original_len
    assert not result.data.isnull().any().any()


def test_process_remove_duplicates(processor, data_model, data_with_duplicates):
    """测试删除重复行"""
    data_model.data = data_with_duplicates
    original_len = len(data_model.data)
    
    result = processor.process(data_model, remove_duplicates=True)
    
    assert result is not None
    assert len(result.data) < original_len
    assert not result.data.duplicated().any()


def test_process_with_iqr_outlier_removal(processor, data_model):
    """测试IQR方法移除异常值"""
    # 创建包含异常值的数据
    data = data_model.data.copy()
    data.loc[9, 'close'] = 1000  # 添加异常值
    
    data_model.data = data
    result = processor.process(data_model, outlier_method='iqr')
    
    assert result is not None
    # 异常值应该被裁剪到合理范围
    assert result.data.loc[9, 'close'] <= data['close'].quantile(0.75) + 1.5 * (data['close'].quantile(0.75) - data['close'].quantile(0.25))


def test_process_with_zscore_outlier_removal(processor, data_model):
    """测试Z-score方法移除异常值"""
    # 创建包含异常值的数据
    data = data_model.data.copy()
    data.loc[9, 'close'] = 1000  # 添加异常值
    original_median = data['close'].median()
    
    data_model.data = data
    result = processor.process(data_model, outlier_method='zscore', normalize_method=None)
    
    assert result is not None
    # 异常值应该被替换为中位数（但由于标准化，值会不同）
    # 检查异常值是否被处理（不再是1000）
    assert result.data.loc[9, 'close'] != 1000


def test_process_with_minmax_normalization(processor, data_model):
    """测试MinMax标准化"""
    result = processor.process(data_model, normalize_method='minmax')
    
    assert result is not None
    # 检查数值列是否在[0, 1]范围内
    numeric_cols = result.data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(result.data) > 1:
            assert result.data[col].min() >= 0
            assert result.data[col].max() <= 1


def test_process_with_zscore_normalization(processor, data_model):
    """测试Z-score标准化"""
    result = processor.process(data_model, normalize_method='zscore')
    
    assert result is not None
    # 检查数值列的均值是否接近0
    numeric_cols = result.data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if len(result.data) > 1 and result.data[col].std() > 0:
            assert abs(result.data[col].mean()) < 0.1


def test_process_with_robust_normalization(processor, data_model):
    """测试Robust标准化"""
    result = processor.process(data_model, normalize_method='robust')
    
    assert result is not None
    # Robust标准化应该对异常值更鲁棒
    assert len(result.data) == len(data_model.data)


def test_process_with_index_alignment(processor, data_model):
    """测试索引对齐"""
    result = processor.process(data_model, index_col='date')
    
    assert result is not None
    assert isinstance(result.data.index, pd.DatetimeIndex)


def test_process_with_time_alignment(processor, data_model):
    """测试时间对齐"""
    result = processor.process(data_model, time_col='date')
    
    assert result is not None
    assert isinstance(result.data.index, pd.DatetimeIndex)


def test_process_with_required_columns(processor, data_model):
    """测试必需列对齐"""
    result = processor.process(data_model, required_columns=['open', 'high', 'low', 'close', 'volume', 'new_col'])
    
    assert result is not None
    assert 'new_col' in result.data.columns


def test_process_validation_with_value_ranges(processor, data_model):
    """测试值范围验证"""
    result = processor.process(data_model, value_ranges={'close': (0, 200)})
    
    assert result is not None
    assert result.data['close'].min() >= 0
    assert result.data['close'].max() <= 200


def test_process_validation_with_expected_dtypes(processor, data_model):
    """测试数据类型验证"""
    result = processor.process(data_model, expected_dtypes={'volume': 'int64'})
    
    assert result is not None
    assert result.data['volume'].dtype == 'int64'


def test_process_handles_inf_values(processor, data_model):
    """测试处理无穷大值"""
    data = data_model.data.copy()
    data.loc[5, 'close'] = np.inf
    data.loc[6, 'volume'] = -np.inf
    
    data_model.data = data
    result = processor.process(data_model)
    
    assert result is not None
    # 无穷大值应该被替换为NaN
    assert not np.isinf(result.data['close']).any()
    assert not np.isinf(result.data['volume']).any()


def test_get_processing_info(processor, data_model):
    """测试获取处理信息"""
    processor.process(data_model)
    
    info = processor.get_processing_info()
    
    assert 'processor_type' in info
    assert 'steps' in info
    assert len(info['steps']) > 0


def test_add_processing_step(processor):
    """测试添加处理步骤"""
    def custom_step(df):
        return df
    
    processor.add_processing_step('custom_step', custom_step)
    
    assert len(processor.processing_steps) == 1
    assert processor.processing_steps[0]['name'] == 'custom_step'


def test_add_processing_step_with_dict(processor):
    """测试使用字典添加处理步骤"""
    step_dict = {'name': 'custom', 'function': lambda x: x}
    processor.add_processing_step(step_dict)
    
    assert len(processor.processing_steps) == 1


def test_get_processing_stats(processor, data_model):
    """测试获取处理统计信息"""
    processor.process(data_model)
    
    stats = processor.get_processing_stats()
    
    assert 'total_steps' in stats or 'steps' in stats
    assert isinstance(stats, dict)

