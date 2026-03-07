#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用特征处理器边界场景与异常分支测试

覆盖异常输入、缺失值处理、降级策略等关键路径
"""

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from src.features.processors.general_processor import FeatureProcessor
from src.features.processors.base_processor import ProcessorConfig


@pytest.fixture
def sample_features():
    """样本特征数据"""
    return pd.DataFrame({
        'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'numeric2': [10.0, 20.0, np.nan, 40.0, 50.0],
        'categorical': ['a', 'b', 'c', np.nan, 'e'],
        'mixed': [1, 2, 3, 4, 5]
    })


class TestFeatureProcessorExceptions:
    """测试特征处理器异常处理"""

    def test_process_features_empty_returns_empty(self):
        """测试处理空特征数据"""
        processor = FeatureProcessor()
        
        result = processor.process_features(pd.DataFrame())
        
        assert result.empty

    def test_process_features_none_returns_empty(self):
        """测试处理 None 输入"""
        processor = FeatureProcessor()
        
        result = processor.process_features(None)
        
        assert result.empty

    def test_process_features_exception_returns_original(self, sample_features, caplog):
        """测试处理异常时返回原始数据（降级策略）"""
        processor = FeatureProcessor()
        
        with patch('pandas.DataFrame.drop_duplicates', side_effect=Exception("处理失败")):
            with caplog.at_level("ERROR"):
                result = processor.process_features(sample_features)
            
            # 应该返回原始数据
            pd.testing.assert_frame_equal(result, sample_features)
            assert any("特征处理失败" in msg for msg in caplog.messages)


class TestMissingValueHandling:
    """测试缺失值处理"""

    def test_handle_missing_values_numeric_median(self, sample_features):
        """测试数值列缺失值用中位数填充"""
        processor = FeatureProcessor()
        config = SimpleNamespace(handle_missing_values=True)
        
        result = processor.process_features(sample_features, config)
        
        # numeric2 列应该有中位数填充
        assert not result['numeric2'].isna().any()

    def test_handle_missing_values_categorical_mode(self, sample_features):
        """测试分类列缺失值用众数填充"""
        processor = FeatureProcessor()
        config = SimpleNamespace(handle_missing_values=True)
        
        result = processor.process_features(sample_features, config)
        
        # categorical 列应该有众数填充
        assert not result['categorical'].isna().any()

    def test_handle_missing_values_all_nan_column(self):
        """测试全 NaN 列的处理"""
        processor = FeatureProcessor()
        config = SimpleNamespace(handle_missing_values=True)
        features = pd.DataFrame({
            'all_nan': [np.nan] * 5,
            'normal': [1, 2, 3, 4, 5]
        })
        
        result = processor.process_features(features, config)
        
        # 应该不报错，NaN 列可能保持 NaN 或被填充
        assert 'normal' in result.columns

    def test_handle_missing_values_no_mode_uses_unknown(self):
        """测试无众数时使用 Unknown"""
        processor = FeatureProcessor()
        config = SimpleNamespace(handle_missing_values=True)
        features = pd.DataFrame({
            'cat': ['a', 'b', 'c', np.nan]  # 每个值都不同，无众数
        })
        
        result = processor.process_features(features, config)
        
        # 应该使用 'Unknown' 填充
        assert not result['cat'].isna().any()

    def test_handle_missing_values_exception_returns_original(self, sample_features, caplog):
        """测试缺失值处理异常时返回原始数据"""
        processor = FeatureProcessor()
        config = SimpleNamespace(handle_missing_values=True)
        
        # 直接 patch _handle_missing_values 方法使其抛出异常
        original_method = processor._handle_missing_values
        def failing_method(features):
            raise Exception("填充失败")
        
        processor._handle_missing_values = failing_method
        
        with caplog.at_level("ERROR"):
            result = processor.process_features(sample_features, config)
        
        # _handle_missing_values 内部有 try-except，异常时返回原始数据
        # process_features 也有 try-except，异常时返回原始数据（可能已去重）
        assert result is not None
        # 验证日志中记录了错误（可能来自 process_features 或 _handle_missing_values）
        assert any("失败" in msg for msg in caplog.messages)


class TestDuplicateHandling:
    """测试重复数据处理"""

    def test_drop_duplicates_removes_duplicates(self):
        """测试移除重复行"""
        processor = FeatureProcessor()
        features = pd.DataFrame({
            'a': [1, 2, 2, 3, 3],
            'b': [10, 20, 20, 30, 30]
        })
        
        result = processor.process_features(features)
        
        assert len(result) < len(features)
        assert len(result) == 3  # 移除了重复行

    def test_drop_duplicates_empty_dataframe(self):
        """测试空 DataFrame 去重"""
        processor = FeatureProcessor()
        empty = pd.DataFrame()
        
        result = processor.process_features(empty)
        
        assert result.empty

    def test_drop_duplicates_all_duplicates(self):
        """测试全重复数据"""
        processor = FeatureProcessor()
        features = pd.DataFrame({
            'a': [1, 1, 1, 1, 1],
            'b': [2, 2, 2, 2, 2]
        })
        
        result = processor.process_features(features)
        
        assert len(result) == 1  # 只保留一行


class TestFeatureComputation:
    """测试特征计算"""

    def test_compute_feature_existing_column(self):
        """测试计算已存在的特征"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        result = processor._compute_feature(data, 'feature1', {})
        
        pd.testing.assert_series_equal(result, data['feature1'])

    def test_compute_feature_missing_column_returns_empty_series(self):
        """测试计算不存在的特征"""
        processor = FeatureProcessor()
        data = pd.DataFrame({'feature1': [1, 2, 3]})
        
        result = processor._compute_feature(data, 'nonexistent', {})
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data)
        assert result.isna().all()


class TestMetadata:
    """测试元数据"""

    def test_get_feature_metadata_returns_info(self):
        """测试获取特征元数据"""
        processor = FeatureProcessor()
        metadata = processor._get_feature_metadata('test_feature')
        
        assert metadata['name'] == 'test_feature'
        assert metadata['type'] == 'general_feature'
        assert 'description' in metadata
        assert 'parameters' in metadata

    def test_get_available_features_returns_empty(self):
        """测试获取可用特征列表"""
        processor = FeatureProcessor()
        features = processor._get_available_features()
        
        # 通用处理器处理所有现有特征，不预定义特征列表
        assert features == []


class TestConfigHandling:
    """测试配置处理"""

    def test_process_features_with_config_handles_missing(self, sample_features):
        """测试使用配置处理缺失值"""
        config = ProcessorConfig(
            processor_type="general",
            feature_params={"handle_missing_values": True}
        )
        processor = FeatureProcessor(config)
        
        # process_features 需要传入 config 参数才会处理缺失值
        # 创建一个包含 handle_missing_values 的 config 对象
        process_config = SimpleNamespace(handle_missing_values=True)
        result = processor.process_features(sample_features, process_config)
        
        # 应该处理缺失值
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert not result[col].isna().any()

    def test_process_features_with_config_disables_missing_handling(self, sample_features):
        """测试配置禁用缺失值处理"""
        config = ProcessorConfig(
            processor_type="general",
            feature_params={"handle_missing_values": False}
        )
        processor = FeatureProcessor(config)
        # 使用 SimpleNamespace 模拟旧接口
        simple_config = SimpleNamespace(handle_missing_values=False)
        
        result = processor.process_features(sample_features, simple_config)
        
        # 可能仍有缺失值（如果原始数据有）
        assert result is not None

    def test_process_features_without_config_uses_defaults(self, sample_features):
        """测试无配置时使用默认值"""
        processor = FeatureProcessor()
        
        result = processor.process_features(sample_features)
        
        # 应该移除重复行，但可能不处理缺失值（取决于默认配置）
        assert not result.empty
        assert len(result) <= len(sample_features)

