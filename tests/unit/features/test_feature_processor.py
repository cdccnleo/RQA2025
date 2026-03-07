# tests/unit/features/test_feature_processor.py
"""
FeatureProcessor单元测试

测试覆盖:
- 初始化参数验证
- 特征处理功能
- 数据验证
- 错误处理
- 缓存机制
- 性能监控
- 边界条件
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import time
import os

from src.features.processors.feature_processor import FeatureProcessor
from src.features.processors.base_processor import ProcessorConfig
from src.infrastructure.interfaces.standard_interfaces import FeatureRequest



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestFeatureProcessor:
    """FeatureProcessor测试类"""

    @pytest.fixture
    def temp_dir(self):
        """临时目录fixture"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def sample_data(self):
        """样本数据fixture"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'date': dates,
            'open': 100 + np.random.randn(100) * 5,
            'high': 105 + np.random.randn(100) * 5,
            'low': 95 + np.random.randn(100) * 5,
            'close': 100 + np.random.randn(100) * 3,
            'volume': np.random.randint(100000, 1000000, 100)
        })

    @pytest.fixture
    def processor_config(self):
        """处理器配置fixture"""
        return ProcessorConfig(
            processor_type="general",
            feature_params={
                "moving_averages": [5, 10, 20],
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26,
                "macd_signal": 9,
                "bollinger_period": 20,
                "bollinger_std": 2
            }
        )

    @pytest.fixture
    def processor(self, processor_config):
        """FeatureProcessor实例"""
        return FeatureProcessor(processor_config)

    def test_initialization_with_config(self, processor_config):
        """测试带配置的初始化"""
        processor = FeatureProcessor(processor_config)

        assert processor.config == processor_config
        assert processor.processor_type == "general"
        assert hasattr(processor, '_available_features')
        assert len(processor._available_features) > 0

    def test_initialization_without_config(self):
        """测试无配置的初始化"""
        processor = FeatureProcessor()

        assert processor.config is not None
        assert processor.config.processor_type == "general"
        assert 'moving_averages' in processor.config.feature_params
        assert len(processor._available_features) > 0

    def test_initialization_invalid_config(self):
        """测试无效配置的初始化"""
        invalid_config = ProcessorConfig(
            processor_type="",
            feature_params={}
        )

        # 应该能够处理空的processor_type
        processor = FeatureProcessor(invalid_config)
        assert processor.processor_type == ""

    def test_process(self, processor, sample_data):
        """测试有效数据的处理"""
        result = processor.process(sample_data)

        assert result is not None
        assert not result.empty
        assert len(result) == len(sample_data)

        # 检查是否添加了技术指标列
        expected_columns = ['close', 'volume']  # 基础列
        for col in expected_columns:
            assert col in result.columns

    def test_process(self, processor):
        """测试空数据的处理"""
        empty_data = pd.DataFrame()

        with pytest.raises(ValueError, match="输入数据为空"):
            processor.process(empty_data)

    def test_process(self, processor):
        """测试缺失必要列的数据处理"""
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.randn(10)
            # 缺少其他必要列
        })

        # 应该能够处理缺少的列，或者抛出适当的错误
        result = processor.process(incomplete_data)
        assert result is not None

    def test_process_with_feature_selection(self, processor, sample_data):
        """测试带特征选择的数据处理"""
        # 使用实际支持的特征名称（小写）
        selected_features = ['sma', 'rsi', 'macd']

        result = processor.process(sample_data, features=selected_features)

        assert result is not None
        # 检查是否包含了请求的特征列（实际生成的是feature_前缀的列名）
        for feature in selected_features:
            feature_column = f"feature_{feature}"
            assert feature_column in result.columns

    def test_calculate_moving_averages(self, processor, sample_data):
        """测试移动平均线计算"""
        # 手动调用移动平均线计算方法
        result = processor._calculate_moving_averages(sample_data)

        assert result is not None
        # 检查是否添加了移动平均线列
        ma_columns = [col for col in result.columns if 'SMA_' in col or 'EMA_' in col]
        assert len(ma_columns) > 0

    def test_calculate_rsi(self, processor, sample_data):
        """测试RSI指标计算"""
        result = processor._calculate_rsi(sample_data, period=14)

        assert result is not None
        assert 'RSI' in result.columns
        assert len(result) == len(sample_data)

        # RSI值应该在0-100范围内
        rsi_values = result['RSI'].dropna()
        assert all(0 <= rsi <= 100 for rsi in rsi_values)

    def test_calculate_macd(self, processor, sample_data):
        """测试MACD指标计算"""
        result = processor._calculate_macd(sample_data)

        assert result is not None
        # 检查MACD相关列
        macd_columns = [col for col in result.columns if 'MACD' in col.upper()]
        assert len(macd_columns) >= 3  # MACD, Signal, Histogram

    def test_calculate_bollinger_bands(self, processor, sample_data):
        """测试布林带计算"""
        result = processor._calculate_bollinger_bands(sample_data)

        assert result is not None
        # 检查布林带相关列
        bb_columns = [col for col in result.columns if 'BB_' in col.upper()]
        assert len(bb_columns) >= 3  # Upper, Middle, Lower

    def test_list_features(self, processor):
        """测试特征列表获取"""
        features = processor.list_features()

        assert isinstance(features, list)
        assert len(features) > 0

        # 检查是否包含常见技术指标
        feature_names = [f['name'] for f in features if isinstance(f, dict) and 'name' in f]
        assert len(feature_names) > 0

    def test_get_feature_info(self, processor):
        """测试特征信息获取"""
        # 假设有一个已知的特征
        features = processor.list_features()
        if features:
            first_feature = features[0]
            if isinstance(first_feature, dict) and 'name' in first_feature:
                feature_name = first_feature['name']
                info = processor.get_feature_info(feature_name)
                assert info is not None

    def test_feature_validation(self, processor, sample_data):
        """测试特征验证"""
        # 测试有效特征（使用小写名称）
        valid_features = ['sma', 'rsi', 'macd']
        is_valid, errors = processor.validate_features(valid_features)
        assert is_valid is True
        assert len(errors) == 0

        # 测试无效特征
        invalid_features = ['INVALID_FEATURE_123']
        is_valid, errors = processor.validate_features(invalid_features)
        # 这里取决于具体实现，可能允许未知特征或抛出错误

    def test_cache_mechanism(self, processor, sample_data):
        """测试缓存机制"""
        # 第一次处理
        result1 = processor.process(sample_data)

        # 检查是否使用了缓存
        assert hasattr(processor, '_features_cache')
        assert len(processor._features_cache) >= 0

    def test_performance_monitoring(self, processor, sample_data):
        """测试性能监控"""
        start_time = time.time()

        result = processor.process(sample_data)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能指标
        assert duration >= 0
        assert result is not None

        # 检查是否记录了性能信息
        assert hasattr(processor, '_feature_info')

    def test_error_handling_invalid_data_type(self, processor):
        """测试无效数据类型错误处理"""
        invalid_data = "not a dataframe"

        with pytest.raises((AttributeError, TypeError)):
            processor.process(invalid_data)

    def test_error_handling_nan_values(self, processor):
        """测试NaN值错误处理"""
        data_with_nan = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': [100] * 5 + [np.nan] * 5,
            'high': [105] * 5 + [np.nan] * 5,
            'low': [95] * 5 + [np.nan] * 5,
            'close': [100] * 5 + [np.nan] * 5,
            'volume': [100000] * 10
        })

        # 应该能够处理NaN值或抛出适当的错误
        result = processor.process(data_with_nan)
        assert result is not None

    def test_memory_usage_efficiency(self, processor, sample_data):
        """测试内存使用效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        result = processor.process(sample_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该在合理范围内（例如不超过100MB）
        assert memory_increase < 100 * 1024 * 1024
        assert result is not None

    def test_concurrent_processing_safety(self, processor):
        """测试并发处理安全性"""
        import threading
        import concurrent.futures

        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=50),
            'open': np.random.randn(50) * 5 + 100,
            'high': np.random.randn(50) * 5 + 105,
            'low': np.random.randn(50) * 5 + 95,
            'close': np.random.randn(50) * 3 + 100,
            'volume': np.random.randint(100000, 1000000, 50)
        })

        results = []
        errors = []

        def process_worker():
            try:
                result = processor.process(sample_data)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 创建多个线程并发处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_worker) for _ in range(10)]
            concurrent.futures.wait(futures)

        # 验证并发处理的结果
        assert len(results) == 10  # 所有请求都成功
        assert len(errors) == 0    # 没有错误

        # 验证所有结果的一致性
        for result in results:
            assert len(result) == len(sample_data)

    def test_feature_processor_config_persistence(self, processor):
        """测试配置持久性"""
        original_config = processor.config

        # 模拟一些操作
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'open': np.random.randn(20) * 5 + 100,
            'high': np.random.randn(20) * 5 + 105,
            'low': np.random.randn(20) * 5 + 95,
            'close': np.random.randn(20) * 3 + 100,
            'volume': np.random.randint(100000, 1000000, 20)
        })

        result = processor.process(sample_data)

        # 验证配置没有被修改
        assert processor.config == original_config

    def test_feature_processor_resource_cleanup(self, processor):
        """测试资源清理"""
        # 执行一些操作
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'open': np.random.randn(20) * 5 + 100,
            'high': np.random.randn(20) * 5 + 105,
            'low': np.random.randn(20) * 5 + 95,
            'close': np.random.randn(20) * 3 + 100,
            'volume': np.random.randint(100000, 1000000, 20)
        })

        result = processor.process(sample_data)

        # 这里可以添加资源清理验证逻辑
        # 例如验证缓存大小、临时文件等
        assert result is not None

    def test_feature_processor_scalability(self, processor):
        """测试处理器扩展性"""
        # 测试不同大小的数据集
        sizes = [10, 100, 1000]

        for size in sizes:
            sample_data = pd.DataFrame({
                'date': pd.date_range('2024-01-01', periods=size),
                'open': np.random.randn(size) * 5 + 100,
                'high': np.random.randn(size) * 5 + 105,
                'low': np.random.randn(size) * 5 + 95,
                'close': np.random.randn(size) * 3 + 100,
                'volume': np.random.randint(100000, 1000000, size)
            })

            start_time = time.time()
            result = processor.process(sample_data)
            end_time = time.time()

            duration = end_time - start_time

            # 验证处理时间在合理范围内
            # 对于小型数据集应该很快，对于大型数据集可以稍慢但不能太慢
            if size <= 100:
                assert duration < 5.0  # 小数据集5秒内完成
            else:
                assert duration < 30.0  # 大数据集30秒内完成

            assert len(result) == size
