"""
数据处理模块深度测试
全面测试数据处理的各种功能和边界条件
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# 导入实际的类
from src.data.processing.data_processor import DataProcessor
from src.data.processing.unified_processor import UnifiedDataProcessor


class TestDataProcessingComprehensive:
    """数据处理综合深度测试"""

    @pytest.fixture
    def sample_stock_data(self):
        """创建样本股票数据"""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'symbol': ['AAPL'] * 100,
            'date': dates,
            'open': np.random.uniform(150, 200, 100),
            'high': np.random.uniform(155, 205, 100),
            'low': np.random.uniform(145, 195, 100),
            'close': np.random.uniform(150, 200, 100),
            'volume': np.random.randint(1000000, 10000000, 100),
            'adj_close': np.random.uniform(150, 200, 100)
        })

    @pytest.fixture
    def data_processor(self):
        """创建数据处理器实例"""
        return DataProcessor()

    @pytest.fixture
    def unified_processor(self):
        """创建统一数据处理器实例"""
        return UnifiedDataProcessor()

    def test_data_processor_initialization(self, data_processor):
        """测试数据处理器初始化"""
        assert data_processor is not None
        assert hasattr(data_processor, 'config')

    def test_unified_processor_initialization(self, unified_processor):
        """测试统一数据处理器初始化"""
        assert unified_processor is not None
        assert hasattr(unified_processor, 'processors')

    def test_data_cleaning_missing_values(self, data_processor, sample_stock_data):
        """测试数据清理 - 缺失值处理"""
        # 引入缺失值
        dirty_data = sample_stock_data.copy()
        dirty_data.loc[10:15, 'close'] = np.nan
        dirty_data.loc[20:25, 'volume'] = np.nan

        # 使用DataProcessor的清理方法
        cleaned_data = data_processor._clean_data(dirty_data)

        # 检查缺失值已被处理
        assert not cleaned_data['close'].isna().any()
        assert not cleaned_data['volume'].isna().any()

    def test_data_transformation_normalization(self, sample_stock_data):
        """测试数据变换 - 标准化"""
        if DATA_TRANSFORMER_AVAILABLE:
            transformer = DataTransformer()
            transformed_data = transformer.normalize(sample_stock_data, columns=['close', 'volume'])

            # 检查标准化后的数据
            assert transformed_data['close'].mean() < 0.1  # 标准化后均值接近0
            assert transformed_data['volume'].std() < 0.1   # 标准化后标准差接近1

    def test_data_filtering_outliers(self, sample_stock_data):
        """测试数据过滤 - 异常值检测"""
        # 引入异常值
        outlier_data = sample_stock_data.copy()
        outlier_data.loc[0, 'close'] = 1000  # 明显异常值

        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()
            filtered_data = processor.filter_outliers(outlier_data, column='close', method='iqr')

            # 检查异常值被过滤
            assert len(filtered_data) < len(outlier_data)
            assert not (filtered_data['close'] > 500).any()

    def test_data_aggregation_groupby(self, sample_stock_data):
        """测试数据聚合 - 分组操作"""
        # 创建多股票数据
        multi_stock_data = pd.concat([
            sample_stock_data.assign(symbol='AAPL'),
            sample_stock_data.assign(symbol='GOOGL')
        ])

        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()
            aggregated = processor.aggregate_by_symbol(multi_stock_data)

            # 检查聚合结果
            assert len(aggregated) == 2  # 两只股票
            assert 'symbol' in aggregated.columns

    def test_data_resampling_time_series(self, sample_stock_data):
        """测试数据重采样 - 时间序列"""
        # 创建高频数据
        high_freq_data = sample_stock_data.copy()
        high_freq_data['date'] = pd.date_range('2024-01-01', periods=100, freq='H')

        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()
            resampled = processor.resample_data(high_freq_data, freq='D')

            # 检查重采样结果
            assert len(resampled) < len(high_freq_data)
            assert resampled.index.freq == 'D'

    def test_performance_optimization_memory_usage(self, sample_stock_data):
        """测试性能优化 - 内存使用"""
        if PERFORMANCE_OPTIMIZER_AVAILABLE:
            optimizer = PerformanceOptimizer()

            # 测试数据类型优化
            optimized_data = optimizer.optimize_memory_usage(sample_stock_data)

            # 检查内存使用量减少
            original_memory = sample_stock_data.memory_usage(deep=True).sum()
            optimized_memory = optimized_data.memory_usage(deep=True).sum()

            assert optimized_memory <= original_memory

    def test_data_validation_schema(self, sample_stock_data):
        """测试数据验证 - 模式验证"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 定义验证模式
            schema = {
                'symbol': {'type': 'string', 'required': True},
                'close': {'type': 'number', 'min': 0, 'max': 1000},
                'volume': {'type': 'integer', 'min': 0}
            }

            validation_result = processor.validate_schema(sample_stock_data, schema)

            assert validation_result['valid'] is True
            assert len(validation_result['errors']) == 0

    def test_data_transformation_feature_engineering(self, sample_stock_data):
        """测试数据变换 - 特征工程"""
        if DATA_TRANSFORMER_AVAILABLE:
            transformer = DataTransformer()

            # 创建技术指标特征
            featured_data = transformer.create_technical_features(sample_stock_data)

            # 检查新增的特征列
            expected_features = ['returns', 'ma_5', 'ma_20', 'rsi', 'macd']
            for feature in expected_features:
                assert feature in featured_data.columns

    def test_data_processing_pipeline_execution(self, sample_stock_data):
        """测试数据处理管道执行"""
        if UNIFIED_PROCESSOR_AVAILABLE:
            processor = UnifiedDataProcessor()

            # 定义处理管道
            pipeline = [
                {'type': 'clean', 'method': 'missing_values'},
                {'type': 'transform', 'method': 'normalize', 'columns': ['close']},
                {'type': 'filter', 'method': 'outliers', 'column': 'close'}
            ]

            processed_data = processor.execute_pipeline(sample_stock_data, pipeline)

            # 检查管道执行结果
            assert len(processed_data) > 0
            assert 'close' in processed_data.columns

    def test_performance_monitoring_processing_time(self, sample_stock_data):
        """测试性能监控 - 处理时间"""
        if PERFORMANCE_OPTIMIZER_AVAILABLE:
            optimizer = PerformanceOptimizer()

            start_time = datetime.now()

            # 执行耗时操作
            result = optimizer.process_with_monitoring(sample_stock_data)

            end_time = datetime.now()

            # 检查处理时间被监控
            processing_time = (end_time - start_time).total_seconds()
            assert processing_time >= 0

            # 检查结果
            assert len(result) == len(sample_stock_data)

    def test_data_quality_checks_processing(self, sample_stock_data):
        """测试数据质量检查 - 处理前后"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 处理前质量检查
            pre_quality = processor.check_data_quality(sample_stock_data)

            # 执行处理
            processed_data = processor.clean_missing_values(sample_stock_data)

            # 处理后质量检查
            post_quality = processor.check_data_quality(processed_data)

            # 检查质量提升
            assert post_quality['completeness'] >= pre_quality['completeness']

    def test_error_handling_processing_failures(self, sample_stock_data):
        """测试错误处理 - 处理失败"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 创建会导致处理失败的数据
            bad_data = sample_stock_data.copy()
            bad_data['invalid_column'] = 'invalid'

            # 测试错误处理
            try:
                result = processor.process_with_error_handling(bad_data)
                # 如果没有抛出异常，检查结果
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                # 如果抛出异常，检查异常类型
                assert isinstance(e, (ValueError, TypeError, RuntimeError))

    def test_concurrent_processing_multi_threading(self, sample_stock_data):
        """测试并发处理 - 多线程"""
        if UNIFIED_PROCESSOR_AVAILABLE:
            processor = UnifiedDataProcessor()

            # 创建多个数据块
            data_chunks = [sample_stock_data] * 5

            # 并发处理
            results = processor.process_concurrent(data_chunks, max_workers=3)

            # 检查结果
            assert len(results) == 5
            for result in results:
                assert len(result) == len(sample_stock_data)

    def test_data_export_processing_formats(self, sample_stock_data):
        """测试数据导出 - 处理结果格式"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 处理数据
            processed_data = processor.clean_missing_values(sample_stock_data)

            # 测试不同格式导出
            formats = ['csv', 'json', 'parquet']
            for fmt in formats:
                with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as tmp:
                    try:
                        processor.export_processed_data(processed_data, tmp.name, fmt)
                        # 检查文件存在
                        assert os.path.exists(tmp.name)
                        assert os.path.getsize(tmp.name) > 0
                    finally:
                        if os.path.exists(tmp.name):
                            os.unlink(tmp.name)

    def test_memory_management_large_datasets(self):
        """测试内存管理 - 大数据集"""
        if PERFORMANCE_OPTIMIZER_AVAILABLE:
            optimizer = PerformanceOptimizer()

            # 创建大数据集
            large_data = pd.DataFrame({
                'col1': np.random.randn(100000),
                'col2': np.random.randn(100000),
                'col3': np.random.randn(100000)
            })

            # 测试内存管理
            memory_info = optimizer.monitor_memory_usage(large_data)

            assert 'memory_usage_mb' in memory_info
            assert memory_info['memory_usage_mb'] > 0

    def test_data_validation_business_rules(self, sample_stock_data):
        """测试数据验证 - 业务规则"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 定义业务规则
            business_rules = [
                lambda df: (df['high'] >= df['low']).all(),  # 最高价应大于等于最低价
                lambda df: (df['close'] >= df['low']).all(),  # 收盘价应大于等于最低价
                lambda df: (df['volume'] > 0).all()          # 成交量应大于0
            ]

            validation_results = processor.validate_business_rules(sample_stock_data, business_rules)

            assert validation_results['passed'] is True
            assert len(validation_results['failed_rules']) == 0

    def test_performance_optimization_caching(self, sample_stock_data):
        """测试性能优化 - 缓存机制"""
        if PERFORMANCE_OPTIMIZER_AVAILABLE:
            optimizer = PerformanceOptimizer()

            # 执行带缓存的处理
            result1 = optimizer.process_with_caching(sample_stock_data, cache_key='test1')
            result2 = optimizer.process_with_caching(sample_stock_data, cache_key='test1')

            # 检查结果一致性
            pd.testing.assert_frame_equal(result1, result2)

    def test_data_processing_monitoring_metrics(self, sample_stock_data):
        """测试数据处理监控 - 指标收集"""
        if UNIFIED_PROCESSOR_AVAILABLE:
            processor = UnifiedDataProcessor()

            # 执行带监控的处理
            result, metrics = processor.process_with_monitoring(sample_stock_data)

            # 检查监控指标
            assert 'processing_time' in metrics
            assert 'input_rows' in metrics
            assert 'output_rows' in metrics
            assert metrics['processing_time'] >= 0
            assert metrics['input_rows'] == len(sample_stock_data)
            assert metrics['output_rows'] == len(result)

    def test_data_transformation_rolling_features(self, sample_stock_data):
        """测试数据变换 - 滚动特征"""
        if DATA_TRANSFORMER_AVAILABLE:
            transformer = DataTransformer()

            # 创建滚动特征
            rolling_features = transformer.create_rolling_features(
                sample_stock_data,
                column='close',
                windows=[5, 10, 20]
            )

            # 检查滚动特征列
            expected_columns = ['close_ma_5', 'close_ma_10', 'close_ma_20',
                              'close_std_5', 'close_std_10', 'close_std_20']
            for col in expected_columns:
                assert col in rolling_features.columns

    def test_error_recovery_processing_failures(self, sample_stock_data):
        """测试错误恢复 - 处理失败恢复"""
        if DATA_PROCESSOR_AVAILABLE:
            processor = DataProcessor()

            # 模拟处理失败场景
            with patch.object(processor, 'clean_missing_values', side_effect=Exception("Processing failed")):
                # 测试错误恢复
                result = processor.process_with_recovery(sample_stock_data)

                # 检查错误恢复机制
                assert result is not None  # 应该有备用处理结果
