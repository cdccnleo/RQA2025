#!/usr/bin/env python3
"""
特征工程层全面边界条件测试
提升features模块覆盖率至85%的关键测试
"""

import pytest
import sys
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import tempfile
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

# 导入异常类
try:
    from src.features.core.exceptions import FeatureEngineeringError, ValidationError
except ImportError:
    # 如果导入失败，定义基本的异常类
    class FeatureEngineeringError(Exception):
        pass

    class ValidationError(FeatureEngineeringError):
        pass


class TestFeaturesComprehensiveBoundary:
    """特征工程层全面边界条件测试"""

    @pytest.fixture
    def feature_engineer(self):
        """特征工程器fixture"""
        try:
            from src.features.core.feature_engineer import FeatureEngineer
            return FeatureEngineer()
        except ImportError:
            pytest.skip("FeatureEngineer不可用")

    @pytest.fixture
    def temp_cache_dir(self):
        """临时缓存目录fixture"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_market_data(self):
        """样本市场数据fixture"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1min')

        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.random.normal(0, 1, 100).cumsum(),
            'high': 101 + np.random.normal(0, 1, 100).cumsum(),
            'low': 99 + np.random.normal(0, 1, 100).cumsum(),
            'close': 100 + np.random.normal(0, 1, 100).cumsum(),
            'volume': np.random.poisson(1000, 100)
        })

        # 确保high >= close >= low
        for i in range(len(data)):
            data.loc[i, 'high'] = max(data.loc[i, ['open', 'high', 'low', 'close']])
            data.loc[i, 'low'] = min(data.loc[i, ['open', 'high', 'low', 'close']])

        return data

    def test_feature_engineer_initialization(self, feature_engineer):
        """测试特征工程器初始化"""
        assert feature_engineer is not None

        # 检查基本属性
        assert hasattr(feature_engineer, 'config_manager')
        assert hasattr(feature_engineer, 'cache_dir')
        assert hasattr(feature_engineer, 'max_workers')
        assert hasattr(feature_engineer, 'executor')

        # 检查配置
        assert feature_engineer.max_workers > 0
        assert feature_engineer.batch_size > 0
        assert feature_engineer.timeout > 0

    def test_feature_engineer_initialization_with_invalid_config(self):
        """测试特征工程器初始化时无效配置"""
        try:
            from src.features.core.feature_engineer import FeatureEngineer

            # 测试无效缓存目录
            with pytest.raises((ValueError, OSError)):
                FeatureEngineer(cache_dir="")

            # 测试无效工作线程数
            with pytest.raises((ValueError, TypeError)):
                FeatureEngineer(max_workers=0)

            # 测试无效重试次数
            with pytest.raises((ValueError, TypeError)):
                FeatureEngineer(max_retries=-1)

        except ImportError:
            pytest.skip("FeatureEngineer不可用")

    def test_process_data_with_none_input(self, feature_engineer):
        """测试处理None数据"""
        with pytest.raises((ValueError, TypeError, AttributeError)):
            feature_engineer.process_data(None)

    def test_process_data_with_empty_dataframe(self, feature_engineer):
        """测试处理空DataFrame"""
        empty_df = pd.DataFrame()

        # 可能抛出异常或返回空结果
        try:
            result = feature_engineer.process_data(empty_df)
            assert result is not None  # 如果不抛出异常，应该返回结果
        except (ValueError, ValidationError):
            # 也接受抛出异常
            pass

    def test_process_data_with_single_row(self, feature_engineer, sample_market_data):
        """测试处理单行数据"""
        single_row = sample_market_data.iloc[:1]

        result = feature_engineer.process_data(single_row)
        assert result is not None
        assert isinstance(result, (pd.DataFrame, dict))

    def test_process_data_with_missing_columns(self, feature_engineer):
        """测试处理缺少必要列的数据"""
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100.0] * 10,
            # 缺少close, high, low, volume等列
        })

        # 应该能够处理或抛出适当的异常
        try:
            result = feature_engineer.process_data(incomplete_data)
            assert result is not None
        except (ValueError, KeyError, ValidationError):
            # 也接受抛出异常
            pass

    def test_process_data_with_nan_values(self, feature_engineer, sample_market_data):
        """测试处理包含NaN值的数据"""
        data_with_nan = sample_market_data.copy()

        # 添加NaN值
        data_with_nan.loc[0, 'close'] = np.nan
        data_with_nan.loc[1, 'volume'] = np.nan
        data_with_nan.loc[2:4, 'high'] = np.nan

        # 应该能够处理NaN值
        result = feature_engineer.process_data(data_with_nan)
        assert result is not None

    def test_process_data_with_extreme_values(self, feature_engineer, sample_market_data):
        """测试处理极值数据"""
        extreme_data = sample_market_data.copy()

        # 添加极值
        extreme_data.loc[0, 'close'] = 1e10  # 极大价格
        extreme_data.loc[1, 'close'] = 1e-10  # 极小价格
        extreme_data.loc[2, 'volume'] = 1e15  # 极大成交量
        extreme_data.loc[3, 'volume'] = 0     # 零成交量

        result = feature_engineer.process_data(extreme_data)
        assert result is not None

    def test_process_data_with_invalid_data_types(self, feature_engineer):
        """测试处理无效数据类型"""
        # 测试非DataFrame输入
        invalid_inputs = [
            "string_input",
            123,
            [1, 2, 3],
            {"key": "value"},
            None
        ]

        for invalid_input in invalid_inputs:
            with pytest.raises((TypeError, ValueError, AttributeError)):
                feature_engineer.process_data(invalid_input)

    def test_generate_technical_indicators_with_insufficient_data(self, feature_engineer):
        """测试生成技术指标时数据不足"""
        # 创建少量数据（不足以计算某些指标）
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })

        try:
            result = feature_engineer.generate_technical_indicators(small_data)
            # 对于少量数据，可能返回部分结果或空结果
            assert result is not None
        except (ValueError, ValidationError):
            # 也接受抛出异常
            pass

    def test_generate_technical_indicators_with_all_nan(self, feature_engineer, sample_market_data):
        """测试生成技术指标时数据全为NaN"""
        nan_data = sample_market_data.copy()

        # 将所有数值列设为NaN
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            nan_data[col] = np.nan

        try:
            result = feature_engineer.generate_technical_indicators(nan_data)
            assert result is not None
        except (ValueError, ValidationError, ZeroDivisionError):
            # 也接受抛出异常
            pass

    def test_cache_functionality(self, feature_engineer, sample_market_data, temp_cache_dir):
        """测试缓存功能"""
        # 设置自定义缓存目录
        feature_engineer.cache_dir = Path(temp_cache_dir)

        # 第一次处理
        result1 = feature_engineer.process_data(sample_market_data)

        # 检查缓存文件是否创建
        cache_files = list(Path(temp_cache_dir).glob("*"))
        has_cache = len(cache_files) > 0

        # 第二次处理（应该使用缓存）
        result2 = feature_engineer.process_data(sample_market_data)

        # 结果应该一致
        if isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
            pd.testing.assert_frame_equal(result1, result2)
        else:
            assert result1 == result2

        # 如果启用了缓存，应该有缓存文件
        if hasattr(feature_engineer, 'cache_metadata') and feature_engineer.cache_metadata:
            assert has_cache or len(feature_engineer.cache_metadata) > 0

    def test_parallel_processing_edge_cases(self, feature_engineer, sample_market_data):
        """测试并行处理边界情况"""
        # 测试单线程处理
        feature_engineer.max_workers = 1
        result_single = feature_engineer.process_data(sample_market_data)

        # 测试多线程处理
        feature_engineer.max_workers = 4
        result_parallel = feature_engineer.process_data(sample_market_data)

        # 结果应该一致
        if isinstance(result_single, pd.DataFrame) and isinstance(result_parallel, pd.DataFrame):
            pd.testing.assert_frame_equal(result_single, result_parallel)
        else:
            assert result_single == result_parallel

    def test_configuration_validation(self, feature_engineer):
        """测试配置验证"""
        # 测试无效配置更新
        invalid_configs = {
            'max_workers': 0,      # 无效工作线程数
            'batch_size': -1,      # 无效批次大小
            'timeout': 0,          # 无效超时时间
            'max_retries': -5      # 无效重试次数
        }

        for key, value in invalid_configs.items():
            try:
                # 尝试设置无效配置
                setattr(feature_engineer, key, value)
                # 如果设置成功，验证应该在后续使用时失败
                if hasattr(feature_engineer, 'validate_config'):
                    with pytest.raises((ValueError, ValidationError)):
                        feature_engineer.validate_config()
            except (ValueError, TypeError, AttributeError):
                # 也接受直接设置失败
                pass

    def test_error_handling_and_recovery(self, feature_engineer, sample_market_data):
        """测试错误处理和恢复"""
        # 先正常处理一次
        result1 = feature_engineer.process_data(sample_market_data)

        # 模拟错误情况
        try:
            feature_engineer.process_data(None)
        except Exception:
            pass  # 忽略错误

        # 应该能够恢复正常处理
        result2 = feature_engineer.process_data(sample_market_data)

        # 结果应该一致
        if isinstance(result1, pd.DataFrame) and isinstance(result2, pd.DataFrame):
            pd.testing.assert_frame_equal(result1, result2)
        else:
            assert result1 == result2

    def test_resource_management(self, feature_engineer, sample_market_data):
        """测试资源管理"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 处理大数据集
        large_data = pd.concat([sample_market_data] * 10, ignore_index=True)  # 扩大10倍

        result = feature_engineer.process_data(large_data)

        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = end_memory - start_memory

        # 内存增加应该在合理范围内 (< 100MB)
        assert memory_increase < 100, f"内存使用过高: {memory_increase:.1f}MB"
        assert result is not None

    def test_batch_processing_edge_cases(self, feature_engineer):
        """测试批处理边界情况"""
        # 测试极小批次
        feature_engineer.batch_size = 1
        small_batch_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1min'),
            'open': [100.0] * 5,
            'high': [101.0] * 5,
            'low': [99.0] * 5,
            'close': [100.0] * 5,
            'volume': [1000] * 5
        })

        result_small = feature_engineer.process_data(small_batch_data)
        assert result_small is not None

        # 测试极大批次
        feature_engineer.batch_size = 10000
        result_large = feature_engineer.process_data(small_batch_data)
        assert result_large is not None

    def test_monitoring_integration(self, feature_engineer, sample_market_data):
        """测试监控集成"""
        # 启用监控
        if hasattr(feature_engineer, 'enable_monitoring'):
            feature_engineer.enable_monitoring = True

        # 处理数据
        result = feature_engineer.process_data(sample_market_data)

        # 检查是否有监控数据
        try:
            metrics = feature_engineer.get_metrics()
            if metrics:
                assert isinstance(metrics, dict)
                # 检查基本监控指标
                expected_keys = ['processed_records', 'processing_time', 'cache_hits']
                for key in expected_keys:
                    if key in metrics:
                        assert isinstance(metrics[key], (int, float))
        except (NotImplementedError, AttributeError):
            # 如果不支持监控，跳过
            pytest.skip("监控功能未实现")

    def test_fallback_mechanisms(self, feature_engineer, sample_market_data):
        """测试降级机制"""
        # 禁用降级
        if hasattr(feature_engineer, 'fallback_enabled'):
            feature_engineer.fallback_enabled = False

        # 处理数据
        result_no_fallback = feature_engineer.process_data(sample_market_data)

        # 启用降级
        feature_engineer.fallback_enabled = True
        result_with_fallback = feature_engineer.process_data(sample_market_data)

        # 结果应该一致或降级版本可用
        assert result_no_fallback is not None
        assert result_with_fallback is not None

    def test_concurrent_processing_safety(self, feature_engineer):
        """测试并发处理安全性"""
        import threading

        results = []
        errors = []

        def process_worker(data_chunk):
            try:
                result = feature_engineer.process_data(data_chunk)
                results.append(result)
            except Exception as e:
                errors.append(str(e))

        # 创建多个数据块
        data_chunks = []
        base_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=20, freq='1min'),
            'open': [100.0] * 20,
            'high': [101.0] * 20,
            'low': [99.0] * 20,
            'close': [100.0] * 20,
            'volume': [1000] * 20
        })

        for i in range(5):
            chunk = base_data.copy()
            chunk['close'] = chunk['close'] + i  # 稍微修改数据
            data_chunks.append(chunk)

        # 并发处理
        threads = []
        for chunk in data_chunks:
            thread = threading.Thread(target=process_worker, args=(chunk,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30)

        # 验证结果
        assert len(results) == len(data_chunks), f"只有{len(results)}/{len(data_chunks)}个处理成功"
        assert len(errors) == 0, f"发生错误: {errors}"

    def test_feature_engineer_shutdown(self, feature_engineer):
        """测试特征工程器关闭"""
        # 先执行一些操作
        sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1min'),
            'open': [100.0] * 10,
            'high': [101.0] * 10,
            'low': [99.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000] * 10
        })

        feature_engineer.process_data(sample_data)

        # 测试关闭
        try:
            feature_engineer.shutdown()
            # 关闭后应该无法执行新操作
            with pytest.raises(Exception):  # 应该抛出异常或返回错误
                feature_engineer.process_data(sample_data)
        except (NotImplementedError, AttributeError):
            # 如果不支持shutdown，跳过测试
            pytest.skip("shutdown功能未实现")

    def test_memory_efficiency_with_large_datasets(self, feature_engineer):
        """测试大数据集的内存效率"""
        import gc

        # 创建大数据集
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='1s'),
            'open': 100 + np.random.normal(0, 1, 10000).cumsum(),
            'high': 101 + np.random.normal(0, 1, 10000).cumsum(),
            'low': 99 + np.random.normal(0, 1, 10000).cumsum(),
            'close': 100 + np.random.normal(0, 1, 10000).cumsum(),
            'volume': np.random.poisson(1000, 10000)
        })

        # 记录初始内存
        import psutil
        import os
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        # 处理大数据集
        result = feature_engineer.process_data(large_data)

        # 强制垃圾回收
        gc.collect()

        # 检查内存使用
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_delta = final_memory - initial_memory

        # 内存增量应该在合理范围内
        assert memory_delta < 200, f"内存使用过高: {memory_delta:.1f}MB"
        assert result is not None

        # 清理结果
        del result
        gc.collect()
