# tests/unit/features/test_volatility_calculator.py
"""
VolatilityCalculator单元测试

测试覆盖:
- 初始化参数验证
- 波动率指标计算功能
- 数据验证
- 错误处理
- 边界条件
- 性能监控
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

from src.features.indicators.volatility_calculator import VolatilityCalculator


class TestVolatilityCalculator:
    """VolatilityCalculator测试类"""

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
    def calculator(self):
        """VolatilityCalculator实例"""
        return VolatilityCalculator()

    @pytest.fixture
    def calculator_with_config(self):
        """带配置的VolatilityCalculator实例"""
        config = {
            'bb_period': 25,
            'kc_period': 25,
            'kc_multiplier': 2.5,
            'vix_period': 35
        }
        return VolatilityCalculator(config)

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        calculator = VolatilityCalculator()

        assert calculator.bb_period == 20
        assert calculator.kc_period == 20
        assert calculator.kc_multiplier == 2
        assert calculator.vix_period == 30

    def test_initialization_custom_config(self, calculator_with_config):
        """测试自定义配置初始化"""
        calculator = calculator_with_config

        assert calculator.bb_period == 25
        assert calculator.kc_period == 25
        assert calculator.kc_multiplier == 2.5
        assert calculator.vix_period == 35

    def test_initialization_empty_config(self):
        """测试空配置初始化"""
        calculator = VolatilityCalculator({})

        assert calculator.bb_period == 20
        assert calculator.kc_period == 20
        assert calculator.kc_multiplier == 2
        assert calculator.kc_multiplier == 2

    def test_calculate_valid_data(self, calculator, sample_data):
        """测试有效数据的计算"""
        result = calculator.calculate(sample_data)

        assert result is not None
        assert not result.empty
        assert len(result) == len(sample_data)

        # 检查是否包含了基础列
        required_columns = ['high', 'low', 'close']
        for col in required_columns:
            assert col in result.columns

    def test_calculate_empty_data(self, calculator):
        """测试空数据的计算"""
        empty_data = pd.DataFrame()

        result = calculator.calculate(empty_data)

        assert result.empty

    def test_calculate_none_data(self, calculator):
        """测试None数据的计算"""
        result = calculator.calculate(None)

        assert result.empty

    def test_calculate_missing_columns(self, calculator):
        """测试缺失必要列的数据计算"""
        incomplete_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'close': np.random.randn(10)
            # 缺少high, low
        })

        result = calculator.calculate(incomplete_data)

        # 应该返回包含原始数据的DataFrame，但不包含波动率指标
        assert len(result) == len(incomplete_data)
        assert 'close' in result.columns

    def test_calculate_with_nan_values(self, calculator):
        """测试包含NaN值的数据计算"""
        data_with_nan = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': [100] * 5 + [np.nan] * 5,
            'high': [105] * 5 + [np.nan] * 5,
            'low': [95] * 5 + [np.nan] * 5,
            'close': [100] * 5 + [np.nan] * 5,
            'volume': [100000] * 10
        })

        result = calculator.calculate(data_with_nan)

        assert len(result) == len(data_with_nan)
        # NaN值应该被适当处理

    def test_calculate_bollinger_bands(self, calculator, sample_data):
        """测试布林带计算"""
        result = calculator._calculate_bollinger_bands(sample_data)

        assert result is not None
        # 检查布林带相关列
        bb_columns = [col for col in result.columns if 'BB_' in col.upper()]
        assert len(bb_columns) >= 3  # Upper, Middle, Lower

        # 检查布林带值的合理性
        if len(result) > calculator.bb_period:
            bb_upper = result['BB_Upper'].iloc[-1]
            bb_middle = result['BB_Middle'].iloc[-1]
            bb_lower = result['BB_Lower'].iloc[-1]

            # 上轨应该高于中轨，中轨应该高于下轨
            assert bb_upper > bb_middle > bb_lower

    def test_calculate_keltner_channels(self, calculator, sample_data):
        """测试肯特纳通道计算"""
        result = calculator._calculate_keltner_channels(sample_data)

        assert result is not None
        # 检查肯特纳通道相关列
        kc_columns = [col for col in result.columns if 'KC_' in col.upper()]
        assert len(kc_columns) >= 3  # Upper, Middle, Lower

    def test_calculate_atr(self, calculator, sample_data):
        """测试ATR计算"""
        result = calculator._calculate_atr(sample_data)

        assert result is not None
        assert 'ATR' in result.columns

        # ATR值应该为正数
        atr_values = result['ATR'].dropna()
        if len(atr_values) > 0:
            assert all(atr >= 0 for atr in atr_values)

    def test_calculate_historical_volatility(self, calculator, sample_data):
        """测试历史波动率计算"""
        result = calculator._calculate_historical_volatility(sample_data)

        assert result is not None
        hv_columns = [col for col in result.columns if 'HV_' in col.upper() or 'HIST_VOL' in col.upper()]
        assert len(hv_columns) >= 1

    def test_calculate_parkinson_volatility(self, calculator, sample_data):
        """测试帕金森波动率计算"""
        result = calculator._calculate_parkinson_volatility(sample_data)

        assert result is not None
        pv_columns = [col for col in result.columns if 'PV_' in col.upper() or 'PARKINSON' in col.upper()]
        assert len(pv_columns) >= 1

    def test_calculate_garman_klass_volatility(self, calculator, sample_data):
        """测试Garman-Klass波动率计算"""
        result = calculator._calculate_garman_klass_volatility(sample_data)

        assert result is not None
        gk_columns = [col for col in result.columns if 'GK_' in col.upper() or 'GARMAN' in col.upper()]
        assert len(gk_columns) >= 1

    def test_calculate_yang_zhang_volatility(self, calculator, sample_data):
        """测试Yang-Zhang波动率计算"""
        result = calculator._calculate_yang_zhang_volatility(sample_data)

        assert result is not None
        yz_columns = [col for col in result.columns if 'YZ_' in col.upper() or 'YANG' in col.upper()]
        assert len(yz_columns) >= 1

    def test_calculate_realized_volatility(self, calculator, sample_data):
        """测试已实现波动率计算"""
        result = calculator._calculate_realized_volatility(sample_data)

        assert result is not None
        rv_columns = [col for col in result.columns if 'RV_' in col.upper() or 'REALIZED' in col.upper()]
        assert len(rv_columns) >= 1

    def test_performance_monitoring(self, calculator, sample_data):
        """测试性能监控"""
        start_time = time.time()

        result = calculator.calculate(sample_data)

        end_time = time.time()
        duration = end_time - start_time

        # 验证性能指标
        assert duration >= 0
        assert result is not None

        # 对于100行数据，应该很快完成
        assert duration < 5.0

    def test_memory_usage_efficiency(self, calculator, sample_data):
        """测试内存使用效率"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        result = calculator.calculate(sample_data)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # 内存增加应该在合理范围内
        assert memory_increase < 50 * 1024 * 1024  # 不超过50MB
        assert result is not None

    def test_error_handling_invalid_periods(self):
        """测试无效周期参数错误处理"""
        # 测试过小的周期
        config = {'bb_period': 1}  # 过小的周期
        calculator = VolatilityCalculator(config)

        # 应该仍然能够工作，或者抛出适当的错误
        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': np.random.randn(10) + 100,
            'high': np.random.randn(10) + 105,
            'low': np.random.randn(10) + 95,
            'close': np.random.randn(10) + 100,
            'volume': np.random.randint(100000, 1000000, 10)
        })

        result = calculator.calculate(sample_data)
        assert isinstance(result, pd.DataFrame)

    def test_error_handling_extreme_values(self, calculator):
        """测试极端值错误处理"""
        # 创建包含极端值的数据
        extreme_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'open': [100, 1e10, 100, 100, 100, 100, 100, 100, 100, 100],  # 包含极大值
            'high': [105, 105, 1e10, 105, 105, 105, 105, 105, 105, 105],  # 包含极大值
            'low': [95, 95, 95, -1e10, 95, 95, 95, 95, 95, 95],  # 包含极小值
            'close': [100, 100, 100, 100, 1e10, 100, 100, 100, 100, 100],  # 包含极大值
            'volume': [100000] * 10
        })

        result = calculator.calculate(extreme_data)

        # 应该能够处理极端值而不崩溃
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(extreme_data)

    def test_calculator_config_persistence(self, calculator_with_config):
        """测试配置持久性"""
        calculator = calculator_with_config

        # 验证配置在多次调用间保持一致
        original_config = calculator.config.copy()

        sample_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=20),
            'open': np.random.randn(20) + 100,
            'high': np.random.randn(20) + 105,
            'low': np.random.randn(20) + 95,
            'close': np.random.randn(20) + 100,
            'volume': np.random.randint(100000, 1000000, 20)
        })

        result1 = calculator.calculate(sample_data)
        result2 = calculator.calculate(sample_data)

        # 配置应该保持不变
        assert calculator.config == original_config

        # 结果应该是一致的
        pd.testing.assert_frame_equal(result1, result2)

    def test_calculator_scalability(self, calculator):
        """测试计算器扩展性"""
        # 测试不同大小的数据集
        sizes = [10, 50, 100, 200]

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
            result = calculator.calculate(sample_data)
            end_time = time.time()

            duration = end_time - start_time

            # 验证处理时间在合理范围内
            assert duration < 10.0  # 应该在10秒内完成
            assert len(result) == size

            # 对于较大的数据集，验证确实计算了波动率指标
            if size >= 20:
                bb_columns = [col for col in result.columns if 'BB_' in col.upper()]
                assert len(bb_columns) >= 3

    def test_calculator_result_validation(self, calculator, sample_data):
        """测试计算结果验证"""
        result = calculator.calculate(sample_data)

        # 验证布林带逻辑
        if 'BB_Upper' in result.columns and 'BB_Middle' in result.columns and 'BB_Lower' in result.columns:
            # 上轨应该始终高于等于中轨，中轨应该始终高于等于下轨
            bb_upper = result['BB_Upper']
            bb_middle = result['BB_Middle']
            bb_lower = result['BB_Lower']

            # 检查数值关系（忽略NaN值）
            valid_indices = bb_upper.notna() & bb_middle.notna() & bb_lower.notna()

            if valid_indices.any():
                assert all(bb_upper[valid_indices] >= bb_middle[valid_indices])
                assert all(bb_middle[valid_indices] >= bb_lower[valid_indices])

        # 验证ATR为正数
        if 'ATR' in result.columns:
            atr_values = result['ATR'].dropna()
            if len(atr_values) > 0:
                assert all(atr >= 0 for atr in atr_values)

    def test_calculator_indicator_combination(self, calculator, sample_data):
        """测试多个指标的组合计算"""
        result = calculator.calculate(sample_data)

        # 检查是否计算了多种波动率指标
        volatility_indicators = [
            'BB_Upper', 'BB_Middle', 'BB_Lower',  # 布林带
            'ATR',  # 真实波动幅度
        ]

        found_indicators = [col for col in result.columns if col in volatility_indicators]
        assert len(found_indicators) >= 3  # 至少应该有3个指标

    def test_calculator_data_types_preservation(self, calculator, sample_data):
        """测试数据类型保持"""
        original_dtypes = sample_data.dtypes.copy()

        result = calculator.calculate(sample_data)

        # 检查原始列的数据类型是否保持
        for col in sample_data.columns:
            if col in result.columns:
                # 数据类型应该保持一致或兼容
                original_dtype = original_dtypes[col]
                result_dtype = result[col].dtype

                # 对于数值类型，应该保持数值类型
                if original_dtype in ['int64', 'float64']:
                    assert result_dtype in ['int64', 'float64']

    def test_calculator_index_preservation(self, calculator, sample_data):
        """测试索引保持"""
        original_index = sample_data.index.copy()

        result = calculator.calculate(sample_data)

        # 索引应该保持一致
        pd.testing.assert_index_equal(result.index, original_index)

    def test_calculator_column_order_preservation(self, calculator, sample_data):
        """测试列顺序保持"""
        original_columns = sample_data.columns.tolist()

        result = calculator.calculate(sample_data)

        # 原始列的顺序应该保持
        result_original_columns = [col for col in result.columns if col in original_columns]
        assert result_original_columns == original_columns

    def test_calculator_with_different_timeframes(self):
        """测试不同时间框架的数据"""
        # 测试日线数据
        daily_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'open': np.random.randn(100) * 5 + 100,
            'high': np.random.randn(100) * 5 + 105,
            'low': np.random.randn(100) * 5 + 95,
            'close': np.random.randn(100) * 3 + 100,
            'volume': np.random.randint(100000, 1000000, 100)
        })

        calculator = VolatilityCalculator()
        result = calculator.calculate(daily_data)

        assert len(result) == len(daily_data)

        # 测试小时数据
        hourly_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=200, freq='H'),
            'open': np.random.randn(200) * 2 + 100,
            'high': np.random.randn(200) * 2 + 102,
            'low': np.random.randn(200) * 2 + 98,
            'close': np.random.randn(200) * 1 + 100,
            'volume': np.random.randint(10000, 100000, 200)
        })

        result_hourly = calculator.calculate(hourly_data)
        assert len(result_hourly) == len(hourly_data)

    def test_calculator_edge_cases(self, calculator):
        """测试边界情况"""
        # 测试只有一个数据点的情况
        single_point_data = pd.DataFrame({
            'date': [pd.Timestamp('2024-01-01')],
            'open': [100.0],
            'high': [105.0],
            'low': [95.0],
            'close': [100.0],
            'volume': [100000]
        })

        result = calculator.calculate(single_point_data)
        assert len(result) == 1

        # 测试两个数据点的情况
        two_points_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=2),
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [95.0, 96.0],
            'close': [100.0, 101.0],
            'volume': [100000, 110000]
        })

        result = calculator.calculate(two_points_data)
        assert len(result) == 2
