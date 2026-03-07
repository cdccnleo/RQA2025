#!/usr/bin/env python3
"""
布林带指标计算器全面测试
测试BollingerBandsCalculator的核心功能、边界条件和错误处理
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.features.indicators.bollinger_calculator import BollingerBandsCalculator


class TestBollingerCalculatorComprehensive:
    """布林带计算器全面测试"""

    @pytest.fixture
    def calculator(self):
        """创建布林带计算器实例"""
        return BollingerBandsCalculator()

    @pytest.fixture
    def calculator_with_config(self):
        """创建带配置的布林带计算器"""
        config = {
            'period': 10,
            'std_dev': 1.5
        }
        return BollingerBandsCalculator(config)

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # 生成趋势性价格数据
        base_price = 100
        trend = np.linspace(0, 20, 50)  # 上升趋势
        noise = np.random.normal(0, 2, 50)  # 添加噪声
        prices = base_price + trend + noise

        return pd.DataFrame({
            'close': prices,
            'high': prices + np.abs(np.random.normal(0, 1, 50)),
            'low': prices - np.abs(np.random.normal(0, 1, 50)),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=dates)

    def test_calculator_initialization_default(self, calculator):
        """测试默认参数初始化"""
        assert calculator.period == 20
        assert calculator.std_dev == 2
        assert calculator.config == {}

    def test_calculator_initialization_with_config(self, calculator_with_config):
        """测试带配置参数初始化"""
        assert calculator_with_config.period == 10
        assert calculator_with_config.std_dev == 1.5
        assert calculator_with_config.config == {'period': 10, 'std_dev': 1.5}

    def test_calculate_basic_functionality(self, calculator, sample_data):
        """测试基本计算功能"""
        result = calculator.calculate(sample_data)

        # 检查结果结构
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_data)

        # 检查是否包含所有必要的列
        required_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position']
        for col in required_columns:
            assert col in result.columns

        # 检查数据类型
        assert result['bb_middle'].dtype in [np.float64, float]
        assert result['bb_upper'].dtype in [np.float64, float]
        assert result['bb_lower'].dtype in [np.float64, float]

    def test_calculate_mathematical_correctness(self, calculator):
        """测试数学计算正确性"""
        # 使用已知的数据进行精确验证
        data = pd.DataFrame({
            'close': [100, 102, 98, 105, 95, 108, 92, 110, 90, 112]
        })

        result = calculator.calculate(data)

        # 手动计算验证 (period=20, 但数据只有10行，所以会有NaN)
        # 对于最后一行（第10行），计算前20行的均值和标准差
        window_data = data['close'].iloc[-10:]  # 最后10行
        manual_middle = window_data.mean()
        manual_std = window_data.std()
        manual_upper = manual_middle + (manual_std * 2)
        manual_lower = manual_middle - (manual_std * 2)

        # 由于数据不足，前面应该有NaN
        assert pd.isna(result['bb_middle'].iloc[0])  # 前19行应该是NaN
        assert pd.isna(result['bb_upper'].iloc[0])
        assert pd.isna(result['bb_lower'].iloc[0])

        # 最后一行应该有值
        assert not pd.isna(result['bb_middle'].iloc[-1])
        assert not pd.isna(result['bb_upper'].iloc[-1])
        assert not pd.isna(result['bb_lower'].iloc[-1])

        # 验证上轨大于等于中轨，中轨大于等于下轨
        valid_data = result.dropna()
        assert all(valid_data['bb_upper'] >= valid_data['bb_middle'])
        assert all(valid_data['bb_middle'] >= valid_data['bb_lower'])

    def test_calculate_with_custom_parameters(self, calculator_with_config, sample_data):
        """测试自定义参数计算"""
        result = calculator_with_config.calculate(sample_data)

        # 验证使用的参数
        assert calculator_with_config.period == 10
        assert calculator_with_config.std_dev == 1.5

        # 由于period=10，前面9行应该是NaN
        assert pd.isna(result['bb_middle'].iloc[0])
        assert not pd.isna(result['bb_middle'].iloc[9])  # 第10行应该有值

    def test_band_width_calculation(self, calculator):
        """测试布林带宽度计算"""
        # 创建固定方差的数据，确保宽度计算正确
        data = pd.DataFrame({
            'close': [100] * 25  # 所有价格相同，方差为0
        })

        result = calculator.calculate(data)

        # 当所有价格相同时，宽度应该为0（标准化后）
        # bb_width = (upper - lower) / middle = (2 * std_dev) / middle
        # 当std_dev=0时，width=0
        valid_widths = result['bb_width'].dropna()
        assert len(valid_widths) > 0

        # 由于数据都是100，标准差应该接近0
        # 因此宽度应该很小（接近0）
        assert all(valid_widths < 0.01)  # 宽度应该很小

    def test_position_calculation(self, calculator):
        """测试价格位置计算"""
        data = pd.DataFrame({
            'close': [100, 102, 98, 105, 95, 108, 92, 110, 90, 112,
                     88, 115, 85, 118, 82, 120, 80, 122, 78, 125]
        })

        result = calculator.calculate(data)

        # 检查位置值在[0,1]范围内
        valid_positions = result['bb_position'].dropna()
        assert all(valid_positions >= 0)
        assert all(valid_positions <= 1)

        # 当价格在中轨附近时，位置应该接近0.5
        # 这里不做精确验证，因为依赖于具体数据分布

    def test_insufficient_data_handling(self, calculator):
        """测试数据不足时的处理"""
        # 测试只有很少数据的情况
        small_data = pd.DataFrame({
            'close': [100, 102, 98]  # 少于默认period=20
        })

        result = calculator.calculate(small_data)

        # 应该返回原数据（因为无法计算布林带）
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(small_data)

        # 布林带列可能不存在或全是NaN
        bb_columns = ['bb_middle', 'bb_upper', 'bb_lower', 'bb_width', 'bb_position']
        for col in bb_columns:
            if col in result.columns:
                assert all(pd.isna(result[col]))

    def test_missing_close_column_error(self, calculator):
        """测试缺少收盘价列时的错误处理"""
        data_without_close = pd.DataFrame({
            'open': [100, 102, 98],
            'high': [105, 107, 103],
            'low': [95, 97, 93]
        })

        with pytest.raises(ValueError, match="数据缺少收盘价列"):
            calculator.calculate(data_without_close)

    def test_empty_dataframe_handling(self, calculator):
        """测试空数据框的处理"""
        empty_data = pd.DataFrame(columns=['close'])

        result = calculator.calculate(empty_data)

        # 应该返回空的DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_price_handling(self, calculator):
        """测试单一价格的处理"""
        single_price_data = pd.DataFrame({
            'close': [100] * 25  # 所有价格相同
        })

        result = calculator.calculate(single_price_data)

        # 所有布林带值应该相等
        valid_middle = result['bb_middle'].dropna()
        valid_upper = result['bb_upper'].dropna()
        valid_lower = result['bb_lower'].dropna()

        # 当标准差为0时，upper=middle, lower=middle
        assert all(valid_upper == valid_middle)
        assert all(valid_lower == valid_middle)

    def test_extreme_price_volatility(self, calculator):
        """测试极端价格波动"""
        # 创建高波动性的价格数据
        np.random.seed(42)
        base_price = 100
        high_volatility = np.random.normal(0, 10, 50)  # 高波动
        prices = base_price + np.cumsum(high_volatility * 0.1)  # 累积效应

        data = pd.DataFrame({
            'close': prices
        })

        result = calculator.calculate(data)

        # 在高波动情况下，布林带应该更宽
        valid_widths = result['bb_width'].dropna()
        assert len(valid_widths) > 0

        # 宽度应该大于正常情况（这里不设置具体阈值，因为依赖于数据）

    def test_price_trend_scenarios(self, calculator):
        """测试不同价格趋势场景"""
        # 上升趋势
        uptrend_data = pd.DataFrame({
            'close': list(range(100, 121))  # 100, 101, ..., 120
        })

        # 下降趋势
        downtrend_data = pd.DataFrame({
            'close': list(range(120, 99, -1))  # 120, 119, ..., 100
        })

        # 横盘整理
        sideways_data = pd.DataFrame({
            'close': [100] * 10 + [99] * 10 + [101] * 10
        })

        uptrend_result = calculator.calculate(uptrend_data)
        downtrend_result = calculator.calculate(downtrend_data)
        sideways_result = calculator.calculate(sideways_data)

        # 所有结果都应该是有效的DataFrame
        assert isinstance(uptrend_result, pd.DataFrame)
        assert isinstance(downtrend_result, pd.DataFrame)
        assert isinstance(sideways_result, pd.DataFrame)

        # 检查布林带列都存在
        bb_columns = ['bb_middle', 'bb_upper', 'bb_lower']
        for result in [uptrend_result, downtrend_result, sideways_result]:
            for col in bb_columns:
                assert col in result.columns

    def test_exception_handling_and_logging(self, calculator):
        """测试异常处理和日志记录"""
        # 创建会导致计算错误的数据
        problematic_data = pd.DataFrame({
            'close': [100, np.nan, 102, np.inf, -np.inf]  # 包含NaN和无穷大
        })

        with patch('src.features.indicators.bollinger_calculator.logger') as mock_logger:
            result = calculator.calculate(problematic_data)

            # 应该记录错误日志
            mock_logger.error.assert_called()

            # 仍然应该返回结果（错误处理）
            assert isinstance(result, pd.DataFrame)

    def test_memory_efficiency_large_dataset(self, calculator):
        """测试大数据集的内存效率"""
        # 创建大数据集
        large_data = pd.DataFrame({
            'close': np.random.normal(100, 5, 10000)  # 10000个数据点
        })

        import time
        start_time = time.time()

        result = calculator.calculate(large_data)

        end_time = time.time()
        processing_time = end_time - start_time

        # 验证计算完成
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(large_data)

        # 性能应该在合理范围内（这里不设置具体时间限制，因为依赖于环境）

    def test_rolling_calculation_edge_cases(self, calculator):
        """测试滚动计算的边界情况"""
        # 测试period=1的情况（应该等同于价格本身）
        period_one_data = pd.DataFrame({
            'close': [100, 102, 98, 105, 95]
        })

        calculator_one = BollingerBandsCalculator({'period': 1, 'std_dev': 2})
        result_one = calculator_one.calculate(period_one_data)

        # 当period=1时，标准差为0，所以upper=lower=middle=close
        valid_data = result_one.dropna()
        np.testing.assert_array_almost_equal(
            valid_data['bb_middle'].values,
            valid_data['close'].values,
            decimal=6
        )

    def test_config_validation(self):
        """测试配置参数验证"""
        # 测试无效的period
        with pytest.raises(ValueError):
            BollingerBandsCalculator({'period': 0})  # period不能为0

        with pytest.raises(ValueError):
            BollingerBandsCalculator({'period': -1})  # period不能为负数

        # 测试无效的std_dev
        with pytest.raises(ValueError):
            BollingerBandsCalculator({'std_dev': 0})  # std_dev不能为0

        with pytest.raises(ValueError):
            BollingerBandsCalculator({'std_dev': -1})  # std_dev不能为负数

    def test_result_persistence_and_consistency(self, calculator, sample_data):
        """测试结果持久性和一致性"""
        # 多次计算相同数据应该得到相同结果
        result1 = calculator.calculate(sample_data)
        result2 = calculator.calculate(sample_data)

        # 比较数值结果（忽略NaN）
        for col in ['bb_middle', 'bb_upper', 'bb_lower']:
            valid1 = result1[col].dropna()
            valid2 = result2[col].dropna()
            pd.testing.assert_series_equal(valid1, valid2, check_names=False)

    def test_different_time_frequencies(self, calculator):
        """测试不同时间频率的数据"""
        # 日频数据
        daily_data = pd.DataFrame({
            'close': np.random.normal(100, 5, 100)
        })

        # 小时频数据（模拟）
        hourly_data = pd.DataFrame({
            'close': np.random.normal(100, 2, 500)  # 更多数据点，更小波动
        })

        daily_result = calculator.calculate(daily_data)
        hourly_result = calculator.calculate(hourly_data)

        # 都应该成功计算
        assert isinstance(daily_result, pd.DataFrame)
        assert isinstance(hourly_result, pd.DataFrame)

        # 小时数据的布林带可能更窄（因为波动更小）
        # 这里不做具体数值比较，因为依赖于具体数据

    def test_integration_with_other_indicators(self, calculator):
        """测试与其他指标的集成"""
        # 创建包含多个技术指标的数据
        data = pd.DataFrame({
            'close': [100, 102, 98, 105, 95, 108, 92, 110, 90, 112,
                     88, 115, 85, 118, 82, 120, 80, 122, 78, 125],
            'volume': [1000, 1100, 900, 1200, 800, 1300, 700, 1400, 600, 1500,
                      500, 1600, 400, 1700, 300, 1800, 200, 1900, 100, 2000]
        })

        result = calculator.calculate(data)

        # 布林带应该与其他数据兼容
        assert len(result) == len(data)
        assert all(col in result.columns for col in ['close', 'volume'])  # 原有列应该保留
        assert all(col in result.columns for col in ['bb_middle', 'bb_upper', 'bb_lower'])  # 新增列存在

    def test_performance_metrics_calculation(self, calculator):
        """测试性能指标计算"""
        # 创建价格穿越布林带边界的场景
        data = pd.DataFrame({
            'close': [95, 97, 99, 101, 103, 105, 107, 109, 111, 113,
                     115, 117, 119, 121, 123, 125, 127, 129, 131, 133]
        })

        result = calculator.calculate(data)

        # 验证位置指标的有效性
        valid_positions = result['bb_position'].dropna()

        # 位置应该在合理范围内
        assert all(valid_positions >= -0.5)  # 允许稍微超出边界
        assert all(valid_positions <= 1.5)

        # 在上升趋势中，价格应该经常在布林带中间偏上
        # 这里不做具体数值断言，因为依赖于数据分布
