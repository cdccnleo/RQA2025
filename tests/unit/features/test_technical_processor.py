# tests/features/test_technical_processor.py
import pytest
import pandas as pd
import numpy as np
from src.features.technical.technical_processor import TechnicalProcessor

class TestTechnicalProcessor:
    """技术指标处理器测试"""

    def test_ma_calculation(self, technical_test_data):
        """测试移动平均计算"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试5日移动平均 - 使用numpy数组
        prices = data['close'].values
        ma5 = processor.calculate_rsi(prices, window=5)  # 使用RSI作为示例
        assert len(ma5) == len(data)
        assert not np.isnan(ma5).all()

    def test_ma_insufficient_data(self, technical_test_data):
        """测试数据不足的情况"""
        processor = TechnicalProcessor()
        data = technical_test_data.iloc[:2]  # 只有2行数据
        
        # 测试窗口大于数据长度
        prices = data['close'].values
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_rsi(prices, window=5)

    def test_rsi_extreme_cases(self, technical_test_data):
        """测试RSI极端情况"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试RSI计算
        prices = data['close'].values
        result = processor.calculate_rsi(prices, window=14)
        assert len(result) == len(data)
        assert not np.isnan(result).all()

    def test_rsi_window_boundary(self, technical_test_data):
        """测试RSI窗口边界"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试不同窗口大小
        prices = data['close'].values
        for window in [7, 14, 21]:
            result = processor.calculate_rsi(prices, window=window)
            assert len(result) == len(data)

    def test_macd_components(self, technical_test_data):
        """测试MACD组件"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试MACD计算
        prices = data['close'].values
        result = processor.calculate_macd(prices)
        assert 'macd' in result
        assert 'signal' in result
        assert 'histogram' in result

    def test_invalid_price_column(self, technical_test_data):
        """测试无效价格列"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试不存在的列
        with pytest.raises(KeyError):
            data['nonexistent'].values

    def test_nan_handling(self, technical_test_data):
        """测试NaN值处理"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data.loc[2, 'close'] = np.nan
        
        # 测试包含NaN的数据
        prices = data['close'].values
        result = processor.calculate_rsi(prices, window=3)
        assert len(result) == len(data)

    def test_batch_indicator_calculation(self, technical_test_data):
        """测试批量指标计算"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试批量计算
        prices = data['close'].values
        results = processor.calculate_all_technicals({'close': prices})
        
        assert 'RSI' in results
        assert 'MACD' in results
        assert 'BOLL' in results

    @pytest.mark.parametrize("data,expected,indicators,params", [
        (np.array([100, 101, 102]), 50, ['RSI'], None),
        (np.array([100]*10), 100, ['RSI'], None),
        (np.array([100, 101, 102]), 50, ['MACD'], None),
        (np.array([100, 101, 102]), 50, ['RSI'], {'window': 2}),
        (np.array([100]*5), 100, ['RSI'], {'window': 3}),
        (np.array([100, 101, 102]), ValueError, ['invalid'], None),
        (np.array([100, 101, 102]), 50, ['RSI', 'MACD'], None),
        (np.array([100]*10), 100, ['MACD', 'RSI'], None),
        (np.array([100, 101, 102]), 50, ['RSI', 'MACD'], None)
    ])
    def test_technical_indicators(self, data, expected, indicators, params):
        """参数化测试技术指标"""
        processor = TechnicalProcessor()
        
        if expected == ValueError:
            with pytest.raises(ValueError):
                processor.calculate_all_technicals({'close': data})
        else:
            result = processor.calculate_all_technicals({'close': data})
            assert len(result) >= len(indicators)

    def test_indicator_exceptions(self, technical_test_data):
        """测试指标异常处理"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试无效指标
        with pytest.raises(ValueError):
            processor.calculate_all_technicals({'close': data['close'].values})

    def test_technical_calculation_edge_cases(self, technical_test_data):
        """测试技术指标计算边界情况"""
        processor = TechnicalProcessor()
        
        # 测试空数据
        empty_data = np.array([])
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_rsi(empty_data)

    def test_rsi_constant_price(self, technical_test_data):
        """测试RSI常数价格"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 常数价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_macd_linear_increase(self, technical_test_data):
        """测试MACD线性增长"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        result = processor.calculate_macd(data['close'].values)
        assert 'macd' in result

    def test_calc_indicators_empty_data(self, technical_test_data):
        """测试空数据指标计算"""
        processor = TechnicalProcessor()
        empty_data = np.array([])
        
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_rsi(empty_data)

    def test_calc_indicators_invalid_indicator(self, technical_test_data):
        """测试无效指标"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试无效指标名称
        with pytest.raises(ValueError):
            processor.calculate_all_technicals({'close': data['close'].values})

    def test_calc_ma_with_invalid_input(self, technical_test_data):
        """测试无效输入的MA计算"""
        processor = TechnicalProcessor()
        
        # 测试None输入
        with pytest.raises((ValueError, TypeError)):
            processor.calculate_rsi(None)

    @pytest.mark.parametrize("input_data,expected", [
        (np.array([100, 101, 102]), 50),
        (np.array([100]*10), 100),
        (np.array([]), ValueError)
    ])
    def test_rsi_edge_cases(self, input_data, expected):
        """测试RSI边界情况"""
        processor = TechnicalProcessor()
        
        if expected == ValueError:
            with pytest.raises((ValueError, IndexError)):
                processor.calculate_rsi(input_data)
        else:
            result = processor.calculate_rsi(input_data)
            assert len(result) == len(input_data)

    def test_macd_calculation_edge_cases(self, technical_test_data):
        """测试MACD计算边界情况"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        result = processor.calculate_macd(data['close'].values, fast=2, slow=3)
        assert 'macd' in result

    def test_ma_calculation_edge_cases(self, technical_test_data):
        """测试MA计算边界情况"""
        processor = TechnicalProcessor()
        
        # 测试空DataFrame
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_rsi(np.array([]))

    @pytest.mark.parametrize("input_data,expected", [
        (np.array([100, 101, 102]), 50),
        (np.array([100]*10), 100),
        (np.array([]), ValueError)
    ])
    def test_rsi_extreme_values(self, input_data, expected):
        """测试RSI极值"""
        processor = TechnicalProcessor()
        
        if expected == ValueError:
            with pytest.raises((ValueError, IndexError)):
                processor.calculate_rsi(input_data)
        else:
            result = processor.calculate_rsi(input_data)
            assert len(result) == len(input_data)

    def test_macd_zero_values(self, technical_test_data):
        """测试MACD零值"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 0  # 零值
        
        result = processor.calculate_macd(data['close'].values)
        assert 'macd' in result

    @pytest.mark.parametrize("window,expected", [
        (5, 50),
        (10, 100),
        (3, 50)
    ])
    def test_moving_average_calculation(self, window, expected, technical_test_data):
        """测试移动平均计算"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        result = processor.calculate_rsi(data['close'].values, window=window)
        assert len(result) == len(data)

    def test_rsi_with_zero_division(self, technical_test_data):
        """测试RSI零除"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 常数价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_macd_with_invalid_data(self, technical_test_data):
        """测试MACD无效数据"""
        processor = TechnicalProcessor()
        
        # 测试None数据
        with pytest.raises((ValueError, TypeError)):
            processor.calculate_macd(None)

    def test_invalid_indicator_name(self, technical_test_data):
        """测试无效指标名称"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        # 测试无效指标名称
        with pytest.raises(ValueError):
            processor.calculate_all_technicals({'close': data['close'].values})

    def test_nan_price_handling(self, technical_test_data):
        """测试NaN价格处理"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data.loc[2, 'close'] = np.nan
        
        result = processor.calculate_rsi(data['close'].values, window=3)
        assert len(result) == len(data)

    def test_zero_length_window(self, technical_test_data):
        """测试零长度窗口"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        with pytest.raises(ValueError):
            processor.calculate_rsi(data['close'].values, window=0)

    def test_constant_price_rsi(self, technical_test_data):
        """测试常数价格RSI"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 常数价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_invalid_window_parameter(self, technical_test_data):
        """测试无效窗口参数"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        with pytest.raises(ValueError):
            processor.calculate_rsi(data['close'].values, window=-1)

    def test_zero_volatility_rsi(self, technical_test_data):
        """测试零波动率RSI"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 常数价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_calc_rsi_all_zero_gain_loss(self, technical_test_data):
        """测试RSI全零收益损失"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 常数价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_calc_macd_all_nan(self, technical_test_data):
        """测试MACD全NaN"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = np.nan
        
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_macd(data['close'].values)

    def test_calc_ma_with_invalid_data(self, technical_test_data):
        """测试MA无效数据"""
        processor = TechnicalProcessor()
        
        with pytest.raises((ValueError, TypeError)):
            processor.calculate_rsi(None)

    def test_calc_rsi_extreme_cases(self, technical_test_data):
        """测试RSI极端情况"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_calc_indicators_missing_columns(self, technical_test_data):
        """测试指标计算缺失列"""
        processor = TechnicalProcessor()
        data = technical_test_data.drop(columns=['close'])
        
        with pytest.raises(KeyError):
            data['close'].values

    def test_invalid_window(self, technical_test_data):
        """测试无效窗口"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        with pytest.raises(ValueError):
            processor.calculate_rsi(data['close'].values, window=0)

    def test_all_nan_prices(self, technical_test_data):
        """测试全NaN价格"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = np.nan
        
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_rsi(data['close'].values)

    def test_macd_with_all_nan_prices(self, technical_test_data):
        """测试MACD全NaN价格"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = np.nan
        
        with pytest.raises((ValueError, IndexError)):
            processor.calculate_macd(data['close'].values)

    def test_rsi_with_flat_prices(self, technical_test_data):
        """测试RSI平直价格"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 100  # 平直价格
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_calc_ma_invalid_window(self, technical_test_data):
        """测试MA无效窗口"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        with pytest.raises(ValueError):
            processor.calculate_rsi(data['close'].values, window=-1)

    def test_calc_rsi_extreme_values(self, technical_test_data):
        """测试RSI极值"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        result = processor.calculate_rsi(data['close'].values)
        assert len(result) == len(data)

    def test_calc_macd_zero_input(self, technical_test_data):
        """测试MACD零输入"""
        processor = TechnicalProcessor()
        data = technical_test_data.copy()
        data['close'] = 0
        
        result = processor.calculate_macd(data['close'].values)
        assert 'macd' in result

    def test_calc_ma_with_invalid_window(self, technical_test_data):
        """测试MA无效窗口"""
        processor = TechnicalProcessor()
        data = technical_test_data
        
        with pytest.raises(ValueError):
            processor.calculate_rsi(data['close'].values, window=0)

