import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.features.technical.technical_processor import TechnicalProcessor

@pytest.fixture
def sample_price_data():
    dates = pd.date_range(start="2023-01-01", periods=20)
    return pd.DataFrame({
        "close": np.linspace(10, 30, 20),
        "high": np.linspace(11, 31, 20),
        "low": np.linspace(9, 29, 20),
        "volume": np.random.randint(100, 1000, 20)
    }, index=dates)

@pytest.fixture
def technical_processor(mock_feature_manager):
    """创建TechnicalProcessor实例"""
    return TechnicalProcessor(register_func=mock_feature_manager.register)

class TestTechnicalProcessor:
    """技术指标处理器测试"""

    @pytest.mark.parametrize("window", [5, 10])
    def test_calc_ma(self, technical_processor, sample_price_data, window):
        """测试移动平均线计算"""
        result = technical_processor.calc_ma(sample_price_data, window=window)
        assert f"MA_{window}" in result.columns

    def test_calc_ma_invalid_window(self, technical_processor, sample_price_data):
        """测试无效窗口参数"""
        with pytest.raises(ValueError):
            technical_processor.calc_ma(sample_price_data, window=0)

    def test_calc_ma_missing_price_col(self, technical_processor, sample_price_data):
        """测试缺失价格列"""
        with pytest.raises(ValueError):
            technical_processor.calc_ma(sample_price_data, price_col="invalid")

    def test_calc_ma_nan_values(self, technical_processor, sample_price_data):
        """测试包含NaN值的价格列"""
        invalid_data = sample_price_data.copy()
        invalid_data.loc[2, "close"] = np.nan
        with pytest.raises(ValueError):
            technical_processor.calc_ma(invalid_data)

    @pytest.mark.parametrize("window", [7, 14])
    def test_calc_rsi(self, technical_processor, sample_price_data, window):
        """测试相对强弱指数计算"""
        result = technical_processor.calc_rsi(sample_price_data, window=window)
        assert "RSI" in result.columns
        # 只对非NaN值进行断言
        rsi_values = result["RSI"].dropna()
        assert (rsi_values >= 0).all() and (rsi_values <= 100).all()

    def test_calc_rsi_extreme_cases(self, technical_processor):
        """测试RSI极端情况"""
        # 测试全上涨情况
        up_data = pd.DataFrame({
            "close": np.linspace(10, 20, 20),
            "high": np.linspace(11, 21, 20),
            "low": np.linspace(9, 19, 20),
            "volume": np.random.randint(100, 1000, 20)
        })
        result = technical_processor.calc_rsi(up_data)
        # 只对非NaN值进行断言
        rsi_values = result["RSI"].dropna()
        assert (rsi_values == 100).all()

        # 测试全下跌情况
        down_data = pd.DataFrame({
            "close": np.linspace(20, 10, 20),
            "high": np.linspace(21, 11, 20),
            "low": np.linspace(19, 9, 20),
            "volume": np.random.randint(100, 1000, 20)
        })
        result = technical_processor.calc_rsi(down_data)
        rsi_values = result["RSI"].dropna()
        assert (rsi_values == 0).all()

    @pytest.mark.parametrize("short_window,long_window,signal_window", [
        (12, 26, 9),
        (5, 20, 5)
    ])
    def test_calc_macd(self, technical_processor, sample_price_data, short_window, long_window, signal_window):
        """测试MACD指标计算"""
        result = technical_processor.calc_macd(
            sample_price_data,
            short_window=short_window,
            long_window=long_window,
            signal_window=signal_window
        )
        assert "MACD_DIF" in result.columns
        assert "MACD_DEA" in result.columns
        assert "MACD_Histogram" in result.columns

    def test_calc_macd_all_nan(self, technical_processor, sample_price_data):
        """测试全NaN价格数据"""
        invalid_data = sample_price_data.copy()
        invalid_data["close"] = np.nan
        with pytest.raises(ValueError):
            technical_processor.calc_macd(invalid_data)

    @pytest.mark.parametrize("window,num_std", [
        (20, 2),
        (10, 1.5)
    ])
    def test_calc_bollinger(self, technical_processor, sample_price_data, window, num_std):
        """测试布林带计算"""
        result = technical_processor.calc_bollinger(
            sample_price_data,
            window=window,
            num_std=num_std
        )
        assert "BOLL_UPPER" in result.columns
        assert "BOLL_MIDDLE" in result.columns
        assert "BOLL_LOWER" in result.columns

    def test_calc_obv(self, technical_processor, sample_price_data):
        """测试能量潮指标计算"""
        result = technical_processor.calc_obv(sample_price_data)
        assert "OBV" in result.columns
        assert not result["OBV"].isnull().any()

    def test_calc_obv_missing_columns(self, technical_processor, sample_price_data):
        """测试缺失成交量列"""
        invalid_data = sample_price_data.drop(columns=["volume"])
        with pytest.raises(ValueError):
            technical_processor.calc_obv(invalid_data)

    @pytest.mark.parametrize("window", [10, 14])
    def test_calc_atr(self, technical_processor, sample_price_data, window):
        """测试平均真实波幅计算"""
        result = technical_processor.calc_atr(sample_price_data, window=window)
        assert "ATR" in result.columns
        # 允许前几行有NaN，但至少有一些有效值
        assert not result["ATR"].isnull().all()

    def test_calc_indicators(self, technical_processor, sample_price_data):
        """测试批量计算技术指标"""
        indicators = ["ma", "rsi", "macd"]
        params = {
            "ma": {"window": [5, 10]},
            "rsi": {"window": 14},
            "macd": {"short_window": 12, "long_window": 26, "signal_window": 9}
        }
        result = technical_processor.calc_indicators(
            sample_price_data,
            indicators=indicators,
            params=params
        )
        assert "MA_5" in result.columns
        assert "MA_10" in result.columns
        assert "RSI" in result.columns
        assert "MACD_DIF" in result.columns

    def test_calc_indicators_empty_input(self, technical_processor):
        """测试空输入数据"""
        with pytest.raises(ValueError):
            technical_processor.calc_indicators(pd.DataFrame(), indicators=["ma"])

    def test_calc_indicators_missing_columns(self, technical_processor):
        """测试缺失必要列"""
        invalid_data = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError):
            technical_processor.calc_indicators(invalid_data, indicators=["ma"])

    def test_calc_indicators_invalid_index(self, technical_processor, sample_price_data):
        """测试无效索引类型"""
        invalid_data = sample_price_data.reset_index()
        with pytest.raises(KeyError):
            technical_processor.calc_indicators(invalid_data, indicators=["ma"])

    def test_calc_indicators_invalid_indicator(self, technical_processor, sample_price_data):
        """测试无效技术指标"""
        with pytest.raises(ValueError):
            technical_processor.calc_indicators(sample_price_data, indicators=["invalid"])

    def test_calculate_volatility_moments(self, technical_processor, sample_price_data):
        """测试波动率矩计算"""
        result = technical_processor.calculate_volatility_moments(sample_price_data)
        assert "VOLATILITY" in result.columns
        assert "SKEWNESS" in result.columns
        assert "KURTOSIS" in result.columns

    def test_extreme_value_analysis(self, technical_processor, sample_price_data):
        """测试极值分析"""
        result = technical_processor.extreme_value_analysis(sample_price_data)
        assert "EXTREME_HIGH" in result.columns
        assert "EXTREME_LOW" in result.columns
        assert "UPPER_THRESHOLD" in result.columns
        assert "LOWER_THRESHOLD" in result.columns
