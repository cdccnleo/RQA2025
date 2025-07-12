import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.features.processors.technical import TechnicalProcessor

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
def technical_processor():
    return TechnicalProcessor()

class TestTechnicalProcessor:

    @pytest.mark.parametrize("window", [5, 10, [5, 10, 20]])
    def test_calc_ma(self, technical_processor, sample_price_data, window):
        """测试移动平均线计算"""
        result = technical_processor.calc_ma(sample_price_data, window=window)
        if isinstance(window, list):
            for w in window:
                assert f"MA_{w}" in result.columns
        else:
            assert f"MA_{window}" in result.columns

    def test_calc_ma_invalid_window(self, technical_processor, sample_price_data):
        """测试无效窗口参数"""
        with pytest.raises(ValueError, match=".*window必须大于0.*"):
            technical_processor.calc_ma(sample_price_data, window=0)

    def test_calc_ma_missing_price_col(self, technical_processor, sample_price_data):
        """测试缺失价格列"""
        with pytest.raises(ValueError, match=".*价格列.*不存在.*"):
            technical_processor.calc_ma(sample_price_data, price_col="invalid")

    def test_calc_ma_nan_values(self, technical_processor, sample_price_data):
        """测试包含NaN值的价格列"""
        invalid_data = sample_price_data.copy()
        invalid_data.loc["2023-01-05", "close"] = np.nan
        with pytest.raises(ValueError, match=".*价格列包含 NaN 值.*"):
            technical_processor.calc_ma(invalid_data)

    @pytest.mark.parametrize("window", [7, 14])
    def test_calc_rsi(self, technical_processor, sample_price_data, window):
        """测试相对强弱指数计算"""
        result = technical_processor.calc_rsi(sample_price_data, window=window)
        assert "RSI" in result.columns
        assert result["RSI"].between(0, 100).all()

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
        assert (result["RSI"] == 100).all()

        # 测试全下跌情况
        down_data = pd.DataFrame({
            "close": np.linspace(20, 10, 20),
            "high": np.linspace(21, 11, 20),
            "low": np.linspace(19, 9, 20),
            "volume": np.random.randint(100, 1000, 20)
        })
        result = technical_processor.calc_rsi(down_data)
        assert (result["RSI"] == 0).all()

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
        with pytest.raises(ValueError, match=".*价格数据全为无效值.*"):
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
        assert "Bollinger_Mid" in result.columns
        assert "Bollinger_Up" in result.columns
        assert "Bollinger_Low" in result.columns

    def test_calc_obv(self, technical_processor, sample_price_data):
        """测试能量潮指标计算"""
        result = technical_processor.calc_obv(sample_price_data)
        assert "OBV" in result.columns
        assert not result["OBV"].isnull().any()

    def test_calc_obv_missing_columns(self, technical_processor, sample_price_data):
        """测试缺失成交量列"""
        invalid_data = sample_price_data.drop(columns=["volume"])
        result = technical_processor.calc_obv(invalid_data)
        assert "OBV" in result.columns

    @pytest.mark.parametrize("window", [10, 14])
    def test_calc_atr(self, technical_processor, sample_price_data, window):
        """测试平均真实波幅计算"""
        result = technical_processor.calc_atr(sample_price_data, window=window)
        assert "ATR" in result.columns
        assert not result["ATR"].isnull().any()

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
        with pytest.raises(ValueError, match=".*输入数据为空.*"):
            technical_processor.calc_indicators(pd.DataFrame())

    def test_calc_indicators_missing_columns(self, technical_processor):
        """测试缺失必要列"""
        invalid_data = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(ValueError, match=".*缺失必要价格列.*"):
            technical_processor.calc_indicators(invalid_data)

    def test_calc_indicators_invalid_index(self, technical_processor, sample_price_data):
        """测试无效索引类型"""
        invalid_data = sample_price_data.reset_index()
        with pytest.raises(KeyError, match=".*输入数据必须包含 'date' 列或日期索引.*"):
            technical_processor.calc_indicators(invalid_data)

    def test_calc_indicators_invalid_indicator(self, technical_processor, sample_price_data):
        """测试无效技术指标"""
        with pytest.raises(ValueError, match=".*无效的技术指标.*"):
            technical_processor.calc_indicators(sample_price_data, indicators=["invalid"])

    def test_calculate_volatility_moments(self, technical_processor, sample_price_data):
        """测试波动率矩计算"""
        result = technical_processor.calculate_volatility_moments(sample_price_data)
        assert "volatility" in result.columns
        assert "volatility_skew" in result.columns
        assert "volatility_kurtosis" in result.columns

    def test_extreme_value_analysis(self, technical_processor, sample_price_data):
        """测试极值分析"""
        result = technical_processor.extreme_value_analysis(sample_price_data)
        assert "evt_threshold_0.95" in result.columns
