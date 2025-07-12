# tests/features/test_technical_processor.py
import pytest
import pandas as pd
import numpy as np

if not hasattr(np, 'bool'):
    np.bool = np.bool_
from src.features.processors.technical import TechnicalProcessor
from src.infrastructure.utils.logger import get_logger

logger = get_logger(__name__)  # 自动继承全局配置


@pytest.fixture
def sample_data():
    # 创建包含20个交易日的示例数据
    dates = pd.date_range(start='2023-01-01', periods=20)
    return pd.DataFrame({
        'close': np.linspace(10.0, 19.5, 20),
        'high': np.linspace(11.0, 20.5, 20),
        'low': np.linspace(9.0, 18.5, 20),
        'volume': [1000] * 20
    }, index=dates)


def test_ma_calculation(sample_data):
    """测试MA计算的正常流程"""
    processor = TechnicalProcessor()
    window = 5

    # 执行计算
    result = processor.calc_ma(
        sample_data,
        window=window,
        price_col="close",
        is_training=True
    )

    # 验证返回类型
    assert isinstance(result, pd.DataFrame), "应返回DataFrame对象"

    # 验证列名是否正确生成
    ma_col = f"MA_{window}"
    assert ma_col in result.columns, f"结果中应包含{ma_col}列"

    # 验证数值计算
    expected_values = sample_data['close'].rolling(window, min_periods=1).mean()
    pd.testing.assert_series_equal(
        result[ma_col],
        expected_values,
        check_names=False
    )


def test_ma_insufficient_data():
    """测试数据长度不足时的处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [10, 11]}, index=pd.date_range(start="2023-01-01", periods=2))
    result = processor.calc_ma(data, window=5, is_training=False)

    # 非训练模式下可用数据不足，应填充NaN
    assert pd.isna(result["MA_5"].iloc[-1])
    logger.info("MA数据不足测试通过")


def test_rsi_extreme_cases():
    """测试RSI计算的极端情况处理"""
    # Case1: 全上涨行情
    data = pd.DataFrame({"close": [100, 101, 102, 103]})
    result = TechnicalProcessor().calc_rsi(data, price_col="close")
    assert result["RSI"].iloc[-1] == 100  # 应触发avg_loss=0条件

    # Case2: 缺失价格数据
    with pytest.raises(ValueError) as excinfo:
        TechnicalProcessor().calc_rsi(pd.DataFrame(columns=["wrong_col"]))

    # 验证错误消息
    assert "Price column 'close' not found in DataFrame" in str(excinfo.value)


def test_rsi_window_boundary(sample_data):
    """测试窗口边界条件（窗口=数据长度）"""
    processor = TechnicalProcessor()
    result = processor.calc_rsi(sample_data, window=len(sample_data))
    assert not result["RSI"].isnull().all(), "RSI不应全为NaN"
    logger.info("RSI窗口边界测试通过")


def test_macd_components(sample_data):
    """验证MACD分量计算逻辑"""
    processor = TechnicalProcessor()
    result = processor.calc_macd(sample_data)

    # 正确计算EMA（无shift）
    ema12 = sample_data["close"].ewm(span=12, adjust=False).mean()
    ema26 = sample_data["close"].ewm(span=26, adjust=False).mean()
    expected_dif = ema12 - ema26
    expected_dif = expected_dif.fillna(0.0)  # 填充初始NaN为0.0

    # 验证DIF
    pd.testing.assert_series_equal(
        result["MACD_DIF"].round(8),
        expected_dif.round(8),
        check_names=False,
        check_dtype=False,
        atol=1e-8
    )


def test_invalid_price_column():
    """测试无效价格列时的异常处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"wrong_col": [10, 11]})

    # 将KeyError改为ValueError
    with pytest.raises(ValueError):  # 修改此行
        processor.calc_ma(data, price_col="close")
    logger.info("无效列名异常处理测试通过")


def test_nan_handling():
    """测试输入包含NaN时的处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({
        "close": [10, np.nan, 12],
        "volume": [1000, 2000, 1500]
    })

    # 测试 calc_ma 方法的 NaN 处理
    with pytest.raises(ValueError):
        processor.calc_ma(data)

    # 测试 calc_rsi 方法的 NaN 处理
    with pytest.raises(ValueError):
        processor.calc_rsi(data)
    logger.info("NaN输入处理测试通过")


# ------------------- 批量计算测试 -------------------
def test_batch_indicator_calculation(sample_data):
    """测试calc_indicators批量计算功能"""
    processor = TechnicalProcessor()
    indicators = ["ma", "rsi"]
    result = processor.calc_indicators(sample_data, indicators)

    assert "MA_20" in result.columns  # 默认窗口为20
    assert "RSI" in result.columns
    logger.info("批量计算功能测试通过")


@pytest.fixture
def mock_stock_data():
    return pd.DataFrame({
        "close": np.random.rand(100) * 100 + 50,
        "high": np.random.rand(100) * 110 + 50,
        "low": np.random.rand(100) * 90 + 40,
        "volume": np.random.randint(10000, 50000, 100)
    }, index=pd.date_range("2023-01-01", periods=100))


@pytest.mark.parametrize("data, expected, indicators, params", [
    # RSI测试
    ([10, 10, 10, 10], 50, ["rsi"], None),  # 无变化
    ([10, 12, 14, 16], 100, ["rsi"], None),  # 全上涨
    ([16, 14, 12, 10], 0, ["rsi"], None),  # 全下跌
    ([10, 12, 12, 14], 100, ["rsi"], None),  # 混合变化

    # MA测试修正后的预期
    ([1, 2, 3, 4], {
        3: [1.0, 1.5, 2.0, 3.0],  # 窗口3，min_periods=1
        5: [1.0, 1.5, 2.0, 2.5]  # 窗口5，min_periods=1
    }, ["ma"], {"ma": {"window": [3, 5]}}),

    # MACD边界测试
    ([0, 0, 0, 0], {
        'MACD_DIF': [0.0, 0.0, 0.0, 0.0],
        'MACD_DEA': [0.0, 0.0, 0.0, 0.0],
        'MACD_Histogram': [0.0, 0.0, 0.0, 0.0]
    }, ["macd"], None),  # 全零输入
    ([np.nan, 1, 2], ValueError, ["macd"], None),  # 无效数据

    # MACD正常测试（重新生成预期值后）
    ([10, 12, 14, 16], {
        'MACD_DIF': [0.0, 0.15954416, 0.44226914, 0.81828137],
        'MACD_DEA': [0.0, 0.03190883, 0.11398089, 0.25484099],
        'MACD_Histogram': [0.0, 0.12763533, 0.32828825, 0.56344038]
    }, ["macd"], None),
    ([10, 12, 10, 14], {
        'MACD_DIF': [0.0, 0.15954416, 0.12318082, 0.41237557],
        'MACD_DEA': [0.0, 0.03190883, 0.05016323, 0.12260570],
        'MACD_Histogram': [0.0, 0.12763533, 0.07301759, 0.28976987]
    }, ["macd"], None),
])
def test_technical_indicators(data, expected, indicators, params):
    processor = TechnicalProcessor()
    df = pd.DataFrame({
        "close": data,
        "high": data,
        "low": data,
        "volume": [1000] * len(data)
    })
    df.index = pd.date_range(start='2023-01-01', periods=len(data))

    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            processor.calc_indicators(df, indicators=indicators, params=params)
    else:
        result = processor.calc_indicators(df, indicators=indicators, params=params)
        if indicators == ["macd"]:
            # 四舍五入到小数点后8位，避免浮点精度问题
            for col, values in expected.items():
                pd.testing.assert_series_equal(
                    result[col].round(8),
                    pd.Series(values, index=df.index, name=col).round(8),
                    check_names=False,
                    check_dtype=False,
                    atol=1e-8
                )


def test_indicator_exceptions():
    processor = TechnicalProcessor()
    with pytest.raises(ValueError):
        processor.calc_ma(pd.DataFrame({"close": [np.nan] * 10}))


def test_technical_calculation_edge_cases():
    processor = TechnicalProcessor()

    # 空数据测试（包含列名）
    empty_data = pd.DataFrame(columns=["close"])
    with pytest.raises(ValueError):
        processor.calc_indicators(empty_data, indicators=["ma"])


def test_rsi_constant_price():
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [100] * 30})
    result = processor.calc_rsi(data)
    assert (result["RSI"] == 50).all()  # 修正预期值为50


def test_macd_linear_increase(sample_data):
    processor = TechnicalProcessor()
    result = processor.calc_macd(sample_data)  # 仅接收DataFrame
    dif = result['MACD_DIF']
    assert (dif > 0).iloc[-1]


def test_calc_indicators_empty_data():
    processor = TechnicalProcessor()
    with pytest.raises(ValueError) as excinfo:
        processor.calc_indicators(pd.DataFrame(columns=["close"]), ["ma"])  # 空数据但包含列名
    assert "Input data is empty" in str(excinfo.value)


def test_calc_indicators_invalid_indicator():
    processor = TechnicalProcessor()
    # 构造包含必要列和日期索引的测试数据
    valid_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"]),
        "close": [100],
        "high": [105],
        "low": [95],
        "volume": [1000]
    }).set_index('date')  # 设置日期索引
    with pytest.raises(ValueError) as excinfo:
        processor.calc_indicators(valid_data, ["invalid"])
    assert "Invalid technical indicator" in str(excinfo.value)


def test_calc_ma_with_invalid_input():
    processor = TechnicalProcessor()
    # 空数据测试
    with pytest.raises(ValueError):
        processor.calc_ma(pd.DataFrame(), window=20)
    # 价格列缺失测试
    with pytest.raises(ValueError):
        processor.calc_ma(pd.DataFrame({"volume": [1, 2]}), price_col="close")


@pytest.mark.parametrize("input_data,expected", [
    ([1, 1, 1, 1], 50),  # 无变化
    ([1, 2, 3, 4], 100),  # 全上涨
    ([4, 3, 2, 1], 0)  # 全下跌
])
def test_rsi_edge_cases(input_data, expected):
    data = pd.DataFrame({"close": input_data})
    result = TechnicalProcessor().calc_rsi(data)
    last_rsi = result["RSI"].dropna().iloc[-1]  # 跳过初始NaN值
    assert np.isclose(last_rsi, expected, atol=1e-2)


def test_macd_calculation_edge_cases():
    """测试MACD计算的边界条件"""
    # 1. 极短时间窗口
    data = pd.DataFrame({"close": np.random.rand(5)})
    result = TechnicalProcessor().calc_macd(data, short_window=2, long_window=3)
    assert "MACD_DIF" in result.columns

    # 2. 全零数据
    zero_data = pd.DataFrame({"close": [0] * 10})
    result = TechnicalProcessor().calc_macd(zero_data)
    assert not result.isnull().any().any()


def test_ma_calculation_edge_cases():
    """测试移动平均计算的边界情况"""
    # 空数据测试
    with pytest.raises(ValueError):
        TechnicalProcessor().calc_ma(pd.DataFrame())

    # 窗口大于数据长度
    data = pd.DataFrame({"close": [1, 2, 3]})
    result = TechnicalProcessor().calc_ma(data, window=5)
    assert "MA_5" in result.columns


def test_rsi_extreme_values():
    """测试RSI极端值处理"""
    # 全上涨行情
    data = pd.DataFrame({"close": [10, 11, 12, 13, 14]})
    result = TechnicalProcessor().calc_rsi(data)
    assert result["RSI"].iloc[-1] == 100

    # 全下跌行情
    data = pd.DataFrame({"close": [14, 13, 12, 11, 10]})
    result = TechnicalProcessor().calc_rsi(data)
    assert result["RSI"].iloc[-1] == 0


def test_macd_zero_values():
    """测试全零价格数据场景"""
    data = pd.DataFrame({"close": [0, 0, 0, 0, 0]})
    result = TechnicalProcessor().calc_macd(data)
    assert (result["MACD_DIF"] == 0).all()


@pytest.mark.parametrize("window, expected", [
    (5, [1.0, 1.5, 2.0, 2.5]),  # 窗口5，数据长度4
    (10, [1.0, 1.5, 2.0, 2.5]),  # 窗口10，数据长度4
    (3, [1.0, 1.5, 2.0, 3.0])  # 窗口3，数据长度4
])
def test_moving_average_calculation(window, expected):
    """参数化测试不同窗口大小的移动平均计算"""
    data = pd.DataFrame({"close": [1, 2, 3, 4]})
    result = TechnicalProcessor().calc_ma(data, window=window)
    ma_col = f"MA_{window}"
    pd.testing.assert_series_equal(
        result[ma_col],
        pd.Series(expected, index=data.index),
        check_names=False,
        check_dtype=False
    )


def test_rsi_with_zero_division():
    """测试RSI计算中的除零场景"""
    data = pd.DataFrame({"close": [10, 10, 10, 10]})  # 价格无变化
    result = TechnicalProcessor().calc_rsi(data)
    assert result["RSI"].iloc[-1] == 50  # 应返回中性值


def test_macd_with_invalid_data():
    data = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError) as e:
        TechnicalProcessor().calc_macd(data)
    assert "All price data is invalid" in str(e.value)


def test_invalid_indicator_name():
    """测试无效技术指标名称异常"""
    processor = TechnicalProcessor()
    # 构造包含日期索引的测试数据
    data = pd.DataFrame({
        "close": [1, 2, 3],
        "high": [1.1, 2.2, 3.3],
        "low": [0.9, 1.9, 2.9],
        "volume": [100, 200, 300]
    }, index=pd.date_range(start='2023-01-01', periods=3))  # 明确添加日期索引

    with pytest.raises(ValueError, match="Invalid technical indicator: invalid_indicator"):
        processor.calc_indicators(data, ["invalid_indicator"])


def test_nan_price_handling():
    """测试价格列包含NaN时的异常处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [1, np.nan, 3]})
    with pytest.raises(ValueError, match="价格列包含 NaN 值"):
        processor.calc_ma(data)


def test_zero_length_window():
    """测试窗口为0时的异常处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="window必须大于0"):
        processor.calc_ma(data, window=0)


def test_constant_price_rsi():
    """测试价格不变时的RSI值"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [5] * 20})
    result = processor.calc_rsi(data)
    assert (result["RSI"] == 50).all()  # 应保持中性值


def test_invalid_window_parameter():
    """测试无效窗口参数"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [100] * 20})  # 无波动数据
    with pytest.raises(ValueError) as e:
        processor.calc_ma(data, window=-5)
    assert "必须大于0" in str(e.value)


def test_zero_volatility_rsi():
    """测试零波动时的RSI计算"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [100]*20})  # 无波动数据
    processed = processor.calc_rsi(data)
    assert processed["RSI"].iloc[-1] == 50  # 中性值


def test_calc_rsi_all_zero_gain_loss():
    """测试全零收益/损失时的RSI计算"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [10, 10, 10, 10]})
    result = processor.calc_rsi(data)
    # 应返回中性值50
    assert np.allclose(result["RSI"].iloc[-1], 50, atol=1e-3)

def test_calc_macd_all_nan():
    """测试全NaN价格数据时的MACD计算"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})
    with pytest.raises(ValueError, match="价格数据全为无效值"):
        processor.calc_macd(data)


def test_calc_ma_with_invalid_data():
    """测试无效价格数据的移动平均计算"""
    tp = TechnicalProcessor()
    data = pd.DataFrame({"close": [np.nan, np.nan, np.nan]})

    with pytest.raises(ValueError, match="包含 NaN 值"):
        tp.calc_ma(data, window=5)


def test_calc_rsi_extreme_cases():
    """测试RSI计算的极端情况处理"""
    tp = TechnicalProcessor()
    # 所有价格相同的情况
    data = pd.DataFrame({"close": [10, 10, 10, 10, 10]})
    result = tp.calc_rsi(data)

    assert result["RSI"].iloc[-1] == 50  # 应返回中性值


def test_calc_indicators_missing_columns():
    """测试缺少必要列的技术指标计算"""
    tp = TechnicalProcessor()
    data = pd.DataFrame({"open": [1, 2, 3]})  # 缺少close列

    with pytest.raises(ValueError, match="缺失必要价格列"):
        tp.calc_indicators(data, ["ma"])


# 测试无效窗口大小
def test_invalid_window():
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [1,2,3]})
    with pytest.raises(ValueError):
        processor.calc_ma(data, window=0)

# 测试全NaN价格数据
def test_all_nan_prices():
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [np.nan, np.nan]})
    with pytest.raises(ValueError):
        processor.calc_macd(data)

# 测试极端值分析
def test_extreme_value_analysis():
    data = pd.DataFrame({"close": np.random.rand(100)})
    result = TechnicalProcessor.extreme_value_analysis(data)
    assert "evt_threshold_0.95" in result.columns


def test_macd_with_all_nan_prices():
    """测试全NaN价格数据时的MACD计算"""
    processor = TechnicalProcessor()

    # 创建全NaN数据
    data = pd.DataFrame({
        'close': [np.nan, np.nan, np.nan],
        'high': [np.nan, np.nan, np.nan],
        'low': [np.nan, np.nan, np.nan],
        'volume': [0, 0, 0]
    })

    with pytest.raises(ValueError, match="价格数据全为无效值"):
        processor.calc_macd(data)


def test_rsi_with_flat_prices():
    """测试价格不变时的RSI计算"""
    processor = TechnicalProcessor()

    # 创建价格不变的数据
    data = pd.DataFrame({
        'close': [100, 100, 100, 100],
        'high': [101, 101, 101, 101],
        'low': [99, 99, 99, 99],
        'volume': [1000, 1000, 1000, 1000]
    })

    result = processor.calc_rsi(data)

    # 验证RSI值为50（中性）
    assert all(result['RSI'] == 50)


def test_calc_ma_invalid_window():
    """测试无效窗口参数"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [10, 11, 12]})

    with pytest.raises(ValueError, match="window必须大于0"):
        processor.calc_ma(data, window=0)

    with pytest.raises(ValueError, match="window必须大于0"):
        processor.calc_ma(data, window=-5)


def test_calc_rsi_extreme_values():
    """测试RSI极端值计算"""
    processor = TechnicalProcessor()

    # 全上涨行情
    data_up = pd.DataFrame({"close": [100, 101, 102, 103]})
    result_up = processor.calc_rsi(data_up)
    assert result_up["RSI"].iloc[-1] == 100

    # 全下跌行情
    data_down = pd.DataFrame({"close": [103, 102, 101, 100]})
    result_down = processor.calc_rsi(data_down)
    assert result_down["RSI"].iloc[-1] == 0

    # 无变化行情
    data_flat = pd.DataFrame({"close": [100, 100, 100, 100]})
    result_flat = processor.calc_rsi(data_flat)
    assert result_flat["RSI"].iloc[-1] == 50


def test_calc_macd_zero_input():
    """测试MACD全零输入"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [0, 0, 0, 0]})
    result = processor.calc_macd(data)

    assert all(result["MACD_DIF"] == 0)
    assert all(result["MACD_DEA"] == 0)
    assert all(result["MACD_Histogram"] == 0)


def test_calc_ma_with_invalid_window():
    """测试无效窗口参数处理"""
    processor = TechnicalProcessor()
    data = pd.DataFrame({"close": [1, 2, 3]})

    with pytest.raises(ValueError, match="必须大于0"):
        processor.calc_ma(data, window=0)


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
        with pytest.raises(ValueError, match=".*window must be greater than 0.*"):
            technical_processor.calc_ma(sample_price_data, window=0)

    def test_calc_ma_missing_price_col(self, technical_processor, sample_price_data):
        """测试缺失价格列"""
        with pytest.raises(ValueError, match=".*价格列.*不存在.*"):
            technical_processor.calc_ma(sample_price_data, price_col="invalid")

    def test_calc_ma_nan_values(self, technical_processor, sample_price_data):
        """测试包含NaN值的价格列"""
        invalid_data = sample_price_data.copy()
        invalid_data.loc["2023-01-05", "close"] = np.nan
        with pytest.raises(ValueError, match=".*Price column contains NaN values.*"):
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
        with pytest.raises(ValueError, match=".*Invalid technical indicator.*"):
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

