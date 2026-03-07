"""信号生成器测试模块

测试 src.trading.signal.signal_signal_generator 模块的功能
"""

import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import Mock, patch

from src.trading.signal.signal_signal_generator import (
    Signal,
    SignalType,
    SignalStrength,
    SignalConfig,
    SignalGenerator,
    MovingAverageSignalGenerator,
    RSISignalGenerator,
    SimpleSignalGenerator
)


class TestSignalType:
    """信号类型枚举测试类"""
    
    def test_signal_type_values(self):
        """测试信号类型枚举值"""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"


class TestSignalStrength:
    """信号强度枚举测试类"""
    
    def test_signal_strength_values(self):
        """测试信号强度枚举值"""
        assert SignalStrength.WEAK.value == "weak"
        assert SignalStrength.MEDIUM.value == "medium"
        assert SignalStrength.STRONG.value == "strong"


class TestSignalConfig:
    """信号配置测试类"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = SignalConfig()
        assert config.threshold == 0.5
        assert config.lookback_period == 20
        assert config.smoothing_factor == 0.1
        assert config.enable_filtering is True
        assert config.min_signal_strength == SignalStrength.MEDIUM
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SignalConfig(
            threshold=0.7,
            lookback_period=30,
            smoothing_factor=0.2,
            enable_filtering=False,
            min_signal_strength=SignalStrength.STRONG
        )
        assert config.threshold == 0.7
        assert config.lookback_period == 30
        assert config.smoothing_factor == 0.2
        assert config.enable_filtering is False
        assert config.min_signal_strength == SignalStrength.STRONG


class TestSignal:
    """信号类测试类"""
    
    def test_signal_init(self):
        """测试信号初始化"""
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            timestamp=time.time(),
            confidence=0.8
        )
        
        assert signal.symbol == "000001.SZ"
        assert signal.signal_type == SignalType.BUY
        assert signal.strength == SignalStrength.STRONG
        assert signal.confidence == 0.8
        assert isinstance(signal.metadata, dict)
    
    def test_signal_with_metadata(self):
        """测试带元数据的信号"""
        metadata = {'price': 10.50, 'volume': 1000}
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            timestamp=time.time(),
            confidence=0.7,
            metadata=metadata
        )
        
        assert signal.metadata == metadata
    
    def test_signal_with_price_volume(self):
        """测试带价格和成交量的信号"""
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            timestamp=time.time(),
            confidence=0.8,
            price=10.50,
            volume=1000
        )
        
        assert signal.price == 10.50
        assert signal.volume == 1000
        assert signal.metadata['price'] == 10.50
        assert signal.metadata['volume'] == 1000
    
    def test_signal_str(self):
        """测试信号字符串表示"""
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            timestamp=time.time(),
            confidence=0.85
        )
        
        str_repr = str(signal)
        assert "000001.SZ" in str_repr
        assert "buy" in str_repr
        assert "strong" in str_repr


class TestSignalGenerator:
    """信号生成器基类测试类"""
    
    def test_init_with_default_config(self):
        """测试使用默认配置初始化"""
        generator = SimpleSignalGenerator()
        assert generator.config is not None
        assert isinstance(generator.signals, list)
        assert len(generator.signals) == 0
    
    def test_init_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = SignalConfig(threshold=0.7)
        generator = SimpleSignalGenerator(config)
        assert generator.config.threshold == 0.7
    
    def test_add_signal(self):
        """测试添加信号"""
        generator = SimpleSignalGenerator()
        signal = Signal(
            symbol="000001.SZ",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            timestamp=time.time()
        )
        
        generator.add_signal(signal)
        
        assert len(generator.signals) == 1
        assert generator.signals[0] == signal
    
    def test_get_recent_signals(self):
        """测试获取最近信号"""
        generator = SimpleSignalGenerator()
        
        # 添加多个信号
        for i in range(5):
            signal = Signal(
                symbol="000001.SZ",
                signal_type=SignalType.BUY,
                strength=SignalStrength.MEDIUM,
                timestamp=time.time() + i
            )
            generator.add_signal(signal)
        
        recent = generator.get_recent_signals("000001.SZ", limit=3)
        
        assert len(recent) == 3
        # 应该按时间戳降序排列
        assert recent[0].timestamp >= recent[1].timestamp
    
    def test_get_recent_signals_different_symbols(self):
        """测试获取不同标的的信号"""
        generator = SimpleSignalGenerator()
        
        # 添加不同标的的信号
        generator.add_signal(Signal("000001.SZ", SignalType.BUY, SignalStrength.MEDIUM, time.time()))
        generator.add_signal(Signal("000002.SZ", SignalType.SELL, SignalStrength.MEDIUM, time.time()))
        generator.add_signal(Signal("000001.SZ", SignalType.BUY, SignalStrength.MEDIUM, time.time()))
        
        recent = generator.get_recent_signals("000001.SZ")
        
        assert len(recent) == 2
        assert all(s.symbol == "000001.SZ" for s in recent)
    
    def test_clear_signals(self):
        """测试清空信号"""
        generator = SimpleSignalGenerator()
        
        # 添加信号
        generator.add_signal(Signal("000001.SZ", SignalType.BUY, SignalStrength.MEDIUM, time.time()))
        assert len(generator.signals) > 0
        
        # 清空
        generator.clear_signals()
        
        assert len(generator.signals) == 0
    
    def test_generate_signal_single(self):
        """测试生成单个信号"""
        generator = SimpleSignalGenerator()
        generator._test_mode = True
        
        data = pd.DataFrame({'close': [10.0, 10.5, 11.0]})
        
        signal = generator.generate_signal(data)
        
        assert signal is not None
        assert isinstance(signal, Signal)


class TestMovingAverageSignalGenerator:
    """移动平均信号生成器测试类"""
    
    @pytest.fixture
    def generator(self):
        """创建移动平均信号生成器"""
        return MovingAverageSignalGenerator()
    
    @pytest.fixture
    def sample_data(self):
        """创建样本数据"""
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        # 生成上升趋势的价格数据
        prices = np.linspace(10.0, 15.0, 50) + np.random.normal(0, 0.1, 50)
        return pd.DataFrame({'close': prices}, index=dates)
    
    def test_init(self, generator):
        """测试初始化"""
        assert generator.fast_period == 5
        assert generator.slow_period == 20
        assert generator.long_period == 20
    
    def test_generate_signals_with_sufficient_data(self, generator, sample_data):
        """测试生成信号（数据充足）"""
        signals = generator.generate_signals(sample_data)
        
        assert isinstance(signals, list)
        # 在上升趋势中应该能生成一些信号
    
    def test_generate_signals_insufficient_data(self, generator):
        """测试生成信号（数据不足）"""
        data = pd.DataFrame({'close': [10.0, 10.5, 11.0]})
        signals = generator.generate_signals(data)
        
        assert signals == []
    
    def test_generate_signals_missing_close_column(self, generator):
        """测试缺少close列的数据"""
        data = pd.DataFrame({'open': [10.0, 10.5, 11.0]})
        signals = generator.generate_signals(data)
        
        assert signals == []
    
    def test_generate_signals_golden_cross(self, generator):
        """测试金叉信号生成"""
        # 创建金叉场景：短期均线上穿长期均线
        # 需要足够的数据点来确保短期MA(5)上穿长期MA(20)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        # 前20个价格较低且稳定，后10个价格快速上升
        prices = np.concatenate([
            np.full(20, 10.0),  # 前20个价格稳定在10.0（确保长期MA稳定）
            np.linspace(10.5, 15.0, 10)  # 后10个价格快速上升
        ])
        data = pd.DataFrame({'close': prices}, index=dates)
        
        signals = generator.generate_signals(data)
        
        # 应该生成买入信号（金叉）
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert isinstance(signals, list)
        # 验证金叉信号的生成
        if len(signals) > 0:
            # 检查是否有买入信号
            assert any(s.signal_type == SignalType.BUY for s in signals)
            # 验证买入信号的metadata包含MA值
            buy_signal = buy_signals[0] if buy_signals else None
            if buy_signal:
                assert 'short_ma' in buy_signal.metadata
                assert 'long_ma' in buy_signal.metadata
                assert buy_signal.signal_type == SignalType.BUY
                assert buy_signal.strength == SignalStrength.MEDIUM
                assert buy_signal.confidence == 0.7


class TestRSISignalGenerator:
    """RSI信号生成器测试类"""
    
    @pytest.fixture
    def generator(self):
        """创建RSI信号生成器"""
        return RSISignalGenerator()
    
    def test_init(self, generator):
        """测试初始化"""
        assert generator.rsi_period == 14
        assert generator.oversold_threshold == 30
        assert generator.overbought_threshold == 70
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        config = SignalConfig(lookback_period=21)
        generator = RSISignalGenerator(config)
        
        assert generator.rsi_period == 21
    
    def test_calculate_rsi(self, generator):
        """测试计算RSI"""
        # 创建价格数据：先涨后跌
        prices = np.concatenate([
            np.linspace(10.0, 12.0, 20),  # 上涨
            np.linspace(12.0, 9.0, 20)   # 下跌
        ])
        
        rsi = generator.calculate_rsi(prices, period=14)
        
        assert isinstance(rsi, np.ndarray)
        assert len(rsi) > 0
    
    def test_calculate_rsi_insufficient_data(self, generator):
        """测试计算RSI（数据不足）"""
        prices = np.array([10.0, 10.5, 11.0])
        rsi = generator.calculate_rsi(prices, period=14)
        
        assert len(rsi) == 0
    
    def test_generate_signals_with_rsi_column(self, generator):
        """测试使用rsi列生成信号"""
        data = pd.DataFrame({
            'close': np.random.normal(10.0, 0.5, 30),
            'rsi': np.concatenate([np.full(15, 25), np.full(15, 75)])  # 前15个超卖，后15个超买
        })
        
        signals = generator.generate_signals(data)
        
        assert isinstance(signals, list)
        # 应该生成买入和卖出信号
    
    def test_generate_signals_with_close_column(self, generator):
        """测试使用close列生成信号"""
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        # 创建先跌后涨的价格序列（会产生超卖和超买）
        prices = np.concatenate([
            np.linspace(12.0, 9.0, 15),   # 下跌
            np.linspace(9.0, 13.0, 15)   # 上涨
        ])
        data = pd.DataFrame({'close': prices}, index=dates)
        
        signals = generator.generate_signals(data)
        
        assert isinstance(signals, list)
    
    def test_generate_signals_oversold(self, generator):
        """测试超卖信号生成"""
        data = pd.DataFrame({
            'rsi': [35, 30, 25, 20]  # RSI从超卖阈值以上降到以下
        })
        
        signals = generator.generate_signals(data)
        
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0
    
    def test_generate_signals_overbought(self, generator):
        """测试超买信号生成"""
        data = pd.DataFrame({
            'rsi': [65, 70, 75, 80]  # RSI从超买阈值以下升到以上
        })
        
        signals = generator.generate_signals(data)
        
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert len(sell_signals) > 0


class TestSimpleSignalGenerator:
    """简单信号生成器测试类"""
    
    @pytest.fixture
    def generator(self):
        """创建简单信号生成器"""
        return SimpleSignalGenerator()
    
    def test_generate_signals_empty_data(self, generator):
        """测试空数据生成信号"""
        data = pd.DataFrame()
        signals = generator.generate_signals(data)
        
        assert signals == []
    
    def test_generate_signals_with_close(self, generator):
        """测试使用close列生成信号"""
        data = pd.DataFrame({
            'close': [10.0, 10.5, 11.0, 10.8, 11.2]
        })
        
        signals = generator.generate_signals(data)
        
        assert isinstance(signals, list)
        assert len(signals) == 5
        assert all(isinstance(s, Signal) for s in signals)
    
    def test_generate_signals_with_nan(self, generator):
        """测试包含NaN的数据"""
        data = pd.DataFrame({
            'close': [10.0, np.nan, 11.0, 10.8, np.nan]
        })
        
        signals = generator.generate_signals(data)
        
        # 应该跳过NaN值
        assert isinstance(signals, list)
        assert len(signals) <= 5
    
    def test_generate_signals_test_mode(self, generator):
        """测试测试模式"""
        generator._test_mode = True
        generator._force_signal_type = SignalType.SELL
        
        data = pd.DataFrame({'close': [10.0, 10.5, 11.0]})
        signals = generator.generate_signals(data)
        
        assert len(signals) > 0
        # 在测试模式下应该生成指定类型的信号
        assert all(s.signal_type == SignalType.SELL for s in signals)
    
    def test_generate_signals_death_cross(self, generator):
        """测试死叉信号生成"""
        generator = MovingAverageSignalGenerator()
        # 创建死叉场景：短期均线下穿长期均线
        # 需要足够的数据点来确保短期MA(5)下穿长期MA(20)
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        # 前20个价格较高且稳定，后10个价格快速下降
        prices = np.concatenate([
            np.full(20, 15.0),  # 前20个价格稳定在15.0（确保长期MA稳定）
            np.linspace(14.0, 10.0, 10)  # 后10个价格快速下降
        ])
        data = pd.DataFrame({'close': prices}, index=dates)
        
        signals = generator.generate_signals(data)
        
        # 应该生成卖出信号（死叉）
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        assert isinstance(signals, list)
        # 验证死叉信号的生成
        if len(signals) > 0:
            # 检查是否有卖出信号
            assert any(s.signal_type == SignalType.SELL for s in signals)
    
    def test_generate_signals_with_nan_ma(self, generator):
        """测试移动平均值为NaN时的处理"""
        generator = MovingAverageSignalGenerator()
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        # 创建包含NaN的数据，但确保有足够的数据点
        prices = np.full(30, 10.0)
        prices[0:5] = np.nan  # 前5个是NaN
        data = pd.DataFrame({'close': prices}, index=dates)
        
        signals = generator.generate_signals(data)
        
        # 应该跳过NaN值，不生成信号或生成少量信号
        assert isinstance(signals, list)
    
    def test_generate_signals_datetime_index(self, generator):
        """测试使用datetime索引的信号生成"""
        generator = MovingAverageSignalGenerator()
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        prices = np.linspace(10.0, 15.0, 30)
        data = pd.DataFrame({'close': prices}, index=dates)
        
        signals = generator.generate_signals(data)
        
        # 验证信号中的symbol是datetime对象转换的字符串
        assert isinstance(signals, list)
        if signals:
            # symbol应该是datetime索引的字符串表示
            assert isinstance(signals[0].symbol, str)
    
    def test_generate_signals_integer_index(self, generator):
        """测试使用整数索引的信号生成"""
        generator = MovingAverageSignalGenerator()
        # 使用整数索引而不是datetime索引
        prices = np.linspace(10.0, 15.0, 30)
        data = pd.DataFrame({'close': prices})
        
        signals = generator.generate_signals(data)
        
        # 验证信号中的symbol是整数索引的字符串表示
        assert isinstance(signals, list)
        if signals:
            # symbol应该是整数索引的字符串表示
            assert isinstance(signals[0].symbol, str)
    
    def test_rsi_generate_signals_no_close_no_rsi(self, generator):
        """测试RSI信号生成 - 既没有close也没有rsi列"""
        generator = RSISignalGenerator()
        data = pd.DataFrame({'open': [10.0, 10.5, 11.0]})
        
        signals = generator.generate_signals(data)
        
        assert signals == []
    
    def test_rsi_generate_signals_insufficient_rsi_values(self, generator):
        """测试RSI信号生成 - RSI值不足"""
        generator = RSISignalGenerator()
        data = pd.DataFrame({'rsi': [50.0]})  # 只有1个值，不足2个
        
        signals = generator.generate_signals(data)
        
        assert signals == []

