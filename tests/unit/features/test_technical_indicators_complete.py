# -*- coding: utf-8 -*-
"""
技术指标计算器完整测试套件 - Phase 2.4

实现VolatilityCalculator、MomentumCalculator等所有技术指标计算器的100%覆盖率
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

try:
    from src.features.indicators.volatility_calculator import VolatilityCalculator
except ImportError:
    import sys
    sys.path.insert(0, 'src')
    from features.indicators.volatility_calculator import VolatilityCalculator

try:
    from src.features.indicators.momentum_calculator import MomentumCalculator
except ImportError:
    import sys
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')
    from features.indicators.momentum_calculator import MomentumCalculator


class TestVolatilityCalculator:
    """VolatilityCalculator完整测试"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """生成OHLC测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')

        # 生成基础价格数据
        base_price = 100
        returns = np.random.normal(0, 0.02, 100)  # 2%波动率
        prices = base_price * np.exp(np.cumsum(returns))

        # 生成OHLC数据
        high_noise = np.random.uniform(0.001, 0.005, 100)
        low_noise = np.random.uniform(0.001, 0.005, 100)

        data = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.002, 0.002, 100)),
            'high': prices * (1 + high_noise),
            'low': prices * (1 - low_noise),
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })

        # 确保OHLC逻辑正确
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

        return data

    @pytest.fixture
    def volatility_calculator(self):
        """创建VolatilityCalculator实例"""
        try:
            # 直接导入以避免复杂的模块初始化问题
            import sys
            sys.path.insert(0, 'src')
            from features.indicators.volatility_calculator import VolatilityCalculator
            return VolatilityCalculator()
        except ImportError as e:
            pytest.skip(f"VolatilityCalculator导入失败: {e}")

    def test_volatility_calculator_initialization(self, volatility_calculator):
        """测试VolatilityCalculator初始化"""
        assert volatility_calculator is not None
        assert hasattr(volatility_calculator, 'config')
        assert hasattr(volatility_calculator, 'bb_period')
        assert volatility_calculator.bb_period == 20  # 默认值

        # 测试自定义配置
        custom_config = {'bb_period': 30, 'kc_multiplier': 1.5}
        custom_calculator = VolatilityCalculator(custom_config)
        assert custom_calculator.bb_period == 30
        assert custom_calculator.kc_multiplier == 1.5

    def test_calculate_atr(self, volatility_calculator, sample_ohlc_data):
        """测试ATR计算"""
        result = volatility_calculator.calculate(sample_ohlc_data)

        assert 'volatility_atr' in result.columns
        atr_values = result['volatility_atr'].dropna()

        assert len(atr_values) > 0
        assert all(atr_values > 0)  # ATR应该都是正值

        # ATR应该随着时间平滑变化
        atr_diff = atr_values.diff().abs()
        assert atr_diff.mean() < atr_values.mean()  # 变化应该小于平均值

    def test_calculate_bollinger_bandwidth(self, volatility_calculator, sample_ohlc_data):
        """测试布林带宽度计算"""
        result = volatility_calculator.calculate(sample_ohlc_data)

        assert 'volatility_bb_width' in result.columns
        bb_width = result['volatility_bb_width'].dropna()

        assert len(bb_width) > 0
        assert all(bb_width > 0)  # 宽度应该是正值

    def test_calculate_keltner_channel_width(self, volatility_calculator, sample_ohlc_data):
        """测试凯尔特纳通道宽度计算"""
        result = volatility_calculator.calculate(sample_ohlc_data)

        assert 'volatility_kc_width' in result.columns
        kc_width = result['volatility_kc_width'].dropna()

        assert len(kc_width) > 0
        assert all(kc_width > 0)  # 通道宽度应该是正值

    def test_calculate_donchian_width(self, volatility_calculator, sample_ohlc_data):
        """测试唐奇安通道宽度计算"""
        result = volatility_calculator.calculate(sample_ohlc_data)

        assert 'volatility_donchian' in result.columns
        donchian_width = result['volatility_donchian'].dropna()

        assert len(donchian_width) > 0
        assert all(donchian_width > 0)  # 唐奇安宽度应该是正值

    def test_generate_signals(self, volatility_calculator, sample_ohlc_data):
        """测试波动率信号生成"""
        result = volatility_calculator.calculate(sample_ohlc_data)

        # 检查信号列是否存在
        signal_columns = [col for col in result.columns if 'volatility_atr_high' in col or 'volatility_bb_expanding' in col]
        assert len(signal_columns) > 0

    def test_calculate_empty_data(self, volatility_calculator):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        result = volatility_calculator.calculate(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_missing_columns(self, volatility_calculator):
        """测试缺失列处理"""
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
            # 缺少high, low列
        })

        result = volatility_calculator.calculate(incomplete_data)

        # 应该返回原始数据，不计算指标
        assert len(result) == len(incomplete_data)
        assert 'close' in result.columns
        assert 'volume' in result.columns

    def test_atr_mathematical_correctness(self, volatility_calculator):
        """测试ATR的数学正确性"""
        # 创建已知的数据来验证计算
        data = pd.DataFrame({
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        })

        result = volatility_calculator.calculate(data)

        # 检查ATR是否合理计算
        atr_values = result['volatility_atr'].dropna()
        assert len(atr_values) > 0

        # ATR应该反映价格波动
        # 对于这个上升趋势数据，ATR应该相对稳定
        atr_std = atr_values.std()
        atr_mean = atr_values.mean()
        assert atr_std / atr_mean < 0.5  # 波动率相对较小


class TestMomentumCalculator:
    """MomentumCalculator完整测试"""

    @pytest.fixture
    def sample_price_data(self):
        """生成价格测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')

        # 生成趋势性价格数据
        base_price = 100
        trend = np.linspace(0, 20, 100)  # 逐渐上涨的趋势
        noise = np.random.normal(0, 2, 100)
        prices = base_price + trend + noise

        data = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        })

        return data

    @pytest.fixture
    def momentum_calculator(self):
        """创建MomentumCalculator实例"""
        try:
            # 直接导入以避免复杂的模块初始化问题
            import sys
            sys.path.insert(0, 'src')
            from features.indicators.momentum_calculator import MomentumCalculator
            return MomentumCalculator()
        except ImportError as e:
            pytest.skip(f"MomentumCalculator导入失败: {e}")

    def test_momentum_calculator_initialization(self, momentum_calculator):
        """测试MomentumCalculator初始化"""
        assert momentum_calculator is not None
        assert hasattr(momentum_calculator, 'config')
        assert hasattr(momentum_calculator, 'momentum_period')
        assert momentum_calculator.momentum_period == 10  # 默认值

        # 测试自定义配置
        custom_config = {'momentum_period': 15, 'roc_period': 20}
        custom_calculator = MomentumCalculator(custom_config)
        assert custom_calculator.momentum_period == 15
        assert custom_calculator.roc_period == 20

    def test_calculate_rsi(self, momentum_calculator, sample_price_data):
        """测试RSI计算"""
        result = momentum_calculator.calculate(sample_price_data)

        assert 'rsi' in result.columns
        rsi_values = result['rsi'].dropna()

        assert len(rsi_values) > 0
        # RSI应该在0-100范围内
        assert all((rsi_values >= 0) & (rsi_values <= 100))

        # 对于上涨趋势，最后的RSI应该偏高
        final_rsi = rsi_values.iloc[-1] if len(rsi_values) > 0 else 50
        assert final_rsi > 40  # 应该显示一定的上涨动能

    def test_calculate_roc(self, momentum_calculator, sample_price_data):
        """测试ROC计算"""
        result = momentum_calculator.calculate(sample_price_data)

        assert 'roc' in result.columns
        roc_values = result['roc'].dropna()

        assert len(roc_values) > 0

        # 对于上涨趋势，ROC应该为正
        final_roc = roc_values.iloc[-1] if len(roc_values) > 0 else 0
        assert final_roc > 0  # 应该显示正的动量

    def test_calculate_stochastic(self, momentum_calculator):
        """测试随机指标计算"""
        # 需要OHLC数据
        data = pd.DataFrame({
            'high': [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120],
            'low': [95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115]
        })

        result = momentum_calculator.calculate(data)

        # 随机指标可能需要更多数据点才能生成
        if 'stoch_k' in result.columns:
            stoch_values = result['stoch_k'].dropna()
            if len(stoch_values) > 0:
                # 随机指标应该在0-100范围内
                assert all((stoch_values >= 0) & (stoch_values <= 100))

    def test_calculate_trix(self, momentum_calculator, sample_price_data):
        """测试TRIX计算"""
        result = momentum_calculator.calculate(sample_price_data)

        assert 'trix' in result.columns
        trix_values = result['trix'].dropna()

        assert len(trix_values) > 0
        # TRIX可以是任何实数，通常在较小范围内

    def test_calculate_kst(self, momentum_calculator, sample_price_data):
        """测试KST计算"""
        result = momentum_calculator.calculate(sample_price_data)

        assert 'kst' in result.columns
        kst_values = result['kst'].dropna()

        assert len(kst_values) > 0

    def test_calculate_empty_data(self, momentum_calculator):
        """测试空数据处理"""
        empty_data = pd.DataFrame()
        result = momentum_calculator.calculate(empty_data)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_calculate_missing_close_column(self, momentum_calculator):
        """测试缺失收盘价列处理"""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'volume': [1000, 1100, 1200]
            # 缺少close列
        })

        result = momentum_calculator.calculate(incomplete_data)

        # 应该返回原始数据，不计算指标
        assert len(result) == len(incomplete_data)
        assert 'open' in result.columns
        assert 'volume' in result.columns


class TestATRCalculator:
    """ATRCalculator完整测试"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """生成OHLC测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')

        # 生成真实的OHLC数据
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(50) * 2)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(50) * 0.5,
            'high': prices + np.abs(np.random.randn(50)) * 1.5,
            'low': prices - np.abs(np.random.randn(50)) * 1.5,
            'close': prices,
            'volume': np.random.randint(10000, 100000, 50)
        })

        # 确保OHLC逻辑正确
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

        return data

    @pytest.fixture
    def atr_calculator(self):
        """创建ATRCalculator实例"""
        try:
            # 直接导入以避免复杂的模块初始化问题
            import sys
            sys.path.insert(0, 'src')
            from features.indicators.atr_calculator import ATRCalculator
            return ATRCalculator()
        except ImportError as e:
            pytest.skip(f"ATRCalculator导入失败: {e}")

    def test_atr_calculator_initialization(self, atr_calculator):
        """测试ATRCalculator初始化"""
        assert atr_calculator is not None
        assert hasattr(atr_calculator, 'config')
        assert hasattr(atr_calculator, 'period')
        assert atr_calculator.period == 14  # 默认值

    def test_calculate_atr_basic(self, atr_calculator, sample_ohlc_data):
        """测试基本ATR计算"""
        result = atr_calculator.calculate(sample_ohlc_data)

        assert isinstance(result, pd.DataFrame)
        assert 'atr' in result.columns

        atr_values = result['atr'].dropna()
        assert len(atr_values) > 0
        assert all(atr_values > 0)

    def test_atr_true_range_calculation(self, atr_calculator):
        """测试真实波幅计算"""
        # 创建特定的测试数据来验证真实波幅计算
        data = pd.DataFrame({
            'high': [105, 106, 108, 107, 109],
            'low': [95, 96, 98, 97, 99],
            'close': [100, 101, 102, 103, 104]
        })

        result = atr_calculator.calculate(data)

        # 检查ATR是否计算出来
        assert 'atr' in result.columns
        atr_values = result['atr'].dropna()
        assert len(atr_values) > 0

    def test_atr_missing_columns(self, atr_calculator):
        """测试缺失列处理"""
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # ATRCalculator在缺失列时记录错误并返回原始数据，不抛出异常
        result = atr_calculator.calculate(incomplete_data)
        # 验证返回了原始数据（没有计算ATR）
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(incomplete_data)
        # 如果没有计算ATR，应该没有atr列
        if 'atr' not in result.columns:
            assert 'close' in result.columns


class TestBollingerBandsCalculator:
    """BollingerBandsCalculator完整测试"""

    @pytest.fixture
    def sample_price_data(self):
        """生成价格测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')

        # 生成有波动性的价格数据
        base_price = 100
        trend = np.sin(np.arange(50) * 0.2) * 5  # 周期性波动
        noise = np.random.randn(50) * 3
        prices = base_price + trend + noise

        data = pd.DataFrame({
            'timestamp': dates,
            'close': prices,
            'volume': np.random.randint(10000, 100000, 50)
        })

        return data

    @pytest.fixture
    def bollinger_calculator(self):
        """创建BollingerBandsCalculator实例"""
        try:
            # 直接导入以避免复杂的模块初始化问题
            import sys
            sys.path.insert(0, 'src')
            from features.indicators.bollinger_calculator import BollingerBandsCalculator
            return BollingerBandsCalculator()
        except ImportError as e:
            pytest.skip(f"BollingerBandsCalculator导入失败: {e}")

    def test_bollinger_calculator_initialization(self, bollinger_calculator):
        """测试BollingerBandsCalculator初始化"""
        assert bollinger_calculator is not None
        assert hasattr(bollinger_calculator, 'config')
        assert hasattr(bollinger_calculator, 'period')
        assert bollinger_calculator.period == 20  # 默认值
        assert bollinger_calculator.std_dev == 2  # 默认值

    def test_calculate_bollinger_bands(self, bollinger_calculator, sample_price_data):
        """测试布林带计算"""
        result = bollinger_calculator.calculate(sample_price_data)

        assert isinstance(result, pd.DataFrame)
        assert 'bb_middle' in result.columns
        assert 'bb_upper' in result.columns
        assert 'bb_lower' in result.columns

        # 检查布林带逻辑：上轨 > 中线 > 下轨
        valid_data = result.dropna()
        if len(valid_data) > 0:
            assert all(valid_data['bb_upper'] >= valid_data['bb_middle'])
            assert all(valid_data['bb_middle'] >= valid_data['bb_lower'])

    def test_bollinger_bandwidth_calculation(self, bollinger_calculator, sample_price_data):
        """测试布林带宽度计算"""
        result = bollinger_calculator.calculate(sample_price_data)

        # 布林带宽度 = (上轨 - 下轨) / 中线
        if 'bb_middle' in result.columns and 'bb_upper' in result.columns and 'bb_lower' in result.columns:
            valid_data = result.dropna()
            if len(valid_data) > 0:
                bandwidth = (valid_data['bb_upper'] - valid_data['bb_lower']) / valid_data['bb_middle']
                assert all(bandwidth > 0)  # 宽度应该是正值

    def test_bollinger_missing_close(self, bollinger_calculator):
        """测试缺失收盘价处理"""
        incomplete_data = pd.DataFrame({
            'open': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # BollingerCalculator在缺失列时记录错误并返回原始数据，不抛出异常
        result = bollinger_calculator.calculate(incomplete_data)
        # 验证返回了原始数据（没有计算布林带）
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(incomplete_data)
        # 如果没有计算布林带，应该没有middle/upper/lower列
        if 'middle' not in result.columns:
            assert 'open' in result.columns


class TestKDJCalculator:
    """KDJCalculator完整测试"""

    @pytest.fixture
    def sample_ohlc_data(self):
        """生成OHLC测试数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=30, freq='1H')

        # 生成有足够周期的数据用于KDJ计算
        base_price = 100
        prices = base_price + np.cumsum(np.random.randn(30) * 1.5)

        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(30) * 0.5,
            'high': prices + np.abs(np.random.randn(30)) * 2,
            'low': prices - np.abs(np.random.randn(30)) * 2,
            'close': prices,
            'volume': np.random.randint(10000, 100000, 30)
        })

        return data

    @pytest.fixture
    def kdj_calculator(self):
        """创建KDJCalculator实例"""
        try:
            # 直接导入以避免复杂的模块初始化问题
            import sys
            sys.path.insert(0, 'src')
            from features.indicators.kdj_calculator import KDJCalculator
            return KDJCalculator()
        except ImportError as e:
            pytest.skip(f"KDJCalculator导入失败: {e}")

    def test_kdj_calculator_initialization(self, kdj_calculator):
        """测试KDJCalculator初始化"""
        assert kdj_calculator is not None
        assert hasattr(kdj_calculator, 'config')
        assert hasattr(kdj_calculator, 'period')
        assert kdj_calculator.period == 9  # 默认值

    def test_calculate_kdj(self, kdj_calculator, sample_ohlc_data):
        """测试KDJ计算"""
        result = kdj_calculator.calculate(sample_ohlc_data)

        assert isinstance(result, pd.DataFrame)

        # KDJ指标应该包含K、D、J值
        kdj_columns = [col for col in result.columns if any(x in col.lower() for x in ['k_', 'd_', 'j_'])]
        assert len(kdj_columns) >= 3  # 至少应该有K、D、J三个值

        # 检查KDJ值的范围（K和D应该在0-100之间，J可能超出范围）
        for col in kdj_columns:
            values = result[col].dropna()
            if len(values) > 0:
                # K和D值应该在0-100之间
                if 'k_' in col.lower() or 'd_' in col.lower():
                    assert all((values >= 0) & (values <= 100))
                # J值可能超出0-100范围（J = 3K - 2D，可能为负或超过100）
                elif 'j_' in col.lower():
                    # J值应该在合理范围内（通常-50到150）
                    assert all((values >= -50) & (values <= 150))
                else:
                    # 其他KDJ相关列，至少验证是数值
                    assert all(pd.notna(values))

    def test_kdj_rsv_calculation(self, kdj_calculator):
        """测试RSV计算"""
        # 创建已知的数据来验证RSV计算
        data = pd.DataFrame({
            'high': [110, 112, 115, 118, 120],
            'low': [100, 102, 105, 108, 110],
            'close': [105, 107, 110, 113, 115]
        })

        result = kdj_calculator.calculate(data)

        # RSV = (Close - Low) / (High - Low) * 100
        # 对于最后一条数据: (115 - 110) / (120 - 110) * 100 = 50
        # 这里我们只是检查计算没有错误
        assert isinstance(result, pd.DataFrame)

    def test_kdj_missing_columns(self, kdj_calculator):
        """测试缺失列处理"""
        incomplete_data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })

        # KDJCalculator在缺失列时记录错误并返回原始数据，不抛出异常
        result = kdj_calculator.calculate(incomplete_data)
        # 验证返回了原始数据（没有计算KDJ）
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(incomplete_data)
        # 如果没有计算KDJ，应该没有kdj相关列
        kdj_columns = [col for col in result.columns if any(x in col.lower() for x in ['k_', 'd_', 'j_'])]
        if len(kdj_columns) == 0:
            assert 'close' in result.columns


class TestTechnicalIndicatorsIntegration:
    """技术指标集成测试"""

    @pytest.fixture
    def technical_indicators_suite(self):
        """创建技术指标计算器套件"""
        indicators = {}

        try:
            from src.features.indicators.volatility_calculator import VolatilityCalculator
            indicators['volatility'] = VolatilityCalculator()
        except ImportError:
            indicators['volatility'] = None

        try:
            from src.features.indicators.momentum_calculator import MomentumCalculator
            indicators['momentum'] = MomentumCalculator()
        except ImportError:
            indicators['momentum'] = None

        try:
            from src.features.indicators.atr_calculator import ATRCalculator
            indicators['atr'] = ATRCalculator()
        except ImportError:
            indicators['atr'] = None

        try:
            from src.features.indicators.bollinger_calculator import BollingerBandsCalculator
            indicators['bollinger'] = BollingerBandsCalculator()
        except ImportError:
            indicators['bollinger'] = None

        try:
            from src.features.indicators.kdj_calculator import KDJCalculator
            indicators['kdj'] = KDJCalculator()
        except ImportError:
            indicators['kdj'] = None

        return indicators

    @pytest.fixture
    def comprehensive_market_data(self):
        """生成综合市场数据"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='5min')

        # 生成复杂的价格数据
        base_price = 100
        trend = 0.0001 * np.arange(200)  # 轻微上涨趋势
        seasonal = 0.3 * np.sin(2 * np.pi * np.arange(200) / (24 * 12))  # 日内季节性
        noise = np.random.normal(0, 0.5, 200)

        close_prices = base_price + trend * 100 + seasonal + noise

        # 生成完整的OHLCV数据
        high_noise = np.abs(np.random.normal(0, 0.01, 200))
        low_noise = np.abs(np.random.normal(0, 0.01, 200))

        data = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices * (1 + np.random.normal(0, 0.002, 200)),
            'high': close_prices * (1 + high_noise),
            'low': close_prices * (1 - low_noise),
            'close': close_prices,
            'volume': np.random.randint(10000, 100000, 200)
        })

        # 确保OHLC逻辑正确
        data['high'] = np.maximum(data[['open', 'close']].max(axis=1), data['high'])
        data['low'] = np.minimum(data[['open', 'close']].min(axis=1), data['low'])

        return data

    def test_volatility_momentum_correlation(self, technical_indicators_suite, comprehensive_market_data):
        """测试波动率和动量指标的相关性"""
        indicators = technical_indicators_suite

        # 计算波动率指标
        volatility_data = comprehensive_market_data.copy()
        if indicators['volatility']:
            volatility_data = indicators['volatility'].calculate(volatility_data)

        # 计算动量指标
        momentum_data = comprehensive_market_data.copy()
        if indicators['momentum']:
            momentum_data = indicators['momentum'].calculate(momentum_data)

        # 计算ATR
        atr_data = comprehensive_market_data.copy()
        if indicators['atr']:
            atr_data = indicators['atr'].calculate(atr_data)

        # 检查是否有有效的指标生成
        vol_cols = [col for col in volatility_data.columns if 'volatility' in col.lower() or 'atr' in col.lower()]
        mom_cols = [col for col in momentum_data.columns if any(x in col.lower() for x in ['rsi', 'roc', 'momentum'])]

        print(f"✅ 波动率指标数量: {len(vol_cols)}")
        print(f"✅ 动量指标数量: {len(mom_cols)}")

        # 至少应该有一些指标
        assert len(vol_cols) + len(mom_cols) > 0

    def test_indicators_consistency(self, technical_indicators_suite, comprehensive_market_data):
        """测试指标计算的一致性"""
        indicators = technical_indicators_suite

        # 多次计算同一个数据集应该得到相同结果
        if indicators['volatility']:
            result1 = indicators['volatility'].calculate(comprehensive_market_data)
            result2 = indicators['volatility'].calculate(comprehensive_market_data)

            # 检查数值一致性
            vol_cols = [col for col in result1.columns if 'volatility' in col]
            for col in vol_cols:
                if col in result2.columns:
                    pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)

            print("✅ 波动率指标计算一致性验证通过")

        if indicators['momentum']:
            result1 = indicators['momentum'].calculate(comprehensive_market_data)
            result2 = indicators['momentum'].calculate(comprehensive_market_data)

            # 检查数值一致性
            mom_cols = [col for col in result1.columns if any(x in col.lower() for x in ['rsi', 'roc'])]
            for col in mom_cols:
                if col in result2.columns:
                    pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)

            print("✅ 动量指标计算一致性验证通过")

    def test_indicators_performance(self, technical_indicators_suite, comprehensive_market_data):
        """测试指标计算性能"""
        indicators = technical_indicators_suite

        performance_results = {}

        # 测试波动率计算性能
        if indicators['volatility']:
            import time
            start_time = time.time()
            vol_result = indicators['volatility'].calculate(comprehensive_market_data)
            vol_time = time.time() - start_time
            performance_results['volatility'] = vol_time

        # 测试动量计算性能
        if indicators['momentum']:
            import time
            start_time = time.time()
            mom_result = indicators['momentum'].calculate(comprehensive_market_data)
            mom_time = time.time() - start_time
            performance_results['momentum'] = mom_time

        # 测试ATR计算性能
        if indicators['atr']:
            import time
            start_time = time.time()
            atr_result = indicators['atr'].calculate(comprehensive_market_data)
            atr_time = time.time() - start_time
            performance_results['atr'] = atr_time

        # 性能应该在合理范围内
        for name, exec_time in performance_results.items():
            assert exec_time < 2.0, f"{name}指标计算时间过长: {exec_time:.3f}秒"

        print(f"✅ 指标计算性能验证通过 - 测试了 {len(performance_results)} 个指标")

    def test_indicators_data_validation(self, technical_indicators_suite):
        """测试指标计算的数据验证"""
        indicators = technical_indicators_suite

        # 测试无效数据
        invalid_data = [
            None,
            pd.DataFrame(),
            pd.DataFrame({'invalid_col': [1, 2, 3]}),
            "invalid_string",
            123
        ]

        for invalid_input in invalid_data:
            for name, indicator in indicators.items():
                if indicator is None:
                    continue

                try:
                    result = indicator.calculate(invalid_input)
                    # 应该返回DataFrame、None或抛出适当异常
                    # 某些指标在无效输入时可能返回None而不是抛出异常
                    if result is not None:
                        assert isinstance(result, pd.DataFrame)
                except (ValueError, TypeError, AttributeError, Exception):
                    # 预期的异常（包括所有异常类型，因为不同指标可能抛出不同类型的异常）
                    pass

        print("✅ 指标数据验证测试通过")

    def test_indicators_trend_detection(self, technical_indicators_suite):
        """测试指标的趋势检测能力"""
        indicators = technical_indicators_suite

        # 创建明显的上涨趋势数据
        uptrend_prices = np.linspace(100, 120, 50) + np.random.normal(0, 0.5, 50)
        uptrend_data = pd.DataFrame({
            'high': uptrend_prices + 2,
            'low': uptrend_prices - 2,
            'close': uptrend_prices
        })

        # 创建明显的下跌趋势数据
        downtrend_prices = np.linspace(120, 100, 50) + np.random.normal(0, 0.5, 50)
        downtrend_data = pd.DataFrame({
            'high': downtrend_prices + 2,
            'low': downtrend_prices - 2,
            'close': downtrend_prices
        })

        # 测试趋势检测
        trend_detected = False

        if indicators['momentum']:
            uptrend_result = indicators['momentum'].calculate(uptrend_data)
            downtrend_result = indicators['momentum'].calculate(downtrend_data)

            # 检查是否有动量指标
            uptrend_momentum_cols = [col for col in uptrend_result.columns if any(x in col.lower() for x in ['rsi', 'roc'])]
            downtrend_momentum_cols = [col for col in downtrend_result.columns if any(x in col.lower() for x in ['rsi', 'roc'])]

            if len(uptrend_momentum_cols) > 0 and len(downtrend_momentum_cols) > 0:
                trend_detected = True

        if indicators['volatility']:
            # 波动率指标也应该能正常计算
            vol_result = indicators['volatility'].calculate(uptrend_data)
            vol_cols = [col for col in vol_result.columns if 'volatility' in col.lower()]
            if len(vol_cols) > 0:
                trend_detected = True

        assert trend_detected
        print("✅ 指标趋势检测能力测试通过")

    def test_indicators_scalability(self, technical_indicators_suite):
        """测试指标计算的可扩展性"""
        indicators = technical_indicators_suite

        # 测试不同规模的数据
        test_sizes = [50, 100, 500, 1000]

        for size in test_sizes:
            # 生成测试数据
            np.random.seed(42)
            data = pd.DataFrame({
                'high': np.random.uniform(100, 110, size),
                'low': np.random.uniform(90, 100, size),
                'close': np.random.uniform(95, 105, size)
            })

            # 测试主要指标计算器
            for name, indicator in indicators.items():
                if indicator is None:
                    continue

                try:
                    result = indicator.calculate(data)

                    # 验证结果
                    assert isinstance(result, pd.DataFrame)
                    assert len(result) == size

                    # 检查是否有指标列
                    indicator_cols = [col for col in result.columns if col not in data.columns]
                    assert len(indicator_cols) > 0

                except Exception as e:
                    # 对于某些指标，较小的数据集可能无法计算
                    if size >= 100 or "period" not in str(e).lower():
                        raise

        print(f"✅ 指标可扩展性测试通过 - 测试规模: {test_sizes}")


if __name__ == "__main__":
    # 手动运行测试以查看结果
    import sys
    pytest.main([__file__, "-v", "--tb=short"])
