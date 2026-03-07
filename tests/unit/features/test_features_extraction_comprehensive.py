#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Features层 - 特征提取综合测试

测试技术指标特征、价格特征、成交量特征、时间序列特征的提取功能
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List


class TestTechnicalIndicatorFeatures:
    """测试技术指标特征提取"""
    
    @pytest.fixture
    def sample_price_data(self):
        """创建示例价格数据"""
        dates = pd.date_range(start='2025-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(90, 110, 100),
            'high': np.random.uniform(100, 120, 100),
            'low': np.random.uniform(80, 100, 100),
            'close': np.random.uniform(90, 110, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })
        return data
    
    def test_extract_moving_average(self, sample_price_data):
        """测试提取移动平均线特征"""
        # 计算5日移动平均
        ma5 = sample_price_data['close'].rolling(window=5).mean()
        
        assert len(ma5) == len(sample_price_data)
        assert not ma5.iloc[4:].isna().any()  # 从第5个值开始不应该有NaN
        assert ma5.iloc[:4].isna().all()  # 前4个值应该是NaN
    
    def test_extract_rsi(self, sample_price_data):
        """测试提取RSI（相对强弱指标）"""
        # 简化的RSI计算
        closes = sample_price_data['close']
        delta = closes.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        assert len(rsi) == len(sample_price_data)
        assert rsi.iloc[14:].between(0, 100).all()  # RSI应该在0-100之间
    
    def test_extract_macd(self, sample_price_data):
        """测试提取MACD指标"""
        close = sample_price_data['close']
        
        # 计算EMA
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        
        # MACD = EMA12 - EMA26
        macd = ema12 - ema26
        
        assert len(macd) == len(sample_price_data)
        assert not macd.isna().all()
    
    def test_extract_bollinger_bands(self, sample_price_data):
        """测试提取布林带"""
        close = sample_price_data['close']
        
        # 计算20日移动平均和标准差
        ma20 = close.rolling(window=20).mean()
        std20 = close.rolling(window=20).std()
        
        # 布林带
        upper_band = ma20 + 2 * std20
        lower_band = ma20 - 2 * std20
        
        assert len(upper_band) == len(sample_price_data)
        assert (upper_band.iloc[20:] >= lower_band.iloc[20:]).all()
    
    def test_extract_atr(self, sample_price_data):
        """测试提取ATR（平均真实波幅）"""
        high = sample_price_data['high']
        low = sample_price_data['low']
        close = sample_price_data['close']
        
        # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        atr = tr.rolling(window=14).mean()
        
        assert len(atr) == len(sample_price_data)
        assert (atr.iloc[14:] > 0).all()


class TestPriceFeatures:
    """测试价格特征提取"""
    
    @pytest.fixture
    def price_data(self):
        """创建价格数据"""
        return pd.Series([100, 102, 101, 105, 103, 108, 107])
    
    def test_extract_price_return(self, price_data):
        """测试提取价格收益率"""
        returns = price_data.pct_change()
        
        assert len(returns) == len(price_data)
        assert returns.iloc[0] is np.nan or pd.isna(returns.iloc[0])
        assert abs(returns.iloc[1] - 0.02) < 0.001  # (102-100)/100 = 0.02
    
    def test_extract_log_return(self, price_data):
        """测试提取对数收益率"""
        log_returns = np.log(price_data / price_data.shift(1))
        
        assert len(log_returns) == len(price_data)
        assert not log_returns.iloc[1:].isna().all()
    
    def test_extract_price_momentum(self, price_data):
        """测试提取价格动量"""
        # 动量 = 当前价格 / N日前价格 - 1
        momentum = (price_data / price_data.shift(3)) - 1
        
        assert len(momentum) == len(price_data)
    
    def test_extract_price_volatility(self, price_data):
        """测试提取价格波动率"""
        returns = price_data.pct_change()
        volatility = returns.rolling(window=5).std()
        
        assert len(volatility) == len(price_data)
        assert (volatility.iloc[5:] >= 0).all()
    
    def test_extract_high_low_range(self):
        """测试提取高低价范围"""
        high = pd.Series([105, 108, 110])
        low = pd.Series([95, 97, 100])
        
        price_range = high - low
        
        assert len(price_range) == 3
        assert (price_range == [10, 11, 10]).all()


class TestVolumeFeatures:
    """测试成交量特征提取"""
    
    @pytest.fixture
    def volume_data(self):
        """创建成交量数据"""
        return pd.Series([1000000, 1200000, 900000, 1500000, 1100000])
    
    def test_extract_volume_ma(self, volume_data):
        """测试提取成交量移动平均"""
        volume_ma = volume_data.rolling(window=3).mean()
        
        assert len(volume_ma) == len(volume_data)
        assert not volume_ma.iloc[2:].isna().any()
    
    def test_extract_volume_ratio(self, volume_data):
        """测试提取成交量比率"""
        volume_ma = volume_data.rolling(window=3).mean()
        volume_ratio = volume_data / volume_ma
        
        assert len(volume_ratio) == len(volume_data)
    
    def test_extract_volume_surge(self, volume_data):
        """测试检测成交量异常"""
        volume_mean = volume_data.mean()
        volume_std = volume_data.std()
        
        # 成交量超过均值+2倍标准差视为异常
        surge_threshold = volume_mean + 2 * volume_std
        volume_surge = volume_data > surge_threshold
        
        assert len(volume_surge) == len(volume_data)
    
    def test_extract_obv(self):
        """测试提取OBV（能量潮）"""
        closes = pd.Series([100, 102, 101, 105])
        volumes = pd.Series([1000, 1200, 900, 1500])
        
        # OBV累计计算
        obv = []
        obv_value = 0
        for i in range(len(closes)):
            if i == 0:
                obv_value = volumes[i]
            elif closes[i] > closes[i-1]:
                obv_value += volumes[i]
            elif closes[i] < closes[i-1]:
                obv_value -= volumes[i]
            obv.append(obv_value)
        
        assert len(obv) == len(closes)


class TestTimeSeriesFeatures:
    """测试时间序列特征提取"""
    
    def test_extract_lag_features(self):
        """测试提取滞后特征"""
        data = pd.Series([10, 20, 30, 40, 50])
        
        lag1 = data.shift(1)
        lag2 = data.shift(2)
        
        assert len(lag1) == len(data)
        assert lag1.iloc[1] == 10  # lag1第2个值是原始第1个值
        assert lag2.iloc[2] == 10  # lag2第3个值是原始第1个值
    
    def test_extract_rolling_statistics(self):
        """测试提取滚动统计特征"""
        data = pd.Series(range(1, 11))  # 1到10
        
        rolling_mean = data.rolling(window=3).mean()
        rolling_std = data.rolling(window=3).std()
        rolling_max = data.rolling(window=3).max()
        rolling_min = data.rolling(window=3).min()
        
        assert rolling_mean.iloc[2] == 2.0  # (1+2+3)/3 = 2
        assert rolling_max.iloc[2] == 3
        assert rolling_min.iloc[2] == 1
    
    def test_extract_time_based_features(self):
        """测试提取基于时间的特征"""
        dates = pd.date_range(start='2025-01-01', periods=10, freq='D')
        df = pd.DataFrame({'date': dates})
        
        # 提取时间特征
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        
        assert df['day_of_week'].between(0, 6).all()
        assert df['month'].iloc[0] == 1
    
    def test_extract_seasonal_features(self):
        """测试提取季节性特征"""
        dates = pd.date_range(start='2025-01-01', periods=365, freq='D')
        df = pd.DataFrame({'date': dates})
        
        # 提取季度特征
        df['quarter'] = df['date'].dt.quarter
        
        assert df['quarter'].between(1, 4).all()
    
    def test_extract_trend_features(self):
        """测试提取趋势特征"""
        data = pd.Series(range(1, 11))  # 上升趋势
        
        # 简单线性趋势
        trend = np.polyfit(range(len(data)), data, deg=1)[0]
        
        assert trend > 0  # 上升趋势


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

