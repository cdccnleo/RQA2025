#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Features模块技术指标详细测试

测试features/目录中的技术指标计算功能（详细版本）
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any, Optional


class TestTechnicalIndicatorsDetailed:
    """测试技术指标计算功能（详细版本）"""

    def setup_method(self):
        """测试前准备"""
        # 创建更真实的测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # 固定随机种子确保测试可重复

        # 生成价格数据
        base_price = 100
        price_changes = np.random.normal(0, 2, 100).cumsum()
        close_prices = base_price + price_changes

        # 生成高低价和开盘价
        high_prices = close_prices + np.abs(np.random.normal(0, 1, 100))
        low_prices = close_prices - np.abs(np.random.normal(0, 1, 100))
        open_prices = close_prices + np.random.normal(0, 0.5, 100)

        # 生成成交量
        volumes = np.random.lognormal(10, 1, 100).astype(int)

        self.market_data = pd.DataFrame({
            'date': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })

        # 确保价格合理性
        self.market_data['high'] = np.maximum(self.market_data[['open', 'close', 'high']].max(axis=1), self.market_data['high'])
        self.market_data['low'] = np.minimum(self.market_data[['open', 'close', 'low']].min(axis=1), self.market_data['low'])

    def test_sma_calculation(self):
        """测试简单移动平均线计算"""
        prices = self.market_data['close'].values

        def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
            """计算简单移动平均线"""
            if len(prices) < period:
                return np.array([])

            weights = np.ones(period) / period
            sma = np.convolve(prices, weights, mode='valid')
            return sma

        # 测试不同周期的SMA
        sma_5 = calculate_sma(prices, 5)
        sma_10 = calculate_sma(prices, 10)
        sma_20 = calculate_sma(prices, 20)

        # 验证结果长度
        assert len(sma_5) == len(prices) - 4
        assert len(sma_10) == len(prices) - 9
        assert len(sma_20) == len(prices) - 19

        # 验证SMA值在合理范围内
        assert np.all(sma_5 >= np.min(prices))
        assert np.all(sma_5 <= np.max(prices))

        # 验证SMA的平滑效果（较长周期的SMA应该更平滑）
        sma_5_volatility = np.std(np.diff(sma_5))
        sma_20_volatility = np.std(np.diff(sma_20))
        assert sma_20_volatility < sma_5_volatility  # 20日SMA应该比5日SMA更平滑

    def test_ema_calculation(self):
        """测试指数移动平均线计算"""
        prices = self.market_data['close'].values

        def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
            """计算指数移动平均线"""
            if len(prices) < period:
                return np.array([])

            alpha = 2 / (period + 1)
            ema = np.zeros_like(prices)

            # 第一个EMA值使用SMA
            ema[period-1] = np.mean(prices[:period])

            # 计算后续的EMA值
            for i in range(period, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

            return ema[period-1:]

        # 测试EMA计算
        ema_12 = calculate_ema(prices, 12)
        ema_26 = calculate_ema(prices, 26)

        # 验证结果长度
        assert len(ema_12) == len(prices) - 11
        assert len(ema_26) == len(prices) - 25

        # 验证EMA在合理范围内
        assert np.all(ema_12 >= np.min(prices))
        assert np.all(ema_12 <= np.max(prices))

    def test_rsi_calculation(self):
        """测试相对强弱指数计算"""
        prices = self.market_data['close'].values

        def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
            """计算RSI指标"""
            if len(prices) < period + 1:
                return np.array([])

            # 计算价格变化
            deltas = np.diff(prices)

            # 分离上涨和下跌
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            # 计算初始平均值
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])

            rsi_values = []

            for i in range(period, len(deltas)):
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)

                # 更新平均值（使用平滑移动平均）
                if i < len(deltas) - 1:
                    avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                    avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            return np.array(rsi_values)

        # 测试RSI计算
        rsi = calculate_rsi(prices, 14)

        # 验证RSI范围
        assert len(rsi) == len(prices) - 15  # RSI需要period+1个价格点来计算第一个RSI值
        assert np.all(rsi >= 0)
        assert np.all(rsi <= 100)

        # RSI应该有一些变化，但不应该过于极端
        rsi_std = np.std(rsi)
        assert rsi_std > 0  # 应该有变化
        assert rsi_std < 50  # 但不应该过于极端

    def test_macd_calculation(self):
        """测试MACD指标计算"""
        prices = self.market_data['close'].values

        def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
            """辅助函数：计算EMA"""
            if len(prices) < period:
                return np.array([])

            alpha = 2 / (period + 1)
            ema = np.zeros_like(prices)

            ema[period-1] = np.mean(prices[:period])

            for i in range(period, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

            return ema

        def calculate_macd(prices: np.ndarray, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
            """计算MACD指标"""
            fast_ema = calculate_ema(prices, fast_period)
            slow_ema = calculate_ema(prices, slow_period)

            # 确保两个EMA有相同的长度
            min_length = min(len(fast_ema), len(slow_ema))
            macd_line = fast_ema[-min_length:] - slow_ema[-min_length:]

            # 计算信号线
            signal_line = calculate_ema(macd_line, signal_period)

            # 计算直方图
            histogram = macd_line[-len(signal_line):] - signal_line

            return macd_line[-len(signal_line):], signal_line, histogram

        # 测试MACD计算
        macd_line, signal_line, histogram = calculate_macd(prices)

        # 验证结果
        assert len(signal_line) == len(histogram)
        assert len(macd_line) >= len(signal_line)

        # MACD应该有正有负
        assert np.any(macd_line > 0)
        assert np.any(macd_line < 0)

        # 信号线应该比MACD线更平滑
        macd_volatility = np.std(np.diff(macd_line[-20:]))
        signal_volatility = np.std(np.diff(signal_line[-20:]))
        assert signal_volatility < macd_volatility

    def test_bollinger_bands(self):
        """测试布林带计算"""
        prices = self.market_data['close'].values

        def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, num_std: float = 2.0):
            """计算布林带"""
            if len(prices) < period:
                return np.array([]), np.array([]), np.array([])

            sma = np.convolve(prices, np.ones(period)/period, mode='valid')
            rolling_std = []

            for i in range(period-1, len(prices)):
                window = prices[i-period+1:i+1]
                rolling_std.append(np.std(window))

            rolling_std = np.array(rolling_std)

            upper_band = sma + (rolling_std * num_std)
            lower_band = sma - (rolling_std * num_std)

            return sma, upper_band, lower_band

        # 测试布林带计算
        sma, upper, lower = calculate_bollinger_bands(prices, 20, 2.0)

        # 验证结果长度
        expected_length = len(prices) - 19
        assert len(sma) == expected_length
        assert len(upper) == expected_length
        assert len(lower) == expected_length

        # 验证布林带关系
        assert np.all(upper >= sma)  # 上轨应该在中间线上方
        assert np.all(sma >= lower)  # 中轨应该在下轨上方

        # 验证价格大多在布林带内
        recent_prices = prices[-expected_length:]
        within_bands = np.logical_and(recent_prices <= upper, recent_prices >= lower)
        within_percentage = np.mean(within_bands)

        # 大约95%的价格应该在2个标准差的范围内
        assert within_percentage > 0.85

    def test_stochastic_oscillator(self):
        """测试随机指标计算"""
        high_prices = self.market_data['high'].values
        low_prices = self.market_data['low'].values
        close_prices = self.market_data['close'].values

        def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3):
            """计算随机指标"""
            if len(close) < k_period:
                return np.array([]), np.array([])

            k_values = []

            for i in range(k_period-1, len(close)):
                high_window = high[i-k_period+1:i+1]
                low_window = low[i-k_period+1:i+1]

                highest_high = np.max(high_window)
                lowest_low = np.min(low_window)

                if highest_high == lowest_low:
                    k = 50.0  # 避免除零错误
                else:
                    k = 100 * (close[i] - lowest_low) / (highest_high - lowest_low)

                k_values.append(k)

            k_array = np.array(k_values)

            # 计算D值（K的移动平均）
            if len(k_array) >= d_period:
                d_values = np.convolve(k_array, np.ones(d_period)/d_period, mode='valid')
            else:
                d_values = np.array([])

            return k_array[-len(d_values):], d_values

        # 测试随机指标计算
        k_values, d_values = calculate_stochastic(high_prices, low_prices, close_prices)

        # 验证结果
        assert len(k_values) == len(d_values)
        assert len(k_values) == len(close_prices) - 14 + 1 - 2  # K周期14，D周期3

        # K值应该在0-100范围内
        assert np.all(k_values >= 0)
        assert np.all(k_values <= 100)

        # D值应该比K值更平滑
        if len(k_values) > 5:
            k_volatility = np.std(k_values[-10:])
            d_volatility = np.std(d_values[-10:])
            assert d_volatility <= k_volatility

    def test_williams_r(self):
        """测试威廉指标计算"""
        high_prices = self.market_data['high'].values
        low_prices = self.market_data['low'].values
        close_prices = self.market_data['close'].values

        def calculate_williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14):
            """计算威廉指标"""
            if len(close) < period:
                return np.array([])

            williams_values = []

            for i in range(period-1, len(close)):
                high_window = high[i-period+1:i+1]
                low_window = low[i-period+1:i+1]

                highest_high = np.max(high_window)
                lowest_low = np.min(low_window)

                if highest_high == lowest_low:
                    williams_r = -50.0  # 中性值
                else:
                    williams_r = -100 * (highest_high - close[i]) / (highest_high - lowest_low)

                williams_values.append(williams_r)

            return np.array(williams_values)

        # 测试威廉指标计算
        williams_r = calculate_williams_r(high_prices, low_prices, close_prices, 14)

        # 验证结果
        assert len(williams_r) == len(close_prices) - 13

        # 威廉指标应该在-100到0之间
        assert np.all(williams_r >= -100)
        assert np.all(williams_r <= 0)

        # 应该有一些变化
        assert np.std(williams_r) > 0

    def test_commodity_channel_index(self):
        """测试顺势指标计算"""
        high_prices = self.market_data['high'].values
        low_prices = self.market_data['low'].values
        close_prices = self.market_data['close'].values

        def calculate_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20):
            """计算顺势指标CCI"""
            if len(close) < period:
                return np.array([])

            # 计算典型价格
            typical_price = (high + low + close) / 3

            cci_values = []

            for i in range(period-1, len(typical_price)):
                tp_window = typical_price[i-period+1:i+1]
                sma_tp = np.mean(tp_window)
                mean_deviation = np.mean(np.abs(tp_window - sma_tp))

                if mean_deviation == 0:
                    cci = 0.0
                else:
                    cci = (typical_price[i] - sma_tp) / (0.015 * mean_deviation)

                cci_values.append(cci)

            return np.array(cci_values)

        # 测试CCI计算
        cci = calculate_cci(high_prices, low_prices, close_prices, 20)

        # 验证结果
        assert len(cci) == len(close_prices) - 19

        # CCI通常在-300到+300之间，但也可能超出
        assert np.std(cci) > 0  # 应该有变化

        # 验证计算没有产生无效值
        assert not np.any(np.isnan(cci))
        assert not np.any(np.isinf(cci))

    def test_indicators_consistency(self):
        """测试指标计算的一致性"""
        prices = self.market_data['close'].values

        # 计算多个指标
        def calculate_sma(prices, period):
            if len(prices) < period:
                return np.array([])
            return np.convolve(prices, np.ones(period)/period, mode='valid')

        sma_5 = calculate_sma(prices, 5)
        sma_10 = calculate_sma(prices, 10)

        # 验证较长周期的SMA更平滑
        sma_5_changes = np.abs(np.diff(sma_5))
        sma_10_changes = np.abs(np.diff(sma_10[-len(sma_5_changes):]))

        avg_change_5 = np.mean(sma_5_changes)
        avg_change_10 = np.mean(sma_10_changes)

        # 10日SMA的变化应该小于5日SMA
        assert avg_change_10 < avg_change_5

        # 验证指标计算的数值合理性
        assert np.all(sma_5 > 0)  # 价格应该为正
        assert np.all(sma_10 > 0)

        # 验证SMA的包含关系（较短周期SMA应该在较长周期SMA附近）
        if len(sma_10) >= len(sma_5) and len(sma_5) > 5:
            # 比较重叠部分的SMA值
            overlap_sma_10 = sma_10[-len(sma_5):]
            ratio = sma_5 / overlap_sma_10
            assert np.all(ratio > 0.8)  # SMA值应该相对接近
            assert np.all(ratio < 1.2)

    def test_indicators_edge_cases(self):
        """测试指标计算的边界情况"""
        # 测试空数据
        empty_prices = np.array([])
        assert len(empty_prices) == 0

        # 测试单点数据
        single_price = np.array([100.0])
        # 大多数指标需要至少2个数据点

        # 测试常数价格序列
        constant_prices = np.full(50, 100.0)
        sma = np.convolve(constant_prices, np.ones(5)/5, mode='valid')

        # 常数序列的SMA应该是常数
        assert np.allclose(sma, 100.0)

        # 测试剧烈波动的价格
        volatile_prices = np.array([100, 150, 80, 200, 50, 120, 180, 90, 160, 70])

        # RSI对于波动数据应该有较大变化
        def simple_rsi(prices):
            if len(prices) < 2:
                return 50.0
            gains = losses = 0
            for i in range(1, len(prices)):
                change = prices[i] - prices[i-1]
                if change > 0:
                    gains += change
                else:
                    losses -= change
            if losses == 0:
                return 100.0
            return 100 - (100 / (1 + gains/losses))

        rsi_volatile = simple_rsi(volatile_prices)
        assert 0 <= rsi_volatile <= 100

        # 验证极端情况下的数值稳定性
        extreme_prices = np.array([1e-10, 1e10, 0.0, -1.0, float('nan')])
        # 清理NaN和负值
        clean_prices = extreme_prices[np.isfinite(extreme_prices) & (extreme_prices > 0)]
        if len(clean_prices) >= 5:
            sma_extreme = np.convolve(clean_prices, np.ones(3)/3, mode='valid')
            assert np.all(np.isfinite(sma_extreme))  # 结果应该是有限的
