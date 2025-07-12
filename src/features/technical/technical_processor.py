"""技术指标处理器模块"""
import numpy as np
import pandas as pd
from numba import jit
from typing import Dict, List, Tuple, Callable, Optional

from ..feature_config import FeatureConfig, FeatureType

class TechnicalProcessor:
    """高性能技术指标处理器"""

    def __init__(self, register_func: Optional[Callable] = None):
        """初始化技术指标处理器
        
        Args:
            register_func: 可选的特征注册函数
        """
        self.register_func = register_func
        if register_func:
            self._register_technical_indicators()

    def _register_technical_indicators(self):
        """注册技术指标到特征引擎"""
        if not self.register_func:
            return

        # RSI指标
        self.register_func(FeatureConfig(
            name="RSI",
            feature_type=FeatureType.TECHNICAL,
            params={"window": 14},
            dependencies=["close"]
        ))

        # MACD指标
        self.register_func(FeatureConfig(
            name="MACD",
            feature_type=FeatureType.TECHNICAL,
            params={"fast":12, "slow":26, "signal":9},
            dependencies=["close"]
        ))

        # 布林带
        self.register_func(FeatureConfig(
            name="BOLL",
            feature_type=FeatureType.TECHNICAL,
            params={"window":20, "num_std":2},
            dependencies=["close"]
        ))

    @staticmethod
    @jit(nopython=True)
    def _calculate_rsi_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """使用numba加速的RSI计算"""
        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:window] = 100. - 100./(1.+rs)

        for i in range(window, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up*(window-1) + upval)/window
            down = (down*(window-1) + downval)/window
            rs = up/down
            rsi[i] = 100. - 100./(1.+rs)

        return rsi

    def calculate_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """计算RSI指标"""
        return self._calculate_rsi_numba(prices, window)

    @staticmethod
    @jit(nopython=True)
    def _calculate_ema_numba(prices: np.ndarray, window: int) -> np.ndarray:
        """使用numba加速的EMA计算"""
        alpha = 2 / (window + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    def calculate_macd(self, prices: np.ndarray,
                      fast: int = 12, slow: int = 26, signal: int = 9
                     ) -> Dict[str, np.ndarray]:
        """计算MACD指标"""
        ema_fast = self._calculate_ema_numba(prices, fast)
        ema_slow = self._calculate_ema_numba(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = self._calculate_ema_numba(macd, signal)
        histogram = macd - signal_line

        return {
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _calculate_bollinger_numba(prices: np.ndarray, window: int, num_std: int
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用numba加速的布林带计算"""
        n = len(prices)
        upper = np.zeros(n)
        middle = np.zeros(n)
        lower = np.zeros(n)

        for i in range(window-1, n):
            window_prices = prices[i-window+1:i+1]
            m = np.mean(window_prices)
            s = np.std(window_prices)

            middle[i] = m
            upper[i] = m + num_std * s
            lower[i] = m - num_std * s

        return upper, middle, lower

    def calculate_bollinger(self, prices: np.ndarray, window: int = 20, num_std: int = 2
                          ) -> Dict[str, np.ndarray]:
        """计算布林带指标"""
        upper, middle, lower = self._calculate_bollinger_numba(prices, window, num_std)
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    def calculate_all_technicals(self, price_data: Dict[str, np.ndarray]
                               ) -> Dict[str, Dict[str, np.ndarray]]:
        """批量计算所有技术指标"""
        closes = price_data['close']

        return {
            'RSI': {'value': self.calculate_rsi(closes)},
            'MACD': self.calculate_macd(closes),
            'BOLL': self.calculate_bollinger(closes)
        }

class AShareTechnicalProcessor(TechnicalProcessor):
    """A股特有技术指标处理器"""

    def __init__(self, register_func: Optional[Callable] = None):
        super().__init__(register_func)
        if register_func:
            self._register_a_share_indicators()

    def _register_a_share_indicators(self):
        """注册A股特有技术指标"""
        if not self.register_func:
            return
            
        # 涨跌停强度指标
        self.register_func(FeatureConfig(
            name="LIMIT_STRENGTH",
            feature_type=FeatureType.TECHNICAL,
            params={"window": 10},
            dependencies=["close", "limit_status"],
            a_share_specific=True
        ))

    def calculate_limit_strength(self, closes: np.ndarray, limit_status: np.ndarray,
                                window: int = 10) -> np.ndarray:
        """计算涨跌停强度指标"""
        strength = np.zeros_like(closes)

        for i in range(window, len(closes)):
            window_status = limit_status[i-window:i]
            up_count = np.sum(window_status == 1)
            down_count = np.sum(window_status == -1)
            strength[i] = (up_count - down_count) / window

        return strength

    def calculate_all_technicals(self, price_data: Dict[str, np.ndarray],
                               a_share_data: Dict[str, np.ndarray] = None
                              ) -> Dict[str, Dict[str, np.ndarray]]:
        """扩展包含A股特有指标的技术指标计算"""
        result = super().calculate_all_technicals(price_data)

        if a_share_data:
            closes = price_data['close']
            limit_status = a_share_data.get('limit_status', np.zeros_like(closes))
            result['LIMIT_STRENGTH'] = {
                'value': self.calculate_limit_strength(closes, limit_status)
            }

        return result
