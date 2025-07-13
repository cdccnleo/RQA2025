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
        if len(prices) < window + 1:
            rsi = np.zeros_like(prices)
            rsi.fill(50.0)  # 数据不足时返回中性值
            return rsi
            
        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed >= 0].sum()/window
        down = -seed[seed < 0].sum()/window
        
        rsi = np.zeros_like(prices)
        
        # 处理初始值
        if down == 0:
            if up == 0:
                rsi[:window].fill(50.0)  # 无变化时返回中性值
            else:
                rsi[:window].fill(100.0)  # 只有上涨时返回100
        else:
            rs = up/down
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
            
            if down == 0:
                if up == 0:
                    rsi[i] = 50.0  # 无变化时返回中性值
                else:
                    rsi[i] = 100.0  # 只有上涨时返回100
            else:
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

    def calc_ma(self, df, window=5, price_col='close'):
        """计算移动平均线"""
        if window <= 0:
            raise ValueError("window must be positive")
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        ma = prices.rolling(window=window, min_periods=window).mean()
        return pd.DataFrame({f"MA_{window}": ma}, index=df.index)

    def calc_rsi(self, df, window=14, price_col='close'):
        """计算相对强弱指数RSI"""
        if window <= 0:
            raise ValueError("window must be positive")
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        delta = prices.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.rolling(window=window, min_periods=window).mean()
        roll_down = down.rolling(window=window, min_periods=window).mean()
        rs = roll_up / roll_down
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(~rsi.isna(), np.nan)
        return pd.DataFrame({"RSI": rsi}, index=df.index)

    def calc_macd(self, df, short_window=12, long_window=26, signal_window=9, price_col='close'):
        """计算MACD指标"""
        if short_window <= 0 or long_window <= 0 or signal_window <= 0:
            raise ValueError("window must be positive")
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        ema_short = prices.ewm(span=short_window, adjust=False).mean()
        ema_long = prices.ewm(span=long_window, adjust=False).mean()
        dif = ema_short - ema_long
        dea = dif.ewm(span=signal_window, adjust=False).mean()
        macd_hist = dif - dea
        return pd.DataFrame({
            "MACD_DIF": dif,
            "MACD_DEA": dea,
            "MACD_Histogram": macd_hist
        }, index=df.index)

    def calc_bollinger(self, df, window=20, num_std=2, price_col='close'):
        """计算布林带指标"""
        if window <= 0 or num_std <= 0:
            raise ValueError("window and num_std must be positive")
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        ma = prices.rolling(window=window, min_periods=window).mean()
        std = prices.rolling(window=window, min_periods=window).std()
        upper = ma + num_std * std
        lower = ma - num_std * std
        return pd.DataFrame({
            "BOLL_UPPER": upper,
            "BOLL_MIDDLE": ma,
            "BOLL_LOWER": lower
        }, index=df.index)

    def calc_obv(self, df, price_col='close', volume_col='volume'):
        """计算能量潮指标"""
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        if volume_col not in df.columns:
            raise ValueError(f"volume_col '{volume_col}' not in DataFrame")
        prices = df[price_col]
        volumes = df[volume_col]
        if prices.isnull().any() or volumes.isnull().any():
            raise ValueError("price or volume column contains NaN")
        
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = volumes.iloc[0]
        
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volumes.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volumes.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return pd.DataFrame({"OBV": obv}, index=df.index)

    def calc_atr(self, df, window=14, high_col='high', low_col='low', close_col='close'):
        """计算平均真实波幅"""
        if high_col not in df.columns or low_col not in df.columns or close_col not in df.columns:
            raise ValueError("high, low, or close column not in DataFrame")
        
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        
        if high.isnull().any() or low.isnull().any() or close.isnull().any():
            raise ValueError("high, low, or close column contains NaN")
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=window).mean()
        
        return pd.DataFrame({"ATR": atr}, index=df.index)

    def calc_indicators(self, df, indicators, params=None):
        """批量计算技术指标"""
        if df.empty:
            raise ValueError("DataFrame is empty")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise KeyError("DataFrame must have DatetimeIndex")
        
        if params is None:
            params = {}
        
        results = {}
        for indicator in indicators:
            if indicator == "ma":
                windows = params.get("ma", {}).get("window", [5])
                for window in windows:
                    result = self.calc_ma(df, window=window)
                    results[f"MA_{window}"] = result[f"MA_{window}"]
            elif indicator == "rsi":
                window = params.get("rsi", {}).get("window", 14)
                result = self.calc_rsi(df, window=window)
                results["RSI"] = result["RSI"]
            elif indicator == "macd":
                short_window = params.get("macd", {}).get("short_window", 12)
                long_window = params.get("macd", {}).get("long_window", 26)
                signal_window = params.get("macd", {}).get("signal_window", 9)
                result = self.calc_macd(df, short_window=short_window, long_window=long_window, signal_window=signal_window)
                results.update(result)
            elif indicator == "bollinger":
                window = params.get("bollinger", {}).get("window", 20)
                num_std = params.get("bollinger", {}).get("num_std", 2)
                result = self.calc_bollinger(df, window=window, num_std=num_std)
                results.update(result)
            else:
                raise ValueError(f"Unknown indicator: {indicator}")
        
        return pd.DataFrame(results, index=df.index)

    def calculate_volatility_moments(self, df, price_col='close', window=20):
        """计算波动率矩"""
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        
        returns = prices.pct_change().dropna()
        volatility = returns.rolling(window=window).std()
        skewness = returns.rolling(window=window).skew()
        kurtosis = returns.rolling(window=window).kurt()
        
        return pd.DataFrame({
            "VOLATILITY": volatility,
            "SKEWNESS": skewness,
            "KURTOSIS": kurtosis
        }, index=df.index)

    def extreme_value_analysis(self, df, price_col='close', threshold=2):
        """极值分析"""
        if price_col not in df.columns:
            raise ValueError(f"price_col '{price_col}' not in DataFrame")
        prices = df[price_col]
        if prices.isnull().any():
            raise ValueError("price column contains NaN")
        
        returns = prices.pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        
        upper_threshold = mean_return + threshold * std_return
        lower_threshold = mean_return - threshold * std_return
        
        extreme_high = returns > upper_threshold
        extreme_low = returns < lower_threshold
        
        return pd.DataFrame({
            "EXTREME_HIGH": extreme_high,
            "EXTREME_LOW": extreme_low,
            "UPPER_THRESHOLD": upper_threshold,
            "LOWER_THRESHOLD": lower_threshold
        }, index=df.index)

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
