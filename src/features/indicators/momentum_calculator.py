#!/usr/bin/env python3
"""
动量指标计算器
计算各种动量相关指标
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MomentumCalculator:

    """动量指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.momentum_period = self.config.get('momentum_period', 10)
        self.roc_period = self.config.get('roc_period', 12)
        self.trix_period = self.config.get('trix_period', 15)
        self.kst_periods = self.config.get('kst_periods', [10, 15, 20, 30])
        self.kst_weights = self.config.get('kst_weights', [1, 2, 3, 4])
        self.rsi_period = self.config.get('rsi_period', 14)
        self.stoch_period = self.config.get('stoch_period', 14)
        self.stoch_signal_period = self.config.get('stoch_signal_period', 3)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算动量指标

        Args:
            data: 包含价格数据的DataFrame

        Returns:
            包含动量指标的DataFrame
        """
        try:
            if data is None or data.empty:
                logger.warning("输入数据为空")
                return pd.DataFrame()

            result_df = data.copy()

            # 确保必要列存在
            if 'close' not in result_df.columns:
                logger.error("数据缺少close列")
                return result_df

            close_price = result_df['close']

            # 计算动量指标
            result_df['momentum'] = self._calculate_momentum(close_price)
            result_df['roc'] = self._calculate_roc(close_price)
            result_df['trix'] = self._calculate_trix(close_price)
            result_df['kst'] = self._calculate_kst(close_price)
            result_df['rsi'] = self._calculate_rsi(close_price)

            if {'high', 'low'}.issubset(result_df.columns):
                stoch_k, stoch_d = self._calculate_stochastic(
                    result_df['high'],
                    result_df['low'],
                    close_price,
                )
                result_df['stoch_k'] = stoch_k
                result_df['stoch_d'] = stoch_d

            # 计算动量信号
            result_df = self._generate_signals(result_df)

            logger.info("动量指标计算完成")
            return result_df

        except Exception as e:
            logger.error(f"动量指标计算失败: {e}")
            return data

    def _calculate_momentum(self, close_price: pd.Series) -> pd.Series:
        """计算动量指标"""
        return close_price - close_price.shift(self.momentum_period)

    def _calculate_roc(self, close_price: pd.Series) -> pd.Series:
        """计算ROC (Rate of Change)"""
        return ((close_price - close_price.shift(self.roc_period))
                / close_price.shift(self.roc_period) * 100)

    def _calculate_trix(self, close_price: pd.Series) -> pd.Series:
        """计算TRIX指标"""
        # TRIX = 3期EMA的3期EMA的ROC
        ema1 = close_price.ewm(span=self.trix_period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.trix_period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.trix_period, adjust=False).mean()

        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1) * 100)
        return trix

    def _calculate_kst(self, close_price: pd.Series) -> pd.Series:
        """计算KST (Know Sure Thing) 指标"""
        # KST = ROC(10) * 1 + ROC(15) * 2 + ROC(20) * 3 + ROC(30) * 4
        roc_values = []

        for i, period in enumerate(self.kst_periods):
            roc = self._calculate_roc(close_price)
            # 应用权重和移动平均
            weighted_roc = roc * self.kst_weights[i]
            window = max(1, period // 3)
            smoothed_roc = weighted_roc.rolling(window=window, min_periods=1).mean()
            roc_values.append(smoothed_roc)

        kst = sum(roc_values)
        return kst

    def _calculate_rsi(self, close_price: pd.Series) -> pd.Series:
        """计算RSI指标 (0-100)"""
        delta = close_price.diff()
        gain = delta.clip(lower=0).rolling(window=self.rsi_period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=self.rsi_period, min_periods=1).mean()

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.clip(lower=0, upper=100)

    def _calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """计算随机指标%K与%D"""
        lowest_low = low.rolling(window=self.stoch_period, min_periods=1).min()
        highest_high = high.rolling(window=self.stoch_period, min_periods=1).max()
        denominator = (highest_high - lowest_low).replace(0, np.nan)

        stoch_k = ((close - lowest_low) / denominator) * 100
        stoch_k = stoch_k.clip(lower=0, upper=100)
        stoch_d = stoch_k.rolling(window=self.stoch_signal_period, min_periods=1).mean()
        return stoch_k, stoch_d

    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成动量信号"""
        # 动量信号
        data['momentum_positive'] = data['momentum'] > 0
        data['momentum_strong'] = data['momentum'] > data['momentum'].rolling(window=20).mean()

        # ROC信号
        data['roc_positive'] = data['roc'] > 0
        data['roc_oversold'] = data['roc'] < -10
        data['roc_overbought'] = data['roc'] > 10

        # TRIX信号
        trix_signal = data['trix'] > 0
        trix_prev = data['trix'].shift(1) <= 0
        data['trix_buy_signal'] = trix_signal & trix_prev

        trix_signal = data['trix'] < 0
        trix_prev = data['trix'].shift(1) >= 0
        data['trix_sell_signal'] = trix_signal & trix_prev

        # KST信号
        kst_signal = data['kst'] > data['kst'].shift(1)
        data['kst_trend_up'] = kst_signal

        # 综合动量信号
        data['momentum_buy_signal'] = (
            (data['momentum'] > 0)
            & (data['roc'] > 0)
            & (data['kst_trend_up'])
        )

        data['momentum_sell_signal'] = (
            (data['momentum'] < 0)
            & (data['roc'] < 0)
            & (~data['kst_trend_up'])
        )

        return data
