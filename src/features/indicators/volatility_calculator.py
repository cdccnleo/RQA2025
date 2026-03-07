#!/usr/bin/env python3
"""
波动率指标计算器
计算各种波动率相关指标
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class VolatilityCalculator:

    """波动率指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.bb_period = self.config.get('bb_period', 20)
        self.kc_period = self.config.get('kc_period', 20)
        self.kc_multiplier = self.config.get('kc_multiplier', 2)
        self.vix_period = self.config.get('vix_period', 30)
        self.atr_period = self.config.get('atr_period', 14)
        self.hv_period = self.config.get('hv_period', 30)
        self.rv_period = self.config.get('rv_period', 20)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含波动率指标的DataFrame
        """
        try:
            if data is None or data.empty:
                logger.warning("输入数据为空")
                return pd.DataFrame()

            result_df = data.copy()

            if not self._ensure_required_columns(result_df):
                return result_df

            # 计算核心指标
            additions = [
                self._calculate_bollinger_bands(result_df),
                self._calculate_keltner_channels(result_df),
                self._calculate_atr(result_df, self.atr_period),
                self._calculate_historical_volatility(result_df),
                self._calculate_parkinson_volatility(result_df),
                self._calculate_garman_klass_volatility(result_df),
                self._calculate_yang_zhang_volatility(result_df),
                self._calculate_realized_volatility(result_df),
            ]

            for addition in additions:
                for column in addition.columns:
                    result_df[column] = addition[column]

            # 追加派生宽度与通道指标
            if 'ATR' in result_df:
                result_df['volatility_atr'] = result_df['ATR']
            else:
                result_df['volatility_atr'] = np.nan
            result_df['volatility_bb_width'] = self._calculate_bollinger_bandwidth(result_df)
            result_df['volatility_kc_width'] = self._calculate_keltner_channel_width(result_df)
            result_df['volatility_donchian'] = self._calculate_donchian_width(result_df)

            # 生成波动率信号
            result_df = self._generate_signals(result_df)

            logger.info("波动率指标计算完成")
            return result_df

        except Exception as e:
            logger.error(f"波动率指标计算失败: {e}")
            return data

    def _ensure_required_columns(self, data: pd.DataFrame) -> bool:
        required_columns = ['high', 'low', 'close']
        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            logger.error(f"数据缺少必要列: {missing}")
            return False
        return True

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算ATR (Average True Range)"""
        high = data['high']
        low = data['low']
        close = data['close']

        # 计算True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = true_range.rolling(window=period).mean()
        return pd.DataFrame({'ATR': atr})

    def _calculate_bollinger_bands(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算布林带（上、中、下轨）"""
        close = data['close']
        sma = close.rolling(window=self.bb_period).mean()
        std = close.rolling(window=self.bb_period).std()

        upper = sma + (std * 2)
        lower = sma - (std * 2)

        return pd.DataFrame({
            'BB_Upper': upper,
            'BB_Middle': sma,
            'BB_Lower': lower
        })

    def _calculate_bollinger_bandwidth(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带宽度"""
        if not {'BB_Upper', 'BB_Lower', 'BB_Middle'}.issubset(data.columns):
            bands = self._calculate_bollinger_bands(data)
            data = data.join(bands, how='left')

        denominator = data['BB_Middle'].replace(0, np.nan).abs()
        bandwidth = (data['BB_Upper'] - data['BB_Lower']) / denominator * 100
        return bandwidth

    def _calculate_keltner_channels(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算凯尔特纳通道"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        ma = typical_price.rolling(window=self.kc_period).mean()
        atr_series = self._calculate_atr(data, self.kc_period)['ATR']
        upper = ma + (atr_series * self.kc_multiplier)
        lower = ma - (atr_series * self.kc_multiplier)

        return pd.DataFrame({
            'KC_Upper': upper,
            'KC_Middle': ma,
            'KC_Lower': lower
        })

    def _calculate_keltner_channel_width(self, data: pd.DataFrame) -> pd.Series:
        """计算凯尔特纳通道宽度"""
        if not {'KC_Upper', 'KC_Lower', 'KC_Middle'}.issubset(data.columns):
            channels = self._calculate_keltner_channels(data)
            data = data.join(channels, how='left')

        denominator = data['KC_Middle'].replace(0, np.nan).abs()
        channel_width = (data['KC_Upper'] - data['KC_Lower']) / denominator * 100
        return channel_width

    def _calculate_historical_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算历史波动率"""
        log_returns = np.log(data['close'] / data['close'].shift(1))
        hv = log_returns.rolling(window=self.hv_period).std() * np.sqrt(252)
        return pd.DataFrame({f'HV_{self.hv_period}': hv})

    def _calculate_parkinson_volatility(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """计算帕金森波动率"""
        high_low_ratio = np.log(data['high'] / data['low'])
        pv = (high_low_ratio ** 2).rolling(window=period).mean()
        pv = np.sqrt(pv / (4 * np.log(2)))
        return pd.DataFrame({f'PV_{period}': pv})

    def _calculate_garman_klass_volatility(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """计算Garman-Klass波动率"""
        open_series = data['open'] if 'open' in data.columns else data['close']
        log_hl = np.log(data['high'] / data['low']) ** 2
        log_co = np.log(data['close'] / open_series) ** 2
        gk = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=period).mean()
        gk = np.sqrt(gk.clip(lower=0))
        return pd.DataFrame({f'GK_{period}': gk})

    def _calculate_yang_zhang_volatility(self, data: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """计算Yang-Zhang波动率"""
        close_prev = data['close'].shift(1)
        o = data['open'] if 'open' in data.columns else data['close']
        h = data['high']
        l = data['low']
        c = data['close']

        log_ho = np.log(h / o)
        log_lo = np.log(l / o)
        log_co = np.log(c / o)
        log_oc = np.log(o / close_prev.replace(0, np.nan))

        rs = (log_ho * log_lo)
        close_vol = log_co ** 2
        open_vol = log_oc ** 2

        yz = (open_vol.rolling(period).mean() +
              0.34 * close_vol.rolling(period).mean() +
              0.64 * rs.rolling(period).mean())
        yz = np.sqrt(yz.clip(lower=0))
        return pd.DataFrame({f'YZ_{period}': yz})

    def _calculate_realized_volatility(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算已实现波动率"""
        log_returns = np.log(data['close'] / data['close'].shift(1)).fillna(0)
        realized = np.sqrt((log_returns ** 2).rolling(window=self.rv_period).sum())
        return pd.DataFrame({f'RV_{self.rv_period}': realized})

    def _calculate_donchian_width(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算唐奇安通道宽度"""
        high = data['high']
        low = data['low']

        upper_channel = high.rolling(window=period).max()
        lower_channel = low.rolling(window=period).min()

        denominator = lower_channel.replace(0, np.nan)
        channel_width = (upper_channel - lower_channel) / denominator * 100
        return channel_width

    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成波动率信号"""
        # ATR信号
        if 'volatility_atr' in data:
            atr = data['volatility_atr']
            atr_ma = atr.rolling(window=20).mean()
            data['volatility_atr_high'] = atr > atr_ma * 1.5
            data['volatility_atr_low'] = atr < atr_ma * 0.5
        else:
            data['volatility_atr_high'] = False
            data['volatility_atr_low'] = False

        # 布林带宽度信号
        if 'volatility_bb_width' in data:
            bb_width = data['volatility_bb_width']
            bb_width_ma = bb_width.rolling(window=20).mean()
            data['volatility_bb_expanding'] = bb_width > bb_width_ma
            data['volatility_bb_contracting'] = bb_width < bb_width_ma
        else:
            data['volatility_bb_expanding'] = False
            data['volatility_bb_contracting'] = False

        # 凯尔特纳通道宽度信号
        if 'volatility_kc_width' in data:
            kc_width = data['volatility_kc_width']
            kc_width_ma = kc_width.rolling(window=20).mean()
            data['volatility_kc_expanding'] = kc_width > kc_width_ma
            data['volatility_kc_contracting'] = kc_width < kc_width_ma
        else:
            data['volatility_kc_expanding'] = False
            data['volatility_kc_contracting'] = False

        # 唐奇安通道宽度信号
        if 'volatility_donchian' in data:
            donchian_width = data['volatility_donchian']
            donchian_ma = donchian_width.rolling(window=20).mean()
            data['volatility_donchian_high'] = donchian_width > donchian_ma * 1.5
        else:
            data['volatility_donchian_high'] = False

        # 综合波动率信号
        data['volatility_high'] = (
            data['volatility_atr_high']
            & data['volatility_bb_expanding']
            & data['volatility_kc_expanding']
        )

        data['volatility_low'] = (
            data['volatility_atr_low']
            & data['volatility_bb_contracting']
            & data['volatility_kc_contracting']
        )

        # 波动率突破信号
        data['volatility_breakout'] = (
            data['volatility_bb_expanding']
            & (bb_width > bb_width.rolling(window=50).quantile(0.95))
        )

        return data
