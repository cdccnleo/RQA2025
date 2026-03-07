#!/usr/bin/env python3
"""
一目均衡表指标计算器
计算一目均衡表的各个组成部分
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class IchimokuCalculator:

    """一目均衡表计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.tenkan_period = self.config.get('tenkan_period', 9)
        self.kijun_period = self.config.get('kijun_period', 26)
        self.senkou_span_b_period = self.config.get('senkou_span_b_period', 52)
        self.displacement = self.config.get('displacement', 26)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算一目均衡表指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含一目均衡表指标的DataFrame
        """
        try:
            if data is None or data.empty:
                logger.warning("输入数据为空")
                return pd.DataFrame()

            result_df = data.copy()

            # 确保必要列存在
            required_columns = ['high', 'low', 'close']
            if not all(col in result_df.columns for col in required_columns):
                logger.error(
                    f"数据缺少必要列: {[col for col in required_columns if col not in result_df.columns]}")
                return result_df

            # 计算转换线 (Tenkan - sen)
            result_df['ichimoku_tenkan'] = self._calculate_tenkan(result_df)

            # 计算基准线 (Kijun - sen)
            result_df['ichimoku_kijun'] = self._calculate_kijun(result_df)

            # 计算前置跨度A (Senkou Span A)
            result_df['ichimoku_senkou_a'] = self._calculate_senkou_span_a(result_df)

            # 计算前置跨度B (Senkou Span B)
            result_df['ichimoku_senkou_b'] = self._calculate_senkou_span_b(result_df)

            # 计算迟行跨度 (Chikou Span)
            result_df['ichimoku_chikou'] = result_df['close'].shift(-self.displacement)

            # 计算云层厚度
            result_df['ichimoku_cloud_thickness'] = (
                result_df['ichimoku_senkou_a'] - result_df['ichimoku_senkou_b']
            ).abs()

            # 计算价格与云层的关系
            result_df['ichimoku_above_cloud'] = result_df['close'] > result_df[[
                'ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
            result_df['ichimoku_below_cloud'] = result_df['close'] < result_df[[
                'ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
            result_df['ichimoku_in_cloud'] = ~(
                result_df['ichimoku_above_cloud'] | result_df['ichimoku_below_cloud'])

            # 计算信号
            result_df = self._generate_signals(result_df)

            logger.info("一目均衡表指标计算完成")
            return result_df

        except Exception as e:
            logger.error(f"一目均衡表计算失败: {e}")
            return data

    def _calculate_tenkan(self, data: pd.DataFrame) -> pd.Series:
        """计算转换线 (Tenkan - sen)"""
        high_max = data['high'].rolling(window=self.tenkan_period).max()
        low_min = data['low'].rolling(window=self.tenkan_period).min()
        return (high_max + low_min) / 2

    def _calculate_kijun(self, data: pd.DataFrame) -> pd.Series:
        """计算基准线 (Kijun - sen)"""
        high_max = data['high'].rolling(window=self.kijun_period).max()
        low_min = data['low'].rolling(window=self.kijun_period).min()
        return (high_max + low_min) / 2

    def _calculate_senkou_span_a(self, data: pd.DataFrame) -> pd.Series:
        """计算前置跨度A (Senkou Span A)"""
        tenkan = self._calculate_tenkan(data)
        kijun = self._calculate_kijun(data)
        return ((tenkan + kijun) / 2).shift(self.displacement)

    def _calculate_senkou_span_b(self, data: pd.DataFrame) -> pd.Series:
        """计算前置跨度B (Senkou Span B)"""
        high_max = data['high'].rolling(window=self.senkou_span_b_period).max()
        low_min = data['low'].rolling(window=self.senkou_span_b_period).min()
        return ((high_max + low_min) / 2).shift(self.displacement)

    def _generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        # 价格突破信号
        data['ichimoku_price_above_tenkan'] = data['close'] > data['ichimoku_tenkan']
        data['ichimoku_tenkan_above_kijun'] = data['ichimoku_tenkan'] > data['ichimoku_kijun']

        # 云层突破信号
        data['ichimoku_price_above_cloud'] = data['close'] > data[[
            'ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        data['ichimoku_price_below_cloud'] = data['close'] < data[[
            'ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)

        # 迟行跨度信号
        data['ichimoku_chikou_above_price'] = data['ichimoku_chikou'] > data['close']

        # 综合买入信号
        data['ichimoku_buy_signal'] = (
            (data['close'] > data[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1))
            & (data['ichimoku_tenkan'] > data['ichimoku_kijun'])
            & (data['ichimoku_chikou'] > data['close'])
        )

        # 综合卖出信号
        data['ichimoku_sell_signal'] = (
            (data['close'] < data[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1))
            & (data['ichimoku_tenkan'] < data['ichimoku_kijun'])
            & (data['ichimoku_chikou'] < data['close'])
        )

        return data
