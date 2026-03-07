#!/usr/bin/env python3
"""
RSI策略
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from ...base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):

    """均值回归策略 - RSI"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.period = self.config.get('period', 14)
        self.overbought_level = self.config.get('overbought', 70)
        self.oversold_level = self.config.get('oversold', 30)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成RSI信号"""
        try:
            if len(data) < self.period + 1:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算RSI
            rsi = self.calculate_rsi(data['close'], self.period)

            # 生成信号
            if rsi <= self.oversold_level:
                return {
                    'signal': 'BUY',
                    'reason': f'RSI超卖信号: {rsi:.2f} <= {self.oversold_level}',
                    'rsi': rsi,
                    'level': 'oversold'
                }
            elif rsi >= self.overbought_level:
                return {
                    'signal': 'SELL',
                    'reason': f'RSI超买信号: {rsi:.2f} >= {self.overbought_level}',
                    'rsi': rsi,
                    'level': 'overbought'
                }
            else:
                return {
                    'signal': 'HOLD',
                    'reason': f'RSI中性: {rsi:.2f}',
                    'rsi': rsi,
                    'level': 'neutral'
                }

        except Exception as e:
            self.logger.error(f"RSI策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}
