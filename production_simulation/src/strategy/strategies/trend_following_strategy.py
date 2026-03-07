#!/usr/bin/env python3
"""
均线交叉策略
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class TrendFollowingStrategy(BaseStrategy):

    """趋势跟踪策略 - 均线交叉"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__(config)
        self.short_period = self.config.get('short_period', 5)
        self.long_period = self.config.get('long_period', 20)

    def generate_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """生成均线交叉信号"""
        try:
            if len(data) < self.long_period:
                return {'signal': 'HOLD', 'reason': '数据不足'}

            # 计算移动平均线
            data_copy = data.copy()
            data_copy['sma_short'] = data_copy['close'].rolling(window=self.short_period).mean()
            data_copy['sma_long'] = data_copy['close'].rolling(window=self.long_period).mean()

            # 获取最新数据
            latest = data_copy.iloc[-1]
            prev = data_copy.iloc[-2] if len(data_copy) > 1 else latest

            current_short = latest['sma_short']
            current_long = latest['sma_long']
            prev_short = prev['sma_short']
            prev_long = prev['sma_long']

            # 生成信号
            if pd.notna(current_short) and pd.notna(current_long) and pd.notna(prev_short) and pd.notna(prev_long):
                # 金叉：短期均线上穿长期均线
                if prev_short <= prev_long and current_short > current_long:
                    return {
                        'signal': 'BUY',
                        'reason': f'金叉信号: {self.short_period}日线穿过{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }
                # 死叉：短期均线下穿长期均线
                elif prev_short >= prev_long and current_short < current_long:
                    return {
                        'signal': 'SELL',
                        'reason': f'死叉信号: {self.short_period}日线跌破{self.long_period}日线',
                        'short_ma': current_short,
                        'long_ma': current_long
                    }

            return {'signal': 'HOLD', 'reason': '无明确交叉信号'}

        except Exception as e:
            self.logger.error(f"均线交叉策略信号生成失败: {e}")
            return {'signal': 'HOLD', 'reason': f'错误: {str(e)}'}
