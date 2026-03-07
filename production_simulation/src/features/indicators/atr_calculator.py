#!/usr/bin/env python3
"""
ATR指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ATRCalculator:

    """ATR指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算ATR指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含ATR指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算真实波幅 (True Range)
            df['high_low'] = df['high'] - df['low']
            df['high_prev_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_prev_close'] = np.abs(df['low'] - df['close'].shift(1))

            df['true_range'] = df[['high_low', 'high_prev_close', 'low_prev_close']].max(axis=1)

            # 计算ATR (使用指数移动平均)
            df['atr'] = df['true_range'].ewm(span=self.period, adjust=False).mean()

            # 计算ATR比率
            df['atr_ratio'] = df['atr'] / df['close']

            # 清理临时列
            df = df.drop(['high_low', 'high_prev_close', 'low_prev_close', 'true_range'], axis=1)

            logger.info(f"ATR指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"ATR指标计算失败: {e}")
            return data
