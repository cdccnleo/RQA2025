#!/usr/bin/env python3
"""
威廉指标计算器
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WilliamsCalculator:

    """威廉指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算威廉指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含威廉指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算周期内最高价和最低价
            df['highest_high'] = df['high'].rolling(window=self.period).max()
            df['lowest_low'] = df['low'].rolling(window=self.period).min()

            # 计算威廉指标 %R
            for i in range(len(df)):
                if i >= self.period - 1:
                    highest = df['highest_high'].iloc[i]
                    lowest = df['lowest_low'].iloc[i]
                    close = df['close'].iloc[i]

                    if highest != lowest:
                        williams_r = (highest - close) / (highest - lowest) * (-100)
                    else:
                        williams_r = 0
                else:
                    williams_r = 0

                df.loc[df.index[i], 'williams_r'] = williams_r

            # 清理临时列
            df = df.drop(['highest_high', 'lowest_low'], axis=1)

            logger.info(f"威廉指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"威廉指标计算失败: {e}")
            return data
