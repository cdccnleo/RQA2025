#!/usr/bin/env python3
"""
CCI指标计算器
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class CCICalculator:

    """CCI指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.period = self.config.get('period', 14)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算CCI指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含CCI指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算典型价格 (TP)
            df['tp'] = (df['high'] + df['low'] + df['close']) / 3

            # 计算MA
            df['tp_ma'] = df['tp'].rolling(window=self.period).mean()

            # 计算平均偏差
            df['mean_deviation'] = df['tp'].rolling(window=self.period).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            )

            # 计算CCI
            df['cci'] = (df['tp'] - df['tp_ma']) / (0.015 * df['mean_deviation'])

            # 清理临时列
            df = df.drop(['tp', 'tp_ma', 'mean_deviation'], axis=1)

            logger.info(f"CCI指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"CCI指标计算失败: {e}")
            return data
