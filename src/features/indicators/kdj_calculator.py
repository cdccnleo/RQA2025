#!/usr/bin/env python3
"""
KDJ指标计算器
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class KDJCalculator:

    """KDJ指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.period = self.config.get('period', 9)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算KDJ指标

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含KDJ指标的DataFrame
        """
        try:
            if not all(col in data.columns for col in ['high', 'low', 'close']):
                raise ValueError("数据缺少必需的OHLC列")

            df = data.copy()

            # 计算RSV (Raw Stochastic Value)
            for i in range(len(df)):
                if i >= self.period - 1:
                    # 获取周期内最高价和最低价
                    high_period = df['high'].iloc[i - self.period + 1:i + 1].max()
                    low_period = df['low'].iloc[i - self.period + 1:i + 1].min()
                    close_current = df['close'].iloc[i]

                    if high_period != low_period:
                        rsv = (close_current - low_period) / (high_period - low_period) * 100
                    else:
                        rsv = 50  # 当最高价等于最低价时，RSV设为50
                else:
                    rsv = 50  # 前period - 1个周期设为50

                df.loc[df.index[i], 'rsv'] = rsv

            # 计算K、D、J值
            df['k_value'] = df['rsv'].ewm(alpha=1 / 3, adjust=False).mean()
            df['d_value'] = df['k_value'].ewm(alpha=1 / 3, adjust=False).mean()
            df['j_value'] = 3 * df['k_value'] - 2 * df['d_value']

            # 重命名列
            df = df.rename(columns={
                'k_value': 'kdj_k',
                'd_value': 'kdj_d',
                'j_value': 'kdj_j'
            })

            logger.info(f"KDJ指标计算完成，周期: {self.period}")
            return df

        except Exception as e:
            logger.error(f"KDJ指标计算失败: {e}")
            return data
