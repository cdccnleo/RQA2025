#!/usr/bin/env python3
"""
布林带指标计算器
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class BollingerBandsCalculator:

    """布林带指标计算器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.period = self.config.get('period', 20)
        self.std_dev = self.config.get('std_dev', 2)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算布林带指标

        Args:
            data: 包含收盘价数据的DataFrame

        Returns:
            包含布林带指标的DataFrame
        """
        try:
            if 'close' not in data.columns:
                raise ValueError("数据缺少收盘价列")

            df = data.copy()

            # 计算简单移动平均线 (中线)
            df['bb_middle'] = df['close'].rolling(window=self.period).mean()

            # 计算标准差
            df['bb_std'] = df['close'].rolling(window=self.period).std()

            # 计算上轨和下轨
            df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * self.std_dev)
            df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * self.std_dev)

            # 计算布林带宽度
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # 计算价格相对位置
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

            logger.info(f"布林带指标计算完成，周期: {self.period}, 标准差: {self.std_dev}")
            return df

        except Exception as e:
            logger.error(f"布林带指标计算失败: {e}")
            return data
