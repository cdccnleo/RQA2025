#!/usr/bin/env python3
"""
斐波那契回撤水平计算器
计算斐波那契回撤水平和扩展水平
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class FibonacciCalculator:

    """斐波那契计算器"""

    # 标准斐波那契水平
    FIB_LEVELS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0]
    FIB_EXTENSIONS = [0.0, 0.236, 0.382, 0.500, 0.618, 0.786, 1.0, 1.272, 1.414, 1.618, 2.0, 2.618]

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        self.config = config or {}
        self.lookback_period = self.config.get('lookback_period', 50)
        self.min_swing_length = self.config.get('min_swing_length', 5)
        self.custom_levels = self.config.get('custom_levels', None)

    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算斐波那契水平

        Args:
            data: 包含OHLC数据的DataFrame

        Returns:
            包含斐波那契水平的DataFrame
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

            # 计算摆动点
            swing_highs, swing_lows = self._find_swing_points(result_df)

            # 计算回撤水平
            fib_levels = self._calculate_fibonacci_levels(result_df, swing_highs, swing_lows)

            # 添加到结果DataFrame
            for level_name, level_values in fib_levels.items():
                result_df[level_name] = level_values

            # 计算价格与斐波那契水平的关系
            result_df = self._calculate_price_fib_relationship(result_df, fib_levels)

            logger.info("斐波那契水平计算完成")
            return result_df

        except Exception as e:
            logger.error(f"斐波那契计算失败: {e}")
            return data

    def _find_swing_points(self, data: pd.DataFrame) -> tuple:
        """寻找摆动点"""
        highs = data['high']
        lows = data['low']

        swing_highs = []
        swing_lows = []

        for i in range(self.min_swing_length, len(data) - self.min_swing_length):
            # 检查高点
            if highs.iloc[i] == highs.iloc[i - self.min_swing_length:i + self.min_swing_length + 1].max():
                swing_highs.append((i, highs.iloc[i]))

            # 检查低点
            if lows.iloc[i] == lows.iloc[i - self.min_swing_length:i + self.min_swing_length + 1].min():
                swing_lows.append((i, lows.iloc[i]))

        return swing_highs, swing_lows

    def _calculate_fibonacci_levels(self, data: pd.DataFrame,


                                    swing_highs: List[tuple],
                                    swing_lows: List[tuple]) -> Dict[str, pd.Series]:
        """计算斐波那契水平"""
        fib_levels = {}

        if not swing_highs or not swing_lows:
            # 如果没有找到摆动点，使用最近的高低点
            recent_high = data['high'].rolling(window=self.lookback_period).max().iloc[-1]
            recent_low = data['low'].rolling(window=self.lookback_period).min().iloc[-1]
            swing_highs = [(len(data) - 1, recent_high)]
            swing_lows = [(len(data) - 1, recent_low)]

        # 找到最近的主要摆动点
        recent_swing_high = max(swing_highs, key=lambda x: x[0])
        recent_swing_low = min(swing_lows, key=lambda x: x[0])

        high_price = recent_swing_high[1]
        low_price = recent_swing_low[1]

        # 计算回撤水平
        levels_to_use = self.custom_levels or self.FIB_LEVELS

        for level in levels_to_use:
            level_name = f'fib_retrace_{level:.3f}'
            fib_levels[level_name] = low_price + (high_price - low_price) * level

        # 计算扩展水平（如果有足够的数据）
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # 找到前一个主要摆动
            prev_swing_high = max([h for h in swing_highs if h[0] < recent_swing_high[0]],
                                  key=lambda x: x[0], default=None)
            prev_swing_low = min([l for l in swing_lows if l[0] < recent_swing_low[0]],
                                 key=lambda x: x[0], default=None)

            if prev_swing_high and prev_swing_low:
                extension_range = abs(high_price - low_price)

                for ext_level in self.FIB_EXTENSIONS:
                    level_name = f'fib_ext_{ext_level:.3f}'
                    if recent_swing_high[0] > recent_swing_low[0]:  # 上涨趋势
                        fib_levels[level_name] = low_price + extension_range * ext_level
                    else:  # 下跌趋势
                        fib_levels[level_name] = high_price - extension_range * ext_level

        # 将水平转换为Series
        for level_name in fib_levels:
            fib_levels[level_name] = pd.Series([fib_levels[level_name]] * len(data),
                                               index=data.index)

        return fib_levels

    def _calculate_price_fib_relationship(self, data: pd.DataFrame,


                                          fib_levels: Dict[str, pd.Series]) -> pd.DataFrame:
        """计算价格与斐波那契水平的关系"""
        close_price = data['close']

        # 计算距离各个水平的距离
        for level_name, level_values in fib_levels.items():
            distance_col = f'{level_name}_distance'
            data[distance_col] = close_price - level_values

            # 计算是否接近水平（在一定范围内）
            tolerance = close_price * 0.005  # 0.5 % 的容差
            near_level_col = f'{level_name}_near'
            data[near_level_col] = abs(data[distance_col]) <= tolerance

        # 计算最接近的水平
        data['fib_nearest_level'] = None
        data['fib_nearest_distance'] = float('inf')

        for level_name, level_values in fib_levels.items():
            distance = abs(close_price - level_values)
            closer_mask = distance < data['fib_nearest_distance']

            data.loc[closer_mask, 'fib_nearest_level'] = level_name.replace(
                'fib_', '').replace('_', ' ')
            data.loc[closer_mask, 'fib_nearest_distance'] = distance[closer_mask]

        # 计算支撑和阻力水平
        data['fib_support_level'] = None
        data['fib_resistance_level'] = None

        # 简单支撑阻力识别（可以根据需要扩展）
        for i in range(len(data)):
            current_price = close_price.iloc[i]

            # 寻找低于当前价格的回撤水平作为支撑
            support_levels = []
            resistance_levels = []

            for level_name, level_values in fib_levels.items():
                level_price = level_values.iloc[i]
                if level_price < current_price and 'retrace' in level_name:
                    support_levels.append((level_name, level_price))
                elif level_price > current_price and 'retrace' in level_name:
                    resistance_levels.append((level_name, level_price))

            # 选择最近的支撑和阻力
            if support_levels:
                nearest_support = max(support_levels, key=lambda x: x[1])
                data.at[data.index[i], 'fib_support_level'] = nearest_support[0]

            if resistance_levels:
                nearest_resistance = min(resistance_levels, key=lambda x: x[1])
                data.at[data.index[i], 'fib_resistance_level'] = nearest_resistance[0]

        return data
