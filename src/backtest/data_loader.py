#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
回测系统历史数据加载引擎
支持多数据源、多频率的历史数据加载和预处理
"""

import os
import pandas as pd
from typing import Dict, List, Optional
from src.data.loader import BaseDataLoader
from src.utils.logger import get_logger
from src.utils.date_utils import convert_timezone

logger = get_logger(__name__)

class BacktestDataLoader:
    def __init__(self, config: Dict):
        """
        初始化回测数据加载器
        :param config: 配置参数
        """
        self.config = config
        self.base_loader = BaseDataLoader(config.get("data", {}))
        self.cache = {}
        self.timezone = config.get("timezone", "Asia/Shanghai")

    def load_ohlcv(self,
                  symbol: str,
                  start: str,
                  end: str,
                  frequency: str = "1d",
                  adjust: str = "none") -> pd.DataFrame:
        """
        加载OHLCV行情数据
        :param symbol: 标的代码
        :param start: 开始日期(YYYY-MM-DD)
        :param end: 结束日期(YYYY-MM-DD)
        :param frequency: 数据频率(1d/1h/1m等)
        :param adjust: 复权方式(none/pre/post)
        :return: DataFrame with columns: [open, high, low, close, volume]
        """
        cache_key = f"{symbol}_{frequency}_{adjust}"

        # 检查缓存
        if cache_key in self.cache:
            data = self.cache[cache_key]
            mask = (data.index >= start) & (data.index <= end)
            return data[mask].copy()

        # 从基础加载器获取数据
        raw_data = self.base_loader.load_ohlcv(
            symbol=symbol,
            start=start,
            end=end,
            frequency=frequency,
            adjust=adjust
        )

        # 数据预处理
        processed = self._preprocess_data(raw_data, frequency)

        # 缓存数据
        self.cache[cache_key] = processed

        return processed.loc[start:end].copy()

    def _preprocess_data(self,
                        data: pd.DataFrame,
                        frequency: str) -> pd.DataFrame:
        """
        数据预处理
        :param data: 原始数据
        :param frequency: 数据频率
        :return: 处理后的数据
        """
        # 1. 确保时间索引
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # 2. 时区转换
        data.index = convert_timezone(data.index, self.timezone)

        # 3. 按频率重采样
        if frequency.endswith(('d', 'h', 'm')):
            freq_map = {
                '1d': 'D',
                '1h': 'H',
                '1m': 'T'
            }
            resample_freq = freq_map.get(frequency, 'D')
            if resample_freq:
                data = data.resample(resample_freq).last()

        # 4. 处理缺失值
        data = data.ffill().bfill()

        # 5. 标准化列名
        data.columns = data.columns.str.lower()

        return data

    def load_tick_data(self,
                      symbol: str,
                      date: str,
                      adjust: str = "none") -> pd.DataFrame:
        """
        加载tick级别数据
        :param symbol: 标的代码
        :param date: 日期(YYYY-MM-DD)
        :param adjust: 复权方式
        :return: DataFrame with tick data
        """
        return self.base_loader.load_tick_data(
            symbol=symbol,
            date=date,
            adjust=adjust
        )

    def load_fundamental(self,
                       symbol: str,
                       start: str,
                       end: str) -> pd.DataFrame:
        """
        加载财务数据
        :param symbol: 标的代码
        :param start: 开始日期
        :param end: 结束日期
        :return: DataFrame with fundamental data
        """
        return self.base_loader.load_fundamental(
            symbol=symbol,
            start=start,
            end=end
        )

    def load_universe(self, date: str) -> List[str]:
        """
        加载指定日期的股票池
        :param date: 日期(YYYY-MM-DD)
        :return: 股票代码列表
        """
        return self.base_loader.load_universe(date)

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
