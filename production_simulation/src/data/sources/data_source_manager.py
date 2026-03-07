#!/usr/bin/env python3
"""
RQA2025数据源管理器
统一管理多个数据源的访问
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class DataSource(ABC):

    """数据源抽象基类"""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):

        self.name = name
        self.config = config or {}
        self.is_available = False

    @abstractmethod
    def check_availability(self) -> bool:
        """检查数据源可用性"""

    @abstractmethod
    def fetch_data(self, symbol: str, start_date: str, end_date: str,


                   interval: str = '1d') -> pd.DataFrame:
        """获取数据"""

    def get_info(self) -> Dict[str, Any]:
        """获取数据源信息"""
        return {
            'name': self.name,
            'available': self.is_available,
            'config': self.config
        }


class YahooFinanceSource(DataSource):

    """Yahoo Finance数据源"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):

        super().__init__('Yahoo Finance', config)

    def check_availability(self) -> bool:

        try:
            import requests
            test_url = "https://query1.finance.yahoo.com / v8 / finance / chart / AAPL"
            response = requests.head(test_url, timeout=5)
            self.is_available = response.status_code == 200
            return self.is_available
        except BaseException:
            self.is_available = False
            return False

    def fetch_data(self, symbol: str, start_date: str, end_date: str,


                   interval: str = '1d') -> pd.DataFrame:
        try:
            import requests

            start_timestamp = int(pd.Timestamp(start_date).timestamp())
            end_timestamp = int(pd.Timestamp(end_date).timestamp())

            url = f"https://query1.finance.yahoo.com / v8 / finance / chart/{symbol}"
            params = {
                'period1': start_timestamp,
                'period2': end_timestamp,
                'interval': interval,
                'includePrePost': 'false',
                'events': 'div,splits'
            }

            headers = {
                'User - Agent': 'Mozilla / 5.0 (Windows NT 10.0; Win64; x64) AppleWebKit / 537.36'
            }

            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()

            data = response.json()

            if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                result = data['chart']['result'][0]
                timestamps = result['timestamp']
                quotes = result['indicators']['quote'][0]

                df_data = {
                    'timestamp': pd.to_datetime(timestamps, unit='s'),
                    'open': quotes.get('open', []),
                    'high': quotes.get('high', []),
                    'low': quotes.get('low', []),
                    'close': quotes.get('close', []),
                    'volume': quotes.get('volume', [])
                }

                df = pd.DataFrame(df_data)
                df = df.dropna(subset=['close'])
                df = df.sort_values('timestamp').reset_index(drop=True)

                logger.info(f"Yahoo Finance获取数据成功: {symbol}, {len(df)} 条记录")
                return df
            else:
                logger.warning(f"Yahoo Finance API返回数据格式错误: {symbol}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Yahoo Finance获取数据失败 {symbol}: {e}")
            return pd.DataFrame()


class DataSourceManager:

    """数据源管理器"""

    def __init__(self):

        self.sources = {}
        self._init_sources()

    def _init_sources(self):
        """初始化数据源"""
        self.sources['yahoo'] = YahooFinanceSource()

    def get_available_sources(self) -> List[str]:
        """获取可用的数据源"""
        available = []
        for name, source in self.sources.items():
            if source.check_availability():
                available.append(name)
        return available

    def fetch_data_with_fallback(self, symbol: str, start_date: str, end_date: str,


                                 interval: str = '1d', preferred_source: str = None) -> pd.DataFrame:
        """使用后备机制获取数据"""
        sources_to_try = []

        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)

        for name in ['yahoo']:
            if name != preferred_source and name in self.sources:
                sources_to_try.append(name)

        for source_name in sources_to_try:
            logger.info(f"尝试从 {source_name} 获取数据: {symbol}")
            source = self.sources[source_name]

            if source.check_availability():
                data = source.fetch_data(symbol, start_date, end_date, interval)
                if not data.empty:
                    logger.info(f"成功从 {source_name} 获取数据: {len(data)} 条记录")
                    return data
                else:
                    logger.warning(f"从 {source_name} 获取数据为空")
            else:
                logger.warning(f"数据源 {source_name} 不可用")

        logger.error(f"所有数据源都无法获取数据: {symbol}")
        return pd.DataFrame()
