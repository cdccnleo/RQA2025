#!/usr/bin/env python3
"""
RQA2025多市场适配器
支持A股、港股、美股、期货、期权、外汇、数字货币等多市场交易
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import requests

from src.infrastructure.utils.logging.logger import get_logger

logger = logging.getLogger(__name__)


class MarketType(Enum):

    """市场类型枚举"""
    A_STOCK = "a_stock"          # A股
    H_STOCK = "h_stock"          # 港股
    US_STOCK = "us_stock"        # 美股
    FUTURES = "futures"          # 期货
    OPTIONS = "options"          # 期权
    FOREX = "forex"              # 外汇
    CRYPTO = "crypto"            # 数字货币
    COMMODITY = "commodity"      # 商品


class AssetClass(Enum):

    """资产类别枚举"""
    EQUITY = "equity"            # 股票
    DERIVATIVE = "derivative"    # 衍生品
    CURRENCY = "currency"        # 货币
    COMMODITY = "commodity"      # 商品


class MarketAdapter(ABC):

    """市场适配器抽象基类"""

    def __init__(self, market_type: MarketType, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.market_type = market_type
        self.config = config or {}
        self.is_connected = False
        self.session = None
        self.logger = get_logger(f"{self.__class__.__name__}_{market_type.value}")

        # 市场特定配置
        self.trading_hours = self.config.get('trading_hours', {})
        self.holidays = self.config.get('holidays', [])
        self.timezone = self.config.get('timezone', 'UTC')

        # 数据缓存
        self._cache = {}
        self._cache_timeout = self.config.get('cache_timeout', 300)  # 5分钟

    @abstractmethod
    def connect(self) -> bool:
        """连接到市场数据源"""

    @abstractmethod
    def disconnect(self) -> bool:
        """断开市场连接"""

    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取历史数据"""

    @abstractmethod
    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""

    @abstractmethod
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""

    def is_market_open(self) -> bool:
        """检查市场是否开放"""
        now = datetime.now()

        # 检查是否为节假日
        if now.date() in self.holidays:
            return False

        # 检查交易时间
        if self.trading_hours:
            current_time = now.time()
            start_time = self.trading_hours.get(
                'start', datetime.strptime('09:30', '%H:%M').time())
            end_time = self.trading_hours.get(
                'end', datetime.strptime('16:00', '%H:%M').time())

            return start_time <= current_time <= end_time

        return True

    def validate_symbol(self, symbol: str) -> bool:
        """验证交易标的格式"""
        return bool(symbol and isinstance(symbol, str))

    def format_symbol(self, symbol: str) -> str:
        """格式化交易标的"""
        return symbol.upper().strip()

    def _cache_data(self, key: str, data: Any):
        """缓存数据"""
        self._cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }

    def _get_cached_data(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        if key in self._cache:
            cached_item = self._cache[key]
            if datetime.now() - cached_item['timestamp'] < timedelta(seconds=self._cache_timeout):
                return cached_item['data']
            else:
                del self._cache[key]
        return None

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        return {
            'market_type': self.market_type.value,
            'is_connected': self.is_connected,
            'is_market_open': self.is_market_open(),
            'cache_size': len(self._cache),
            'status': 'healthy' if self.is_connected else 'disconnected'
        }


class AStockAdapter(MarketAdapter):

    """A股市场适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(MarketType.A_STOCK, config)

        # A股特定配置
        self.trading_hours = {
            'start': datetime.strptime('09:30', '%H:%M').time(),
            'end': datetime.strptime('15:00', '%H:%M').time()
        }
        self.timezone = 'Asia / Shanghai'

        # 数据源配置
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://api.wmcloud.com')

    def connect(self) -> bool:
        """连接到A股数据源"""
        try:
            # 初始化HTTP会话
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content - Type': 'application / json'
            })

            # 测试连接
            test_response = self.session.get(f"{self.base_url}/health")
            if test_response.status_code == 200:
                self.is_connected = True
                self.logger.info("A股适配器连接成功")
                return True
            else:
                self.logger.error(f"A股适配器连接失败: {test_response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"A股适配器连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开A股连接"""
        if self.session:
            self.session.close()
        self.is_connected = False
        self.logger.info("A股适配器已断开")
        return True

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取A股历史数据"""
        if not self.is_connected:
            raise ConnectionError("A股适配器未连接")

        # 检查缓存
        cache_key = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 构建API请求
            params = {
                'symbol': self.format_symbol(symbol),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe,
                'market': 'CN'
            }

            response = self.session.get(
                f"{self.base_url}/data / historical",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()

                # 转换为DataFrame
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                # 缓存数据
                self._cache_data(cache_key, df)

                self.logger.info(f"获取A股历史数据成功: {symbol}, {len(df)}条记录")
                return df
            else:
                self.logger.error(f"获取A股历史数据失败: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取A股历史数据异常: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取A股实时数据"""
        if not self.is_connected:
            raise ConnectionError("A股适配器未连接")

        try:
            params = {'symbol': self.format_symbol(symbol)}

            response = self.session.get(
                f"{self.base_url}/data / realtime",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"获取A股实时数据成功: {symbol}")
                return data
            else:
                self.logger.error(f"获取A股实时数据失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.error(f"获取A股实时数据异常: {e}")
            return {}

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取A股市场信息"""
        return {
            'symbol': symbol,
            'market_type': self.market_type.value,
            'trading_hours': self.trading_hours,
            'timezone': self.timezone,
            'currency': 'CNY',
            'lot_size': 100,  # 手数
            'price_precision': 2,
            'volume_precision': 0
        }


class HStockAdapter(MarketAdapter):

    """港股市场适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(MarketType.H_STOCK, config)

        # 港股特定配置
        self.trading_hours = {
            'start': datetime.strptime('09:30', '%H:%M').time(),
            'end': datetime.strptime('16:00', '%H:%M').time()
        }
        self.timezone = 'Asia / Hong_Kong'

        # 数据源配置
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://api.hkex.com')

    def connect(self) -> bool:
        """连接到港股数据源"""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content - Type': 'application / json'
            })

            # 测试连接
            test_response = self.session.get(f"{self.base_url}/health")
            if test_response.status_code == 200:
                self.is_connected = True
                self.logger.info("港股适配器连接成功")
                return True
            else:
                self.logger.error(f"港股适配器连接失败: {test_response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"港股适配器连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开港股连接"""
        if self.session:
            self.session.close()
        self.is_connected = False
        self.logger.info("港股适配器已断开")
        return True

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取港股历史数据"""
        if not self.is_connected:
            raise ConnectionError("港股适配器未连接")

        cache_key = f"hk_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            params = {
                'symbol': self.format_symbol(symbol),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe,
                'market': 'HK'
            }

            response = self.session.get(
                f"{self.base_url}/data / historical",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                self._cache_data(cache_key, df)
                self.logger.info(f"获取港股历史数据成功: {symbol}, {len(df)}条记录")
                return df
            else:
                self.logger.error(f"获取港股历史数据失败: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取港股历史数据异常: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取港股实时数据"""
        if not self.is_connected:
            raise ConnectionError("港股适配器未连接")

        try:
            params = {'symbol': self.format_symbol(symbol)}

            response = self.session.get(
                f"{self.base_url}/data / realtime",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"获取港股实时数据成功: {symbol}")
                return data
            else:
                self.logger.error(f"获取港股实时数据失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.error(f"获取港股实时数据异常: {e}")
            return {}

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取港股市场信息"""
        return {
            'symbol': symbol,
            'market_type': self.market_type.value,
            'trading_hours': self.trading_hours,
            'timezone': self.timezone,
            'currency': 'HKD',
            'lot_size': 100,  # 手数
            'price_precision': 3,
            'volume_precision': 0
        }


class USStockAdapter(MarketAdapter):

    """美股市场适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(MarketType.US_STOCK, config)

        # 美股特定配置
        self.trading_hours = {
            'start': datetime.strptime('09:30', '%H:%M').time(),
            'end': datetime.strptime('16:00', '%H:%M').time()
        }
        self.timezone = 'America / New_York'

        # 数据源配置
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://api.polygon.io')

    def connect(self) -> bool:
        """连接到美股数据源"""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content - Type': 'application / json'
            })

            # 测试连接
            test_response = self.session.get(f"{self.base_url}/v1 / marketstatus / now")
            if test_response.status_code == 200:
                self.is_connected = True
                self.logger.info("美股适配器连接成功")
                return True
            else:
                self.logger.error(f"美股适配器连接失败: {test_response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"美股适配器连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开美股连接"""
        if self.session:
            self.session.close()
        self.is_connected = False
        self.logger.info("美股适配器已断开")
        return True

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取美股历史数据"""
        if not self.is_connected:
            raise ConnectionError("美股适配器未连接")

        cache_key = f"us_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            params = {
                'symbol': self.format_symbol(symbol),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe,
                'market': 'US'
            }

            response = self.session.get(
                f"{self.base_url}/v2 / aggs / ticker/{symbol}/range / 1/{timeframe}/{start_date}/{end_date}",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['results'])
                df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # 重命名列
                df = df.rename(columns={
                    'o': 'open',
                    'h': 'high',
                    'l': 'low',
                    'c': 'close',
                    'v': 'volume'
                })

                self._cache_data(cache_key, df)
                self.logger.info(f"获取美股历史数据成功: {symbol}, {len(df)}条记录")
                return df
            else:
                self.logger.error(f"获取美股历史数据失败: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取美股历史数据异常: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取美股实时数据"""
        if not self.is_connected:
            raise ConnectionError("美股适配器未连接")

        try:
            response = self.session.get(
                f"{self.base_url}/v2 / last / trade/{symbol}",
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"获取美股实时数据成功: {symbol}")
                return data
            else:
                self.logger.error(f"获取美股实时数据失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.error(f"获取美股实时数据异常: {e}")
            return {}

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取美股市场信息"""
        return {
            'symbol': symbol,
            'market_type': self.market_type.value,
            'trading_hours': self.trading_hours,
            'timezone': self.timezone,
            'currency': 'USD',
            'lot_size': 1,  # 股数
            'price_precision': 2,
            'volume_precision': 0
        }


class FuturesAdapter(MarketAdapter):

    """期货市场适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(MarketType.FUTURES, config)

        # 期货特定配置
        self.trading_hours = {
            'start': datetime.strptime('09:00', '%H:%M').time(),
            'end': datetime.strptime('23:00', '%H:%M').time()
        }
        self.timezone = 'Asia / Shanghai'

        # 数据源配置
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://api.futures.com')

    def connect(self) -> bool:
        """连接到期货数据源"""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'Authorization': f'Bearer {self.api_key}',
                'Content - Type': 'application / json'
            })

            # 测试连接
            test_response = self.session.get(f"{self.base_url}/health")
            if test_response.status_code == 200:
                self.is_connected = True
                self.logger.info("期货适配器连接成功")
                return True
            else:
                self.logger.error(f"期货适配器连接失败: {test_response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"期货适配器连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开期货连接"""
        if self.session:
            self.session.close()
        self.is_connected = False
        self.logger.info("期货适配器已断开")
        return True

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取期货历史数据"""
        if not self.is_connected:
            raise ConnectionError("期货适配器未连接")

        cache_key = f"futures_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            params = {
                'symbol': self.format_symbol(symbol),
                'start_date': start_date,
                'end_date': end_date,
                'timeframe': timeframe,
                'market': 'FUTURES'
            }

            response = self.session.get(
                f"{self.base_url}/data / historical",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data['data'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

                self._cache_data(cache_key, df)
                self.logger.info(f"获取期货历史数据成功: {symbol}, {len(df)}条记录")
                return df
            else:
                self.logger.error(f"获取期货历史数据失败: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取期货历史数据异常: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取期货实时数据"""
        if not self.is_connected:
            raise ConnectionError("期货适配器未连接")

        try:
            params = {'symbol': self.format_symbol(symbol)}

            response = self.session.get(
                f"{self.base_url}/data / realtime",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"获取期货实时数据成功: {symbol}")
                return data
            else:
                self.logger.error(f"获取期货实时数据失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.error(f"获取期货实时数据异常: {e}")
            return {}

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取期货市场信息"""
        return {
            'symbol': symbol,
            'market_type': self.market_type.value,
            'trading_hours': self.trading_hours,
            'timezone': self.timezone,
            'currency': 'CNY',
            'lot_size': 1,  # 手数
            'price_precision': 2,
            'volume_precision': 0,
            'margin_required': True,
            'leverage_available': True
        }


class CryptoAdapter(MarketAdapter):

    """数字货币市场适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(MarketType.CRYPTO, config)

        # 数字货币特定配置
        self.trading_hours = {}  # 24 / 7交易
        self.timezone = 'UTC'

        # 数据源配置
        self.api_key = self.config.get('api_key', '')
        self.base_url = self.config.get('base_url', 'https://api.binance.com')

    def connect(self) -> bool:
        """连接到数字货币数据源"""
        try:
            self.session = requests.Session()
            self.session.headers.update({
                'X - MBX - APIKEY': self.api_key,
                'Content - Type': 'application / json'
            })

            # 测试连接
            test_response = self.session.get(f"{self.base_url}/api / v3 / ping")
            if test_response.status_code == 200:
                self.is_connected = True
                self.logger.info("数字货币适配器连接成功")
                return True
            else:
                self.logger.error(f"数字货币适配器连接失败: {test_response.status_code}")
                return False

        except Exception as e:
            self.logger.error(f"数字货币适配器连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开数字货币连接"""
        if self.session:
            self.session.close()
        self.is_connected = False
        self.logger.info("数字货币适配器已断开")
        return True

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取数字货币历史数据"""
        if not self.is_connected:
            raise ConnectionError("数字货币适配器未连接")

        cache_key = f"crypto_{symbol}_{start_date}_{end_date}_{timeframe}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data is not None:
            return cached_data

        try:
            # 转换时间格式
            start_time = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_time = int(pd.Timestamp(end_date).timestamp() * 1000)

            params = {
                'symbol': self.format_symbol(symbol),
                'interval': timeframe,
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }

            response = self.session.get(
                f"{self.base_url}/api / v3 / klines",
                params=params,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data, columns=[
                    'timestamp', 'open', 'high', 'low', 'close',
                    'volume', 'close_time', 'quote_volume', 'count',
                    'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
                ])

                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # 数据类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    df[col] = pd.to_numeric(df[col])

                self._cache_data(cache_key, df)
                self.logger.info(f"获取数字货币历史数据成功: {symbol}, {len(df)}条记录")
                return df
            else:
                self.logger.error(f"获取数字货币历史数据失败: {response.status_code}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"获取数字货币历史数据异常: {e}")
            return pd.DataFrame()

    def get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """获取数字货币实时数据"""
        if not self.is_connected:
            raise ConnectionError("数字货币适配器未连接")

        try:
            params = {'symbol': self.format_symbol(symbol)}

            response = self.session.get(
                f"{self.base_url}/api / v3 / ticker / price",
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                self.logger.debug(f"获取数字货币实时数据成功: {symbol}")
                return data
            else:
                self.logger.error(f"获取数字货币实时数据失败: {response.status_code}")
                return {}

        except Exception as e:
            self.logger.error(f"获取数字货币实时数据异常: {e}")
            return {}

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """获取数字货币市场信息"""
        return {
            'symbol': symbol,
            'market_type': self.market_type.value,
            'trading_hours': self.trading_hours,  # 24 / 7
            'timezone': self.timezone,
            'currency': 'USDT',  # 假设稳定币计价
            'lot_size': 0.0001,  # 小数点位数
            'price_precision': 4,
            'volume_precision': 6,
            'margin_required': False,
            'leverage_available': True
        }


class MarketAdapterManager:

    """市场适配器管理器"""

    def __init__(self) -> Any:
        """__init__ 函数的文档字符串"""

        self.adapters: Dict[MarketType, MarketAdapter] = {}
        self.active_adapters: List[MarketType] = []
        self.logger = get_logger(__name__)

    def register_adapter(self, adapter: MarketAdapter):
        """注册适配器"""
        self.adapters[adapter.market_type] = adapter
        self.logger.info(f"适配器已注册: {adapter.market_type.value}")

    def get_adapter(self, market_type: MarketType) -> Optional[MarketAdapter]:
        """获取适配器"""
        return self.adapters.get(market_type)

    def connect_all(self) -> Dict[str, bool]:
        """连接所有适配器"""
        results = {}
        for market_type, adapter in self.adapters.items():
            try:
                success = adapter.connect()
                results[market_type.value] = success
                if success:
                    self.active_adapters.append(market_type)
            except Exception as e:
                self.logger.error(f"连接适配器失败 {market_type.value}: {e}")
                results[market_type.value] = False

        return results

    def disconnect_all(self) -> Dict[str, bool]:
        """断开所有适配器连接"""
        results = {}
        for market_type, adapter in self.adapters.items():
            try:
                success = adapter.disconnect()
                results[market_type.value] = success
            except Exception as e:
                self.logger.error(f"断开适配器失败 {market_type.value}: {e}")
                results[market_type.value] = False

        self.active_adapters.clear()
        return results

    def get_historical_data(self, market_type: MarketType, symbol: str,


                            start_date: str, end_date: str, timeframe: str = '1d') -> pd.DataFrame:
        """获取历史数据"""
        adapter = self.get_adapter(market_type)
        if not adapter:
            raise ValueError(f"未找到 {market_type.value} 适配器")

        return adapter.get_historical_data(symbol, start_date, end_date, timeframe)

    def get_real_time_data(self, market_type: MarketType, symbol: str) -> Dict[str, Any]:
        """获取实时数据"""
        adapter = self.get_adapter(market_type)
        if not adapter:
            raise ValueError(f"未找到 {market_type.value} 适配器")

        return adapter.get_real_time_data(symbol)

    def get_market_info(self, market_type: MarketType, symbol: str) -> Dict[str, Any]:
        """获取市场信息"""
        adapter = self.get_adapter(market_type)
        if not adapter:
            raise ValueError(f"未找到 {market_type.value} 适配器")

        return adapter.get_market_info(symbol)

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有适配器健康状态"""
        status = {}
        for market_type, adapter in self.adapters.items():
            status[market_type.value] = adapter.health_check()
        return status

    def create_default_adapters(self) -> Dict[MarketType, MarketAdapter]:
        """创建默认适配器集合"""
        adapters = {
            MarketType.A_STOCK: AStockAdapter(),
            MarketType.H_STOCK: HStockAdapter(),
            MarketType.US_STOCK: USStockAdapter(),
            MarketType.FUTURES: FuturesAdapter(),
            MarketType.CRYPTO: CryptoAdapter(),
        }

        # 注册所有适配器
        for adapter in adapters.values():
            self.register_adapter(adapter)

        return adapters
