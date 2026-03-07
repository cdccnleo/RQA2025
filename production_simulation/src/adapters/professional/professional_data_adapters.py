#!/usr/bin/env python3
"""
RQA2025 专业数据源适配器
集成Bloomberg、Refinitiv、加密货币交易所等专业数据源
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import requests
import json
import hmac
import hashlib

from src.infrastructure.utils.logging.logger import get_logger

# 尝试导入websocket，如果失败则设置为None
try:
    import websocket
except ImportError:
    websocket = None

logger = logging.getLogger(__name__)


class ProfessionalDataSource(Enum):

    """专业数据源枚举"""
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    BINANCE = "binance"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    HUOBI = "huobi"
    CME = "cme"
    CBOE = "cboe"
    NASDAQ = "nasdaq"
    LSE = "lse"


@dataclass
class ProfessionalMarketData:

    """专业市场数据"""
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    high: float
    low: float
    open: float
    close: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    spread: Optional[float] = None
    volatility: Optional[float] = None
    market_depth: Optional[int] = None
    order_book_imbalance: Optional[float] = None


class ProfessionalDataAdapter:

    """专业数据适配器基类"""

    def __init__(self, data_source: ProfessionalDataSource, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.data_source = data_source
        self.config = config or {}
        self.is_connected = False
        self.session = None
        self.ws_connection = None

        # API配置
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.base_url = self.config.get('base_url', '')
        self.ws_url = self.config.get('ws_url', '')

        # 认证配置
        self.auth_token = None
        self.refresh_token = None

        # 数据缓存
        self._data_cache: Dict[str, List] = {}
        self._cache_size = self.config.get('cache_size', 1000)

        # 订阅管理
        self.subscriptions: Dict[str, Dict[str, Any]] = {}
        self.subscription_callbacks: Dict[str, callable] = {}

        self.logger = get_logger(f"{self.__class__.__name__}_{data_source.value}")

    def connect(self) -> bool:
        """连接数据源"""
        raise NotImplementedError

    def disconnect(self) -> bool:
        """断开连接"""
        raise NotImplementedError

    def authenticate(self) -> bool:
        """认证"""
        raise NotImplementedError

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取历史数据"""
        raise NotImplementedError

    def subscribe_real_time_data(self, symbols: List[str], callback: callable) -> bool:
        """订阅实时数据"""
        raise NotImplementedError

    def unsubscribe_real_time_data(self, symbols: List[str]) -> bool:
        """取消订阅实时数据"""
        raise NotImplementedError

    def get_market_depth(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """获取市场深度"""
        raise NotImplementedError

    def get_option_chain(self, underlying: str) -> Dict[str, Any]:
        """获取期权链"""
        raise NotImplementedError

    def _cache_data(self, symbol: str, data: Any):
        """缓存数据"""
        if symbol not in self._data_cache:
            self._data_cache[symbol] = []

        self._data_cache[symbol].append({
            'timestamp': datetime.now(),
            'data': data
        })

        # 保持缓存大小
        if len(self._data_cache[symbol]) > self._cache_size:
            self._data_cache[symbol] = self._data_cache[symbol][-self._cache_size:]

    def _get_cached_data(self, symbol: str) -> Optional[List]:
        """获取缓存数据"""
        return self._data_cache.get(symbol, [])

    def _make_request(self, endpoint: str, params: Dict[str, Any] = None,


                      method: str = 'GET') -> Dict[str, Any]:
        """发送HTTP请求"""
        url = f"{self.base_url}{endpoint}"

        headers = {'Content - Type': 'application / json'}
        if self.auth_token:
            headers['Authorization'] = f'Bearer {self.auth_token}'

        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=params, timeout=30)
            else:
                raise ValueError(f"不支持的HTTP方法: {method}")

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"请求失败: {response.status_code} - {response.text}")
                return {}

        except Exception as e:
            self.logger.error(f"请求异常: {e}")
            return {}

    def _generate_signature(self, payload: str, secret: str) -> str:
        """生成签名"""
        return hmac.new(
            secret.encode('utf - 8'),
            payload.encode('utf - 8'),
            hashlib.sha256
        ).hexdigest()


class BloombergAdapter(ProfessionalDataAdapter):

    """Bloomberg数据适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(ProfessionalDataSource.BLOOMBERG, config)

        # Bloomberg特定配置
        self.base_url = self.config.get('base_url', 'https://api.bloomberg.com')
        self.ws_url = self.config.get('ws_url', 'wss://api.bloomberg.com / ws')
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')
        self.session_id = None

    def connect(self) -> bool:
        """连接Bloomberg"""
        try:
            if self.authenticate():
                self.is_connected = True
                self.logger.info("Bloomberg连接成功")
                return True
            else:
                self.logger.error("Bloomberg认证失败")
                return False

        except Exception as e:
            self.logger.error(f"Bloomberg连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开Bloomberg连接"""
        if self.session:
            try:
                self.session.close()
            except BaseException:
                pass

        self.is_connected = False
        self.logger.info("Bloomberg连接已断开")
        return True

    def authenticate(self) -> bool:
        """Bloomberg认证"""
        try:
            auth_data = {
                'api_key': self.api_key,
                'api_secret': self.api_secret,
                'grant_type': 'client_credentials'
            }

            response = self._make_request('/oauth2 / token', auth_data, 'POST')

            if 'access_token' in response:
                self.auth_token = response['access_token']
                self.refresh_token = response.get('refresh_token')
                return True
            else:
                return False

        except Exception as e:
            self.logger.error(f"Bloomberg认证异常: {e}")
            return False

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取Bloomberg历史数据"""
        if not self.is_connected:
            raise ConnectionError("Bloomberg未连接")

        params = {
            'symbols': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'periodicity': timeframe,
            'fields': ['PX_LAST', 'PX_HIGH', 'PX_LOW', 'PX_OPEN', 'VOLUME']
        }

        response = self._make_request('/data / historical', params)

        if response and 'data' in response:
            data = response['data']
            df = pd.DataFrame(data)

            # 重命名列
            column_mapping = {
                'PX_LAST': 'close',
                'PX_HIGH': 'high',
                'PX_LOW': 'low',
                'PX_OPEN': 'open',
                'VOLUME': 'volume'
            }
            df = df.rename(columns=column_mapping)

            df['timestamp'] = pd.to_datetime(df['date'])
            df.set_index('timestamp', inplace=True)
            df.drop('date', axis=1, inplace=True)

            return df

        return pd.DataFrame()

    def subscribe_real_time_data(self, symbols: List[str], callback: callable) -> bool:
        """订阅Bloomberg实时数据"""
        if not self.is_connected:
            return False

        try:
            # 注册回调
            subscription_id = f"sub_{datetime.now().strftime('%H % M % S % f')}"
            self.subscription_callbacks[subscription_id] = callback

            # 发送订阅请求
            # 这里应该通过WebSocket发送订阅请求
            # 简化的实现
            self.logger.info(f"Bloomberg订阅请求: {symbols}")
            return True

        except Exception as e:
            self.logger.error(f"Bloomberg订阅异常: {e}")
            return False

    def unsubscribe_real_time_data(self, symbols: List[str]) -> bool:
        """取消订阅Bloomberg实时数据"""
        if not self.is_connected:
            return False

        try:
            # 发送取消订阅请求
            self.logger.info(f"Bloomberg取消订阅: {symbols}")
            return True

        except Exception as e:
            self.logger.error(f"Bloomberg取消订阅异常: {e}")
            return False

    def get_market_depth(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """获取Bloomberg市场深度"""
        if not self.is_connected:
            return {}

        params = {
            'symbol': symbol,
            'levels': levels
        }

        response = self._make_request('/data / depth', params)

        return response

    def get_option_chain(self, underlying: str) -> Dict[str, Any]:
        """获取Bloomberg期权链"""
        if not self.is_connected:
            return {}

        params = {
            'underlying': underlying,
            'include_expirations': True,
            'include_strikes': True
        }

        response = self._make_request('/data / option - chain', params)

        return response


class BinanceAdapter(ProfessionalDataAdapter):

    """Binance加密货币适配器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        super().__init__(ProfessionalDataSource.BINANCE, config)

        # Binance特定配置
        self.base_url = self.config.get('base_url', 'https://api.binance.com')
        self.ws_url = self.config.get('ws_url', 'wss://stream.binance.com:9443 / ws')
        self.api_key = self.config.get('api_key', '')
        self.api_secret = self.config.get('api_secret', '')

    def connect(self) -> bool:
        """连接Binance"""
        try:
            # 测试连接
            response = self._make_request('/api / v3 / ping')
            if response == {}:  # Binance ping返回空对象
                self.is_connected = True
                self.logger.info("Binance连接成功")
                return True
            else:
                self.logger.error("Binance连接失败")
                return False

        except Exception as e:
            self.logger.error(f"Binance连接异常: {e}")
            return False

    def disconnect(self) -> bool:
        """断开Binance连接"""
        self.is_connected = False
        self.logger.info("Binance连接已断开")
        return True

    def authenticate(self) -> bool:
        """Binance认证"""
        # Binance使用API key进行认证
        if self.api_key and self.api_secret:
            return True
        return False

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d') -> pd.DataFrame:
        """获取Binance历史数据"""
        if not self.is_connected:
            raise ConnectionError("Binance未连接")

        # 转换时间格式
        start_time = int(pd.Timestamp(start_date).timestamp() * 1000)
        end_time = int(pd.Timestamp(end_date).timestamp() * 1000)

        params = {
            'symbol': symbol.upper(),
            'interval': timeframe,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }

        response = self._make_request('/api / v3 / klines', params)

        if response:
            # 解析K线数据
            columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                       'close_time', 'quote_volume', 'count', 'taker_buy_volume',
                       'taker_buy_quote_volume', 'ignore']

            df = pd.DataFrame(response, columns=columns)

            # 数据类型转换
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])

            return df

        return pd.DataFrame()

    def subscribe_real_time_data(self, symbols: List[str], callback: callable) -> bool:
        """订阅Binance实时数据"""
        if not self.is_connected:
            return False

        try:
            # 构建WebSocket流
            streams = [f"{symbol.lower()}@ticker" for symbol in symbols]

            def on_message(ws, message) -> Any:
                """on_message 函数的文档字符串"""

                data = json.loads(message)
                callback(data)

            # 这里应该启动WebSocket连接
            # 简化的实现
            self.logger.info(f"Binance订阅请求: {symbols}")
            return True

        except Exception as e:
            self.logger.error(f"Binance订阅异常: {e}")
            return False

    def unsubscribe_real_time_data(self, symbols: List[str]) -> bool:
        """取消订阅Binance实时数据"""
        # 简化的实现
        self.logger.info(f"Binance取消订阅: {symbols}")
        return True

    def get_market_depth(self, symbol: str, levels: int = 10) -> Dict[str, Any]:
        """获取Binance市场深度"""
        if not self.is_connected:
            return {}

        params = {
            'symbol': symbol.upper(),
            'limit': levels
        }

        response = self._make_request('/api / v3 / depth', params)

        if response:
            return {
                'bids': [[float(price), float(qty)] for price, qty in response['bids']],
                'asks': [[float(price), float(qty)] for price, qty in response['asks']],
                'timestamp': response.get('timestamp', datetime.now().timestamp() * 1000)
            }

        return {}

    def get_option_chain(self, underlying: str) -> Dict[str, Any]:
        """获取Binance期权链（如果支持）"""
        # Binance目前不支持期权
        return {'error': 'Binance does not support options'}


class ProfessionalDataManager:

    """专业数据管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """__init__ 函数的文档字符串"""

        self.config = config or {}
        self.adapters: Dict[ProfessionalDataSource, ProfessionalDataAdapter] = {}
        self.active_adapters: List[ProfessionalDataSource] = []

        # 数据路由配置
        self.data_routes: Dict[str, ProfessionalDataSource] = {}

        # 聚合缓存
        self.aggregated_data: Dict[str, List] = {}

        self.logger = get_logger(__name__)

    def register_adapter(self, adapter: ProfessionalDataAdapter):
        """注册适配器"""
        self.adapters[adapter.data_source] = adapter
        self.logger.info(f"注册专业数据适配器: {adapter.data_source.value}")

    def connect_adapter(self, data_source: ProfessionalDataSource) -> bool:
        """连接适配器"""
        if data_source not in self.adapters:
            self.logger.error(f"适配器未注册: {data_source.value}")
            return False

        adapter = self.adapters[data_source]
        if adapter.connect():
            self.active_adapters.append(data_source)
            self.logger.info(f"连接成功: {data_source.value}")
            return True
        else:
            self.logger.error(f"连接失败: {data_source.value}")
            return False

    def disconnect_all(self) -> Any:
        """断开所有连接"""
        for adapter in self.adapters.values():
            try:
                adapter.disconnect()
            except Exception as e:
                self.logger.error(f"断开连接异常: {e}")

        self.active_adapters.clear()

    def get_historical_data(self, symbol: str, start_date: str, end_date: str,


                            timeframe: str = '1d', sources: List[ProfessionalDataSource] = None) -> pd.DataFrame:
        """获取历史数据（多源聚合）"""
        if sources is None:
            sources = self.active_adapters

        all_data = []

        for source in sources:
            if source in self.adapters:
                try:
                    adapter = self.adapters[source]
                    data = adapter.get_historical_data(
                        symbol, start_date, end_date, timeframe)

                    if not data.empty:
                        data['source'] = source.value
                        all_data.append(data)

                except Exception as e:
                    self.logger.error(f"获取 {source.value} 数据失败: {e}")

        if all_data:
            # 合并数据
            combined_data = pd.concat(all_data)

            # 按时间排序并去重
            combined_data = combined_data.sort_index()
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]

            return combined_data

        return pd.DataFrame()

    def subscribe_multi_source_data(self, symbol: str, callback: callable) -> bool:
        """订阅多源实时数据"""
        success_count = 0

        for source in self.active_adapters:
            adapter = self.adapters[source]
            try:
                if adapter.subscribe_real_time_data([symbol], callback):
                    success_count += 1
            except Exception as e:
                self.logger.error(f"订阅 {source.value} 数据失败: {e}")

        return success_count > 0

    def get_market_overview(self, symbols: List[str]) -> Dict[str, Any]:
        """获取市场概览"""
        overview = {
            'timestamp': datetime.now(),
            'symbols': {},
            'market_health': {}
        }

        for symbol in symbols:
            symbol_data = {}

            # 收集各个数据源的数据
            for source in self.active_adapters:
                adapter = self.adapters[source]
                try:
                    # 尝试获取最新数据
                    cached_data = adapter._get_cached_data(symbol)
                    if cached_data:
                        latest = cached_data[-1]['data']
                        symbol_data[source.value] = latest

                except Exception as e:
                    self.logger.error(f"获取 {symbol} {source.value} 数据失败: {e}")

            overview['symbols'][symbol] = symbol_data

        return overview

    def create_default_adapters(self) -> Dict[ProfessionalDataSource, ProfessionalDataAdapter]:
        """创建默认适配器集合"""
        adapters = {}

        # Bloomberg适配器
        bloomberg_config = {
            'api_key': self.config.get('bloomberg', {}).get('api_key', ''),
            'api_secret': self.config.get('bloomberg', {}).get('api_secret', ''),
            'base_url': 'https://api.bloomberg.com'
        }
        adapters[ProfessionalDataSource.BLOOMBERG] = BloombergAdapter(bloomberg_config)

        # Binance适配器
        binance_config = {
            'api_key': self.config.get('binance', {}).get('api_key', ''),
            'api_secret': self.config.get('binance', {}).get('api_secret', ''),
            'base_url': 'https://api.binance.com'
        }
        adapters[ProfessionalDataSource.BINANCE] = BinanceAdapter(binance_config)

        # 注册所有适配器
        for adapter in adapters.values():
            self.register_adapter(adapter)

        return adapters

    def get_adapter_status(self) -> Dict[str, Any]:
        """获取适配器状态"""
        status = {}

        for source, adapter in self.adapters.items():
            status[source.value] = {
                'connected': adapter.is_connected,
                'subscriptions': len(adapter.subscriptions),
                'cache_size': len(adapter._data_cache)
            }

        return status
