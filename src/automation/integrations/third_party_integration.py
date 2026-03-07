"""
Third Party Integration Module
第三方集成模块

This module provides third - party service integration capabilities for quantitative trading systems
此模块为量化交易系统提供第三方服务集成能力

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import time

logger = logging.getLogger(__name__)


class ThirdPartyProvider(Enum):

    """Third - party service providers"""
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    ALPHA_VANTAGE = "alpha_vantage"
    TWELVE_DATA = "twelve_data"
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    QUANDL = "quandl"
    CRYPTOCOMPARE = "cryptocompare"
    COINGECKO = "coingecko"
    NEWSAPI = "newsapi"
    ALPHA_NEWS = "alpha_news"


class DataType(Enum):

    """Data types"""
    STOCK_PRICES = "stock_prices"
    CRYPTO_PRICES = "crypto_prices"
    ECONOMIC_DATA = "economic_data"
    NEWS = "news"
    FUNDAMENTALS = "fundamentals"
    TECHNICAL_INDICATORS = "technical_indicators"
    MARKET_NEWS = "market_news"


@dataclass
class ThirdPartyConnection:

    """
    Third - party connection configuration
    第三方连接配置
    """
    connection_id: str
    provider: str
    api_key: str
    api_secret: Optional[str] = None
    base_url: str
    rate_limit: int = 100  # requests per minute
    timeout: int = 30


@dataclass
class APIRequest:

    """
    API request data class
    API请求数据类
    """
    request_id: str
    provider: str
    endpoint: str
    method: str
    params: Dict[str, Any] = None
    headers: Dict[str, str] = None
    data: Any = None
    timestamp: datetime = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class APIResponse:

    """
    API response data class
    API响应数据类
    """
    request_id: str
    status_code: int
    response_time: float
    data: Any = None
    error_message: Optional[str] = None
    timestamp: datetime = None
    rate_limit_remaining: Optional[int] = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class RateLimiter:

    """
    Rate Limiter Class
    速率限制器类

    Manages API rate limiting for third - party services
    管理第三方服务的API速率限制
    """

    def __init__(self):
        """
        Initialize rate limiter
        初始化速率限制器
        """
        self.requests = {}
        self.limits = {}

    def set_limit(self, provider: str, requests_per_minute: int) -> None:
        """
        Set rate limit for a provider
        为提供商设置速率限制

        Args:
            provider: Provider name
                     提供商名称
            requests_per_minute: Maximum requests per minute
                               每分钟最大请求数
        """
        self.limits[provider] = requests_per_minute
        self.requests[provider] = []

    def check_rate_limit(self, provider: str) -> bool:
        """
        Check if request is within rate limit
        检查请求是否在速率限制内

        Args:
            provider: Provider name
                     提供商名称

        Returns:
            bool: True if within limit
                  如果在限制内则返回True
        """
        if provider not in self.limits:
            return True

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        # Remove old requests
        self.requests[provider] = [
            req_time for req_time in self.requests[provider]
            if req_time > one_minute_ago
        ]

        # Check current count
        if len(self.requests[provider]) >= self.limits[provider]:
            return False

        # Add current request
        self.requests[provider].append(now)
        return True


class AuthenticationManager:

    """
    Authentication Manager Class
    认证管理器类

    Handles authentication for third - party APIs
    处理第三方API的认证
    """

    def __init__(self):
        """
        Initialize authentication manager
        初始化认证管理器
        """
        self.api_keys = {}
        self.api_secrets = {}

    def set_credentials(self,


                        provider: str,
                        api_key: str,
                        api_secret: Optional[str] = None) -> None:
        """
        Set API credentials for a provider
        为提供商设置API凭据

        Args:
            provider: Provider name
                     提供商名称
            api_key: API key
                    API密钥
            api_secret: API secret (optional)
                       API密钥（可选）
        """
        self.api_keys[provider] = api_key
        if api_secret:
            self.api_secrets[provider] = api_secret

    def get_auth_headers(self, provider: str, endpoint: str = "") -> Dict[str, str]:
        """
        Get authentication headers for a provider
        获取提供商的认证头

        Args:
            provider: Provider name
                     提供商名称
            endpoint: API endpoint (for specific auth methods)
                     API端点（用于特定认证方法）

        Returns:
            dict: Authentication headers
                  认证头
        """
        headers = {}

        if provider in self.api_keys:
            if provider == ThirdPartyProvider.ALPHA_VANTAGE.value:
                headers['X - RapidAPI - Key'] = self.api_keys[provider]
            elif provider == ThirdPartyProvider.TWELVE_DATA.value:
                headers['Authorization'] = f"apikey {self.api_keys[provider]}"
            elif provider == ThirdPartyProvider.NEWSAPI.value:
                headers['X - API - Key'] = self.api_keys[provider]
            elif provider == ThirdPartyProvider.CRYPTOCOMPARE.value:
                # CryptoCompare uses API key in URL params
                pass
            else:
                headers['Authorization'] = f"Bearer {self.api_keys[provider]}"

        return headers

    def get_auth_params(self, provider: str) -> Dict[str, str]:
        """
        Get authentication parameters for a provider
        获取提供商的认证参数

        Args:
            provider: Provider name
                     提供商名称

        Returns:
            dict: Authentication parameters
                  认证参数
        """
        params = {}

        if provider in self.api_keys:
            if provider == ThirdPartyProvider.CRYPTOCOMPARE.value:
                params['api_key'] = self.api_keys[provider]
            elif provider == ThirdPartyProvider.ALPHA_VANTAGE.value:
                params['apikey'] = self.api_keys[provider]

        return params


class DataFetcher:

    """
    Data Fetcher Class
    数据获取器类

    Fetches data from third - party APIs
    从第三方API获取数据
    """

    def __init__(self,


                 auth_manager: AuthenticationManager,
                 rate_limiter: RateLimiter):
        """
        Initialize data fetcher
        初始化数据获取器

        Args:
            auth_manager: Authentication manager
                         认证管理器
            rate_limiter: Rate limiter
                         速率限制器
        """
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.endpoints = self._setup_endpoints()

    def _setup_endpoints(self) -> Dict[str, Dict[str, str]]:
        """
        Setup API endpoints for different providers
        为不同提供商设置API端点

        Returns:
            dict: Provider endpoints
                  提供商端点
        """
        return {
            ThirdPartyProvider.ALPHA_VANTAGE.value: {
                'base_url': 'https://www.alphavantage.co / query',
                'endpoints': {
                    'stock_quote': '',
                    'intraday': '',
                    'daily': '',
                    'weekly': '',
                    'monthly': ''
                }
            },
            ThirdPartyProvider.TWELVE_DATA.value: {
                'base_url': 'https://api.twelvedata.com',
                'endpoints': {
                    'quote': '/quote',
                    'time_series': '/time_series',
                    'complex_data': '/complex_data'
                }
            },
            ThirdPartyProvider.CRYPTOCOMPARE.value: {
                'base_url': 'https://min - api.cryptocompare.com / data',
                'endpoints': {
                    'price': '/price',
                    'histominute': '/histominute',
                    'histohour': '/histohour',
                    'histoday': '/histoday'
                }
            },
            ThirdPartyProvider.NEWSAPI.value: {
                'base_url': 'https://newsapi.org / v2',
                'endpoints': {
                    'everything': '/everything',
                    'top_headlines': '/top - headlines',
                    'sources': '/sources'
                }
            }
        }

    def fetch_data(self,


                   provider: str,
                   data_type: DataType,
                   params: Dict[str, Any]) -> APIResponse:
        """
        Fetch data from third - party API
        从第三方API获取数据

        Args:
            provider: Provider name
                     提供商名称
            data_type: Type of data to fetch
                      要获取的数据类型
            params: Request parameters
                   请求参数

        Returns:
            APIResponse: API response
                        API响应
        """
        request_id = f"req_{provider}_{data_type.value}_{datetime.now().strftime('%Y % m % d_ % H % M % S_ % f')}"

        # Check rate limit
        if not self.rate_limiter.check_rate_limit(provider):
            return APIResponse(
                request_id=request_id,
                status_code=429,
                response_time=0.0,
                error_message="Rate limit exceeded"
            )

        start_time = time.time()

        try:
            if provider == ThirdPartyProvider.ALPHA_VANTAGE.value:
                response = self._fetch_alpha_vantage(data_type, params)
            elif provider == ThirdPartyProvider.TWELVE_DATA.value:
                response = self._fetch_twelve_data(data_type, params)
            elif provider == ThirdPartyProvider.CRYPTOCOMPARE.value:
                response = self._fetch_cryptocompare(data_type, params)
            elif provider == ThirdPartyProvider.NEWSAPI.value:
                response = self._fetch_newsapi(data_type, params)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            response_time = time.time() - start_time

            return APIResponse(
                request_id=request_id,
                status_code=response.status_code,
                response_time=response_time,
                data=response.json() if response.status_code == 200 else None,
                error_message=response.text if response.status_code != 200 else None
            )

        except Exception as e:
            response_time = time.time() - start_time
            return APIResponse(
                request_id=request_id,
                status_code=0,
                response_time=response_time,
                error_message=str(e)
            )

    def _fetch_alpha_vantage(self, data_type: DataType, params: Dict[str, Any]) -> requests.Response:
        """Fetch data from Alpha Vantage"""
        base_url = "https://www.alphavantage.co / query"

        request_params = {
            'symbol': params.get('symbol', ''),
            'apikey': self.auth_manager.api_keys.get(ThirdPartyProvider.ALPHA_VANTAGE.value, '')
        }

        if data_type == DataType.STOCK_PRICES:
            if 'interval' in params:
                request_params['function'] = 'TIME_SERIES_INTRADAY'
                request_params['interval'] = params['interval']
            else:
                request_params['function'] = 'TIME_SERIES_DAILY'

        headers = self.auth_manager.get_auth_headers(ThirdPartyProvider.ALPHA_VANTAGE.value)

        return requests.get(base_url, params=request_params, headers=headers, timeout=30)

    def _fetch_twelve_data(self, data_type: DataType, params: Dict[str, Any]) -> requests.Response:
        """Fetch data from Twelve Data"""
        base_url = "https://api.twelvedata.com"

        endpoint = "/time_series" if data_type == DataType.STOCK_PRICES else "/quote"
        url = base_url + endpoint

        request_params = {
            'symbol': params.get('symbol', ''),
            'apikey': self.auth_manager.api_keys.get(ThirdPartyProvider.TWELVE_DATA.value, '')
        }

        if data_type == DataType.STOCK_PRICES:
            request_params.update({
                'interval': params.get('interval', '1day'),
                'outputsize': params.get('outputsize', '30')
            })

        headers = self.auth_manager.get_auth_headers(ThirdPartyProvider.TWELVE_DATA.value)

        return requests.get(url, params=request_params, headers=headers, timeout=30)

    def _fetch_cryptocompare(self, data_type: DataType, params: Dict[str, Any]) -> requests.Response:
        """Fetch data from CryptoCompare"""
        base_url = "https://min - api.cryptocompare.com / data"

        if data_type == DataType.CRYPTO_PRICES:
            endpoint = "/price" if 'fsym' in params and 'tsyms' in params else "/histoday"
        else:
            endpoint = "/histoday"

        url = base_url + endpoint

        request_params = self.auth_manager.get_auth_params(ThirdPartyProvider.CRYPTOCOMPARE.value)
        request_params.update(params)

        return requests.get(url, params=request_params, timeout=30)

    def _fetch_newsapi(self, data_type: DataType, params: Dict[str, Any]) -> requests.Response:
        """Fetch data from NewsAPI"""
        base_url = "https://newsapi.org / v2"

        if data_type == DataType.MARKET_NEWS:
            endpoint = "/everything"
        else:
            endpoint = "/top - headlines"

        url = base_url + endpoint

        request_params = {
            'q': params.get('query', 'finance OR stocks OR trading'),
            'apiKey': self.auth_manager.api_keys.get(ThirdPartyProvider.NEWSAPI.value, '')
        }

        if 'from_date' in params:
            request_params['from'] = params['from_date']
        if 'to_date' in params:
            request_params['to'] = params['to_date']

        headers = self.auth_manager.get_auth_headers(ThirdPartyProvider.NEWSAPI.value)

        return requests.get(url, params=request_params, headers=headers, timeout=30)


class DataProcessor:

    """
    Data Processor Class
    数据处理器类

    Processes and normalizes data from third - party APIs
    处理和规范化来自第三方API的数据
    """

    def __init__(self):
        """
        Initialize data processor
        初始化数据处理器
        """
        self.processors = {
            ThirdPartyProvider.ALPHA_VANTAGE.value: self._process_alpha_vantage_data,
            ThirdPartyProvider.TWELVE_DATA.value: self._process_twelve_data,
            ThirdPartyProvider.CRYPTOCOMPARE.value: self._process_cryptocompare_data,
            ThirdPartyProvider.NEWSAPI.value: self._process_newsapi_data
        }

    def process_data(self, provider: str, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw data from third - party API
        处理来自第三方API的原始数据

        Args:
            provider: Provider name
                     提供商名称
            raw_data: Raw data from API
                     来自API的原始数据

        Returns:
            dict: Processed data
                  处理后的数据
        """
        if provider in self.processors:
            return self.processors[provider](raw_data)
        else:
            logger.warning(f"No processor found for provider: {provider}")
            return raw_data

    def _process_alpha_vantage_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Alpha Vantage data"""
        processed_data = {
            'provider': 'alpha_vantage',
            'timestamp': datetime.now(),
            'data': []
        }

        # Extract time series data
        time_series_key = None
        for key in raw_data.keys():
            if 'Time Series' in key:
                time_series_key = key
                break

        if time_series_key:
            time_series = raw_data[time_series_key]
            for date_str, values in time_series.items():
                processed_data['data'].append({
                    'date': date_str,
                    'open': float(values.get('1. open', 0)),
                    'high': float(values.get('2. high', 0)),
                    'low': float(values.get('3. low', 0)),
                    'close': float(values.get('4. close', 0)),
                    'volume': int(values.get('5. volume', 0))
                })

        return processed_data

    def _process_twelve_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process Twelve Data"""
        processed_data = {
            'provider': 'twelve_data',
            'timestamp': datetime.now(),
            'data': []
        }

        if 'values' in raw_data:
            for item in raw_data['values']:
                processed_data['data'].append({
                    'date': item.get('datetime'),
                    'open': float(item.get('open', 0)),
                    'high': float(item.get('high', 0)),
                    'low': float(item.get('low', 0)),
                    'close': float(item.get('close', 0)),
                    'volume': int(item.get('volume', 0))
                })

        return processed_data

    def _process_cryptocompare_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process CryptoCompare data"""
        processed_data = {
            'provider': 'cryptocompare',
            'timestamp': datetime.now(),
            'data': []
        }

        if 'Data' in raw_data:
            # Historical data
            for item in raw_data['Data']:
                processed_data['data'].append({
                    'timestamp': item.get('time'),
                    'open': item.get('open', 0),
                    'high': item.get('high', 0),
                    'low': item.get('low', 0),
                    'close': item.get('close', 0),
                    'volume': item.get('volumeto', 0)
                })
        else:
            # Current price data
            processed_data['data'] = raw_data

        return processed_data

    def _process_newsapi_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process NewsAPI data"""
        processed_data = {
            'provider': 'newsapi',
            'timestamp': datetime.now(),
            'articles': []
        }

        if 'articles' in raw_data:
            for article in raw_data['articles']:
                processed_data['articles'].append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'content': article.get('content', '')
                })

        return processed_data


class ThirdPartyIntegrationManager:

    """
    Third Party Integration Manager Class
    第三方集成管理器类

    Main manager for third - party service integrations
    第三方服务集成的主要管理器
    """

    def __init__(self, manager_name: str = "default_third_party_integration_manager"):
        """
        Initialize third - party integration manager
        初始化第三方集成管理器

        Args:
            manager_name: Name of the manager
                        管理器名称
        """
        self.manager_name = manager_name
        self.connections: Dict[str, ThirdPartyConnection] = {}
        self.rate_limiter = RateLimiter()
        self.auth_manager = AuthenticationManager()
        self.data_fetcher = DataFetcher(self.auth_manager, self.rate_limiter)
        self.data_processor = DataProcessor()

        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }

        logger.info(f"Third - party integration manager {manager_name} initialized")

    def add_connection(self, connection: ThirdPartyConnection) -> None:
        """
        Add third - party connection
        添加第三方连接

        Args:
            connection: Third - party connection configuration
                       第三方连接配置
        """
        self.connections[connection.connection_id] = connection
        self.auth_manager.set_credentials(
            connection.provider,
            connection.api_key,
            connection.api_secret
        )
        self.rate_limiter.set_limit(connection.provider, connection.rate_limit)

        logger.info(f"Added third - party connection: {connection.connection_id}")

    def fetch_market_data(self,


                          provider: str,
                          data_type: DataType,
                          params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch market data from third - party provider
        从第三方提供商获取市场数据

        Args:
            provider: Provider name
                     提供商名称
            data_type: Type of data to fetch
                      要获取的数据类型
            params: Request parameters
                   请求参数

        Returns:
            dict: Market data
                  市场数据
        """
        response = self.data_fetcher.fetch_data(provider, data_type, params)

        # Update statistics
        self._update_stats(response)

        if response.status_code == 200 and response.data:
            # Process the data
            processed_data = self.data_processor.process_data(provider, response.data)

            return {
                'success': True,
                'data': processed_data,
                'response_time': response.response_time,
                'provider': provider,
                'data_type': data_type.value
            }
        else:
            return {
                'success': False,
                'error': response.error_message,
                'status_code': response.status_code,
                'response_time': response.response_time
            }

    def fetch_news_data(self,


                        provider: str,
                        query: str,
                        from_date: Optional[str] = None,
                        to_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch news data from third - party provider
        从第三方提供商获取新闻数据

        Args:
            provider: Provider name
                     提供商名称
            query: Search query
                  搜索查询
            from_date: Start date (optional)
                      开始日期（可选）
            to_date: End date (optional)
                    结束日期（可选）

        Returns:
            dict: News data
                  新闻数据
        """
        params = {'query': query}

        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date

        response = self.data_fetcher.fetch_data(provider, DataType.MARKET_NEWS, params)

        # Update statistics
        self._update_stats(response)

        if response.status_code == 200 and response.data:
            processed_data = self.data_processor.process_data(provider, response.data)

            return {
                'success': True,
                'news': processed_data,
                'response_time': response.response_time
            }
        else:
            return {
                'success': False,
                'error': response.error_message,
                'status_code': response.status_code
            }

    def get_provider_status(self, provider: str) -> Dict[str, Any]:
        """
        Get provider status and health
        获取提供商状态和健康状况

        Args:
            provider: Provider name
                     提供商名称

        Returns:
            dict: Provider status
                  提供商状态
        """
        if provider not in [conn.provider for conn in self.connections.values()]:
            return {'status': 'not_configured'}

        # Simple health check
        try:
            # Try a simple request to test connectivity
            test_response = self.data_fetcher.fetch_data(
                provider, DataType.STOCK_PRICES, {'symbol': 'AAPL'}
            )

            return {
                'status': 'healthy' if test_response.status_code == 200 else 'unhealthy',
                'response_time': test_response.response_time,
                'last_check': datetime.now()
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'last_check': datetime.now()
            }

    def list_providers(self) -> List[Dict[str, Any]]:
        """
        List configured third - party providers
        列出配置的第三方提供商

        Returns:
            list: List of providers
                  提供商列表
        """
        providers = []

        for connection in self.connections.values():
            status = self.get_provider_status(connection.provider)
            providers.append({
                'connection_id': connection.connection_id,
                'provider': connection.provider,
                'status': status['status'],
                'rate_limit': connection.rate_limit,
                'last_check': status.get('last_check')
            })

        return providers

    def get_integration_stats(self) -> Dict[str, Any]:
        """
        Get third - party integration statistics
        获取第三方集成统计信息

        Returns:
            dict: Integration statistics
                  集成统计信息
        """
        return {
            'manager_name': self.manager_name,
            'total_connections': len(self.connections),
            'stats': self.stats
        }

    def _update_stats(self, response: APIResponse) -> None:
        """
        Update integration statistics
        更新集成统计信息

        Args:
            response: API response
                     API响应
        """
        self.stats['total_requests'] += 1

        if response.status_code and response.status_code < 400:
            self.stats['successful_requests'] += 1
        else:
            self.stats['failed_requests'] += 1

        # Update average response time
        total_requests = self.stats['total_requests']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total_requests - 1)) + response.response_time
        ) / total_requests


# Global third - party integration manager instance
# 全局第三方集成管理器实例
third_party_integration_manager = ThirdPartyIntegrationManager()

__all__ = [
    'ThirdPartyProvider',
    'DataType',
    'ThirdPartyConnection',
    'APIRequest',
    'APIResponse',
    'RateLimiter',
    'AuthenticationManager',
    'DataFetcher',
    'DataProcessor',
    'ThirdPartyIntegrationManager',
    'third_party_integration_manager'
]
