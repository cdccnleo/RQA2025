#!/usr/bin/env python3
"""
RQA2025 加密货币数据加载器

支持从多个加密货币数据源获取数据：
- CoinGecko API: 免费加密货币数据
- Binance API: 实时交易数据
- 本地缓存: 减少API调用频率
"""

# 使用基础设施层日志

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import aiohttp
import pandas as pd
import requests
from dataclasses import dataclass

from ..core.base_loader import BaseDataLoader, LoaderConfig
from ..cache.cache_manager import CacheManager, CacheConfig

# 配置日志
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
requests = requests  # 兼容测试对 `crypto_loader.requests` 的打补丁路径

@dataclass
class CryptoData:

    """加密货币数据结构"""
    symbol: str
    name: str
    price: float
    volume_24h: float
    market_cap: float
    price_change_24h: float
    price_change_percentage_24h: float
    high_24h: float
    low_24h: float
    timestamp: datetime
    source: str


@dataclass
class CryptoMarketData:

    """加密货币市场数据结构"""
    total_market_cap: float
    total_volume_24h: float
    market_cap_percentage: Dict[str, float]
    market_cap_change_24h: float
    timestamp: datetime


class CoinGeckoLoader(BaseDataLoader):

    """CoinGecko API数据加载器"""

    def __init__(self, api_key: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key
        }
        super().__init__(config)
        self.base_url = "https://api.coingecko.com / api / v3"
        self.api_key = api_key
        self.session = None
        # 为CacheManager提供配置
        cache_config = CacheConfig(
            max_size=1000,
            ttl=3600,
            enable_disk_cache=True,
            disk_cache_dir="cache",
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        config_obj = self.config if isinstance(self.config, dict) else getattr(self.config, "__dict__", {})
        return all(config_obj.get(field) is not None for field in self.get_required_config_fields())

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "coingecko",
            "version": "1.0.0",
            "description": "CoinGecko API数据加载器",
            "supported_sources": ["coingecko"],
            "supported_frequencies": ["1d"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - CryptoLoader / 1.0',
                'Accept': 'application / json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_top_coins(self, limit: int = 100) -> List[CryptoData]:
        """获取市值排名前N的加密货币"""
        cache_key = f"coingecko_top_coins_{limit}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            logger.info(f"从缓存获取CoinGecko数据: {limit}个币种")
            return [CryptoData(**coin) for coin in cached_data]

        try:
            url = f"{self.base_url}/coins / markets"
            params = {
                'vs_currency': 'usd',
                "sequence": 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    crypto_data = []
                    for coin in data:
                        crypto_data.append(CryptoData(
                            symbol=coin['symbol'].upper(),
                            name=coin['name'],
                            price=coin['current_price'],
                            volume_24h=coin['total_volume'],
                            market_cap=coin['market_cap'],
                            price_change_24h=coin['price_change_24h'],
                            price_change_percentage_24h=coin['price_change_percentage_24h'],
                            high_24h=coin['high_24h'],
                            low_24h=coin['low_24h'],
                            timestamp=datetime.fromtimestamp(coin['last_updated'] / 1000),
                            source='coingecko'
                        ))

                    # 缓存数据（5分钟）
                    await self.cache_manager.set(cache_key, [coin.__dict__ for coin in crypto_data], ttl=300)

                    logger.info(f"成功获取CoinGecko数据: {len(crypto_data)}个币种")
                    return crypto_data
                else:
                    logger.error(f"CoinGecko API请求失败: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"获取CoinGecko数据时发生错误: {e}")
            return []

    async def get_coin_detail(self, coin_id: str) -> Optional[Dict[str, Any]]:
        """获取特定币种的详细信息"""
        cache_key = f"coingecko_coin_detail_{coin_id}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/coins/{coin_id}"
            params = {
                'localization': False,
                'tickers': False,
                'market_data': True,
                'community_data': False,
                'developer_data': False,
                'sparkline': False
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（10分钟）
                    await self.cache_manager.set(cache_key, data, ttl=600)

                    logger.info(f"成功获取币种详情: {coin_id}")
                    return data
                else:
                    logger.error(f"获取币种详情失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取币种详情时发生错误: {e}")
            return None

    async def get_market_data(self) -> Optional[CryptoMarketData]:
        """获取市场整体数据"""
        cache_key = "coingecko_market_data"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return CryptoMarketData(**cached_data)

        try:
            url = f"{self.base_url}/global"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    market_data = data['data']

                    market_info = CryptoMarketData(
                        total_market_cap=market_data['total_market_cap']['usd'],
                        total_volume_24h=market_data['total_volume']['usd'],
                        market_cap_percentage=market_data['market_cap_percentage'],
                        market_cap_change_24h=market_data['market_cap_change_percentage_24h_usd'],
                        timestamp=datetime.now()
                    )

                    # 缓存数据（5分钟）
                    await self.cache_manager.set(cache_key, market_info.__dict__, ttl=300)

                    logger.info("成功获取市场整体数据")
                    return market_info
                else:
                    logger.error(f"获取市场数据失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取市场数据时发生错误: {e}")
            return None


class BinanceLoader(BaseDataLoader):

    """Binance API数据加载器"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):

        config = {
            'cache_dir': 'cache',
            'max_retries': 3,
            'api_key': api_key,
            'api_secret': api_secret
        }
        super().__init__(config)
        self.base_url = "https://api.binance.com / api / v3"
        self.api_key = api_key
        self.api_secret = api_secret
        self.session = None
        # 为CacheManager提供配置
        cache_config = CacheConfig(
            max_size=1000,
            ttl=3600,
            enable_disk_cache=True,
            disk_cache_dir="cache",
            compression=False,
            encryption=False,
            enable_stats=True,
            cleanup_interval=300,
            max_file_size=10 * 1024 * 1024,
            backup_enabled=False,
            backup_interval=3600
        )
        self.cache_manager = CacheManager(cache_config)

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        config_obj = self.config if isinstance(self.config, dict) else getattr(self.config, "__dict__", {})
        return all(config_obj.get(field) is not None for field in self.get_required_config_fields())

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "binance",
            "version": "1.0.0",
            "description": "Binance API数据加载器",
            "supported_sources": ["binance"],
            "supported_frequencies": ["1m", "5m", "15m", "1h", "4h", "1d"]
        }

    def load(self, start_date: str, end_date: str, frequency: str) -> Any:
        """
        统一的数据加载接口
        """
        # 这里实现同步加载逻辑，或者抛出异常提示使用异步方法
        raise NotImplementedError("Use load_data() method for async loading")

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={
                'User - Agent': 'RQA2025 - BinanceLoader / 1.0',
                'Accept': 'application / json'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def get_ticker_24hr(self, symbol: str = "BTCUSDT") -> Optional[Dict[str, Any]]:
        """获取24小时价格变动统计"""
        cache_key = f"binance_ticker_24hr_{symbol}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/ticker / 24hr"
            params = {'symbol': symbol}

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（1分钟）
                    await self.cache_manager.set(cache_key, data, ttl=60)

                    logger.info(f"成功获取Binance 24小时数据: {symbol}")
                    return data
                else:
                    logger.error(f"Binance API请求失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取Binance数据时发生错误: {e}")
            return None

    async def get_klines(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 100) -> List[Dict[str, Any]]:
        """获取K线数据"""
        cache_key = f"binance_klines_{symbol}_{interval}_{limit}"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }

            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()

                    # 转换为结构化数据
                    klines = []
                    for kline in data:
                        klines.append({
                            'open_time': datetime.fromtimestamp(kline[0] / 1000),
                            'open': float(kline[1]),
                            'high': float(kline[2]),
                            'low': float(kline[3]),
                            'close': float(kline[4]),
                            'volume': float(kline[5]),
                            'close_time': datetime.fromtimestamp(kline[6] / 1000),
                            'quote_volume': float(kline[7]),
                            'trades': int(kline[8]),
                            'taker_buy_base': float(kline[9]),
                            'taker_buy_quote': float(kline[10])
                        })

                    # 缓存数据（5分钟）
                    await self.cache_manager.set(cache_key, klines, ttl=300)

                    logger.info(f"成功获取Binance K线数据: {symbol} {interval}")
                    return klines
                else:
                    logger.error(f"获取K线数据失败: {response.status}")
                    return []

        except Exception as e:
            logger.error(f"获取K线数据时发生错误: {e}")
            return []

    async def get_exchange_info(self) -> Optional[Dict[str, Any]]:
        """获取交易所信息"""
        cache_key = "binance_exchange_info"

        # 检查缓存
        cached_data = await self.cache_manager.get(cache_key)
        if cached_data:
            return cached_data

        try:
            url = f"{self.base_url}/exchangeInfo"

            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()

                    # 缓存数据（1小时）
                    await self.cache_manager.set(cache_key, data, ttl=3600)

                    logger.info("成功获取Binance交易所信息")
                    return data
                else:
                    logger.error(f"获取交易所信息失败: {response.status}")
                    return None

        except Exception as e:
            logger.error(f"获取交易所信息时发生错误: {e}")
            return None


class CryptoDataLoader(BaseDataLoader):

    """统一的加密货币数据加载器"""

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str, Path]] = None,
        save_path: Union[str, Path, None] = None,
        max_retries: int = 3,
        cache_days: int = 1,
        timeout: int = 30,
    ):

        if isinstance(config, (str, Path)):
            config = {"cache_dir": str(config)}
        config = dict(config) if isinstance(config, dict) else (config or {})

        loader_config = LoaderConfig(
            name="crypto_loader",
            max_retries=max_retries,
            timeout=timeout,
        )
        super().__init__(loader_config)

        resolved_cache_dir = Path(save_path) if save_path else Path(config.get("cache_dir", Path.cwd() / "data" / "crypto"))
        self.save_path = resolved_cache_dir
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.cache_dir = self.save_path / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_retries = max_retries
        self.cache_days = cache_days
        self.timeout = timeout
        self.config = config
        self.config.setdefault("cache_dir", str(self.save_path))

        self.coingecko_loader = None
        self.binance_loader = None
        cache_config = CacheConfig(
            max_size=config.get('max_size', 512),
            ttl=max(int(cache_days * 86400), 3600),
            enable_disk_cache=True,
            disk_cache_dir=str(self.cache_dir),
            compression=False,
            encryption=False,
            enable_stats=True,
        )
        self.cache_manager = CacheManager(cache_config)
        self.supported_sources = ["coingecko", "binance"]

    async def initialize(self):
        """初始化数据加载器"""
        logger.info("初始化加密货币数据加载器...")

        # 初始化CoinGecko加载器
        api_key = self.config.get('coingecko_api_key')
        self.coingecko_loader = CoinGeckoLoader(api_key)

        # 初始化Binance加载器
        binance_api_key = self.config.get('binance_api_key')
        binance_api_secret = self.config.get('binance_api_secret')
        self.binance_loader = BinanceLoader(binance_api_key, binance_api_secret)

    async def get_crypto_data(self, symbols: List[str] = None, source: str = "coingecko") -> List[CryptoData]:
        """获取加密货币数据"""
        if not symbols:
            symbols = ["BTC", "ETH", "BNB", "ADA", "SOL"]

        if source == "coingecko":
            return await self._get_coingecko_data(symbols)
        elif source == "binance":
            return await self._get_binance_data(symbols)
        else:
            logger.error(f"不支持的数据源: {source}")
            return []

    async def _get_coingecko_data(self, symbols: List[str]) -> List[CryptoData]:
        """从CoinGecko获取数据"""
        async with self.coingecko_loader as loader:
            # 获取所有币种数据
            all_coins = await loader.get_top_coins(limit=100)

            # 过滤指定币种
            filtered_coins = [coin for coin in all_coins if coin.symbol in symbols]

            return filtered_coins

    async def _get_binance_data(self, symbols: List[str]) -> List[CryptoData]:
        """从Binance获取数据"""
        async with self.binance_loader as loader:
            crypto_data = []

            for symbol in symbols:
                # 转换为Binance格式
                binance_symbol = f"{symbol}USDT"

                ticker_data = await loader.get_ticker_24hr(binance_symbol)
                if ticker_data:
                    crypto_data.append(CryptoData(
                        symbol=symbol,
                        name=symbol,  # Binance API不提供名称
                        price=float(ticker_data['lastPrice']),
                        volume_24h=float(ticker_data['volume']),
                        market_cap=0,  # Binance API不提供市值
                        price_change_24h=float(ticker_data['priceChange']),
                        price_change_percentage_24h=float(ticker_data['priceChangePercent']),
                        high_24h=float(ticker_data['highPrice']),
                        low_24h=float(ticker_data['lowPrice']),
                        timestamp=datetime.now(),
                        source='binance'
                    ))

            return crypto_data

    async def get_market_overview(self) -> Optional[CryptoMarketData]:
        """获取市场概览"""
        async with self.coingecko_loader as loader:
            return await loader.get_market_data()

    async def get_historical_data(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """获取历史数据"""
        async with self.binance_loader as loader:
            return await loader.get_klines(f"{symbol}USDT", "1d", days)

    async def validate_data(self, data: List[CryptoData]) -> Dict[str, Any]:
        """验证加密货币数据"""
        validation_result = {
            'valid': True,
            'total_records': len(data),
            'valid_records': 0,
            'invalid_records': 0,
            'errors': []
        }

        for crypto in data:
            try:
                # 基本验证
                if crypto.price <= 0:
                    validation_result['errors'].append(f"{crypto.symbol}: 价格无效")
                    validation_result['invalid_records'] += 1
                    continue

                if crypto.volume_24h < 0:
                    validation_result['errors'].append(f"{crypto.symbol}: 交易量无效")
                    validation_result['invalid_records'] += 1
                    continue

                # 价格合理性检查
                if crypto.price > 1000000:  # 价格超过100万美元
                    validation_result['errors'].append(f"{crypto.symbol}: 价格异常高")
                    validation_result['invalid_records'] += 1
                    continue

                validation_result['valid_records'] += 1

            except Exception as e:
                validation_result['errors'].append(f"{crypto.symbol}: 验证异常 - {str(e)}")
                validation_result['invalid_records'] += 1

        validation_result['valid'] = validation_result['invalid_records'] == 0

        return validation_result

    def get_required_config_fields(self) -> list:
        """获取必需的配置字段列表"""
        return ['cache_dir', 'max_retries']

    def validate_config(self) -> bool:
        """验证配置有效性"""
        return self._validate_config()

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据加载器的元数据"""
        return {
            "loader_type": "crypto",
            "version": "1.0.0",
            "description": "加密货币数据加载器",
            "supported_sources": ["coingecko", "binance"],
            "supported_frequencies": ["1m", "5m", "15m", "1h", "4h", "1d"],
            "max_retries": self.max_retries,
            "cache_days": self.cache_days,
            "timeout": self.timeout,
        }

    def load(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "coingecko",
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """同步加载单个加密货币行情。"""
        start_str, end_str = self._normalize_dates(start_date, end_date)
        cache_key = f"{symbol}_{start_str}_{end_str}_{source}"

        if not force_refresh:
            cached_df = self.cache_manager.get(cache_key)
            if isinstance(cached_df, pd.DataFrame):
                return cached_df.copy()

            csv_path = self._cache_file_path(symbol, start_str, end_str, source)
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
                    self.cache_manager.set(cache_key, df.copy())
                    self._update_stats()
                    return df
                except Exception:
                    logger.debug("读取加密货币缓存文件失败: %s", csv_path)

        records = self._fetch_snapshot(symbol, start_str, end_str, source)
        df = self._records_to_dataframe(symbol, records)
        self.cache_manager.set(cache_key, df.copy())
        self._persist_to_disk(symbol, start_str, end_str, source, df)
        self._update_stats()
        return df

    def load_batch(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        source: str = "coingecko",
        max_workers: int = 4,
        force_refresh: bool = False,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        if not symbols:
            return {}

        def task(sym: str):
            try:
                data = self.load(
                    symbol=sym,
                    start_date=start_date,
                    end_date=end_date,
                    source=source,
                    force_refresh=force_refresh,
                )
                return sym, data
            except Exception as exc:  # pragma: no cover - 容错路径
                logger.warning("加载加密货币 %s 数据失败: %s", sym, exc)
                return sym, None

        results: Dict[str, Optional[pd.DataFrame]] = {}
        with ThreadPoolExecutor(max_workers=max_workers or 1) as executor:
            futures = [executor.submit(task, sym) for sym in symbols]
            for future in as_completed(futures):
                sym, value = future.result()
                results[sym] = value
        return results

    async def load_data(self, **kwargs) -> Dict[str, Any]:
        """实现IDataLoader接口的load_data方法"""
        try:
            await self.initialize()

            symbols = kwargs.get('symbols', ["BTC", "ETH", "BNB", "ADA", "SOL"])
            source = kwargs.get('source', 'coingecko')

            # 获取数据
            crypto_data = await self.get_crypto_data(symbols, source)

            # 验证数据
            validation_result = await self.validate_data(crypto_data)

            # 转换为DataFrame
            df = pd.DataFrame([{
                'symbol': coin.symbol,
                'name': coin.name,
                'price': coin.price,
                'volume_24h': coin.volume_24h,
                'market_cap': coin.market_cap,
                'price_change_24h': coin.price_change_24h,
                'price_change_percentage_24h': coin.price_change_percentage_24h,
                'high_24h': coin.high_24h,
                'low_24h': coin.low_24h,
                'timestamp': coin.timestamp,
                'source': coin.source
            } for coin in crypto_data])

            return {
                'data': df,
                'metadata': {
                    'source': source,
                    'symbols': symbols,
                    'total_records': len(crypto_data),
                    'timestamp': datetime.now().isoformat(),
                    'validation_result': validation_result
                }
            }

        except Exception as e:
            logger.error(f"加载加密货币数据时发生错误: {e}")
            return {
                'data': pd.DataFrame(),
                'metadata': {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
            }

    # ------------------------------------------------------------------ #
    # 同步加载辅助方法
    # ------------------------------------------------------------------ #
    def _normalize_dates(
        self,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
    ) -> tuple[str, str]:
        def _to_str(value: Union[str, datetime]) -> str:
            if isinstance(value, datetime):
                return value.strftime("%Y-%m-%d")
            return str(value)

        return _to_str(start_date), _to_str(end_date)

    def _fetch_snapshot(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str,
    ) -> List[Dict[str, Any]]:
        url = f"https://api.mock-crypto.local/{source}/prices"
        params = {
            "symbol": symbol,
            "start_date": start,
            "end_date": end,
        }
        response = requests.get(
            url,
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json() or {}
        return payload.get("data", []) or []

    def _records_to_dataframe(self, symbol: str, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()

        rows = []
        for item in records:
            ts = item.get("timestamp")
            timestamp = pd.to_datetime(ts) if ts else None
            def _to_float(key: str, fallback: float) -> float:
                value = item.get(key, fallback)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return fallback

            close_val = _to_float("close", 0.0)
            open_val = _to_float("open", close_val)
            high_val = _to_float("high", max(open_val, close_val))
            low_val = _to_float("low", min(open_val, close_val))
            volume_val = _to_float("volume", 0.0)
            rows.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "open": open_val,
                    "high": max(high_val, open_val, close_val, low_val),
                    "low": min(low_val, open_val, close_val, high_val),
                    "close": close_val,
                    "volume": volume_val,
                }
            )

        df = pd.DataFrame(rows)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp").sort_index()
        return df[["open", "high", "low", "close", "volume"]]

    def _cache_file_path(self, symbol: str, start: str, end: str, source: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{start}_{end}_{source}.csv"

    def _persist_to_disk(
        self,
        symbol: str,
        start: str,
        end: str,
        source: str,
        df: pd.DataFrame,
    ) -> None:
        try:
            csv_path = self._cache_file_path(symbol, start, end, source)
            df.to_csv(csv_path)
        except Exception:
            logger.debug("Failed to persist crypto data for %s %s-%s", symbol, start, end)


# 便捷函数
async def get_crypto_data(symbols: List[str] = None, source: str = "coingecko") -> Dict[str, Any]:
    """获取加密货币数据的便捷函数"""
    loader = CryptoDataLoader()
    return await loader.load_data(symbols=symbols, source=source)


async def get_market_overview() -> Optional[CryptoMarketData]:
    """获取市场概览的便捷函数"""
    loader = CryptoDataLoader()
    await loader.initialize()
    return await loader.get_market_overview()


if __name__ == "__main__":
    # 测试代码
    async def test_crypto_loader():
        """测试加密货币数据加载器"""
        print("测试加密货币数据加载器...")

        loader = CryptoDataLoader()
        result = await loader.load_data(symbols=["BTC", "ETH"], source="coingecko")

        print(f"加载结果: {len(result['data'])} 条记录")
        print(f"元数据: {result['metadata']}")

        if not result['data'].empty:
            print("\n数据预览:")
            print(result['data'].head())

    asyncio.run(test_crypto_loader())
