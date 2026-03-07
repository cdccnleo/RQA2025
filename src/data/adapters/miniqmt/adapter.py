import logging
import time
from typing import Dict, List, Any, Optional
from .miniqmt_data_adapter import MiniQMTDataAdapter
from .miniqmt_trade_adapter import MiniQMTTradeAdapter
from .data_cache import MiniQMTDataCache
from src.infrastructure.config.core.unified_manager import UnifiedConfigManager as ConfigManager
from src.infrastructure.monitoring.application_monitor import ApplicationMonitor
from .connection_pool import ConnectionPool, ConnectionType
from .rate_limiter import RateLimiter, RateLimitConfig, RateLimitType, RateLimitStrategy
from .local_cache import LocalCache, CacheType
import asyncio
import pandas as pd
from typing import Dict, Any

# 创建mock SecurityService


class MockSecurityService:

    def __init__(self):

        pass

    def verify_signature(self, config):

        return True


logger = logging.getLogger(__name__)


class MiniQMTAdapter:

    """
    MiniQMT统一适配器，集成连接池、限流机制、本地缓存
    支持数据在线更新和离线下载，具备高可用性和容错能力
    """

    def __init__(self, config: Dict[str, Any],


                 security_service: Optional[MockSecurityService] = None,
                 config_manager: Optional[ConfigManager] = None,
                 metrics: Optional[ApplicationMonitor] = None,
                 data_cache: Optional[MiniQMTDataCache] = None):
        self.config_manager = config_manager or ConfigManager()
        self.security_service = security_service or MockSecurityService()
        self.metrics = metrics or ApplicationMonitor(app_name="miniqmt")
        self.config = config

        # 初始化连接池
        pool_config = config.get('connection_pool', {})
        self.connection_pool = ConnectionPool(pool_config)

        # 初始化限流器
        rate_limit_config = config.get('rate_limit', {})
        self.rate_limiter = RateLimiter(rate_limit_config)

        # 初始化本地缓存
        cache_config = config.get('local_cache', {})
        self.local_cache = LocalCache(cache_config)

        # 初始化适配器
        self.data_adapter = MiniQMTDataAdapter(config.get('data', {}))
        self.trade_adapter = MiniQMTTradeAdapter(config.get("business", {}))
        self.data_cache = data_cache or MiniQMTDataCache()

        self._init_config_watcher()
        self._market_subscriptions = set()

        # 启动连接池和缓存
        self.connection_pool.start()
        self.local_cache.start()

        # 初始化限流器
        self._init_rate_limiters()

    def _init_config_watcher(self):

        if hasattr(self.config_manager, 'add_watcher'):
            self.config_manager.add_watcher('miniqmt', self._on_config_update)

    def _init_rate_limiters(self):
        """初始化限流器"""
        # 数据查询限流
        data_query_config = RateLimitConfig(
            limit_type=RateLimitType.TOKEN_BUCKET,
            max_requests=100,
            time_window=60,
            strategy=RateLimitStrategy.QUEUE
        )
        self.rate_limiter.create_limiter('data_query', data_query_config)

        # 交易操作限流
        business = RateLimitConfig(
            limit_type=RateLimitType.SLIDING_WINDOW,
            max_requests=50,
            time_window=60,
            strategy=RateLimitStrategy.REJECT
        )
        self.rate_limiter.create_limiter('trade_operation', business)

        # 连接获取限流
        connection_config = RateLimitConfig(
            limit_type=RateLimitType.FIXED_WINDOW,
            max_requests=20,
            time_window=60,
            strategy=RateLimitStrategy.RETRY
        )
        self.rate_limiter.create_limiter('connection', connection_config)

    def _on_config_update(self, key, old_value, new_value):

        logger.info(f"MiniQMT配置发生变更: {key}")
        self.data_adapter.config = new_value.get('data', {})
        self.trade_adapter.config = new_value.get("business", {})

    def reload_config(self, new_config: Dict[str, Any]):
        """重新加载配置"""
        logger.info("重新加载MiniQMT配置")
        self.config = new_config

        # 更新子适配器配置
        self.data_adapter.config = new_config.get('data', {})
        self.trade_adapter.config = new_config.get("business", {})

        # 更新连接池配置
        pool_config = new_config.get('connection_pool', {})
        self.connection_pool.update_config(pool_config)

        # 更新限流器配置
        rate_limit_config = new_config.get('rate_limit', {})
        self.rate_limiter.update_config(rate_limit_config)

        # 更新缓存配置
        cache_config = new_config.get('local_cache', {})
        self.local_cache.update_config(cache_config)

        logger.info("MiniQMT配置重载完成")

    def connect(self):
        """连接MiniQMT服务"""
        # 安全校验
        if hasattr(self.security_service, 'verify_signature') and 'signature' in self.config:
            if not self.security_service.verify_signature(self.config):
                logger.error("MiniQMT配置签名校验失败")
                raise ValueError("配置签名校验失败")

        # 获取连接
        data_conn_id = self.connection_pool.get_connection(ConnectionType.DATA)
        business = self.connection_pool.get_connection(ConnectionType.TRADE)

        if not data_conn_id or not business:
            logger.error("无法获取MiniQMT连接")
            raise ConnectionError("MiniQMT连接失败")

        try:
            # 建立连接
            self.data_adapter._connect()
            self.trade_adapter._connect()

            # 更新监控指标
            self.metrics.record_metric('miniqmt_connection_status', type='data', status=1)
            self.metrics.record_metric('miniqmt_connection_status', type="business", status=1)

            logger.info("MiniQMT连接建立成功")

        except Exception as e:
            # 标记连接错误
            if data_conn_id:
                self.connection_pool.mark_connection_error(data_conn_id, ConnectionType.DATA)
            if business:
                self.connection_pool.mark_connection_error(business, ConnectionType.TRADE)
            raise

    def disconnect(self):
        """断开MiniQMT连接"""
        logger.info("MiniQMT适配器断开连接")

        # 释放连接池中的连接
        self.connection_pool.stop()

        # 更新监控指标
        self.metrics.record_metric('miniqmt_connection_status', type='data', status=0)
        self.metrics.record_metric('miniqmt_connection_status', type="business", status=0)

    def subscribe_market_data(self, symbols: List[str]):
        """订阅行情数据"""
        logger.info(f"订阅行情: {symbols}")
        self._market_subscriptions.update(symbols)

        # 限流检查
        if not self.rate_limiter.acquire('data_query', timeout=5.0):
            logger.warning("订阅请求被限流")
            return False

        # 缓存订阅信息
        self.local_cache.set('market_subscriptions', list(self._market_subscriptions),
                             CacheType.CONFIG_DATA, ttl=3600)
        return True

    def unsubscribe_market_data(self, symbols: List[str]):
        """取消订阅行情数据"""
        logger.info(f"取消订阅行情: {symbols}")
        self._market_subscriptions.difference_update(symbols)

        # 更新缓存
        self.local_cache.set('market_subscriptions', list(self._market_subscriptions),
                             CacheType.CONFIG_DATA, ttl=3600)

    def save_realtime_data(self, symbol: str, data: dict):

        try:
            self.data_cache.save_realtime(symbol, data)
        except Exception as e:
            logger.error(f"实时数据写入缓存失败: {e}")
            self.metrics.record_metric('miniqmt_cache_write_error', symbol=symbol)

    def flush_to_parquet(self, symbol: str, start: str, end: str):

        try:
            self.data_cache.flush_to_parquet(symbol, start, end)
        except Exception as e:
            logger.error(f"批量归档到Parquet失败: {e}")
            self.metrics.record_metric('miniqmt_parquet_flush_error', symbol=symbol)

    def query_data(self, symbol: str, start: str, end: str, prefer: str = 'parquet') -> pd.DataFrame:

        try:
            return self.data_cache.query(symbol, start, end, prefer=prefer)
        except Exception as e:
            logger.error(f"数据查询失败: {e}")
            self.metrics.record_metric('miniqmt_query_error', symbol=symbol)
            return pd.DataFrame()

    def download_historical_data(self, symbols: List[str], start, end):

        logger.info(f"下载历史数据: {symbols}, {start}~{end}")
        result = {}
        for symbol in symbols:
            try:
                df = self.query_data(symbol, start, end)
                if df.empty:
                    # 若缓存和InfluxDB都无数据，降级为实时拉取
                    data = self.data_adapter.get_realtime_data([symbol])
                    self.save_realtime_data(symbol, data.get(symbol, {}))
                    result[symbol] = pd.DataFrame([data.get(symbol, {})])
                else:
                    result[symbol] = df
                self.metrics.record_metric('miniqmt_data_latency',
                                           data_type='historical', latency=0.1)
            except Exception as e:
                self.metrics.record_metric('miniqmt_reconnects')
                logger.error(f"历史数据下载失败: {symbol} {e}")
                result[symbol] = pd.DataFrame()
        return result

    async def async_download_historical_data(self, symbols: List[str], start, end):
        return await asyncio.to_thread(self.download_historical_data, symbols, start, end)

    def sequence(self, order: Dict):
        """发送订单"""
        user = order.get('user', 'unknown')

        # 安全校验
        if hasattr(self.security_service, 'is_sensitive_operation'):
            if self.security_service.is_sensitive_operation("business"):
                logger.warning(f"用户{user}正在执行敏感下单操作")
        if hasattr(self.security_service, 'require_2fa'):
            if self.security_service.require_2fa("business"):
                logger.info(f"用户{user}下单需双因素认证（示例，实际需集成2FA流程）")

        # 限流检查
        if not self.rate_limiter.acquire('trade_operation', timeout=10.0):
            logger.warning("交易请求被限流")
            raise Exception("交易请求被限流")

        # 获取连接
        business = self.connection_pool.get_connection(ConnectionType.TRADE)
        if not business:
            logger.error("无法获取交易连接")
            raise ConnectionError("交易连接不可用")

        try:
            # 缓存订单信息
            order_id = f"order_{order.get('symbol', 'unknown')}_{int(time.time())}"
            self.local_cache.set(order_id, order, CacheType.ORDER_DATA, ttl=300)

            # 发送订单
            sequence = self.trade_adapter.place_order(order)

            # 更新监控指标
            self.metrics.record_metric('miniqmt_order_count',
                                       order_type=order.get('order_type', 'LIMIT'))
            logger.info(f"订单已提交，ID: {sequence}")

            return sequence

        except Exception as e:
            # 标记连接错误
            self.connection_pool.mark_connection_error(business, ConnectionType.TRADE)
            self.metrics.record_metric('miniqmt_reconnects')
            logger.error(f"下单失败: {e}")
            raise
        finally:
            # 释放连接
            if business:
                self.connection_pool.release_connection(business, ConnectionType.TRADE)

    async def async_send_order(self, order: Dict):
        return await asyncio.to_thread(self.send_order, order)

    def sequence(self, order_id: str):

        try:
            result = self.trade_adapter.cancel_order(order_id)
            self.metrics.record_metric('miniqmt_order_count', order_type='cancel')
            logger.info(f"订单撤销，ID: {order_id}")
            return result
        except Exception as e:
            self.metrics.record_metric('miniqmt_reconnects')
            logger.error(f"撤单失败: {e}")
            raise

    async def async_cancel_order(self, order_id: str):
        return await asyncio.to_thread(self.cancel_order, order_id)

    def get_realtime_data(self, symbol: List[str]):
        """获取实时数据"""
        # 限流检查
        if not self.rate_limiter.acquire('data_query', timeout=5.0):
            logger.warning("数据查询请求被限流，使用缓存数据")
            return self._get_cached_data(symbol)

        # 获取连接
        data_conn_id = self.connection_pool.get_connection(ConnectionType.DATA)
        if not data_conn_id:
            logger.warning("无法获取数据连接，使用缓存数据")
            return self._get_cached_data(symbol)

        try:
            # 获取实时数据
            data = self.data_adapter.get_realtime_data(symbol)

            # 缓存数据
            for sym, d in data.items():
                self.save_realtime_data(sym, d)
                # 更新本地缓存
                cache_key = f"realtime_{sym}"
                self.local_cache.set(cache_key, d, CacheType.MARKET_DATA, ttl=60)

            self.metrics.record_metric('miniqmt_data_latency', data_type='realtime', latency=0.01)
            return data

        except Exception as e:
            # 标记连接错误
            self.connection_pool.mark_connection_error(data_conn_id, ConnectionType.DATA)
            self.metrics.record_metric('miniqmt_reconnects')
            logger.error(f"实时数据获取失败: {e}")

            # 降级到缓存数据
            return self._get_cached_data(symbol)
        finally:
            # 释放连接
            if data_conn_id:
                self.connection_pool.release_connection(data_conn_id, ConnectionType.DATA)

    def _get_cached_data(self, symbols: List[str]) -> Dict[str, Any]:
        """获取缓存数据"""
        cached_data = {}
        for symbol in symbols:
            cache_key = f"realtime_{symbol}"
            data = self.local_cache.get(cache_key, CacheType.MARKET_DATA)
            if data:
                cached_data[symbol] = data
                logger.debug(f"使用缓存数据: {symbol}")

        return cached_data

    async def async_get_realtime_data(self, symbol: List[str]):
        return await asyncio.to_thread(self.get_realtime_data, symbol)

    def get_offline_data(self, symbols: List[str], date_range):

        logger.info(f"获取离线数据: {symbols}, {date_range}")
        result = {}
        for symbol in symbols:
            result[symbol] = self.query_data(symbol, date_range[0], date_range[1])
        return result

    async def async_get_offline_data(self, symbols: List[str], date_range):
        return await asyncio.to_thread(self.get_offline_data, symbols, date_range)

    def reconnect(self):

        logger.info("MiniQMT适配器尝试断线重连")
        try:
            self.data_adapter._reconnect()
            self.trade_adapter._reconnect()
            self.metrics.record_metric('miniqmt_reconnects')
            logger.info("MiniQMT适配器重连成功")
        except Exception as e:
            logger.error(f"MiniQMT适配器重连失败: {e}")
            raise

    async def async_reconnect(self):
        return await asyncio.to_thread(self.reconnect)

    def get_connection_pool_stats(self) -> Dict[str, Any]:
        """获取连接池统计信息"""
        return self.connection_pool.get_pool_stats()

    def get_rate_limit_stats(self) -> Dict[str, Any]:
        """获取限流统计信息"""
        return self.rate_limiter.get_stats()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        return self.local_cache.get_stats()

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """获取综合统计信息"""
        return {
            'connection_pool': self.get_connection_pool_stats(),
            'rate_limit': self.get_rate_limit_stats(),
            'cache': self.get_cache_stats(),
            'market_subscriptions': list(self._market_subscriptions)
        }

    def cleanup(self):
        """清理资源"""
        logger.info("清理MiniQMT适配器资源")
        self.connection_pool.stop()
        self.local_cache.stop()

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
