"""
RQA2025高级缓存系统

实现多级缓存策略、缓存预热、智能失效等高级缓存功能。
"""

from typing import Dict, Any, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
import asyncio
import logging
import json
import hashlib
from abc import ABC, abstractmethod
import redis.asyncio as redis
import pickle
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """缓存后端抽象基类"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """清空缓存"""
        pass


class MemoryCache(CacheBackend):
    """内存缓存实现"""

    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.executor = ThreadPoolExecutor(max_workers=2)

    async def get(self, key: str) -> Optional[Any]:
        """获取内存缓存值"""
        def _get():
            if key in self.cache:
                entry = self.cache[key]
                if datetime.utcnow() < entry['expires_at']:
                    return entry['value']
                else:
                    # 过期删除
                    del self.cache[key]
            return None

        return await asyncio.get_event_loop().run_in_executor(self.executor, _get)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置内存缓存值"""
        def _set():
            # 检查是否需要清理过期项
            self._cleanup_expired()

            # 检查容量限制
            if len(self.cache) >= self.max_size:
                self._evict_lru()

            expires_at = datetime.utcnow() + timedelta(seconds=ttl) if ttl else datetime.utcnow() + timedelta(days=365)
            self.cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'access_time': datetime.utcnow()
            }
            return True

        return await asyncio.get_event_loop().run_in_executor(self.executor, _set)

    async def delete(self, key: str) -> bool:
        """删除内存缓存值"""
        def _delete():
            if key in self.cache:
                del self.cache[key]
                return True
            return False

        return await asyncio.get_event_loop().run_in_executor(self.executor, _delete)

    async def exists(self, key: str) -> bool:
        """检查内存缓存键是否存在"""
        def _exists():
            if key in self.cache:
                entry = self.cache[key]
                if datetime.utcnow() < entry['expires_at']:
                    return True
                else:
                    del self.cache[key]
            return False

        return await asyncio.get_event_loop().run_in_executor(self.executor, _exists)

    async def clear(self) -> bool:
        """清空内存缓存"""
        def _clear():
            self.cache.clear()
            return True

        return await asyncio.get_event_loop().run_in_executor(self.executor, _clear)

    def _cleanup_expired(self):
        """清理过期缓存项"""
        now = datetime.utcnow()
        expired_keys = [
            key for key, entry in self.cache.items()
            if now >= entry['expires_at']
        ]
        for key in expired_keys:
            del self.cache[key]

    def _evict_lru(self):
        """LRU淘汰策略"""
        if not self.cache:
            return

        # 找到最少使用的项
        lru_key = min(self.cache.keys(),
                     key=lambda k: self.cache[k]['access_time'])
        del self.cache[lru_key]


class RedisCache(CacheBackend):
    """Redis缓存实现"""

    def __init__(self, host: str = 'localhost', port: int = 6379,
                 password: Optional[str] = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self._redis: Optional[redis.Redis] = None
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def _get_connection(self) -> redis.Redis:
        """获取Redis连接"""
        if self._redis is None:
            self._redis = redis.Redis(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=False,  # 保持二进制数据
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """获取Redis缓存值"""
        try:
            redis_conn = await self._get_connection()
            value = await redis_conn.get(key)
            if value:
                # 反序列化
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis获取缓存失败: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置Redis缓存值"""
        try:
            redis_conn = await self._get_connection()
            # 序列化
            serialized_value = pickle.dumps(value)

            if ttl:
                await redis_conn.setex(key, ttl, serialized_value)
            else:
                await redis_conn.set(key, serialized_value)

            return True
        except Exception as e:
            logger.error(f"Redis设置缓存失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除Redis缓存值"""
        try:
            redis_conn = await self._get_connection()
            await redis_conn.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis删除缓存失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查Redis缓存键是否存在"""
        try:
            redis_conn = await self._get_connection()
            return await redis_conn.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis检查缓存存在失败: {e}")
            return False

    async def clear(self) -> bool:
        """清空Redis缓存"""
        try:
            redis_conn = await self._get_connection()
            await redis_conn.flushdb()
            return True
        except Exception as e:
            logger.error(f"Redis清空缓存失败: {e}")
            return False


class MultiLevelCache:
    """多级缓存实现"""

    def __init__(self):
        self.l1_cache = MemoryCache(max_size=10000)  # L1: 内存缓存
        self.l2_cache = RedisCache()                  # L2: Redis缓存
        self.cache_stats = {
            'l1_hits': 0,
            'l1_misses': 0,
            'l2_hits': 0,
            'l2_misses': 0,
            'sets': 0,
            'deletes': 0
        }

    async def get(self, key: str) -> Optional[Any]:
        """多级缓存获取"""
        # L1缓存检查
        value = await self.l1_cache.get(key)
        if value is not None:
            self.cache_stats['l1_hits'] += 1
            return value

        self.cache_stats['l1_misses'] += 1

        # L2缓存检查
        value = await self.l2_cache.get(key)
        if value is not None:
            self.cache_stats['l2_hits'] += 1
            # 回填L1缓存
            await self.l1_cache.set(key, value, ttl=300)  # 5分钟TTL
            return value

        self.cache_stats['l2_misses'] += 1
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """多级缓存设置"""
        self.cache_stats['sets'] += 1

        # 同时设置L1和L2缓存
        l1_success = await self.l1_cache.set(key, value, ttl=min(ttl or 3600, 300))  # L1最多5分钟
        l2_success = await self.l2_cache.set(key, value, ttl)

        return l1_success and l2_success

    async def delete(self, key: str) -> bool:
        """多级缓存删除"""
        self.cache_stats['deletes'] += 1

        # 同时从L1和L2删除
        l1_success = await self.l1_cache.delete(key)
        l2_success = await self.l2_cache.delete(key)

        return l1_success or l2_success  # 只要一个成功就认为成功

    async def exists(self, key: str) -> bool:
        """多级缓存存在检查"""
        # 先检查L1，如果不存在再检查L2
        return await self.l1_cache.exists(key) or await self.l2_cache.exists(key)

    async def clear(self) -> bool:
        """清空多级缓存"""
        l1_success = await self.l1_cache.clear()
        l2_success = await self.l2_cache.clear()

        return l1_success and l2_success

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.cache_stats['l1_hits'] + self.cache_stats['l1_misses']
        hit_rate = (self.cache_stats['l1_hits'] + self.cache_stats['l2_hits']) / total_requests if total_requests > 0 else 0

        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'overall_hit_rate': hit_rate,
            'l1_hit_rate': self.cache_stats['l1_hits'] / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self.cache_stats['l2_hits'] / (self.cache_stats['l1_misses'] or 1)
        }


class CacheWarmer:
    """缓存预热器"""

    def __init__(self, cache: MultiLevelCache, data_providers: Dict[str, Callable]):
        self.cache = cache
        self.data_providers = data_providers
        self.warmup_stats = {
            'total_items': 0,
            'successful_warups': 0,
            'failed_warups': 0,
            'total_time': 0
        }

    async def warmup_cache(self, keys: List[str]) -> Dict[str, Any]:
        """缓存预热"""
        start_time = asyncio.get_event_loop().time()
        results = []

        # 批量预热
        tasks = []
        for key in keys:
            if key in self.data_providers:
                tasks.append(self._warmup_single_key(key))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time

        # 更新统计
        self.warmup_stats['total_items'] = len(keys)
        self.warmup_stats['successful_warups'] = sum(1 for r in results if not isinstance(r, Exception))
        self.warmup_stats['failed_warups'] = sum(1 for r in results if isinstance(r, Exception))
        self.warmup_stats['total_time'] = total_time

        return {
            'results': results,
            'stats': self.warmup_stats.copy(),
            'total_time': total_time
        }

    async def _warmup_single_key(self, key: str) -> str:
        """预热单个缓存键"""
        try:
            provider = self.data_providers[key]
            data = await provider()
            await self.cache.set(key, data, ttl=3600)  # 1小时TTL
            return f"成功预热: {key}"
        except Exception as e:
            logger.error(f"缓存预热失败 {key}: {e}")
            raise

    async def get_popular_keys(self) -> List[str]:
        """获取热门缓存键（需要根据业务逻辑实现）"""
        # 这里应该根据实际业务逻辑获取热门数据键
        # 示例实现
        popular_keys = [
            'market_data_aapl',
            'market_data_goog',
            'market_data_msft',
            'popular_stocks_list',
            'market_indices'
        ]
        return popular_keys


class SmartCacheInvalidator:
    """智能缓存失效器"""

    def __init__(self, cache: MultiLevelCache):
        self.cache = cache
        self.invalidation_patterns: Dict[str, List[str]] = {}
        self.invalidation_stats = {
            'invalidations_triggered': 0,
            'keys_invalidated': 0,
            'patterns_matched': 0
        }

    def register_invalidation_pattern(self, pattern: str, related_keys: List[str]):
        """注册缓存失效模式"""
        self.invalidation_patterns[pattern] = related_keys

    async def invalidate_by_pattern(self, pattern: str) -> int:
        """按模式失效缓存"""
        if pattern not in self.invalidation_patterns:
            return 0

        related_keys = self.invalidation_patterns[pattern]
        invalidated_count = 0

        for key in related_keys:
            if await self.cache.exists(key):
                await self.cache.delete(key)
                invalidated_count += 1

        self.invalidation_stats['invalidations_triggered'] += 1
        self.invalidation_stats['keys_invalidated'] += invalidated_count
        self.invalidation_stats['patterns_matched'] += 1

        logger.info(f"按模式 '{pattern}' 失效了 {invalidated_count} 个缓存键")
        return invalidated_count

    async def invalidate_market_data(self, symbol: str):
        """失效市场数据缓存"""
        patterns = [
            f"market_data_{symbol}",
            f"market_data_{symbol}_*",
            "market_summary",
            "popular_stocks"
        ]

        total_invalidated = 0
        for pattern in patterns:
            count = await self.invalidate_by_pattern(pattern)
            total_invalidated += count

        return total_invalidated

    async def invalidate_user_portfolio(self, user_id: str):
        """失效用户投资组合缓存"""
        patterns = [
            f"user_portfolio_{user_id}",
            f"user_positions_{user_id}",
            f"user_pnl_{user_id}"
        ]

        total_invalidated = 0
        for pattern in patterns:
            count = await self.invalidate_by_pattern(pattern)
            total_invalidated += count

        return total_invalidated

    def get_stats(self) -> Dict[str, Any]:
        """获取失效统计"""
        return self.invalidation_stats.copy()


# 缓存配置
CACHE_CONFIG = {
    'memory_cache': {
        'max_size': 10000,
        'default_ttl': 300  # 5分钟
    },
    'redis_cache': {
        'host': 'localhost',
        'port': 6379,
        'password': None,
        'db': 0,
        'default_ttl': 3600  # 1小时
    },
    'warmup': {
        'batch_size': 100,
        'concurrency': 10,
        'retry_attempts': 3
    },
    'invalidation': {
        'batch_size': 50,
        'async_invalidation': True
    }
}


class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.cache = MultiLevelCache()
        self.warmer = None
        self.invalidator = SmartCacheInvalidator(self.cache)
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """设置默认缓存失效模式"""
        # 市场数据失效模式
        self.invalidator.register_invalidation_pattern(
            'market_data_update',
            ['market_summary', 'popular_stocks', 'market_indices']
        )

        # 用户操作失效模式
        self.invalidator.register_invalidation_pattern(
            'user_trade',
            ['user_portfolio', 'user_positions', 'user_pnl']
        )

        # 系统配置失效模式
        self.invalidator.register_invalidation_pattern(
            'system_config',
            ['system_settings', 'feature_flags', 'rate_limits']
        )

    async def initialize(self):
        """初始化缓存管理器"""
        # 这里可以添加初始化逻辑
        logger.info("缓存管理器初始化完成")

    async def warmup_popular_data(self) -> Dict[str, Any]:
        """预热热门数据"""
        if not self.warmer:
            # 初始化预热器
            data_providers = {
                'market_data_aapl': self._get_market_data_provider('AAPL'),
                'market_data_goog': self._get_market_data_provider('GOOG'),
                'market_data_msft': self._get_market_data_provider('MSFT'),
                'popular_stocks': self._get_popular_stocks_provider,
                'market_indices': self._get_market_indices_provider
            }
            self.warmer = CacheWarmer(self.cache, data_providers)

        popular_keys = await self.warmer.get_popular_keys()
        return await self.warmer.warmup_cache(popular_keys)

    async def _get_market_data_provider(self, symbol: str):
        """获取市场数据提供者（示例实现）"""
        # 这里应该连接实际的数据源
        return {
            'symbol': symbol,
            'price': 150.0,  # 示例数据
            'volume': 1000000,
            'timestamp': datetime.utcnow().isoformat()
        }

    async def _get_popular_stocks_provider(self):
        """获取热门股票提供者（示例实现）"""
        return ['AAPL', 'GOOG', 'MSFT', 'TSLA', 'AMZN']

    async def _get_market_indices_provider(self):
        """获取市场指数提供者（示例实现）"""
        return {
            'sp500': 4200.50,
            'nasdaq': 13500.75,
            'dow': 34000.25
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'cache_stats': self.cache.get_stats(),
            'warmup_stats': self.warmer.warmup_stats if self.warmer else {},
            'invalidation_stats': self.invalidator.get_stats()
        }
