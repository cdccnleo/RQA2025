#!/usr/bin/env python3
"""
分布式缓存管理器

实现多级缓存架构优化数据访问性能，支持L1内存缓存、L2分布式缓存、L3持久化缓存"""

import logging
import json
import pickle
import threading
import time
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import hashlib
import zlib

logger = logging.getLogger(__name__)

# 尝试导入Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
logger.warning("Redis不可用，将使用内存缓存")

# 尝试导入Memcached
try:
    import pymemcache
    MEMCACHED_AVAILABLE = True
except ImportError:
    MEMCACHED_AVAILABLE = False
    logger.warning("Memcached不可用，将使用内存缓存")


class CacheLevel(Enum):

    """缓存级别"""
    L1_MEMORY = "l1_memory"          # L1内存缓存
    L2_DISTRIBUTED = "l2_distributed"  # L2分布式缓存
    L3_PERSISTENT = "l3_persistent"  # L3持久化缓存


class CacheStrategy(Enum):

    """缓存策略"""
    LRU = "lru"                      # 最近最少使用
    LFU = "lfu"                      # 最少使用频率
    FIFO = "fifo"                   # 先进先出
    TTL = "ttl"                     # 基于时间的过期


class CacheBackend(Enum):

    """缓存后端"""
    MEMORY = "memory"               # 内存后端
    REDIS = "redis"                 # Redis后端
    MEMCACHED = "memcached"         # Memcached后端
    FILE = "file"                   # 文件后端
    DATABASE = "database"           # 数据库后端


@dataclass
class CacheConfig:

    """缓存配置"""
    level: CacheLevel = CacheLevel.L1_MEMORY
    backend: CacheBackend = CacheBackend.MEMORY
    strategy: CacheStrategy = CacheStrategy.LRU

    # 容量配置
    max_size: int = 10000              # 最大缓存条目数
    max_memory_mb: int = 512          # 最大内存使用(MB)

    # 时间配置

    default_ttl: int = 3600           # 默认TTL(秒)
    cleanup_interval: int = 300       # 清理间隔(秒)

    # 连接配置
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None

    memcached_host: str = "localhost"
    memcached_port: int = 11211

    # 压缩配置
    enable_compression: bool = True
    compression_threshold: int = 1024  # 压缩阈值(字节)

    # 预热配置
    enable_prewarm: bool = True
    prewarm_batch_size: int = 100


@dataclass
class CacheEntry:

    """缓存条目"""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:

    """缓存统计信息"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    sets: int = 0
    deletes: int = 0

    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """缓存未命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_misses / self.total_requests


class MemoryCache:

    """内存缓存实现"""

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.access_frequency: Dict[str, int] = {}
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self._cleanup_running = False

        # 启动清理线程
        if config.cleanup_interval > 0:
            self._cleanup_running = True
            self.cleanup_thread = threading.Thread(
                target=self._cleanup_worker,
                daemon=True
            )
            self.cleanup_thread.start()

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            self.stats.total_requests += 1

        if key not in self.cache:
            self.stats.cache_misses += 1
            return None

        entry = self.cache[key]

        # 检查是否过期
        if self._is_expired(entry):
            self._remove_entry(key)
            self.stats.cache_misses += 1
            return None

        # 更新访问信息
        entry.accessed_at = datetime.now()
        entry.access_count += 1
        self.access_frequency[key] += 1

        # 根据策略调整位置
        if self.config.strategy == CacheStrategy.LRU:
            self.cache.move_to_end(key)
        elif self.config.strategy == CacheStrategy.LFU:
            # LFU需要重新排序，这里简化处理
            pass

        self.stats.cache_hits += 1
        return self._deserialize_value(entry.value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        with self.lock:
            try:
                # 序列化值
                serialized_value = self._serialize_value(value)

                # 检查内存使用
                if self._would_exceed_memory_limit(serialized_value):
                    self._evict_entries()
                else:
                    entry = CacheEntry(
                        key=key,
                        value=serialized_value,
                        ttl=ttl or self.config.default_ttl
                    )

                    self.cache[key] = entry
                    self.access_frequency[key] = 1
                    self.stats.sets += 1

                # 根据策略维护缓存
                if self.config.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)

                # 检查容量限制
                if len(self.cache) > self.config.max_size:
                    self._evict_entries()

                return True

            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
                return False

    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        with self.lock:
            if key in self.cache:
                self._remove_entry(key)
                self.stats.deletes += 1
                return True
        return False


    def clear(self) -> bool:

        """清空缓存"""
        with self.lock:
            self.cache.clear()
        self.access_frequency.clear()
        self.stats = CacheStats()
        return True


    def get_stats(self) -> CacheStats:

        """获取统计信息"""
        return self.stats


    def _remove_entry(self, key: str):

        """移除缓存条目"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_frequency:
            del self.access_frequency[key]


    def _is_expired(self, entry: CacheEntry) -> bool:

        """检查条目是否过误"""
        if entry.ttl is None:
            return False

        return (datetime.now() - entry.created_at).seconds > entry.ttl


    def _evict_entries(self):

        """驱逐条目"""
        if self.config.strategy == CacheStrategy.LRU:
            # LRU: 移除最久未使用条目
            while len(self.cache) > 0 and len(self.cache) >= self.config.max_size * 0.9:
                key, _ = self.cache.popitem(last=False)
                self._remove_entry(key)
                self.stats.evictions += 1

        elif self.config.strategy == CacheStrategy.LFU:
            # LFU: 移除使用频率最低的
            while len(self.cache) > 0 and len(self.cache) >= self.config.max_size * 0.9:
                if self.access_frequency:
                    key = min(self.access_frequency, key=self.access_frequency.get)
                    self._remove_entry(key)
                    self.stats.evictions += 1


    def _would_exceed_memory_limit(self, new_value: bytes) -> bool:

        """检查是否会超过内存限制"""
        current_memory = sum(len(entry.value) for entry in self.cache.values())
        new_memory = current_memory + len(new_value)
        return new_memory > self.config.max_memory_mb * 1024 * 1024


    def _serialize_value(self, value: Any) -> bytes:

        """序列化值"""
        try:
            # 先JSON序列化
            json_str = json.dumps(value, default=str)
            json_bytes = json_str.encode('utf-8')

            # 压缩
            if self.config.enable_compression and len(json_bytes) > self.config.compression_threshold:
                return zlib.compress(json_bytes)
            else:
                return json_bytes

        except Exception as e:
            logger.error(f"序列化失败: {e}")
            return pickle.dumps(value)


    def _deserialize_value(self, data: bytes) -> Any:

        """反序列化值"""
        try:
            # 尝试解压
            if self.config.enable_compression:
                try:
                    decompressed = zlib.decompress(data)
                except zlib.error:
                    decompressed = data
            else:
                decompressed = data

            # JSON反序列化
            json_str = decompressed.decode('utf-8')
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            try:
                return pickle.loads(data)
            except Exception:
                return None


    def _cleanup_worker(self):

        """清理工作线程"""
        while self._cleanup_running:
            time.sleep(min(self.config.cleanup_interval, 5.0))  # 最长等待5秒
            try:
                with self.lock:
                    expired_keys = []
                    for key, entry in self.cache.items():
                        if self._is_expired(entry):
                            expired_keys.append(key)

                    for key in expired_keys:
                        self._remove_entry(key)

                    if expired_keys:
                        logger.info(f"清理了{len(expired_keys)} 个过期缓存条目")

            except Exception as e:
                logger.error(f"缓存清理失败: {e}")

    def stop_cleanup(self):
        """停止清理线程"""
        self._cleanup_running = False
        if hasattr(self, 'cleanup_thread') and self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5.0)
            logger.info("缓存清理线程已停止")


class RedisCache:

    """Redis缓存实现"""


    def __init__(self, config: CacheConfig):

        if not REDIS_AVAILABLE:
            raise ImportError("Redis不可用")

        self.config = config
        self.client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port,
            db=config.redis_db,
            password=config.redis_password,
            decode_responses=False
        )
        self.stats = CacheStats()

        # 测试连接
        try:
            self.client.ping()
            logger.info("Redis连接成功")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Redis连接失败: {e}")


    def get(self, key: str) -> Optional[Any]:

        """获取缓存误"""
        self.stats.total_requests += 1

        try:
            value = self.client.get(key)
            if value is None:
                self.stats.cache_misses += 1
                return None

            self.stats.cache_hits += 1
            return self._deserialize_value(value)

        except Exception as e:
            logger.error(f"Redis获取失败: {e}")
            self.stats.cache_misses += 1
            return None


    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:

        """设置缓存误"""
        try:
            serialized_value = self._serialize_value(value)
            ttl_value = ttl or self.config.default_ttl

            result = self.client.setex(key, ttl_value, serialized_value)
            if result:
                self.stats.sets += 1
            return bool(result)

        except Exception as e:
            logger.error(f"Redis设置失败: {e}")
            return False


    def delete(self, key: str) -> bool:

        """删除缓存条目"""
        try:
            result = self.client.delete(key)
            if result:
                self.stats.deletes += 1
            return bool(result)

        except Exception as e:
            logger.error(f"Redis删除失败: {e}")
            return False


    def clear(self) -> bool:

        """清空缓存"""
        try:
            self.client.flushdb()
            self.stats = CacheStats()
            return True

        except Exception as e:
            logger.error(f"Redis清空失败: {e}")
            return False


    def get_stats(self) -> CacheStats:

        """获取统计信息"""
        return self.stats


    def _serialize_value(self, value: Any) -> bytes:

        """序列化误"""
        try:
            json_str = json.dumps(value, default=str)
            json_bytes = json_str.encode('utf - 8')

            if self.config.enable_compression and len(json_bytes) > self.config.compression_threshold:
                return zlib.compress(json_bytes)
            else:
                return json_bytes

        except Exception as e:
            logger.error(f"序列化失误 {e}")
            return pickle.dumps(value)


    def _deserialize_value(self, data: bytes) -> Any:

        """反序列化值"""
        try:
            if self.config.enable_compression:
                try:
                    decompressed = zlib.decompress(data)
                except zlib.error:
                    decompressed = data
            else:
                decompressed = data

            json_str = decompressed.decode('utf-8')
            return json.loads(json_str)

        except Exception as e:
            logger.error(f"反序列化失败: {e}")
            try:
                return pickle.loads(data)
            except Exception:
                return None


class DistributedCacheManager:

    """分布式缓存管理器"""


    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.lock = threading.RLock()

        # 多级缓存
        self.l1_cache: Optional[MemoryCache] = None
        self.l2_cache: Optional[Union[RedisCache, MemoryCache]] = None
        self.l3_cache: Optional[Any] = None

        # 缓存统计
        self.stats = CacheStats()
        self.level_stats = {
            CacheLevel.L1_MEMORY: CacheStats(),
            CacheLevel.L2_DISTRIBUTED: CacheStats(),
            CacheLevel.L3_PERSISTENT: CacheStats()
        }

        # 预热数据
        self.prewarm_data: Dict[str, Any] = {}

        # 初始化缓存层
        self._initialize_cache_levels()

        logger.info("分布式缓存管理器初始化完成")


    def _initialize_cache_levels(self):
        """初始化缓存层"""
        try:
            # L1缓存配置
            l1_config = CacheConfig(
                level=CacheLevel.L1_MEMORY,
                backend=CacheBackend.MEMORY,
                strategy=CacheStrategy.LRU,
                max_size=self.config.get('l1_max_size', 1000),
                max_memory_mb=self.config.get('l1_max_memory_mb', 256),

                default_ttl=self.config.get('l1_ttl', 300),
                enable_compression=self.config.get('enable_compression', True)
            )

            self.l1_cache = MemoryCache(l1_config)

            # L2缓存配置
            if REDIS_AVAILABLE and self.config.get('enable_redis', True):
                l2_config = CacheConfig(
                    level=CacheLevel.L2_DISTRIBUTED,
                    backend=CacheBackend.REDIS,
                    strategy=CacheStrategy.TTL,

                    default_ttl=self.config.get('l2_ttl', 1800),
                    redis_host=self.config.get('redis_host', 'localhost'),
                    redis_port=self.config.get('redis_port', 6379),
                    redis_db=self.config.get('redis_db', 0),
                    redis_password=self.config.get('redis_password'),
                    enable_compression=self.config.get('enable_compression', True)
                )

                try:
                    self.l2_cache = RedisCache(l2_config)
                    logger.info("L2 Redis缓存初始化成误")
                except Exception as e:
                    logger.warning(f"L2 Redis缓存初始化失误 {e}，使用内存缓误")
                    self.l2_cache = MemoryCache(l1_config)
                else:
                    # L2内存缓存
                    l2_config = CacheConfig(
                        level=CacheLevel.L2_DISTRIBUTED,
                        backend=CacheBackend.MEMORY,
                        strategy=CacheStrategy.LRU,
                        max_size=self.config.get('l2_max_size', 5000),
                        max_memory_mb=self.config.get('l2_max_memory_mb', 1024),

                        default_ttl=self.config.get('l2_ttl', 1800),
                        enable_compression=self.config.get('enable_compression', True)
                    )
                    self.l2_cache = MemoryCache(l2_config)

            logger.info("缓存层级初始化完误")

        except Exception as e:
            logger.error(f"缓存层级初始化失误 {e}")


    def get(self, key: str, use_cache_levels: Optional[List[CacheLevel]] = None) -> Optional[Any]:
        """
        多级缓存获取

        Args:
            key: 缓存键
            use_cache_levels: 要使用的缓存级别，默认使用所有级别
        Returns:
            缓存值
        """
        if use_cache_levels is None:
            use_cache_levels = [CacheLevel.L1_MEMORY,
                CacheLevel.L2_DISTRIBUTED, CacheLevel.L3_PERSISTENT]

        self.stats.total_requests += 1

        # L1缓存查找
        if CacheLevel.L1_MEMORY in use_cache_levels and self.l1_cache:
            self.level_stats[CacheLevel.L1_MEMORY].total_requests += 1
            value = self.l1_cache.get(key)
            if value is not None:
                self.level_stats[CacheLevel.L1_MEMORY].cache_hits += 1
                self.stats.cache_hits += 1
                logger.debug(f"L1缓存命中: {key}")
                return value
            else:
                self.level_stats[CacheLevel.L1_MEMORY].cache_misses += 1

        # L2缓存查找
        if CacheLevel.L2_DISTRIBUTED in use_cache_levels and self.l2_cache:
            self.level_stats[CacheLevel.L2_DISTRIBUTED].total_requests += 1
            value = self.l2_cache.get(key)
            if value is not None:
                self.level_stats[CacheLevel.L2_DISTRIBUTED].cache_hits += 1
                self.stats.cache_hits += 1

                # 回填L1缓存
                if self.l1_cache:
                    self.l1_cache.set(key, value)

                logger.debug(f"L2缓存命中: {key}")
                return value
            else:
                self.level_stats[CacheLevel.L2_DISTRIBUTED].cache_misses += 1

        # L3缓存查找 (TODO: 实现数据库缓误
        if CacheLevel.L3_PERSISTENT in use_cache_levels and self.l3_cache:
            # 实现L3缓存逻辑
            pass

        self.stats.cache_misses += 1
        logger.debug(f"缓存未命误 {key}")
        return None


    def set(self, key: str, value: Any, ttl: Optional[int] = None,
             cache_levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        多级缓存设置

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间
            cache_levels: 要设置的缓存级别

        Returns:
            是否设置成功
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_DISTRIBUTED]

        success = False

        # 设置L1缓存
        if CacheLevel.L1_MEMORY in cache_levels and self.l1_cache:
            if self.l1_cache.set(key, value, ttl):
                self.stats.sets += 1
                success = True

        # 设置L2缓存
        if CacheLevel.L2_DISTRIBUTED in cache_levels and self.l2_cache:
            if self.l2_cache.set(key, value, ttl):
                self.stats.sets += 1
                success = True

        # 设置L3缓存 (TODO)
        if CacheLevel.L3_PERSISTENT in cache_levels and self.l3_cache:
            pass

        return success


    def delete(self, key: str, cache_levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        多级缓存删除

        Args:
            key: 缓存键
            cache_levels: 要删除的缓存级别

        Returns:
            是否删除成功
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY,
                CacheLevel.L2_DISTRIBUTED, CacheLevel.L3_PERSISTENT]

        success = False

        # 删除L1缓存
        if CacheLevel.L1_MEMORY in cache_levels and self.l1_cache:
            if self.l1_cache.delete(key):
                self.stats.deletes += 1
                success = True

        # 删除L2缓存
        if CacheLevel.L2_DISTRIBUTED in cache_levels and self.l2_cache:
            if self.l2_cache.delete(key):
                self.stats.deletes += 1
                success = True

        # 删除L3缓存 (TODO)
        if CacheLevel.L3_PERSISTENT in cache_levels and self.l3_cache:
            pass

        return success


    def clear(self, cache_levels: Optional[List[CacheLevel]] = None) -> bool:
        """
        多级缓存清空

        Args:
            cache_levels: 要清空的缓存级别

        Returns:
            是否清空成功
        """
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY,
                CacheLevel.L2_DISTRIBUTED, CacheLevel.L3_PERSISTENT]

        success = False

        # 清空L1缓存
        if CacheLevel.L1_MEMORY in cache_levels and self.l1_cache:
            if self.l1_cache.clear():
                success = True

        # 清空L2缓存
        if CacheLevel.L2_DISTRIBUTED in cache_levels and self.l2_cache:
            if self.l2_cache.clear():
                success = True

        # 清空L3缓存 (TODO)
        if CacheLevel.L3_PERSISTENT in cache_levels and self.l3_cache:
            pass

        return success


    def prewarm_cache(self, data_generator: Callable[[], Dict[str, Any]],
                       batch_size: int = 100) -> bool:
        """
        缓存预热

        Args:
            data_generator: 数据生成器函数
            batch_size: 批处理大小
        Returns:
            是否预热成功
        """
        try:
            logger.info("开始缓存预热")

            # 生成预热数据
            prewarm_data = data_generator()

            # 分批设置缓存
            items = list(prewarm_data.items())
            for i in range(0, len(items), batch_size):
                batch = dict(items[i:i + batch_size])

                for key, value in batch.items():
                    self.set(key, value, cache_levels=[
                             CacheLevel.L1_MEMORY, CacheLevel.L2_DISTRIBUTED])

                logger.info(f"预热批次 {i // batch_size + 1}: 处理了{len(batch)} 个条目")

            self.prewarm_data = prewarm_data
            logger.info(f"缓存预热完成，共预热 {len(prewarm_data)} 个条目")
            return True

        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
            return False


    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            return {
                'overall': {
                    'total_requests': self.stats.total_requests,
                    'cache_hits': self.stats.cache_hits,
                    'cache_misses': self.stats.cache_misses,
                    'hit_rate': self.stats.hit_rate,
                    'miss_rate': self.stats.miss_rate,
                    'sets': self.stats.sets,
                    'deletes': self.stats.deletes,
                    'evictions': self.stats.evictions
                },
                'by_level': {
                    level.value: {
                        'total_requests': stats.total_requests,
                        'cache_hits': stats.cache_hits,
                        'cache_misses': stats.cache_misses,
                        'hit_rate': stats.hit_rate,
                        'sets': stats.sets,
                        'deletes': stats.deletes,
                        'evictions': stats.evictions
                    }
                    for level, stats in self.level_stats.items()
                },
                'cache_info': {
                    'l1_enabled': self.l1_cache is not None,
                    'l2_enabled': self.l2_cache is not None,
                    'l3_enabled': self.l3_cache is not None,
                    'l2_backend': self.l2_cache.config.backend.value if self.l2_cache else None,
                    'prewarm_items': len(self.prewarm_data)
                },
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}


    def optimize_cache(self) -> Dict[str, Any]:
        """优化缓存配置"""
        try:
            stats = self.get_stats()

            recommendations = []

            # 分析命中率
            overall_hit_rate = stats['overall']['hit_rate']

            if overall_hit_rate < 0.5:
                recommendations.append({
                    'type': 'hit_rate_low',
                    'priority': 'high',
                    'message': f'整体缓存命中率过低({overall_hit_rate:.2%})',
                    'action': '考虑增加缓存容量或调整缓存策略'
                })

            # 分析L1缓存性能
            l1_stats = stats['by_level'].get('l1_memory', {})
            if l1_stats.get('total_requests', 0) > 0:
                l1_hit_rate = l1_stats.get('hit_rate', 0)
                if l1_hit_rate > 0.8:
                    recommendations.append({
                        'type': 'l1_performance_good',
                        'priority': 'low',
                        'message': f'L1缓存性能良好 (命中率 {l1_hit_rate:.2%})',
                        'action': '保持当前配置'
                    })

            # 分析驱逐情况
            if stats['overall']['evictions'] > stats['overall']['sets'] * 0.1:
                recommendations.append({
                    'type': 'high_eviction_rate',
                    'priority': 'medium',
                    'message': '缓存驱逐率较高',
                    'action': '考虑增加缓存容量'
                })

            return {
                'current_performance': stats,
                'recommendations': recommendations,
                'optimization_timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"缓存优化失败: {e}")
            return {}


    def create_cache_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        try:
            # 将所有参数转换为字符串
            key_parts = [str(arg) for arg in args]
            key_parts.extend([f"{k}:{v}" for k, v in sorted(kwargs.items())])

            # 创建哈希
            key_string = "|".join(key_parts)
            key_hash = hashlib.md5(key_string.encode()).hexdigest()

            return f"cache:{key_hash}"

        except Exception as e:
            logger.error(f"生成缓存键失败: {e}")
            return f"cache:fallback:{time.time()}"


    def batch_get(self, keys: List[str],
                   use_cache_levels: Optional[List[CacheLevel]] = None) -> Dict[str, Any]:
        """批量获取缓存"""
        results = {}

        for key in keys:
            value = self.get(key, use_cache_levels)
            if value is not None:
                results[key] = value

        return results

    def batch_set(self, key_value_pairs: Dict[str, Any],
                  ttl: Optional[int] = None,
                  cache_levels: Optional[List[CacheLevel]] = None) -> bool:
        """批量设置缓存"""
        success = True

        for key, value in key_value_pairs.items():
            if not self.set(key, value, ttl, cache_levels):
                success = False

        return success


    def get_cache_health(self) -> Dict[str, Any]:
        """获取缓存健康状态"""
        try:
            health = {
                'overall_status': 'healthy',
                'issues': [],
                'l1_health': 'unknown',
                'l2_health': 'unknown',
                'l3_health': 'unknown'
            }

            # 检查L1缓存健康
            if self.l1_cache:
                try:
                    # 执行一个简单的测试
                    test_key = "health_check_l1"
                    self.l1_cache.set(test_key, "test", 60)
                    value = self.l1_cache.get(test_key)
                    self.l1_cache.delete(test_key)

                    if value == "test":
                        health['l1_health'] = 'healthy'
                    else:
                        health['l1_health'] = 'degraded'
                        health['issues'].append('L1缓存读写异常')
                except Exception as e:
                    health['l1_health'] = 'unhealthy'
                    health['issues'].append(f'L1缓存异常: {e}')

            # 检查L2缓存健康
            if self.l2_cache:
                try:
                    test_key = "health_check_l2"
                    self.l2_cache.set(test_key, "test", 60)
                    value = self.l2_cache.get(test_key)
                    self.l2_cache.delete(test_key)

                    if value == "test":
                        health['l2_health'] = 'healthy'
                    else:
                        health['l2_health'] = 'degraded'
                        health['issues'].append('L2缓存读写异常')
                except Exception as e:
                    health['l2_health'] = 'unhealthy'
                    health['issues'].append(f'L2缓存异常: {e}')

            # 确定整体状态
            if 'unhealthy' in [health['l1_health'], health['l2_health'], health['l3_health']]:
                health['overall_status'] = 'unhealthy'
            elif 'degraded' in [health['l1_health'], health['l2_health'], health['l3_health']]:
                health['overall_status'] = 'degraded'
            elif health['issues']:
                health['overall_status'] = 'warning'

            health['timestamp'] = datetime.now().isoformat()
            return health

        except Exception as e:
            logger.error(f"获取缓存健康状态失败: {e}")
            return {
                'overall_status': 'error',
                'issues': [f'健康检查失败: {e}'],
                'timestamp': datetime.now().isoformat()
            }
