"""
multi_tier_cache.py

多级缓存优化组件 - MultiTierCache

提供高性能的多级缓存实现，支持：
- L1: 本地内存缓存 (最快，容量最小，TTL 5-60秒)
- L2: Redis分布式缓存 (中等速度，中等容量，TTL 5分钟-1小时)
- L3: 数据库持久化缓存 (最慢，容量最大，TTL 1-24小时)

特性：
- 智能分层存储策略
- 自动数据提升/降级
- 缓存预热和预加载
- 命中率统计和监控
- 故障自动降级

作者: RQA2025 Team
日期: 2026-02-13
"""

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from functools import wraps

import numpy as np

from .core.multi_level_cache import (
    CacheTier, MultiLevelCache, TierConfig,
    MemoryTier, RedisTier, DiskTier
)
from .interfaces.cache_interfaces import CacheEvictionStrategy


# 配置日志
logger = logging.getLogger(__name__)


class CacheTierLevel(Enum):
    """缓存层级枚举"""
    L1_MEMORY = "l1_memory"      # L1: 本地内存缓存
    L2_REDIS = "l2_redis"        # L2: Redis分布式缓存
    L3_DISK = "l3_disk"          # L3: 磁盘/数据库缓存


@dataclass
class MultiTierCacheConfig:
    """
    多级缓存配置类
    
    Attributes:
        l1_enabled: 是否启用L1缓存
        l1_max_size: L1最大条目数
        l1_ttl_seconds: L1默认TTL（秒）
        l2_enabled: 是否启用L2缓存
        l2_max_size: L2最大条目数
        l2_ttl_seconds: L2默认TTL（秒）
        l2_redis_host: Redis主机地址
        l2_redis_port: Redis端口
        l3_enabled: 是否启用L3缓存
        l3_max_size: L3最大条目数
        l3_ttl_seconds: L3默认TTL（秒）
        l3_cache_dir: L3缓存目录
        enable_compression: 是否启用压缩
        enable_stats: 是否启用统计
        warmup_keys: 预热键列表
    """
    # L1 内存缓存配置
    l1_enabled: bool = True
    l1_max_size: int = 10000
    l1_ttl_seconds: int = 60  # 默认60秒
    
    # L2 Redis缓存配置
    l2_enabled: bool = True
    l2_max_size: int = 100000
    l2_ttl_seconds: int = 300  # 默认5分钟
    l2_redis_host: str = "localhost"
    l2_redis_port: int = 6379
    l2_redis_db: int = 0
    
    # L3 磁盘缓存配置
    l3_enabled: bool = True
    l3_max_size: int = 1000000
    l3_ttl_seconds: int = 3600  # 默认1小时
    l3_cache_dir: str = "./cache_data"
    
    # 全局配置
    enable_compression: bool = False
    enable_stats: bool = True
    eviction_policy: str = "LRU"
    warmup_keys: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """配置验证"""
        if self.l1_max_size <= 0:
            raise ValueError("L1 max_size 必须大于0")
        if self.l2_max_size <= 0:
            raise ValueError("L2 max_size 必须大于0")
        if self.l3_max_size <= 0:
            raise ValueError("L3 max_size 必须大于0")


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    avg_response_time_ms: float = 0.0
    
    def update_hit_rate(self):
        """更新命中率"""
        total = self.hits + self.misses
        if total > 0:
            self.hit_rate = self.hits / total


class CacheDecorator:
    """缓存装饰器 - 为函数提供自动缓存功能"""
    
    def __init__(self, cache_instance: 'MultiTierCache', ttl: int = 300):
        self.cache = cache_instance
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """
        装饰器实现
        
        Args:
            func: 被装饰的函数
            
        Returns:
            包装后的函数
        """
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self._generate_key(func, args, kwargs)
            
            # 尝试从缓存获取
            cached_value = await self.cache.aget(cache_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_value
            
            # 执行函数
            result = await func(*args, **kwargs)
            
            # 存入缓存
            await self.cache.aset(cache_key, result, self.ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = self._generate_key(func, args, kwargs)
            
            # 尝试从缓存获取
            cached_value = self.cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            self.cache.set(cache_key, result, self.ttl)
            return result
        
        # 根据函数类型返回合适的包装器
        if asyncio.iscoroutinefunction(func):
            wrapper = async_wrapper
        else:
            wrapper = sync_wrapper
        
        # 添加缓存清除方法
        wrapper.cache_clear = lambda: self.cache.delete(self._generate_key(func, (), {}))
        return wrapper
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = {
            'func': func.__qualname__,
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return f"cache:{hashlib.md5(key_str.encode()).hexdigest()}"


class MultiTierCache:
    """
    多级缓存优化组件
    
    提供L1/L2/L3三级缓存架构，支持智能数据分层和自动故障转移。
    
    Attributes:
        config: 缓存配置
        l1_cache: L1内存缓存
        l2_cache: L2 Redis缓存
        l3_cache: L3磁盘缓存
        stats: 统计信息
        
    Example:
        >>> config = MultiTierCacheConfig(
        ...     l1_max_size=10000,
        ...     l2_max_size=100000,
        ...     l3_max_size=1000000
        ... )
        >>> cache = MultiTierCache(config)
        >>> cache.set("key", "value", ttl=300)
        >>> value = cache.get("key")
    """
    
    def __init__(self, config: Optional[MultiTierCacheConfig] = None):
        """
        初始化多级缓存
        
        Args:
            config: 缓存配置，如果为None则使用默认配置
        """
        self.config = config or MultiTierCacheConfig()
        self._lock = threading.RLock()
        self._initialized = False
        
        # 统计信息
        self._stats = {
            'l1': CacheStats(),
            'l2': CacheStats(),
            'l3': CacheStats(),
            'total': CacheStats()
        }
        
        # 响应时间记录
        self._response_times: List[float] = []
        
        # 初始化各层缓存
        self._init_caches()
        
        self._initialized = True
        logger.info("MultiTierCache 初始化完成")
    
    def _init_caches(self):
        """初始化各级缓存"""
        # L1: 内存缓存
        if self.config.l1_enabled:
            l1_config = TierConfig(
                tier=CacheTier.L1_MEMORY,
                capacity=self.config.l1_max_size,
                ttl=self.config.l1_ttl_seconds,
                eviction_policy=self.config.eviction_policy
            )
            self.l1_cache = MemoryTier(l1_config)
            logger.info(f"L1内存缓存初始化完成，容量: {self.config.l1_max_size}")
        else:
            self.l1_cache = None
        
        # L2: Redis缓存
        if self.config.l2_enabled:
            l2_config = TierConfig(
                tier=CacheTier.L2_REDIS,
                capacity=self.config.l2_max_size,
                ttl=self.config.l2_ttl_seconds,
                host=self.config.l2_redis_host,
                port=self.config.l2_redis_port,
                eviction_policy=self.config.eviction_policy
            )
            self.l2_cache = RedisTier(l2_config)
            logger.info(f"L2 Redis缓存初始化完成")
        else:
            self.l2_cache = None
        
        # L3: 磁盘缓存
        if self.config.l3_enabled:
            l3_config = TierConfig(
                tier=CacheTier.L3_DISK,
                capacity=self.config.l3_max_size,
                ttl=self.config.l3_ttl_seconds,
                file_dir=self.config.l3_cache_dir,
                eviction_policy=self.config.eviction_policy
            )
            self.l3_cache = DiskTier(l3_config)
            logger.info(f"L3磁盘缓存初始化完成，目录: {self.config.l3_cache_dir}")
        else:
            self.l3_cache = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值（多级查找）
        
        查找顺序: L1 -> L2 -> L3
        如果在下层找到，会自动提升到上层
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在则返回None
        """
        start_time = time.time()
        
        with self._lock:
            # L1 查找
            if self.l1_cache:
                value = self.l1_cache.get(key)
                if value is not None:
                    self._record_hit('l1')
                    self._record_response_time(time.time() - start_time)
                    return value
            
            # L2 查找
            if self.l2_cache:
                value = self.l2_cache.get(key)
                if value is not None:
                    self._record_hit('l2')
                    # 提升到L1
                    self._promote_to_l1(key, value)
                    self._record_response_time(time.time() - start_time)
                    return value
            
            # L3 查找
            if self.l3_cache:
                value = self.l3_cache.get(key)
                if value is not None:
                    self._record_hit('l3')
                    # 提升到L1和L2
                    self._promote_to_l1(key, value)
                    self._promote_to_l2(key, value)
                    self._record_response_time(time.time() - start_time)
                    return value
            
            # 未命中
            self._record_miss()
            self._record_response_time(time.time() - start_time)
            return None
    
    async def aget(self, key: str) -> Optional[Any]:
        """
        异步获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在则返回None
        """
        # 在事件循环中运行同步方法
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.get, key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值（多级存储）
        
        默认存储到所有启用的层级
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒），如果为None则使用各层默认值
            
        Returns:
            是否设置成功
        """
        start_time = time.time()
        success = True
        
        with self._lock:
            # 存储到L1
            if self.l1_cache:
                try:
                    l1_ttl = ttl or self.config.l1_ttl_seconds
                    self.l1_cache.set(key, value, l1_ttl)
                except Exception as e:
                    logger.warning(f"L1缓存设置失败: {e}")
                    success = False
            
            # 存储到L2
            if self.l2_cache:
                try:
                    l2_ttl = ttl or self.config.l2_ttl_seconds
                    self.l2_cache.set(key, value, l2_ttl)
                except Exception as e:
                    logger.warning(f"L2缓存设置失败: {e}")
                    success = False
            
            # 存储到L3
            if self.l3_cache:
                try:
                    l3_ttl = ttl or self.config.l3_ttl_seconds
                    self.l3_cache.set(key, value, l3_ttl)
                except Exception as e:
                    logger.warning(f"L3缓存设置失败: {e}")
                    success = False
            
            # 更新统计
            self._stats['total'].sets += 1
            self._record_response_time(time.time() - start_time)
            
            return success
    
    async def aset(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        异步设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）
            
        Returns:
            是否设置成功
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.set, key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值（多级删除）
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功（任一层级删除成功即返回True）
        """
        start_time = time.time()
        success = False
        
        with self._lock:
            # 从L1删除
            if self.l1_cache:
                try:
                    if self.l1_cache.delete(key):
                        success = True
                except Exception as e:
                    logger.warning(f"L1缓存删除失败: {e}")
            
            # 从L2删除
            if self.l2_cache:
                try:
                    if self.l2_cache.delete(key):
                        success = True
                except Exception as e:
                    logger.warning(f"L2缓存删除失败: {e}")
            
            # 从L3删除
            if self.l3_cache:
                try:
                    if self.l3_cache.delete(key):
                        success = True
                except Exception as e:
                    logger.warning(f"L3缓存删除失败: {e}")
            
            # 更新统计
            if success:
                self._stats['total'].deletes += 1
            self._record_response_time(time.time() - start_time)
            
            return success
    
    async def adelete(self, key: str) -> bool:
        """
        异步删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.delete, key)
    
    def exists(self, key: str) -> bool:
        """
        检查键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        with self._lock:
            # L1 检查
            if self.l1_cache and self.l1_cache.exists(key):
                return True
            
            # L2 检查
            if self.l2_cache and self.l2_cache.exists(key):
                return True
            
            # L3 检查
            if self.l3_cache and self.l3_cache.exists(key):
                return True
            
            return False
    
    def clear(self, tier: Optional[str] = None) -> bool:
        """
        清空缓存
        
        Args:
            tier: 指定层级 ('l1', 'l2', 'l3')，如果为None则清空所有
            
        Returns:
            是否清空成功
        """
        success = True
        
        with self._lock:
            if tier is None or tier == 'l1':
                if self.l1_cache:
                    try:
                        self.l1_cache.clear()
                    except Exception as e:
                        logger.error(f"清空L1缓存失败: {e}")
                        success = False
            
            if tier is None or tier == 'l2':
                if self.l2_cache:
                    try:
                        self.l2_cache.clear()
                    except Exception as e:
                        logger.error(f"清空L2缓存失败: {e}")
                        success = False
            
            if tier is None or tier == 'l3':
                if self.l3_cache:
                    try:
                        self.l3_cache.clear()
                    except Exception as e:
                        logger.error(f"清空L3缓存失败: {e}")
                        success = False
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {
                'l1': self._get_tier_stats('l1'),
                'l2': self._get_tier_stats('l2'),
                'l3': self._get_tier_stats('l3'),
                'total': {
                    'hits': self._stats['total'].hits,
                    'misses': self._stats['total'].misses,
                    'sets': self._stats['total'].sets,
                    'deletes': self._stats['total'].deletes,
                    'hit_rate': self._calculate_hit_rate(),
                    'avg_response_time_ms': self._calculate_avg_response_time()
                }
            }
            return stats
    
    def _get_tier_stats(self, tier: str) -> Dict[str, Any]:
        """获取指定层级的统计信息"""
        tier_stats = self._stats[tier]
        cache = getattr(self, f'{tier}_cache')
        
        if cache:
            cache_stats = cache.get_stats()
            return {
                'enabled': True,
                'hits': tier_stats.hits,
                'misses': tier_stats.misses,
                'hit_rate': tier_stats.hit_rate,
                'size': cache_stats.get('size', 0),
                'capacity': cache_stats.get('capacity', 0),
                'memory_usage_mb': cache_stats.get('memory_usage_mb', 0)
            }
        else:
            return {'enabled': False}
    
    def _record_hit(self, tier: str):
        """记录缓存命中"""
        self._stats[tier].hits += 1
        self._stats[tier].update_hit_rate()
        self._stats['total'].hits += 1
        self._stats['total'].update_hit_rate()
    
    def _record_miss(self):
        """记录缓存未命中"""
        self._stats['total'].misses += 1
        self._stats['total'].update_hit_rate()
    
    def _record_response_time(self, response_time: float):
        """记录响应时间"""
        self._response_times.append(response_time)
        # 只保留最近1000条记录
        if len(self._response_times) > 1000:
            self._response_times = self._response_times[-1000:]
    
    def _calculate_hit_rate(self) -> float:
        """计算总命中率"""
        total = self._stats['total'].hits + self._stats['total'].misses
        if total > 0:
            return self._stats['total'].hits / total
        return 0.0
    
    def _calculate_avg_response_time(self) -> float:
        """计算平均响应时间（毫秒）"""
        if self._response_times:
            return (sum(self._response_times) / len(self._response_times)) * 1000
        return 0.0
    
    def _promote_to_l1(self, key: str, value: Any):
        """将数据提升到L1缓存"""
        if self.l1_cache:
            try:
                ttl = min(self.config.l1_ttl_seconds, 300)  # 最多5分钟
                self.l1_cache.set(key, value, ttl)
            except Exception as e:
                logger.debug(f"提升到L1失败: {e}")
    
    def _promote_to_l2(self, key: str, value: Any):
        """将数据提升到L2缓存"""
        if self.l2_cache:
            try:
                ttl = min(self.config.l2_ttl_seconds, 3600)  # 最多1小时
                self.l2_cache.set(key, value, ttl)
            except Exception as e:
                logger.debug(f"提升到L2失败: {e}")
    
    def warmup(self, keys: Optional[List[str]] = None):
        """
        缓存预热
        
        从L3加载指定键到L1和L2
        
        Args:
            keys: 要预热的键列表，如果为None则使用配置中的warmup_keys
        """
        keys = keys or self.config.warmup_keys
        if not keys:
            logger.info("没有配置预热键")
            return
        
        logger.info(f"开始缓存预热，键数量: {len(keys)}")
        warmed_count = 0
        
        for key in keys:
            if self.l3_cache:
                try:
                    value = self.l3_cache.get(key)
                    if value is not None:
                        self._promote_to_l1(key, value)
                        self._promote_to_l2(key, value)
                        warmed_count += 1
                except Exception as e:
                    logger.warning(f"预热键 {key} 失败: {e}")
        
        logger.info(f"缓存预热完成，成功: {warmed_count}/{len(keys)}")
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            键值对字典（只包含存在的键）
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        批量设置缓存值
        
        Args:
            data: 键值对字典
            ttl: 过期时间（秒）
            
        Returns:
            是否全部设置成功
        """
        success = True
        for key, value in data.items():
            if not self.set(key, value, ttl):
                success = False
        return success
    
    def delete_many(self, keys: List[str]) -> int:
        """
        批量删除缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            成功删除的数量
        """
        count = 0
        for key in keys:
            if self.delete(key):
                count += 1
        return count
    
    def decorator(self, ttl: int = 300) -> CacheDecorator:
        """
        获取缓存装饰器
        
        Args:
            ttl: 默认缓存时间（秒）
            
        Returns:
            缓存装饰器实例
            
        Example:
            >>> cache = MultiTierCache()
            >>> @cache.decorator(ttl=600)
            ... def expensive_function(x):
            ...     return x * 2
        """
        return CacheDecorator(self, ttl)
    
    def close(self):
        """关闭缓存，释放资源"""
        logger.info("关闭MultiTierCache")
        
        with self._lock:
            if self.l3_cache:
                try:
                    # DiskTier可能有资源需要释放
                    pass
                except Exception as e:
                    logger.error(f"关闭L3缓存失败: {e}")
            
            self._initialized = False
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close()


# 全局缓存实例（单例模式）
_global_cache: Optional[MultiTierCache] = None
_global_cache_lock = threading.Lock()


def get_global_cache(config: Optional[MultiTierCacheConfig] = None) -> MultiTierCache:
    """
    获取全局缓存实例
    
    Args:
        config: 缓存配置，仅在第一次调用时使用
        
    Returns:
        全局缓存实例
    """
    global _global_cache
    
    if _global_cache is None:
        with _global_cache_lock:
            if _global_cache is None:
                _global_cache = MultiTierCache(config)
    
    return _global_cache


def clear_global_cache():
    """清除全局缓存实例"""
    global _global_cache
    
    with _global_cache_lock:
        if _global_cache:
            _global_cache.close()
            _global_cache = None
