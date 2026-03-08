#!/usr/bin/env python3
"""
多级缓存系统
实现L1内存缓存 + L2 Redis缓存 + L3数据库的级联缓存策略
"""

import time
import json
import hashlib
import pickle
from typing import Any, Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from functools import wraps
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def touch(self):
        """更新访问信息"""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    """
    LRU (Least Recently Used) 缓存
    使用OrderedDict实现O(1)的get和put操作
    """
    
    def __init__(self, max_size: int = 128, default_ttl: Optional[int] = None):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        if key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[key]
        
        # 检查是否过期
        if entry.is_expired():
            del self._cache[key]
            self._misses += 1
            return None
        
        # 更新访问信息
        entry.touch()
        self._cache.move_to_end(key)
        self._hits += 1
        
        return entry.value
    
    def put(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存值"""
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl if ttl else None
        
        entry = CacheEntry(
            key=key,
            value=value,
            expires_at=expires_at
        )
        
        # 如果key已存在，更新值
        if key in self._cache:
            self._cache.move_to_end(key)
        
        self._cache[key] = entry
        
        # 如果超出容量，移除最久未使用的
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
    
    def delete(self, key: str) -> bool:
        """删除缓存"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'entries': [
                {
                    'key': k,
                    'access_count': v.access_count,
                    'expires_at': v.expires_at
                }
                for k, v in list(self._cache.items())[:10]  # 只显示前10个
            ]
        }


class MultiLevelCache:
    """
    多级缓存系统
    
    L1: 内存LRU缓存 (最快)
    L2: Redis缓存 (分布式)
    L3: 数据库/原始数据源 (最慢)
    
    读取顺序: L1 -> L2 -> L3
    写入顺序: L3 -> L2 -> L1
    """
    
    def __init__(
        self,
        l1_size: int = 128,
        l1_ttl: int = 300,  # 5分钟
        l2_ttl: int = 3600,  # 1小时
        redis_client=None
    ):
        self.l1_cache = LRUCache(max_size=l1_size, default_ttl=l1_ttl)
        self.l1_ttl = l1_ttl
        self.l2_ttl = l2_ttl
        self.redis = redis_client
        
        # 统计信息
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._misses = 0
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        key_data = f"{func_name}:{str(args)}:{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值（级联查询）
        
        顺序: L1 -> L2 -> L3
        """
        # 1. 尝试L1缓存
        value = self.l1_cache.get(key)
        if value is not None:
            self._l1_hits += 1
            logger.debug(f"L1 cache hit: {key}")
            return value
        
        # 2. 尝试L2缓存 (Redis)
        if self.redis:
            try:
                value = self.redis.get(key)
                if value:
                    # 反序列化
                    value = pickle.loads(value)
                    # 回填L1缓存
                    self.l1_cache.put(key, value, self.l1_ttl)
                    self._l2_hits += 1
                    logger.debug(f"L2 cache hit: {key}")
                    return value
            except Exception as e:
                logger.error(f"L2 cache error: {e}")
        
        self._misses += 1
        return None
    
    async def set(self, key: str, value: Any, l1_ttl: Optional[int] = None, l2_ttl: Optional[int] = None):
        """
        设置缓存值
        
        写入L1和L2缓存
        """
        l1_ttl = l1_ttl or self.l1_ttl
        l2_ttl = l2_ttl or self.l2_ttl
        
        # 1. 写入L1缓存
        self.l1_cache.put(key, value, l1_ttl)
        
        # 2. 写入L2缓存 (Redis)
        if self.redis:
            try:
                serialized = pickle.dumps(value)
                self.redis.setex(key, l2_ttl, serialized)
            except Exception as e:
                logger.error(f"L2 cache set error: {e}")
    
    async def delete(self, key: str):
        """删除缓存"""
        # 删除L1
        self.l1_cache.delete(key)
        
        # 删除L2
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.error(f"L2 cache delete error: {e}")
    
    async def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        l1_ttl: Optional[int] = None,
        l2_ttl: Optional[int] = None
    ) -> Any:
        """
        获取或设置缓存
        
        如果缓存不存在，调用loader函数获取数据并缓存
        """
        # 尝试获取缓存
        value = await self.get(key)
        if value is not None:
            return value
        
        # 加载数据
        try:
            if asyncio.iscoroutinefunction(loader):
                value = await loader()
            else:
                value = loader()
        except Exception as e:
            logger.error(f"Cache loader error: {e}")
            raise
        
        # 写入缓存
        await self.set(key, value, l1_ttl, l2_ttl)
        self._l3_hits += 1
        
        return value
    
    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        skip_args: Optional[List[int]] = None
    ):
        """
        缓存装饰器
        
        用法:
            @cache.cached(ttl=300)
            async def get_user(user_id: int):
                return await db.get_user(user_id)
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(
                    func, args, kwargs, key_prefix, skip_args
                )
                
                # 尝试获取缓存
                result = await self.get(cache_key)
                if result is not None:
                    return result
                
                # 执行函数
                result = await func(*args, **kwargs)
                
                # 写入缓存
                await self.set(cache_key, result, ttl)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # 同步版本（使用asyncio.run）
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            # 根据函数类型返回适当的包装器
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def _generate_cache_key(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        key_prefix: str = "",
        skip_args: Optional[List[int]] = None
    ) -> str:
        """生成缓存键"""
        skip_args = skip_args or []
        
        # 过滤掉不需要参与生成key的参数
        filtered_args = tuple(
            arg for i, arg in enumerate(args)
            if i not in skip_args
        )
        
        # 生成key
        key_data = {
            'prefix': key_prefix,
            'func': func.__qualname__,
            'args': filtered_args,
            'kwargs': kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def invalidate(self, pattern: str):
        """
        批量失效缓存
        
        根据模式删除匹配的缓存键
        """
        # 清除L1缓存中匹配的键
        keys_to_delete = [
            key for key in self.l1_cache._cache.keys()
            if pattern in key
        ]
        for key in keys_to_delete:
            self.l1_cache.delete(key)
        
        logger.info(f"Invalidated {len(keys_to_delete)} L1 cache entries matching '{pattern}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_hits = self._l1_hits + self._l2_hits + self._l3_hits
        total_requests = total_hits + self._misses
        
        hit_rate = total_hits / total_requests if total_requests > 0 else 0
        
        return {
            'l1_cache': self.l1_cache.get_stats(),
            'l1_hits': self._l1_hits,
            'l2_hits': self._l2_hits,
            'l3_hits': self._l3_hits,
            'misses': self._misses,
            'total_hits': total_hits,
            'total_requests': total_requests,
            'hit_rate': hit_rate,
            'l1_hit_rate': self._l1_hits / total_hits if total_hits > 0 else 0,
            'l2_hit_rate': self._l2_hits / total_hits if total_hits > 0 else 0,
        }
    
    def clear(self):
        """清空所有缓存"""
        self.l1_cache.clear()
        
        if self.redis:
            try:
                # 注意：这会清空整个Redis数据库
                # 生产环境应该使用更精细的删除策略
                self.redis.flushdb()
            except Exception as e:
                logger.error(f"L2 cache clear error: {e}")
        
        logger.info("All caches cleared")


# 全局缓存实例
_cache_instance: Optional[MultiLevelCache] = None


def get_cache() -> MultiLevelCache:
    """获取全局缓存实例"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = MultiLevelCache()
    return _cache_instance


def setup_cache(redis_client=None, **kwargs):
    """设置缓存"""
    global _cache_instance
    _cache_instance = MultiLevelCache(redis_client=redis_client, **kwargs)
    logger.info("Multi-level cache initialized")


# 示例用法
if __name__ == "__main__":
    import asyncio
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    async def test_cache():
        print("=== 多级缓存测试 ===\n")
        
        # 创建缓存实例
        cache = MultiLevelCache(l1_size=10, l1_ttl=5)
        
        # 测试1: 基本操作
        print("测试1: 基本操作")
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        print(f"设置key1=value1, 获取: {value}")
        
        # 测试2: 缓存装饰器
        print("\n测试2: 缓存装饰器")
        
        @cache.cached(ttl=10, key_prefix="user")
        async def get_user(user_id: int):
            print(f"  (从数据库获取用户 {user_id})")
            return {"id": user_id, "name": f"User_{user_id}"}
        
        # 第一次调用（未缓存）
        user1 = await get_user(1)
        print(f"第一次获取: {user1}")
        
        # 第二次调用（已缓存）
        user1_cached = await get_user(1)
        print(f"第二次获取: {user1_cached}")
        
        # 测试3: 缓存统计
        print("\n测试3: 缓存统计")
        stats = cache.get_stats()
        print(f"L1缓存命中率: {stats['l1_hit_rate']:.2%}")
        print(f"总命中率: {stats['hit_rate']:.2%}")
        print(f"L1缓存大小: {stats['l1_cache']['size']}")
        
        # 测试4: LRU淘汰
        print("\n测试4: LRU淘汰")
        for i in range(15):
            await cache.set(f"key_{i}", f"value_{i}")
        
        stats = cache.get_stats()
        print(f"插入15个key后，L1缓存大小: {stats['l1_cache']['size']} (max=10)")
        
        print("\n=== 测试完成 ===")
    
    # 运行测试
    asyncio.run(test_cache())
