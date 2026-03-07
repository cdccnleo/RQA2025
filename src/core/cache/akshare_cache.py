#!/usr/bin/env python3
"""
AKShare API响应缓存管理器

提供AKShare API响应的缓存功能，减少API调用频率，提高数据采集稳定性。
支持内存缓存和Redis缓存两种模式。
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AKShareCacheManager:
    """
    AKShare缓存管理器

    支持以下功能：
    - API响应缓存，避免重复调用
    - 缓存过期管理
    - 缓存命中率统计
    - 内存和Redis双重缓存
    """

    def __init__(self, redis_client=None, default_ttl: int = 300):
        """
        初始化缓存管理器

        Args:
            redis_client: Redis客户端实例
            default_ttl: 默认缓存时间（秒），默认5分钟
        """
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'expires': 0
        }

    def _generate_cache_key(self, api_name: str, params: Dict[str, Any]) -> str:
        """
        生成缓存键

        Args:
            api_name: API函数名
            params: API参数

        Returns:
            str: 缓存键
        """
        # 对参数进行排序和序列化，确保相同参数生成相同键
        sorted_params = json.dumps(params, sort_keys=True, default=str)
        key_content = f"{api_name}:{sorted_params}"
        return f"akshare:{hashlib.md5(key_content.encode()).hexdigest()}"

    def _is_expired(self, cache_entry: Dict[str, Any]) -> bool:
        """
        检查缓存是否过期

        Args:
            cache_entry: 缓存条目

        Returns:
            bool: 是否过期
        """
        expires_at = cache_entry.get('expires_at', 0)
        return time.time() > expires_at

    def _cleanup_expired_cache(self):
        """清理过期的内存缓存"""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.memory_cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)

        for key in expired_keys:
            del self.memory_cache[key]
            self.cache_stats['expires'] += 1

        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期的缓存条目")

    async def get(self, api_name: str, params: Dict[str, Any], ttl: Optional[int] = None) -> Optional[Any]:
        """
        获取缓存的API响应

        Args:
            api_name: API函数名
            params: API参数
            ttl: 自定义TTL（可选）

        Returns:
            Optional[Any]: 缓存的数据，如果不存在或过期返回None
        """
        cache_key = self._generate_cache_key(api_name, params)

        # 首先尝试Redis缓存
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(cache_key)
                if cached_data:
                    self.cache_stats['hits'] += 1
                    logger.debug(f"Redis缓存命中: {api_name}")
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis缓存读取失败: {e}")

        # 然后尝试内存缓存
        if cache_key in self.memory_cache:
            cache_entry = self.memory_cache[cache_key]
            if not self._is_expired(cache_entry):
                self.cache_stats['hits'] += 1
                logger.debug(f"内存缓存命中: {api_name}")
                return cache_entry['data']
            else:
                # 清理过期条目
                del self.memory_cache[cache_key]
                self.cache_stats['expires'] += 1

        self.cache_stats['misses'] += 1
        logger.debug(f"缓存未命中: {api_name}")
        return None

    async def set(self, api_name: str, params: Dict[str, Any], data: Any, ttl: Optional[int] = None):
        """
        设置缓存

        Args:
            api_name: API函数名
            params: API参数
            data: 要缓存的数据
            ttl: 缓存时间（秒），如果不指定使用默认值
        """
        if ttl is None:
            ttl = self.default_ttl

        cache_key = self._generate_cache_key(api_name, params)
        expires_at = time.time() + ttl

        cache_entry = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time(),
            'api_name': api_name,
            'params': params
        }

        # 设置内存缓存
        self.memory_cache[cache_key] = cache_entry

        # 设置Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(cache_key, ttl, json.dumps(data, default=str))
            except Exception as e:
                logger.warning(f"Redis缓存设置失败: {e}")

        self.cache_stats['sets'] += 1
        logger.debug(f"缓存已设置: {api_name}, TTL: {ttl}秒")

    async def invalidate(self, api_name: str, params: Optional[Dict[str, Any]] = None):
        """
        使缓存失效

        Args:
            api_name: API函数名，如果为None则清除所有缓存
            params: API参数，如果为None则清除该API的所有缓存
        """
        if api_name is None:
            # 清除所有缓存
            self.memory_cache.clear()
            if self.redis_client:
                try:
                    # 注意：这里需要更复杂的模式匹配来清除Redis缓存
                    logger.info("清除所有内存缓存")
                except Exception as e:
                    logger.warning(f"Redis缓存清除失败: {e}")
        else:
            cache_key = self._generate_cache_key(api_name, params or {})

            # 清除内存缓存
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]

            # 清除Redis缓存
            if self.redis_client:
                try:
                    await self.redis_client.delete(cache_key)
                except Exception as e:
                    logger.warning(f"Redis缓存删除失败: {e}")

            logger.debug(f"缓存已清除: {api_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, Any]: 缓存统计数据
        """
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0

        return {
            'memory_cache_size': len(self.memory_cache),
            'hit_rate': f"{hit_rate:.1f}%",
            'total_requests': total_requests,
            'cache_operations': self.cache_stats.copy(),
            'redis_enabled': self.redis_client is not None
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        缓存健康检查

        Returns:
            Dict[str, Any]: 健康检查结果
        """
        health_status = {
            'status': 'healthy',
            'memory_cache': True,
            'redis_cache': False,
            'issues': []
        }

        # 检查内存缓存
        try:
            test_key = 'health_check_test'
            test_data = {'test': True, 'timestamp': time.time()}
            await self.set('health_check', {'test': True}, test_data, ttl=10)

            cached_data = await self.get('health_check', {'test': True})
            if cached_data != test_data:
                health_status['issues'].append('内存缓存读写不一致')
                health_status['memory_cache'] = False
        except Exception as e:
            health_status['issues'].append(f'内存缓存异常: {e}')
            health_status['memory_cache'] = False

        # 检查Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.ping()
                health_status['redis_cache'] = True
            except Exception as e:
                health_status['issues'].append(f'Redis缓存异常: {e}')

        if health_status['issues']:
            health_status['status'] = 'unhealthy'

        return health_status

# 全局缓存管理器实例
_cache_manager = None

def get_akshare_cache_manager(redis_client=None) -> AKShareCacheManager:
    """
    获取AKShare缓存管理器实例（单例模式）

    Args:
        redis_client: Redis客户端实例

    Returns:
        AKShareCacheManager: 缓存管理器实例
    """
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = AKShareCacheManager(redis_client=redis_client)
        logger.info("AKShare缓存管理器已初始化")
    return _cache_manager