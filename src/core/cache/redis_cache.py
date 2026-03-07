#!/usr/bin/env python3
"""
Redis缓存核心模块
提供异步Redis缓存操作接口
"""

import asyncio
import logging
from typing import Any, Optional, Union
import redis
from redis.asyncio import Redis as AsyncRedis

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis缓存核心类"""

    def __init__(self, host: str = 'redis', port: int = 6379,
                 password: Optional[str] = None, db: int = 0):
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self._redis: Optional[AsyncRedis] = None
        self._lock = asyncio.Lock()

    async def _get_connection(self) -> AsyncRedis:
        """获取异步Redis连接"""
        if self._redis is None:
            async with self._lock:
                if self._redis is None:
                    self._redis = AsyncRedis(
                        host=self.host,
                        port=self.port,
                        password=self.password,
                        db=self.db,
                        decode_responses=False,
                        retry_on_timeout=True,
                        socket_timeout=5,
                        socket_connect_timeout=5,
                        max_connections=20
                    )
        return self._redis

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        try:
            redis_conn = await self._get_connection()
            data = await redis_conn.get(key)
            if data is None:
                return None

            # 如果是字符串，解码为字符串
            if isinstance(data, bytes):
                try:
                    return data.decode('utf-8')
                except UnicodeDecodeError:
                    return data
            return data
        except Exception as e:
            logger.error(f"Redis获取缓存失败: {e}")
            return None

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            redis_conn = await self._get_connection()
            if isinstance(value, str):
                value = value.encode('utf-8')
            elif not isinstance(value, (bytes, int, float)):
                # 对于复杂对象，使用JSON序列化
                import json
                value = json.dumps(value).encode('utf-8')

            result = await redis_conn.set(key, value, ex=expire)
            return result is True
        except Exception as e:
            logger.error(f"Redis设置缓存失败: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存数据"""
        try:
            redis_conn = await self._get_connection()
            result = await redis_conn.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis删除缓存失败: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        try:
            redis_conn = await self._get_connection()
            result = await redis_conn.exists(key)
            return result > 0
        except Exception as e:
            logger.error(f"Redis检查键存在失败: {e}")
            return False

    async def expire(self, key: str, seconds: int) -> bool:
        """设置键过期时间"""
        try:
            redis_conn = await self._get_connection()
            result = await redis_conn.expire(key, seconds)
            return result is True
        except Exception as e:
            logger.error(f"Redis设置过期时间失败: {e}")
            return False

    async def ttl(self, key: str) -> int:
        """获取键的剩余生存时间"""
        try:
            redis_conn = await self._get_connection()
            result = await redis_conn.ttl(key)
            return result
        except Exception as e:
            logger.error(f"Redis获取TTL失败: {e}")
            return -1

    async def close(self):
        """关闭Redis连接"""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None

    async def ping(self) -> bool:
        """测试Redis连接"""
        try:
            redis_conn = await self._get_connection()
            result = await redis_conn.ping()
            return result is True
        except Exception as e:
            logger.error(f"Redis连接测试失败: {e}")
            return False

    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.close()