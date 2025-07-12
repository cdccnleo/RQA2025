"""
增强缓存实现，支持多级缓存和智能失效策略
"""
from datetime import datetime, timedelta
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple, Union
import pickle
import zlib

from src.data.interfaces import ICacheBackend
from src.infrastructure.utils.exceptions import CacheError

logger = logging.getLogger(__name__)


class EnhancedCache:
    """
    增强型缓存实现，提供：
    - 多级缓存支持（内存+持久化）
    - 智能缓存失效策略
    - 缓存压缩和序列化
    - 缓存统计和监控
    """
    def __init__(self,
                 primary_backend: ICacheBackend,
                 secondary_backend: Optional[ICacheBackend] = None,
                 compression: bool = True,
                 default_ttl: int = 3600):
        """
        初始化缓存

        Args:
            primary_backend: 主缓存后端（必须实现ICacheBackend接口）
            secondary_backend: 二级缓存后端（可选）
            compression: 是否启用压缩
            default_ttl: 默认缓存时间（秒）
        """
        self.primary = primary_backend
        self.secondary = secondary_backend
        self.compression = compression
        self.default_ttl = default_ttl

        # 缓存统计
        self.stats = {
            'hits': 0,
            'misses': 0,
            'primary_hits': 0,
            'secondary_hits': 0,
            'expired': 0,
            'compression_saved': 0,
            'total_size': 0
        }

        logger.info("EnhancedCache initialized with primary: %s, secondary: %s",
                  primary_backend.__class__.__name__,
                  secondary_backend.__class__.__name__ if secondary_backend else None)

    def _serialize(self, value: Any) -> bytes:
        """序列化值"""
        try:
            serialized = pickle.dumps(value)
            if self.compression:
                compressed = zlib.compress(serialized)
                self.stats['compression_saved'] += len(serialized) - len(compressed)
                return compressed
            return serialized
        except Exception as e:
            raise CacheError(f"Serialization failed: {str(e)}")

    def _deserialize(self, value: bytes) -> Any:
        """反序列化值"""
        try:
            if self.compression:
                try:
                    decompressed = zlib.decompress(value)
                except zlib.error:
                    decompressed = value  # 可能是未压缩的数据
                return pickle.loads(decompressed)
            return pickle.loads(value)
        except Exception as e:
            raise CacheError(f"Deserialization failed: {str(e)}")

    def _generate_key(self, *args, **kwargs) -> str:
        """生成缓存键"""
        parts = [str(arg) for arg in args]
        parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "&".join(parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值

        Args:
            key: 缓存键
            default: 默认值（当缓存未命中时返回）

        Returns:
            缓存的值或默认值
        """
        try:
            # 先从主缓存获取
            value = self.primary.get(key)
            if value is not None:
                self.stats['hits'] += 1
                self.stats['primary_hits'] += 1
                return self._deserialize(value)

            # 主缓存未命中，尝试二级缓存
            if self.secondary:
                value = self.secondary.get(key)
                if value is not None:
                    self.stats['hits'] += 1
                    self.stats['secondary_hits'] += 1

                    # 回填到主缓存
                    self.primary.set(key, value, ttl=self.default_ttl)
                    return self._deserialize(value)

            # 缓存未命中
            self.stats['misses'] += 1
            return default
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {str(e)}")
            return default

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 要缓存的值
            ttl: 缓存时间（秒），None表示使用默认ttl

        Returns:
            是否设置成功
        """
        try:
            serialized = self._serialize(value)
            ttl = ttl if ttl is not None else self.default_ttl

            # 更新到主缓存
            primary_success = self.primary.set(key, serialized, ttl=ttl)

            # 如果有二级缓存，也更新
            secondary_success = True
            if self.secondary:
                secondary_success = self.secondary.set(key, serialized, ttl=ttl)

            # 更新统计
            self.stats['total_size'] += len(serialized)

            return primary_success and secondary_success
        except Exception as e:
            logger.error(f"Cache set failed for key {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存值

        Args:
            key: 要删除的缓存键

        Returns:
            是否删除成功
        """
        try:
            # 从主缓存删除
            primary_success = self.primary.delete(key)

            # 从二级缓存删除
            secondary_success = True
            if self.secondary:
                secondary_success = self.secondary.delete(key)

            return primary_success and secondary_success
        except Exception as e:
            logger.error(f"Cache delete failed for key {key}: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            是否清空成功
        """
        try:
            # 清空主缓存
            primary_success = self.primary.clear()

            # 清空二级缓存
            secondary_success = True
            if self.secondary:
                secondary_success = self.secondary.clear()

            # 重置统计
            self.stats = {
                'hits': 0,
                'misses': 0,
                'primary_hits': 0,
                'secondary_hits': 0,
                'expired': 0,
                'compression_saved': 0,
                'total_size': 0
            }

            return primary_success and secondary_success
        except Exception as e:
            logger.error(f"Cache clear failed: {str(e)}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            包含缓存统计信息的字典
        """
        return self.stats.copy()

    def get_memory_usage(self) -> int:
        """
        获取缓存内存使用量（字节）

        Returns:
            内存使用量（字节）
        """
        return self.stats['total_size']

    def get_hit_rate(self) -> float:
        """
        获取缓存命中率

        Returns:
            命中率（0.0到1.0之间）
        """
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0

    def expire_old_entries(self, max_age: int) -> int:
        """
        使超过指定年龄的缓存条目失效

        Args:
            max_age: 最大年龄（秒）

        Returns:
            失效的条目数
        """
        # 此方法需要后端支持才能实现
        logger.warning("expire_old_entries not implemented for this backend")
        return 0
