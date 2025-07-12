import time
from typing import Any, Dict, Optional
import zlib
from dataclasses import dataclass

@dataclass
class CacheItem:
    value: Any
    expire_at: float
    is_compressed: bool

class MemoryCache:
    """内存缓存实现"""

    def __init__(self):
        self._store: Dict[str, CacheItem] = {}

    def set(self, key: str, value: Any, ttl: int = 0, compress: bool = False) -> None:
        """设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 存活时间(秒)，0表示永不过期
            compress: 是否压缩存储
        """
        if compress:
            value = self._compress(value)

        expire_at = time.time() + ttl if ttl > 0 else float('inf')
        self._store[key] = CacheItem(value, expire_at, compress)

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        item = self._store.get(key)
        if not item:
            return None

        if time.time() > item.expire_at:
            del self._store[key]
            return None

        value = item.value
        if item.is_compressed:
            value = self._decompress(value)

        return value

    def delete(self, key: str) -> None:
        """删除缓存键"""
        if key in self._store:
            del self._store[key]

    def clear(self) -> None:
        """清空所有缓存"""
        self._store.clear()

    def _compress(self, data: Any) -> bytes:
        """压缩数据"""
        return zlib.compress(str(data).encode())

    def _decompress(self, data: bytes) -> Any:
        """解压数据"""
        return zlib.decompress(data).decode()
