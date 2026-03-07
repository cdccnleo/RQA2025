#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轻量级特征缓存存储实现

提供线程安全的内存级缓存，支持 TTL、命中统计等能力，
用于在单元测试或本地开发场景中快速验证特征读写链路。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple


@dataclass
class CacheEntry:
    """缓存条目"""

    value: Any
    expires_at: Optional[datetime]


class CacheStore:
    """线程安全的内存缓存"""

    def __init__(self, default_ttl: Optional[int] = 300):
        """
        Args:
            default_ttl: 默认过期时间（秒），None 表示不过期
        """
        self.default_ttl = default_ttl
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "clears": 0,
        }

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """写入缓存"""
        expires_at = self._compute_expiry(ttl)
        with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=expires_at)
            self._stats["sets"] += 1
        return True

    def get(self, key: str) -> Any:
        """读取缓存，自动处理过期条目"""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.expires_at and entry.expires_at < datetime.utcnow():
                # 过期移除
                del self._store[key]
                self._stats["misses"] += 1
                return None

            self._stats["hits"] += 1
            return entry.value

    def delete(self, key: str) -> bool:
        """删除指定缓存"""
        with self._lock:
            self._store.pop(key, None)
        return True

    def clear(self) -> bool:
        """清空所有缓存"""
        with self._lock:
            self._store.clear()
            self._stats["clears"] += 1
        return True

    def stats(self) -> Dict[str, int]:
        """获取缓存命中统计"""
        with self._lock:
            return dict(self._stats)

    def cleanup_expired(self) -> int:
        """主动清理过期条目"""
        now = datetime.utcnow()
        removed = 0
        with self._lock:
            keys = list(self._store.keys())
            for key in keys:
                entry = self._store.get(key)
                if entry and entry.expires_at and entry.expires_at < now:
                    del self._store[key]
                    removed += 1
        return removed

    def _compute_expiry(self, ttl: Optional[int]) -> Optional[datetime]:
        """计算过期时间"""
        ttl_to_use = ttl if ttl is not None else self.default_ttl
        if ttl_to_use is None or ttl_to_use <= 0:
            return None
        return datetime.utcnow() + timedelta(seconds=ttl_to_use)

