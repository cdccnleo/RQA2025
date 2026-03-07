#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 访问控制组件 - 缓存管理器

负责权限检查结果的缓存管理，提高系统性能
"""

import logging
import threading
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .access_checker import AccessDecision


class CacheEvictionPolicy(Enum):
    """缓存淘汰策略"""
    LRU = "lru"  # 最近最少使用
    LFU = "lfu"  # 最少使用频率
    FIFO = "fifo"  # 先进先出
    TTL = "ttl"  # 基于时间的淘汰


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: AccessDecision
    created_at: datetime
    last_accessed: datetime
    access_count: int
    ttl_seconds: Optional[int] = None

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl_seconds is None:
            return False
        return datetime.now() - self.created_at > timedelta(seconds=self.ttl_seconds)

    def touch(self):
        """更新访问时间和计数"""
        self.last_accessed = datetime.now()
        self.access_count += 1


class CacheManager:
    """
    缓存管理器

    负责权限检查结果的缓存，支持多种淘汰策略和性能监控
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600,
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU,
                 enable_cleanup: bool = True, cleanup_interval: int = 300):
        """
        初始化缓存管理器

        Args:
            max_size: 最大缓存条目数
            ttl_seconds: 默认TTL时间（秒）
            eviction_policy: 淘汰策略
            enable_cleanup: 是否启用自动清理
            cleanup_interval: 清理间隔（秒）
        """
        self.max_size = max_size
        self.default_ttl_seconds = ttl_seconds
        self.eviction_policy = eviction_policy
        self.enable_cleanup = enable_cleanup
        self.cleanup_interval = cleanup_interval

        # 缓存存储：user_id -> resource -> permission -> CacheEntry
        self._cache: Dict[str, Dict[str, Dict[str, CacheEntry]]] = {}
        self._lock = threading.Lock()

        # 统计信息
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0,
            "cleanups": 0,
            "start_time": datetime.now()
        }

        # 清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False

        if self.enable_cleanup:
            self._start_cleanup_thread()

        logging.info(f"缓存管理器初始化完成: max_size={max_size}, policy={eviction_policy.value}")

    def get(self, user_id: str, resource: str, permission: str) -> Optional[AccessDecision]:
        """
        获取缓存的权限检查结果

        Args:
            user_id: 用户ID
            resource: 资源标识
            permission: 权限名

        Returns:
            缓存的决策结果或None
        """
        with self._lock:
            entry = self._get_entry(user_id, resource, permission)
            if entry and not entry.is_expired():
                entry.touch()
                self._stats["hits"] += 1
                return entry.value
            else:
                self._stats["misses"] += 1
                return None

    def set(self, user_id: str, resource: str, permission: str,
            decision: AccessDecision, ttl_seconds: Optional[int] = None):
        """
        设置缓存条目

        Args:
            user_id: 用户ID
            resource: 资源标识
            permission: 权限名
            decision: 权限决策
            ttl_seconds: 过期时间（秒），如果为None则使用默认值
        """
        ttl = ttl_seconds or self.default_ttl_seconds
        entry = CacheEntry(
            key=f"{user_id}:{resource}:{permission}",
            value=decision,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            ttl_seconds=ttl
        )

        with self._lock:
            # 初始化嵌套字典
            if user_id not in self._cache:
                self._cache[user_id] = {}
            if resource not in self._cache[user_id]:
                self._cache[user_id][resource] = {}

            # 设置条目
            self._cache[user_id][resource][permission] = entry
            self._stats["sets"] += 1

            # 检查是否需要淘汰
            self._enforce_size_limit()

    def delete(self, user_id: str, resource: Optional[str] = None,
              permission: Optional[str] = None):
        """
        删除缓存条目

        Args:
            user_id: 用户ID
            resource: 资源标识（可选）
            permission: 权限名（可选）
        """
        with self._lock:
            if user_id not in self._cache:
                return

            if resource is None:
                # 删除整个用户的缓存
                deleted_count = self._count_user_entries(user_id)
                del self._cache[user_id]
                self._stats["deletes"] += deleted_count
                logging.info(f"删除用户缓存: {user_id} ({deleted_count} 条)")

            elif permission is None:
                # 删除用户特定资源的缓存
                if resource in self._cache[user_id]:
                    deleted_count = len(self._cache[user_id][resource])
                    del self._cache[user_id][resource]
                    self._stats["deletes"] += deleted_count

            else:
                # 删除特定条目
                if resource in self._cache[user_id] and permission in self._cache[user_id][resource]:
                    del self._cache[user_id][resource][permission]
                    self._stats["deletes"] += 1

    def clear(self):
        """清空所有缓存"""
        with self._lock:
            total_entries = self._count_all_entries()
            self._cache.clear()
            self._stats["deletes"] += total_entries
            logging.info(f"清空所有缓存: {total_entries} 条")

    def cleanup(self):
        """手动执行缓存清理"""
        with self._lock:
            removed_count = 0

            # 清理过期条目
            for user_id in list(self._cache.keys()):
                for resource in list(self._cache[user_id].keys()):
                    for permission in list(self._cache[user_id][resource].keys()):
                        entry = self._cache[user_id][resource][permission]
                        if entry.is_expired():
                            del self._cache[user_id][resource][permission]
                            removed_count += 1

            # 如果仍然超过大小限制，则使用淘汰策略
            if self._count_all_entries() > self.max_size:
                removed_count += self._evict_entries()

            self._stats["cleanups"] += 1

            if removed_count > 0:
                logging.info(f"缓存清理完成: 移除 {removed_count} 条过期条目")

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            total_entries = self._count_all_entries()
            hit_rate = 0.0
            total_requests = self._stats["hits"] + self._stats["misses"]
            if total_requests > 0:
                hit_rate = self._stats["hits"] / total_requests

            uptime = datetime.now() - self._stats["start_time"]

            return {
                "total_entries": total_entries,
                "max_size": self.max_size,
                "current_size_percent": (total_entries / self.max_size) * 100 if self.max_size > 0 else 0,
                "hit_rate": hit_rate,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "sets": self._stats["sets"],
                "deletes": self._stats["deletes"],
                "evictions": self._stats["evictions"],
                "cleanups": self._stats["cleanups"],
                "uptime_seconds": uptime.total_seconds(),
                "eviction_policy": self.eviction_policy.value,
                "default_ttl_seconds": self.default_ttl_seconds,
                "cleanup_enabled": self.enable_cleanup,
                "cleanup_interval": self.cleanup_interval
            }

    def get_entries_for_user(self, user_id: str) -> Dict[str, Dict[str, AccessDecision]]:
        """
        获取指定用户的所有缓存条目

        Args:
            user_id: 用户ID

        Returns:
            用户的缓存条目字典
        """
        with self._lock:
            if user_id not in self._cache:
                return {}

            result = {}
            for resource, permissions in self._cache[user_id].items():
                result[resource] = {}
                for permission, entry in permissions.items():
                    if not entry.is_expired():
                        result[resource][permission] = entry.value

            return result

    def _get_entry(self, user_id: str, resource: str, permission: str) -> Optional[CacheEntry]:
        """
        获取缓存条目（内部方法，不加锁）

        Args:
            user_id: 用户ID
            resource: 资源标识
            permission: 权限名

        Returns:
            缓存条目或None
        """
        try:
            return self._cache[user_id][resource][permission]
        except KeyError:
            return None

    def _count_user_entries(self, user_id: str) -> int:
        """计算用户的缓存条目数量"""
        if user_id not in self._cache:
            return 0

        count = 0
        for resource in self._cache[user_id].values():
            count += len(resource)
        return count

    def _count_all_entries(self) -> int:
        """计算所有缓存条目数量"""
        count = 0
        for user_cache in self._cache.values():
            for resource_cache in user_cache.values():
                count += len(resource_cache)
        return count

    def _enforce_size_limit(self):
        """强制执行大小限制"""
        while self._count_all_entries() > self.max_size:
            self._evict_entries()

    def _evict_entries(self) -> int:
        """
        根据淘汰策略移除条目

        Returns:
            移除的条目数量
        """
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            return self._evict_lru()
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            return self._evict_lfu()
        elif self.eviction_policy == CacheEvictionPolicy.FIFO:
            return self._evict_fifo()
        elif self.eviction_policy == CacheEvictionPolicy.TTL:
            return self._evict_expired()
        else:
            return self._evict_lru()  # 默认使用LRU

    def _evict_lru(self) -> int:
        """LRU淘汰：移除最久未访问的条目"""
        all_entries = []

        for user_id, user_cache in self._cache.items():
            for resource, resource_cache in user_cache.items():
                for permission, entry in resource_cache.items():
                    all_entries.append((user_id, resource, permission, entry.last_accessed))

        if not all_entries:
            return 0

        # 按最后访问时间排序，移除最旧的
        all_entries.sort(key=lambda x: x[3])

        # 移除最旧的条目（可以移除多个以确保低于限制）
        entries_to_remove = max(1, len(all_entries) - self.max_size + 1)

        for i in range(min(entries_to_remove, len(all_entries))):
            user_id, resource, permission, _ = all_entries[i]
            if permission in self._cache[user_id][resource]:
                del self._cache[user_id][resource][permission]
                self._stats["evictions"] += 1

        return min(entries_to_remove, len(all_entries))

    def _evict_lfu(self) -> int:
        """LFU淘汰：移除访问频率最低的条目"""
        all_entries = []

        for user_id, user_cache in self._cache.items():
            for resource, resource_cache in user_cache.items():
                for permission, entry in resource_cache.items():
                    all_entries.append((user_id, resource, permission, entry.access_count))

        if not all_entries:
            return 0

        # 按访问次数排序，移除访问最少的
        all_entries.sort(key=lambda x: x[3])

        entries_to_remove = max(1, len(all_entries) - self.max_size + 1)

        for i in range(min(entries_to_remove, len(all_entries))):
            user_id, resource, permission, _ = all_entries[i]
            if permission in self._cache[user_id][resource]:
                del self._cache[user_id][resource][permission]
                self._stats["evictions"] += 1

        return min(entries_to_remove, len(all_entries))

    def _evict_fifo(self) -> int:
        """FIFO淘汰：移除最先创建的条目"""
        all_entries = []

        for user_id, user_cache in self._cache.items():
            for resource, resource_cache in user_cache.items():
                for permission, entry in resource_cache.items():
                    all_entries.append((user_id, resource, permission, entry.created_at))

        if not all_entries:
            return 0

        # 按创建时间排序，移除最旧的
        all_entries.sort(key=lambda x: x[3])

        entries_to_remove = max(1, len(all_entries) - self.max_size + 1)

        for i in range(min(entries_to_remove, len(all_entries))):
            user_id, resource, permission, _ = all_entries[i]
            if permission in self._cache[user_id][resource]:
                del self._cache[user_id][resource][permission]
                self._stats["evictions"] += 1

        return min(entries_to_remove, len(all_entries))

    def _evict_expired(self) -> int:
        """TTL淘汰：移除过期的条目"""
        removed_count = 0

        for user_id in list(self._cache.keys()):
            for resource in list(self._cache[user_id].keys()):
                for permission in list(self._cache[user_id][resource].keys()):
                    entry = self._cache[user_id][resource][permission]
                    if entry.is_expired():
                        del self._cache[user_id][resource][permission]
                        removed_count += 1
                        self._stats["evictions"] += 1

        return removed_count

    def _start_cleanup_thread(self):
        """启动清理线程"""
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        """清理循环"""
        while not self._stop_cleanup:
            time.sleep(self.cleanup_interval)
            if not self._stop_cleanup:
                self.cleanup()

    def shutdown(self):
        """关闭缓存管理器"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        logging.info("缓存管理器已关闭")
