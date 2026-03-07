"""
任务缓存模块

提供任务预取和结果缓存功能，减少重复计算
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000                # 最大缓存条目数
    default_ttl_seconds: int = 300      # 默认TTL（秒）
    enable_prefetch: bool = True        # 启用预取
    prefetch_threshold: int = 3         # 预取阈值（访问次数）
    cleanup_interval_seconds: int = 60  # 清理间隔
    compression_enabled: bool = False   # 启用压缩


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    last_accessed_at: datetime = field(default_factory=datetime.now)
    task_type: str = ""


class TaskCache:
    """
    任务缓存管理器

    提供以下功能：
    - 任务结果缓存（避免重复执行相同任务）
    - 智能预取（基于访问模式预测）
    - LRU淘汰策略
    - TTL过期管理

    使用场景：
    - 特征计算结果缓存
    - 模型预测结果缓存
    - 数据查询结果缓存
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化任务缓存

        Args:
            config: 缓存配置
        """
        self._config = config or CacheConfig()

        # 使用OrderedDict实现LRU
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # 预取管理
        self._access_patterns: Dict[str, List[datetime]] = {}  # 访问模式记录
        self._prefetch_candidates: Set[str] = set()  # 预取候选
        self._prefetch_handlers: Dict[str, Callable] = {}  # 预取处理器

        # 统计
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'prefetches': 0,
            'total_requests': 0
        }

        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None

    async def start(self):
        """启动缓存管理器"""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("✅ 任务缓存管理器已启动")

    async def stop(self):
        """停止缓存管理器"""
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 清空缓存
        async with self._lock:
            self._cache.clear()
            self._access_patterns.clear()
            self._prefetch_candidates.clear()

        logger.info("✅ 任务缓存管理器已停止")

    def _generate_key(
        self,
        task_type: str,
        payload: Dict[str, Any]
    ) -> str:
        """
        生成缓存键

        Args:
            task_type: 任务类型
            payload: 任务数据

        Returns:
            str: 缓存键
        """
        # 使用任务类型和payload的hash作为键
        content = f"{task_type}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    async def get(
        self,
        task_type: str,
        payload: Dict[str, Any],
        ttl_seconds: Optional[int] = None
    ) -> Optional[Any]:
        """
        获取缓存值

        Args:
            task_type: 任务类型
            payload: 任务数据
            ttl_seconds: 自定义TTL

        Returns:
            Optional[Any]: 缓存值，不存在或过期则返回None
        """
        key = self._generate_key(task_type, payload)

        async with self._lock:
            self._stats['total_requests'] += 1

            if key not in self._cache:
                self._stats['misses'] += 1
                return None

            entry = self._cache[key]

            # 检查是否过期
            if datetime.now() > entry.expires_at:
                # 过期，移除
                del self._cache[key]
                self._stats['misses'] += 1
                return None

            # 命中
            entry.access_count += 1
            entry.last_accessed_at = datetime.now()
            self._stats['hits'] += 1

            # 移动到末尾（LRU）
            self._cache.move_to_end(key)

            # 记录访问模式
            await self._record_access(key, task_type)

            return entry.value

    async def set(
        self,
        task_type: str,
        payload: Dict[str, Any],
        value: Any,
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """
        设置缓存值

        Args:
            task_type: 任务类型
            payload: 任务数据
            value: 缓存值
            ttl_seconds: TTL（秒）

        Returns:
            bool: 是否成功
        """
        key = self._generate_key(task_type, payload)
        ttl = ttl_seconds or self._config.default_ttl_seconds

        async with self._lock:
            # 检查是否需要淘汰
            if len(self._cache) >= self._config.max_size and key not in self._cache:
                await self._evict_lru()

            # 创建或更新条目
            now = datetime.now()
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=now + timedelta(seconds=ttl),
                task_type=task_type
            )

            self._cache[key] = entry
            self._cache.move_to_end(key)

            return True

    async def delete(
        self,
        task_type: str,
        payload: Dict[str, Any]
    ) -> bool:
        """
        删除缓存

        Args:
            task_type: 任务类型
            payload: 任务数据

        Returns:
            bool: 是否成功删除
        """
        key = self._generate_key(task_type, payload)

        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def clear(self) -> int:
        """
        清空缓存

        Returns:
            int: 清空的条目数
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_patterns.clear()
            self._prefetch_candidates.clear()
            return count

    async def _evict_lru(self):
        """淘汰最久未使用的条目"""
        if not self._cache:
            return

        # 弹出第一个（最久未使用）
        key, entry = self._cache.popitem(last=False)
        self._stats['evictions'] += 1

        # 清理相关记录
        if key in self._access_patterns:
            del self._access_patterns[key]
        if key in self._prefetch_candidates:
            self._prefetch_candidates.discard(key)

    async def _record_access(self, key: str, task_type: str):
        """
        记录访问模式

        Args:
            key: 缓存键
            task_type: 任务类型
        """
        now = datetime.now()

        if key not in self._access_patterns:
            self._access_patterns[key] = []

        self._access_patterns[key].append(now)

        # 只保留最近10次访问
        self._access_patterns[key] = self._access_patterns[key][-10:]

        # 检查是否成为预取候选
        if len(self._access_patterns[key]) >= self._config.prefetch_threshold:
            self._prefetch_candidates.add(key)

            # 触发预取
            if self._config.enable_prefetch:
                asyncio.create_task(self._prefetch_related(task_type, key))

    async def _prefetch_related(self, task_type: str, current_key: str):
        """
        预取相关任务

        Args:
            task_type: 任务类型
            current_key: 当前访问的键
        """
        if task_type not in self._prefetch_handlers:
            return

        handler = self._prefetch_handlers[task_type]

        try:
            # 获取预取键列表
            prefetch_keys = await handler(current_key)

            if prefetch_keys:
                self._stats['prefetches'] += len(prefetch_keys)
                logger.debug(f"预取 {len(prefetch_keys)} 个相关任务")

        except Exception as e:
            logger.error(f"预取失败: {e}")

    async def _cleanup_loop(self):
        """清理循环 - 定期移除过期条目"""
        while self._running:
            try:
                await asyncio.sleep(self._config.cleanup_interval_seconds)
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理错误: {e}")

    async def _cleanup_expired(self) -> int:
        """
        清理过期条目

        Returns:
            int: 清理的条目数
        """
        now = datetime.now()
        expired_keys = []

        async with self._lock:
            for key, entry in list(self._cache.items()):
                if now > entry.expires_at:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_patterns:
                    del self._access_patterns[key]
                if key in self._prefetch_candidates:
                    self._prefetch_candidates.discard(key)

        if expired_keys:
            logger.debug(f"清理 {len(expired_keys)} 个过期缓存条目")

        return len(expired_keys)

    def register_prefetch_handler(
        self,
        task_type: str,
        handler: Callable[[str], Any]
    ):
        """
        注册预取处理器

        Args:
            task_type: 任务类型
            handler: 预取处理器函数，接收当前键，返回预取键列表
        """
        self._prefetch_handlers[task_type] = handler

    def get_cache_key(
        self,
        task_type: str,
        payload: Dict[str, Any]
    ) -> str:
        """
        获取缓存键（不访问缓存）

        Args:
            task_type: 任务类型
            payload: 任务数据

        Returns:
            str: 缓存键
        """
        return self._generate_key(task_type, payload)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            Dict[str, Any]: 统计信息
        """
        hit_rate = 0
        if self._stats['total_requests'] > 0:
            hit_rate = self._stats['hits'] / self._stats['total_requests']

        return {
            **self._stats,
            'hit_rate': hit_rate,
            'size': len(self._cache),
            'max_size': self._config.max_size,
            'utilization': len(self._cache) / self._config.max_size if self._config.max_size > 0 else 0,
            'prefetch_candidates': len(self._prefetch_candidates)
        }

    def get_cache_info(self, key: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存条目信息

        Args:
            key: 缓存键

        Returns:
            Optional[Dict[str, Any]]: 条目信息
        """
        if key not in self._cache:
            return None

        entry = self._cache[key]
        now = datetime.now()

        return {
            'key': key,
            'task_type': entry.task_type,
            'created_at': entry.created_at.isoformat(),
            'expires_at': entry.expires_at.isoformat(),
            'ttl_remaining_seconds': (entry.expires_at - now).total_seconds(),
            'access_count': entry.access_count,
            'last_accessed_at': entry.last_accessed_at.isoformat(),
            'is_expired': now > entry.expires_at
        }

    async def get_all_keys(self) -> List[str]:
        """
        获取所有缓存键

        Returns:
            List[str]: 缓存键列表
        """
        async with self._lock:
            return list(self._cache.keys())

    async def get_keys_by_type(self, task_type: str) -> List[str]:
        """
        获取指定类型的所有缓存键

        Args:
            task_type: 任务类型

        Returns:
            List[str]: 缓存键列表
        """
        async with self._lock:
            return [
                key for key, entry in self._cache.items()
                if entry.task_type == task_type
            ]
