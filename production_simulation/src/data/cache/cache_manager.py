#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器模块
"""

import json
import pickle
import hashlib
import time
import threading
import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from pathlib import Path

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
    logger = get_infrastructure_logger('data_cache_manager')
except ImportError:
    # 降级到标准logging
    logger = logging.getLogger('data_cache_manager')
    logger.warning("无法导入基础设施层日志，使用标准logging")

# 导入全局接口
try:
    from src.infrastructure.cache.global_interfaces import ICacheStrategy, CacheEvictionStrategy
except ImportError:
    # 如果全局接口不可用，创建本地定义
    from typing import Protocol

    class ICacheStrategy(Protocol):

        """缓存策略接口"""

        def should_evict(self, key: str, value: Any, cache_size: int) -> bool: ...

        def on_access(self, key: str, value: Any) -> None: ...

        def on_evict(self, key: str, value: Any) -> None: ...

        def on_get(self, cache: Dict[str, Any], key: str, entry: Any, config: Any) -> None: ...

        def on_set(self, cache: Dict[str, Any], key: str, entry: Any, config: Any) -> None: ...

    from enum import Enum

    class CacheEvictionStrategy(Enum):

        LRU = "lru"
        LFU = "lfu"
        FIFO = "fifo"
        RANDOM = "random"
        TTL = "ttl"

# 日志降级处理


def get_data_logger(name: str):
    """获取数据层日志器，支持降级"""
    try:
        from src.infrastructure.logging import UnifiedLogger
        return UnifiedLogger(name)
    except ImportError:
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


logger = get_data_logger('data_cache_manager')


@dataclass
class CacheConfig:

    """缓存配置"""
    max_size: int = 1000
    ttl: int = 3600  # 默认1小时
    enable_disk_cache: bool = True
    disk_cache_dir: str = "cache"
    compression: bool = False
    encryption: bool = False
    encryption_key: Optional[str] = None
    enable_stats: bool = True
    cleanup_interval: int = 300  # 5分钟清理一次
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_enabled: bool = False
    backup_interval: int = 3600  # 1小时备份一次


class CacheEntry:

    """缓存条目"""

    def __init__(self, key: str, value: Any, ttl: Optional[int] = None, created_at: Optional[float] = None):

        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = created_at or time.time()
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def access(self):
        """访问缓存条目"""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'key': self.key,
            'value': self.value,
            'ttl': self.ttl,
            'created_at': self.created_at,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """从字典创建"""
        entry = cls(
            key=data['key'],
            value=data['value'],
            ttl=data.get('ttl'),
            created_at=data.get('created_at')
        )
        entry.access_count = data.get('access_count', 0)
        entry.last_accessed = data.get('last_accessed', entry.created_at)
        return entry


class CacheStats:

    """缓存统计"""

    def __init__(self):

        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        self.evictions = 0
        self.errors = 0
        self.start_time = time.time()
        # 添加cache属性以兼容测试
        self.cache = {
            'size': 0,
            'hit_rate': 0.0,
            'total_entries': 0
        }
        # 添加config属性以兼容测试
        self.config = CacheConfig()

    def hit(self):
        """命中"""
        self.hits += 1

    def miss(self):
        """未命中"""
        self.misses += 1

    def set(self):
        """设置"""
        self.sets += 1

    def delete(self):
        """删除"""
        self.deletes += 1

    def evict(self):
        """驱逐"""
        self.evictions += 1

    def error(self):
        """错误"""
        self.errors += 1

    @property
    def hit_rate(self) -> float:
        """获取命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def total_requests(self) -> int:
        """获取总请求数"""
        return self.hits + self.misses

    def get_hit_rate(self) -> float:
        """获取命中率（向后兼容）"""
        return self.hit_rate

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self.start_time
        return {
            'hits': self.hits,
            'misses': self.misses,
            'sets': self.sets,
            'deletes': self.deletes,
            'evictions': self.evictions,
            'errors': self.errors,
            'hit_rate': self.get_hit_rate(),
            'uptime': uptime,
            'total_requests': self.hits + self.misses,
            'memory_cache': {
                'size': len(self.cache),
                'max_size': self.config.max_size,
                'usage_ratio': len(self.cache) / self.config.max_size if self.config.max_size > 0 else 0
            },
            'disk_cache': {
                'enabled': self.config.enable_disk_cache,
                'size': 0,  # 简化实现
                'path': self.config.disk_cache_dir
            },
            'cache': {
                'size': len(self.cache),
                'hit_rate': self.get_hit_rate(),
                'total_entries': len(self.cache)
            }
        }


class approach:

    """
    智能缓存策略接口，支持自定义淘汰、预热、优先级等策略
    """

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig):
        """设置缓存时的策略钩子，可用于预热、优先级调整等"""

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig):
        """获取缓存时的策略钩子，可用于动态调整"""

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key，支持自定义淘汰逻辑"""
        return None


class CacheManager:

    """缓存管理器"""

    def __init__(self, config: CacheConfig = None, strategy: Optional[ICacheStrategy] = None):

        self.config = config or CacheConfig()

        # 内存缓存
        self._cache: Dict[str, CacheEntry] = {}
        self.memory_cache = self._cache  # 为测试提供访问接口
        self._lock = threading.RLock()

        # 统计
        self._stats = CacheStats()

        # 磁盘缓存
        self.disk_cache = None
        self._cache_dir = Path(self.config.disk_cache_dir)
        if self.config.enable_disk_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            from .disk_cache import DiskCache, DiskCacheConfig
            disk_config = DiskCacheConfig(
                cache_dir=self.config.disk_cache_dir,
                max_file_size=self.config.max_file_size,
                compression=self.config.compression,
                encryption=self.config.encryption,
                backup_enabled=self.config.backup_enabled,
                cleanup_interval=self.config.cleanup_interval
            )
            self.disk_cache = DiskCache(disk_config)
        else:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        # 启动清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False
        if self.config.enable_stats:
            self._start_cleanup_thread()

        self.strategy = strategy or None  # 智能策略扩展点

        # 添加logger属性
        self.logger = logger
        self._last_clear_time = 0.0

    def stop(self):
        """停止缓存管理器，清理所有资源"""
        self.logger.info("正在停止缓存管理器...")

        # 设置停止标志
        self._stop_cleanup = True

        # 等待清理线程结束，使用更短的超时时间
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=2.0)
            if self._cleanup_thread.is_alive():
                self.logger.warning("缓存管理器清理线程未能及时停止，强制终止")
            else:
                self.logger.info("缓存管理器清理线程已正常停止")

        # 停止磁盘缓存
        if self.disk_cache:
            try:
                self.disk_cache.stop()
            except Exception as e:
                self.logger.error(f"停止磁盘缓存失败: {e}")

        self.logger.info("缓存管理器已停止")

    def __del__(self):
        """析构函数，确保资源被清理"""
        try:
            self.stop()
        except Exception:
            pass  # 忽略析构时的异常

    def _start_cleanup_thread(self):
        """启动清理线程"""

        def cleanup_worker():
            """清理工作线程，避免死锁"""
            last_cleanup_time = time.time()

            while not self._stop_cleanup:
                try:
                    current_time = time.time()

                    # 检查是否需要清理（避免频繁清理）
                    if current_time - last_cleanup_time >= self.config.cleanup_interval:
                        self._cleanup_expired()
                        last_cleanup_time = current_time

                    # 使用更短的睡眠时间，避免长时间阻塞
                    time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"Cleanup error: {e}")
                    self._stats.error()
                    # 出现异常时稍等一下再试
                    time.sleep(1)

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self) -> int:
        """清理过期条目"""
        with self._lock:
            expired_keys = []
            for key, entry in list(self._cache.items()):
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                self._stats.evict()

            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired entries")

            return len(expired_keys)

    def _get_disk_path(self, key: str) -> Path:
        """获取磁盘缓存路径"""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.cache"

    def _save_to_disk(self, key: str, entry: CacheEntry) -> bool:
        """保存到磁盘"""
        if not self.config.enable_disk_cache:
            return False

        try:
            disk_path = self._get_disk_path(key)
            data = {
                'entry': entry.to_dict(),
                'config': {
                    'compression': self.config.compression,
                    'encryption': self.config.encryption
                }
            }

            with open(disk_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save to disk: {e}")
            self._stats.error()
            return False

    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """从磁盘加载"""
        if not self.config.enable_disk_cache:
            return None

        try:
            disk_path = self._get_disk_path(key)
            if not disk_path.exists():
                return None

            with open(disk_path, 'rb') as f:
                data = pickle.load(f)

            entry = CacheEntry.from_dict(data['entry'])
            if entry.is_expired():
                disk_path.unlink()
                return None

            return entry
        except Exception as e:
            self.logger.error(f"Failed to load from disk: {e}")
            self._stats.error()
            return None

    def _evict_if_needed(self):
        """如果需要则驱逐缓存"""
        if len(self._cache) <= self.config.max_size:
            return

        # LRU驱逐策略
        entries = sorted(self._cache.items(), key=lambda x: x[1].last_accessed)
        to_evict = len(self._cache) - self.config.max_size

        for i in range(to_evict):
            key, entry = entries[i]
            del self._cache[key]
            self._stats.evict()

            # 删除磁盘缓存
            if self.config.enable_disk_cache:
                disk_path = self._get_disk_path(key)
                if disk_path.exists():
                    disk_path.unlink()

        if self.strategy:
            evict_key = self.strategy.on_evict(self._cache, self.config)
            if evict_key and evict_key in self._cache:
                del self._cache[evict_key]
                self._stats.evictions += 1
                return

    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self._lock:
            # 先检查内存缓存
            entry = self._cache.get(key)
            if self.strategy:
                self.strategy.on_get(self._cache, key, entry, self.config)
            if entry:
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.miss()
                    return None

                entry.access()
                self._stats.hit()
                return entry.value

            # 检查磁盘缓存
            if self.disk_cache:
                disk_entry = self.disk_cache.get_entry(key, update_metadata=False)
                if disk_entry is not None:
                    if disk_entry.created_at < self._last_clear_time:
                        # 清空操作之后的陈旧数据，不再回填
                        self.disk_cache.delete(key)
                        self._stats.miss()
                        return None

                    disk_entry.access()
                    self._cache[key] = disk_entry
                    self._stats.hit()
                    self._evict_if_needed()
                    return disk_entry.value

            self._stats.miss()
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            with self._lock:
                ttl = ttl or self.config.ttl
                entry = CacheEntry(key=key, value=value, ttl=ttl)

                self._cache[key] = entry
                self._stats.set()

                # 保存到磁盘
                if self.disk_cache:
                    self.disk_cache.set(key, value, ttl)

                self._evict_if_needed()
                if self.strategy:
                    self.strategy.on_set(self._cache, key, entry, self.config)
                return True
        except Exception as e:
            self.logger.error(f"Failed to set cache: {e}")
            self._stats.error()
            return False

    def delete(self, key: str) -> bool:
        """删除缓存值"""
        try:
            with self._lock:
                if key in self._cache:
                    del self._cache[key]
                    self._stats.delete()

                # 删除磁盘缓存（磁盘删除失败不影响整体删除结果）
                if self.disk_cache:
                    try:
                        self.disk_cache.delete(key)
                    except Exception as e:
                        self.logger.error(f"Failed to delete cache: {e}")
                        self._stats.error()
                        # 继续返回 True，表示内存侧已删除且整体操作不被磁盘失败阻断

                return True
        except Exception as e:
            self.logger.error(f"Failed to delete cache: {e}")
            self._stats.error()
            return False

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired():
                    del self._cache[key]
                    return False
                return True

            if self.disk_cache:
                return self.disk_cache.exists(key)

            return False

    def has(self, key: str) -> bool:
        """向后兼容的exists别名"""
        return self.exists(key)

    def clear(self) -> int:
        """清空缓存"""
        try:
            with self._lock:
                self._last_clear_time = time.time()
                cleared = len(self._cache)
                self._cache.clear()

                # 清空磁盘缓存
                if self.disk_cache:
                    self.disk_cache.clear()

                if self.config.enable_stats:
                    return cleared
                return True
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            self._stats.error()
            return 0 if self.config.enable_stats else False

    def list_keys(self) -> List[str]:
        """列出所有键"""
        with self._lock:
            # 清理过期条目
            self._cleanup_expired()
            return list(self._cache.keys())

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            stats = self._stats.get_stats()
            stats.update({
                'cache_size': len(self._cache),
                'current_size': len(self._cache),
                'max_size': self.config.max_size,
                'disk_cache_enabled': self.config.enable_disk_cache,
                'compression_enabled': self.config.compression,
                'encryption_enabled': self.config.encryption
            })

            # 添加磁盘缓存统计
            if self.disk_cache:
                disk_stats = self.disk_cache.get_stats()
                stats['disk_cache'] = disk_stats.get('disk_cache', {})

            return stats

    def set_max_size(self, max_size: int) -> bool:
        """设置最大容量"""
        with self._lock:
            self.config.max_size = max_size
            self._evict_if_needed()
            return True

    def cleanup_expired(self) -> int:
        """公开的过期清理接口"""
        return self._cleanup_expired()

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试基本操作
            test_key = "_health_check"
            test_value = {"status": "ok", "timestamp": time.time()}

            # 测试设置
            if not self.set(test_key, test_value, ttl=10):
                return {"status": "error", "message": "Set operation failed"}

            # 测试获取
            retrieved = self.get(test_key)
            if retrieved != test_value:
                return {"status": "error", "message": "Get operation failed"}

            # 测试删除
            if not self.delete(test_key):
                return {"status": "error", "message": "Delete operation failed"}

            return {
                "status": "healthy",
                "stats": self.get_stats(),
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "timestamp": time.time()
            }

    def close(self):
        """关闭缓存管理器"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)

        # 关闭磁盘缓存
        if self.disk_cache:
            self.disk_cache.close()

        # 保存统计信息
        if self.config.enable_stats and hasattr(self, '_cache_dir'):
            stats_path = self._cache_dir / "cache_stats.json"
            try:
                with open(stats_path, 'w') as f:
                    json.dump(self.get_stats(), f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save stats: {e}")


# 导出主要类
__all__ = ['CacheConfig', 'CacheManager', 'CacheEntry', 'CacheStats']
