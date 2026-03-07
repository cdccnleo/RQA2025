"""
增强版缓存管理器
实现实际的缓存存储和检索逻辑
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):
        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import os
import pickle
import hashlib
import time
import threading
from collections import defaultdict
from typing import Any, Optional, Dict
import pandas as pd
import numpy as np

logger = get_infrastructure_logger(__name__)


class EnhancedCacheManager:

    """
    增强版缓存管理器
    支持多级缓存、智能过期、性能监控
    """

    def __init__(self, cache_dir: str = "cache", max_memory_size: int = 100 * 1024 * 1024,  # 100MB


                 max_disk_size: int = 1024 * 1024 * 1024):  # 1GB
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
            max_memory_size: 最大内存缓存大小（字节）
            max_disk_size: 最大磁盘缓存大小（字节）
        """
        self.cache_dir = cache_dir
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size

        # 内存缓存
        self.memory_cache = {}
        self.memory_size = 0
        self.prefix_index = defaultdict(set)

        # 磁盘缓存
        self.disk_cache_dir = cache_dir
        self._ensure_cache_dir()

        # 缓存统计
        self.stats = {
            'memory_hits': 0,
            'disk_hits': 0,
            'misses': 0,
            'memory_size': 0,
            'disk_size': 0,
            'total_operations': 0
        }

        # 线程锁
        self._lock = threading.RLock()

        # 清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_thread()

        logger.info(f"EnhancedCacheManager initialized with cache_dir: {cache_dir}")

    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def _generate_cache_key(self, key: str, prefix: str = "") -> str:
        """
        生成缓存键

        Args:
            key: 原始键
            prefix: 前缀

        Returns:
            str: 生成的缓存键
        """
        if prefix:
            full_key = f"{prefix}_{key}"
        else:
            full_key = key

        # 使用MD5生成固定长度的键
        return hashlib.md5(full_key.encode()).hexdigest()

    def _get_memory_size(self, obj: Any) -> int:
        """
        估算对象的内存大小

        Args:
            obj: 要估算的对象

        Returns:
            int: 估算的内存大小（字节）
        """
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum()
            elif isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (dict, list, tuple, set)):
                try:
                    return len(pickle.dumps(obj))
                except Exception:
                    return sum(len(str(item)) for item in obj)
            else:
                return len(str(obj))
        except BaseException:
            return 1024  # 默认1KB

    def set(self, key: str, value: Any, expire: int = 3600, prefix: str = "") -> bool:
        """
        设置缓存

        Args:
            key: 缓存键
            value: 缓存值
            expire: 过期时间（秒）
            prefix: 键前缀

        Returns:
            bool: 是否设置成功
        """
        # 输入验证
        if not key or not isinstance(key, str):
            raise ValueError("Key must be a non - empty string")

        if value is None:
            raise ValueError("Value cannot be None")

        if expire < 0:
            raise ValueError("Expire time must be non - negative")

        try:
            cache_key = self._generate_cache_key(key, prefix)
            expire_time = time.time() + expire

            cache_item = {
                'value': value,
                'expire_time': expire_time,
                'size': self._get_memory_size(value),
                'created_time': time.time(),
                'access_count': 0
            }

            with self._lock:
                stored_in_memory = self._try_memory_cache(cache_key, cache_item)
                stored_on_disk = self._try_disk_cache(cache_key, cache_item)

                if stored_in_memory or stored_on_disk:
                    prefix_key = prefix or ""
                    self.prefix_index[prefix_key].add(cache_key)
                    self.stats['total_operations'] += 1
                    return True

            logger.warning(f"Failed to cache key: {key}")
            return False

        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {e}")
            return False

    def _try_memory_cache(self, cache_key: str, cache_item: Dict) -> bool:
        """
        尝试存储到内存缓存

        Args:
            cache_key: 缓存键
            cache_item: 缓存项

        Returns:
            bool: 是否成功存储
        """
        item_size = cache_item['size']

        # 检查内存大小限制
        if self.memory_size + item_size > self.max_memory_size:
            # 清理一些旧数据
            self._cleanup_memory_cache()

            # 再次检查
            if self.memory_size + item_size > self.max_memory_size:
                return False

        # 存储到内存
        self.memory_cache[cache_key] = cache_item
        self.memory_size += item_size
        self.stats['memory_size'] = self.memory_size

        logger.debug(f"Cached in memory: {cache_key}, size: {item_size}")
        return True

    def _try_disk_cache(self, cache_key: str, cache_item: Dict) -> bool:
        """
        尝试存储到磁盘缓存

        Args:
            cache_key: 缓存键
            cache_item: 缓存项

        Returns:
            bool: 是否成功存储
        """
        try:
            # 检查磁盘大小
            current_disk_size = self._get_disk_cache_size()
            if current_disk_size + cache_item['size'] > self.max_disk_size:
                self._cleanup_disk_cache()

                # 再次检查
                if self._get_disk_cache_size() + cache_item['size'] > self.max_disk_size:
                    return False

            # 存储到磁盘
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_item, f)

            self.stats['disk_size'] = self._get_disk_cache_size()
            logger.debug(f"Cached on disk: {cache_key}")
            return True

        except Exception as e:
            logger.error(f"Error storing to disk cache: {e}")
            return False

    def get(self, key: str, prefix: str = "") -> Optional[Any]:
        """
        获取缓存

        Args:
            key: 缓存键
            prefix: 键前缀

        Returns:
            Optional[Any]: 缓存值，如果不存在或已过期则返回None
        """
        try:
            cache_key = self._generate_cache_key(key, prefix)

            with self._lock:
                # 首先检查内存缓存
                if cache_key in self.memory_cache:
                    cache_item = self.memory_cache[cache_key]
                    if self._is_valid_cache_item(cache_item):
                        cache_item['access_count'] += 1
                        self.stats['memory_hits'] += 1
                        self.stats['total_operations'] += 1
                        return cache_item['value']
                    else:
                        # 移除过期项
                        del self.memory_cache[cache_key]
                        self.memory_size -= cache_item['size']

                # 检查磁盘缓存
                cache_item = self._get_from_disk_cache(cache_key)
                if cache_item and self._is_valid_cache_item(cache_item):
                    cache_item['access_count'] += 1
                    self.stats['disk_hits'] += 1
                    self.stats['total_operations'] += 1
                    self._promote_to_memory(cache_key, cache_item)

                    return cache_item['value']

                self.stats['misses'] += 1
                self.stats['total_operations'] += 1
                return None

        except Exception as e:
            logger.error(f"Error getting cache for key {key}: {e}")
            return None

    def _get_from_disk_cache(self, cache_key: str) -> Optional[Dict]:
        """
        从磁盘缓存获取数据

        Args:
            cache_key: 缓存键

        Returns:
            Optional[Dict]: 缓存项
        """
        try:
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")

        return None

    def _is_valid_cache_item(self, cache_item: Dict) -> bool:
        """
        检查缓存项是否有效

        Args:
            cache_item: 缓存项

        Returns:
            bool: 是否有效
        """
        return cache_item['expire_time'] > time.time()

    def _promote_to_memory(self, cache_key: str, cache_item: Dict):
        """
        将磁盘缓存项提升到内存

        Args:
            cache_key: 缓存键
            cache_item: 缓存项
        """
        if self._try_memory_cache(cache_key, cache_item):
            # 从磁盘删除
            cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
            try:
                os.remove(cache_file)
            except BaseException:
                pass

    def _cleanup_memory_cache(self):
        """清理内存缓存"""
        current_time = time.time()

        # 移除过期项
        expired_keys = []
        for key, item in self.memory_cache.items():
            if not self._is_valid_cache_item(item):
                expired_keys.append(key)

        for key in expired_keys:
            item = self.memory_cache[key]
            self.memory_size -= item['size']
            del self.memory_cache[key]

        # 如果仍然超过限制，移除最少访问的项
        if self.memory_size > self.max_memory_size * 0.8:
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1]['access_count']
            )

            for key, item in sorted_items:
                if self.memory_size <= self.max_memory_size * 0.7:
                    break

                self.memory_size -= item['size']
                del self.memory_cache[key]

    def _cleanup_disk_cache(self):
        """清理磁盘缓存"""
        try:
            # 检查磁盘缓存目录是否存在
            if not os.path.exists(self.disk_cache_dir):
                return

            current_time = time.time()

            for filename in os.listdir(self.disk_cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.disk_cache_dir, filename)
                    try:
                        with open(filepath, 'rb') as f:
                            cache_item = pickle.load(f)

                        if not self._is_valid_cache_item(cache_item):
                            os.remove(filepath)
                    except BaseException:
                        # 删除损坏的文件
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass  # 忽略删除失败的情况
        except Exception as e:
            # 只在目录存在时才记录错误
            if os.path.exists(self.disk_cache_dir):
                try:
                    logger.error(f"Error cleaning up disk cache: {e}")
                except BaseException:
                    pass  # 忽略日志记录失败

    def _get_disk_cache_size(self) -> int:
        """
        获取磁盘缓存大小

        Returns:
            int: 磁盘缓存大小（字节）
        """
        try:
            # 检查磁盘缓存目录是否存在
            if not os.path.exists(self.disk_cache_dir):
                return 0

            total_size = 0
            for filename in os.listdir(self.disk_cache_dir):
                if filename.endswith('.pkl'):
                    filepath = os.path.join(self.disk_cache_dir, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except OSError:
                        continue  # 忽略无法获取大小的文件
            return total_size
        except BaseException:
            return 0

    def _start_cleanup_thread(self):
        """启动清理线程"""

        def cleanup_worker():

            while not self._stop_cleanup:
                try:
                    # 使用可中断的睡眠机制
                    for _ in range(300):  # 每5分钟检查一次，但每秒检查退出标志
                        if self._stop_cleanup:
                            break
                        time.sleep(1)

                    if not self._stop_cleanup:
                        self._cleanup_memory_cache()
                        self._cleanup_disk_cache()
                except Exception as e:
                    try:
                        logger.error(f"Error in cleanup thread: {e}")
                    except BaseException:
                        pass  # 忽略日志记录失败

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计

        Returns:
            Dict[str, Any]: 缓存统计
        """
        with self._lock:
            stats = self.stats.copy()

            # 计算命中率
            total_hits = stats['memory_hits'] + stats['disk_hits']
            total_accesses = total_hits + stats['misses']
            if total_accesses > 0:
                stats['hit_rate'] = total_hits / total_accesses
                stats['memory_hit_rate'] = stats['memory_hits'] / total_accesses
                stats['disk_hit_rate'] = stats['disk_hits'] / total_accesses
            else:
                stats['hit_rate'] = 0.0
                stats['memory_hit_rate'] = 0.0
                stats['disk_hit_rate'] = 0.0

            # 添加当前大小信息（使用测试期望的字段名）
            stats['memory_size'] = self.memory_size
            stats['disk_size'] = self._get_disk_cache_size()
            stats['memory_cache_count'] = len(self.memory_cache)

            return stats

    def clear(self, prefix: str = ""):
        """
        清除缓存

        Args:
            prefix: 要清除的前缀或完整的key，如果为空则清除所有
        """
        with self._lock:
            if prefix:
                keys = list(self.prefix_index.get(prefix, set()))

                if not keys:
                    cache_key = self._generate_cache_key(prefix, "")
                    keys = [cache_key]

                for cache_key in keys:
                    if cache_key in self.memory_cache:
                        item = self.memory_cache.pop(cache_key)
                        self.memory_size -= item['size']

                    cache_file = os.path.join(self.disk_cache_dir, f"{cache_key}.pkl")
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                        except OSError:
                            pass

                self.prefix_index.pop(prefix, None)
            else:
                # 清除所有缓存
                self.memory_cache.clear()
                self.memory_size = 0
                self.prefix_index.clear()

                for filename in os.listdir(self.disk_cache_dir):
                    if filename.endswith('.pkl'):
                        filepath = os.path.join(self.disk_cache_dir, filename)
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass

    def shutdown(self):
        """关闭缓存管理器"""
        logger.info("正在关闭增强版缓存管理器...")

        # 设置停止标志
        self._stop_cleanup = True

        # 等待主清理线程结束
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
            if self._cleanup_thread.is_alive():
                logger.warning("增强版缓存管理器清理线程未能及时停止")

        # 清空缓存统计
        with self._lock:
            self.stats = {
                'memory_hits': 0,
                'disk_hits': 0,
                'misses': 0,
                'memory_size': 0,
                'disk_size': 0,
                'total_operations': 0
            }

        logger.info("增强版缓存管理器已关闭")

    def __del__(self):
        """析构函数，确保资源清理"""
        try:
            self.shutdown()
        except BaseException:
            pass  # 忽略析构时的错误
