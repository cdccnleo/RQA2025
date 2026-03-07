#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
磁盘缓存模块
提供持久化缓存功能，支持数据序列化和反序列化
"""

# 日志降级处理
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Dict, List, Tuple
import threading
import time
import hashlib
import pickle
import os

from .cache_manager import CacheEntry, CacheStats


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


logger = get_data_logger('disk_cache')


# logger 已在上面定义


@dataclass
class DiskCacheConfig:

    """磁盘缓存配置"""
    cache_dir: str = "cache"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    compression: bool = False
    encryption: bool = False
    encryption_key: Optional[str] = None
    backup_enabled: bool = False
    backup_interval: int = 3600  # 1小时备份一次
    cleanup_interval: int = 300  # 5分钟清理一次


def _safe_logger_log(log_instance: Any, level: int, message: str) -> None:
    """安全地写日志，移除已关闭的handler并确保至少有一个有效handler。"""
    try:
        handlers = list(getattr(log_instance, "handlers", []))
        for handler in handlers:
            stream = getattr(handler, "stream", None)
            if stream is not None and getattr(stream, "closed", False):
                try:
                    log_instance.removeHandler(handler)
                except Exception:
                    pass
        if not getattr(log_instance, "handlers", []):
            fallback_handler = logging.StreamHandler()
            fallback_handler.setLevel(logging.INFO)
            log_instance.addHandler(fallback_handler)
        log_instance.log(level, message)
    except Exception:
        try:
            logging.getLogger(getattr(log_instance, "name", "disk_cache")).log(level, message)
        except Exception:
            pass


class DiskCache:

    """
    磁盘缓存实现
    提供持久化缓存功能，支持数据序列化和反序列化
    """

    def __init__(self, config: DiskCacheConfig):
        """
        初始化磁盘缓存

        Args:
            config: 磁盘缓存配置
        """
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.stats = CacheStats()
        self._lock = threading.RLock()
        self.logger = logger

        # 启动清理线程
        self._cleanup_thread = None
        self._stop_cleanup = False
        self._start_cleanup_thread()

        _safe_logger_log(self.logger, logging.INFO, f"DiskCache initialized with cache_dir: {self.cache_dir}")

    def stop(self):
        """停止磁盘缓存，清理所有资源"""
        _safe_logger_log(self.logger, logging.INFO, "正在停止磁盘缓存...")

        # 设置停止标志
        self._stop_cleanup = True

        # 等待清理线程结束
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
            if self._cleanup_thread.is_alive():
                _safe_logger_log(self.logger, logging.WARNING, "磁盘缓存清理线程未能及时停止")

        _safe_logger_log(self.logger, logging.INFO, "磁盘缓存已停止")

    def __del__(self):
        """析构函数，确保资源被清理"""
        try:
            self.stop()
        except Exception:
            pass  # 忽略析构时的异常

    def _get_file_path(self, key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键

        Returns:
            Path: 文件路径
        """
        # 使用MD5哈希避免文件名冲突
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"

    def _serialize_entry(self, entry: CacheEntry) -> bytes:
        """
        序列化缓存条目

        Args:
            entry: 缓存条目

        Returns:
            bytes: 序列化后的数据
        """
        data = entry.to_dict()
        return pickle.dumps(data)

    def _deserialize_entry(self, data: bytes) -> Optional[CacheEntry]:
        """
        反序列化缓存条目

        Args:
            data: 序列化数据

        Returns:
            Optional[CacheEntry]: 缓存条目，如果反序列化失败则返回None
        """
        try:
            data_dict = pickle.loads(data)
            if not isinstance(data_dict, dict):
                raise TypeError(f"Deserialized cache entry type invalid: {type(data_dict)}")
            return CacheEntry.from_dict(data_dict)
        except (pickle.UnpicklingError, KeyError, TypeError, ValueError) as e:
            _safe_logger_log(self.logger, logging.ERROR, f"Failed to deserialize cache entry: {e}")
            return None
        except Exception as e:
            _safe_logger_log(self.logger, logging.ERROR, f"Unexpected error during deserialization: {e}")
            return None

    def get_entry(self, key: str, *, update_metadata: bool = True) -> Optional[CacheEntry]:
        """
        获取缓存条目，必要时可跳过元数据更新，供上层自定义接管。

        Args:
            key: 缓存键
            update_metadata: 是否更新访问统计与写回磁盘

        Returns:
            Optional[CacheEntry]: 缓存条目
        """
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                if not file_path.exists():
                    self.stats.miss()
                    return None

                # 检查文件大小
                if file_path.stat().st_size > self.config.max_file_size:
                    _safe_logger_log(self.logger, logging.WARNING, f"Cache file too large: {file_path}")
                    self.delete(key)  # 删除过大的文件
                    self.stats.error()
                    return None

                # 读取文件
                try:
                    with open(file_path, 'rb') as f:
                        data = f.read()
                except (IOError, OSError) as e:
                    _safe_logger_log(self.logger, logging.ERROR, f"Failed to read cache file {file_path}: {e}")
                    self.delete(key)  # 删除无法读取的文件
                    self.stats.error()
                    return None

                entry = self._deserialize_entry(data)
                if entry is None:
                    # 反序列化失败，删除损坏的缓存文件
                    _safe_logger_log(self.logger, logging.WARNING, f"Deleting corrupted cache file for key: {key}")
                    self.delete(key)
                    self.stats.error()
                    return None

                # 检查是否过期
                if entry.is_expired():
                    self.delete(key)
                    self.stats.miss()
                    return None

                # 更新访问统计
                if update_metadata:
                    entry.access()
                    self._save_to_disk(key, entry)
                self.stats.hit()

                return entry

            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error getting cache entry for key {key}: {e}")
                self.stats.error()
                return None

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            Optional[Any]: 缓存数据，如果不存在或过期则返回None
        """
        entry = self.get_entry(key, update_metadata=True)
        return entry.value if entry else None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间（秒）

        Returns:
            bool: 是否设置成功
        """
        with self._lock:
            try:
                entry = CacheEntry(key=key, value=value, ttl=ttl)
                return self._save_to_disk(key, entry)
            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error setting cache entry for key {key}: {e}")
                self.stats.error()
                return False

    def _save_to_disk(self, key: str, entry: CacheEntry) -> bool:
        """
        保存到磁盘

        Args:
            key: 缓存键
            entry: 缓存条目

        Returns:
            bool: 是否保存成功
        """
        try:
            file_path = self._get_file_path(key)

            # 序列化数据
            data = self._serialize_entry(entry)

            # 写入文件
            with open(file_path, 'wb') as f:
                f.write(data)

            self.stats.set()
            return True

        except Exception as e:
            _safe_logger_log(self.logger, logging.ERROR, f"Error saving to disk for key {key}: {e}")
            self.stats.error()
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存数据

        Args:
            key: 缓存键

        Returns:
            bool: 是否删除成功
        """
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                if file_path.exists():
                    file_path.unlink()
                    self.stats.delete()
                    return True
                return False
            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error deleting cache entry for key {key}: {e}")
                self.stats.error()
                return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            bool: 是否存在
        """
        with self._lock:
            try:
                file_path = self._get_file_path(key)
                if not file_path.exists():
                    return False

                # 检查是否过期
                entry = self.get(key)
                return entry is not None

            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error checking existence for key {key}: {e}")
                return False

    def clear(self) -> bool:
        """
        清空所有缓存

        Returns:
            bool: 是否清空成功
        """
        with self._lock:
            try:
                for file_path in self.cache_dir.glob("*.cache"):
                    file_path.unlink()
                _safe_logger_log(self.logger, logging.INFO, "Disk cache cleared")
                return True
            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error clearing disk cache: {e}")
                return False

    def list_keys(self) -> List[str]:
        """
        列出所有缓存键

        Returns:
            List[str]: 缓存键列表
        """
        with self._lock:
            try:
                keys = []
                for file_path in self.cache_dir.glob("*.cache"):
                    # 这里无法从文件名反推原始键，返回文件路径
                    keys.append(str(file_path))
                return keys
            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error listing cache keys: {e}")
                return []

    def _collect_disk_usage(self) -> Tuple[int, int]:
        """计算磁盘缓存中文件数量和总大小，容忍中途被删除的文件"""
        file_count = 0
        total_size = 0
        for file_path in self.cache_dir.glob("*.cache"):
            try:
                total_size += file_path.stat().st_size
                file_count += 1
            except FileNotFoundError:
                # 文件可能在统计过程中被清理，直接忽略
                continue
            except OSError as exc:
                _safe_logger_log(self.logger, logging.DEBUG, f"Failed to stat cache file {file_path}: {exc}")
        return file_count, total_size

    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        with self._lock:
            stats = self.stats.get_stats()
            file_count, total_size = self._collect_disk_usage()
            stats.update({
                'disk_cache_enabled': True,
                'disk_cache': {
                    'cache_dir': str(self.cache_dir),
                    'file_count': file_count,
                    'total_size': total_size,
                    'max_file_size': self.config.max_file_size,
                    'compression': self.config.compression,
                    'encryption': self.config.encryption,
                    'backup_enabled': self.config.backup_enabled
                }
            })
            return stats

    def health_check(self) -> Dict[str, Any]:
        """
        健康检查

        Returns:
            Dict[str, Any]: 健康状态
        """
        with self._lock:
            try:
                # 检查目录是否存在
                if not self.cache_dir.exists():
                    return {'status': 'error', 'message': 'Cache directory does not exist'}

                # 检查目录权限
                if not os.access(self.cache_dir, os.W_OK):
                    return {'status': 'error', 'message': 'No write permission to cache directory'}

                # 检查文件数量
                file_count, total_size = self._collect_disk_usage()

                return {
                    'status': 'healthy',
                    'cache_dir': str(self.cache_dir),
                    'file_count': file_count,
                    'total_size': total_size,
                    'stats': self.stats.get_stats()
                }

            except Exception as e:
                return {'status': 'error', 'message': str(e)}

    def _start_cleanup_thread(self):
        """启动清理线程"""
        if self.config.cleanup_interval <= 0:
            return

        def cleanup_worker():

            while not self._stop_cleanup:
                try:
                    self._cleanup_expired()
                    # 使用可中断的睡眠机制
                    for _ in range(self.config.cleanup_interval):
                        if self._stop_cleanup:
                            break
                        time.sleep(1)
                except Exception as e:
                    _safe_logger_log(self.logger, logging.ERROR, f"Error in cleanup worker: {e}")
                    # 出错后使用可中断的睡眠
                    for _ in range(60):
                        if self._stop_cleanup:
                            break
                        time.sleep(1)

        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """清理过期缓存"""
        with self._lock:
            try:
                expired_count = 0
                for file_path in self.cache_dir.glob("*.cache"):
                    try:
                        with open(file_path, 'rb') as f:
                            data = f.read()

                        entry = self._deserialize_entry(data)
                        if entry and entry.is_expired():
                            file_path.unlink()
                            expired_count += 1
                        elif entry is None:
                            file_path.unlink()
                            expired_count += 1

                    except Exception as e:
                        _safe_logger_log(self.logger, logging.WARNING, f"Error processing cache file {file_path}: {e}")
                        # 删除损坏的文件
                        file_path.unlink()
                        expired_count += 1

                if expired_count > 0:
                    _safe_logger_log(self.logger, logging.INFO, f"Cleaned up {expired_count} expired cache files")

            except Exception as e:
                _safe_logger_log(self.logger, logging.ERROR, f"Error in cleanup: {e}")

    def close(self):
        """关闭磁盘缓存"""
        self._stop_cleanup = True
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        _safe_logger_log(self.logger, logging.INFO, "DiskCache closed")
