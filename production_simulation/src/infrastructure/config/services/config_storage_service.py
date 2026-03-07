
from typing import Any, Dict, Optional, List
from ..core.imports import (
    logging, threading, time
)
from .iconfig_service import (
    IConfigStorageService, BaseConfigService
)
from ..storage.types.iconfigstorage import IConfigStorage
#!/usr/bin/env python3
"""
配置存储服务

Phase 1重构：使用组合模式替代多重继承
提供配置的存储、加载、缓存功能
"""

logger = logging.getLogger(__name__)


class ConfigStorageService(BaseConfigService, IConfigStorageService):
    """配置存储服务

    使用组合模式替代继承，提供配置的存储、加载、缓存功能
    """

    def __init__(self, storage_backend: Optional[IConfigStorage] = None,
                 cache_enabled: bool = True, cache_size: int = 1000):
        """初始化存储服务

        Args:
            storage_backend: 存储后端实现
            cache_enabled: 是否启用缓存
            cache_size: 缓存大小
        """
        super().__init__("config_storage_service")

        # 存储后端
        self._storage_backend = storage_backend

        # 缓存配置
        self._cache_enabled = cache_enabled
        self._cache_size = cache_size

        # 缓存数据
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._cache_access_times: Dict[str, float] = {}

        # 统计信息 - 添加total_operations和cache_size
        self._stats = {
            'loads': 0,
            'saves': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_operations': 0,
            'cache_size': 0  # 当前缓存大小
        }

        # 初始化锁
        self._lock = threading.RLock()

    def _initialize(self):
        """初始化服务"""
        self._start_time = time.time()

        # 如果启用缓存但没有缓存服务，则创建简单的内存缓存
        if self._cache_enabled and not hasattr(self, '_cache_service'):
            self._cache_service = self._create_memory_cache()

        logger.info(f"配置存储服务已初始化，缓存: {'启用' if self._cache_enabled else '禁用'}")

    def _create_memory_cache(self) -> Dict[str, Any]:
        """创建简单的内存缓存"""
        return {}

    def set_storage_backend(self, storage: IConfigStorage):
        """设置存储后端

        Args:
            storage: 存储后端实现
        """
        with self._lock:
            self._ensure_initialized()
            self._storage_backend = storage
            logger.info(f"存储后端已设置为: {storage.__class__.__name__}")

    def load(self, source: str) -> Dict[str, Any]:
        """从指定源加载配置

        Args:
            source: 配置源（文件路径、URL等）

        Returns:
            配置字典
        """
        self._ensure_initialized()

        try:
            # 检查缓存
            if self._cache_enabled and source in self._cache:
                cache_time = self._cache_timestamps.get(source, 0)
                # 简单的缓存过期检查（5分钟）
                if time.time() - cache_time < 300:
                    self._stats['cache_hits'] += 1
                    self._update_cache_access(source)
                    logger.debug(f"缓存命中: {source}")
                    return self._cache[source].copy()

            # 从存储后端加载
            if not self._storage_backend:
                raise ValueError("未设置存储后端")

            config = self._storage_backend.load(source)
            self._stats['loads'] += 1

            # 更新缓存
            if self._cache_enabled:
                self._update_cache(source, config)

            logger.info(f"配置已加载: {source}")
            return config

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"加载配置失败 {source}: {e}")
            raise

    def save(self, config: Dict[str, Any], target: str) -> bool:
        """保存配置到指定目标

        Args:
            config: 配置字典
            target: 保存目标

        Returns:
            保存是否成功
        """
        self._ensure_initialized()

        try:
            if not self._storage_backend:
                raise ValueError("未设置存储后端")

            success = self._storage_backend.save(config, target)
            if success:
                self._stats['saves'] += 1

                # 更新缓存
                if self._cache_enabled:
                    self._update_cache(target, config.copy())

                logger.info(f"配置已保存: {target}")

            return success

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"保存配置失败 {target}: {e}")
            raise

    def reload(self) -> bool:
        """重新加载配置

        Returns:
            重新加载是否成功
        """
        self._ensure_initialized()

        try:
            with self._lock:
                # 清理缓存
                if self._cache_enabled:
                    self._cache.clear()
                    self._cache_timestamps.clear()
                    self._cache_access_times.clear()

                # 如果有存储后端，重新加载
                if self._storage_backend and hasattr(self._storage_backend, 'reload'):
                    return self._storage_backend.reload()

                logger.info("配置已重新加载")
                return True

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"重新加载配置失败: {e}")
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self._cache_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hit_rate": self._calculate_hit_rate(),
            "entries": list(self._cache.keys())
        }

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if not self._cache_enabled:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": len(self._cache),
            "max_size": self._cache_size,
            "hit_rate": self._calculate_hit_rate(),
            "loads": self._stats['loads'],
            "saves": self._stats['saves'],
            "cache_hits": self._stats['cache_hits'],
            "cache_misses": self._stats['cache_misses'],
            "errors": self._stats['errors'],
            "total_operations": self._stats.get('total_operations', 0),  # 确保键存在
            "cache_size": len(self._cache)  # 当前缓存大小
        }

    def _update_cache(self, key: str, value: Any = None):
        """更新缓存

        Args:
            key: 缓存键
            value: 缓存值，如果为None则删除
        """
        if not self._cache_enabled:
            return

        with self._lock:
            if value is None:
                # 删除缓存
                self._cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
                self._cache_access_times.pop(key, None)
            else:
                # 检查缓存大小限制
                if len(self._cache) >= self._cache_size:
                    self._evict_cache_entry()

                # 添加/更新缓存
                self._cache[key] = value
                self._cache_timestamps[key] = time.time()
                self._update_cache_access(key)

    def _update_cache_access(self, key: str):
        """更新缓存访问时间"""
        self._cache_access_times[key] = time.time()

    def _evict_cache_entry(self):
        """逐出缓存条目（LRU策略）"""
        if not self._cache_access_times:
            return

        # 找到最少使用的条目
        lru_key = min(self._cache_access_times.items(), key=lambda x: x[1])[0]

        # 删除该条目
        self._cache.pop(lru_key, None)
        self._cache_timestamps.pop(lru_key, None)
        self._cache_access_times.pop(lru_key, None)

    def _calculate_hit_rate(self) -> float:
        """计算缓存命中率"""
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        if total_requests == 0:
            return 0.0
        return self._stats['cache_hits'] / total_requests

    def cleanup(self):
        """清理资源"""
        with self._lock:
            if self._cache_enabled:
                self._cache.clear()
                self._cache_timestamps.clear()
                self._cache_access_times.clear()

            logger.info("配置存储服务已清理")

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键
            default: 默认值

        Returns:
            配置值或默认值
        """
        self._ensure_initialized()

        with self._lock:
            # 检查缓存
            if self._cache_enabled and key in self._cache:
                cache_time = self._cache_timestamps.get(key, 0)
                # 简单的缓存过期检查（5分钟）
                if time.time() - cache_time < 300:
                    self._stats['cache_hits'] += 1
                    self._update_cache_access(key)
                    logger.debug(f"缓存命中: {key}")
                    return self._cache[key]

            # 从后端获取
            if self._storage_backend:
                try:
                    value = self._storage_backend.get(key, default)
                    if value is not None:
                        self._update_cache(key, value)
                        self._stats['cache_misses'] += 1
                    return value
                except Exception as e:
                    self._stats['errors'] += 1
                    logger.error(f"从后端获取配置失败 {key}: {e}")
                    raise
            else:
                self._stats['errors'] += 1
                logger.warning(f"未设置存储后端，无法获取 {key}")
                return default

    def set(self, key: str, value: Any) -> bool:
        """设置配置值

        Args:
            key: 配置键
            value: 配置值

        Returns:
            设置是否成功
        """
        self._ensure_initialized()

        with self._lock:
            if not self._storage_backend:
                self._stats['errors'] += 1
                logger.warning("未设置存储后端，无法设置配置")
                return False

            try:
                success = self._storage_backend.set(key, value)
                if success:
                    self._update_cache(key, value)
                    self._stats['saves'] += 1
                else:
                    self._stats['errors'] += 1

                logger.info(f"配置已设置: {key}")
                return success

            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"设置配置失败 {key}: {e}")
                raise

    def delete(self, key: str) -> bool:
        """删除配置项

        Args:
            key: 配置键

        Returns:
            删除是否成功
        """
        self._ensure_initialized()

        with self._lock:
            if not self._storage_backend:
                self._stats['errors'] += 1
                logger.warning("未设置存储后端，无法删除配置")
                return False

            try:
                success = self._storage_backend.delete(key)
                if success:
                    # 移除缓存
                    self._cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)
                    self._cache_access_times.pop(key, None)
                    self._stats['saves'] += 1  # 视为保存操作

                logger.info(f"配置已删除: {key}")
                return success

            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"删除配置失败 {key}: {e}")
                raise

    def exists(self, key: str) -> bool:
        """检查配置项是否存在

        Args:
            key: 配置键

        Returns:
            是否存在
        """
        self._ensure_initialized()

        with self._lock:
            # 先检查缓存
            if self._cache_enabled and key in self._cache:
                self._update_cache_access(key)
                return True

            if not self._storage_backend:
                self._stats['errors'] += 1
                logger.warning("未设置存储后端，无法检查存在性")
                return False

            try:
                exists = self._storage_backend.exists(key)
                return exists

            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"检查存在性失败 {key}: {e}")
                raise

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配模式的配置键

        Args:
            pattern: 匹配模式

        Returns:
            配置键列表
        """
        self._ensure_initialized()

        if not self._storage_backend:
            self._stats['errors'] += 1
            logger.warning("未设置存储后端，无法获取键列表")
            return []

        try:
            keys = self._storage_backend.keys(pattern)
            return keys

        except Exception as e:
            self._stats['errors'] += 1
            logger.error(f"获取键列表失败 {pattern}: {e}")
            raise

    def clear(self) -> bool:
        """清空所有配置

        Returns:
            清空是否成功
        """
        self._ensure_initialized()

        with self._lock:
            if not self._storage_backend:
                self._stats['errors'] += 1
                logger.warning("未设置存储后端，无法清空配置")
                return False

            try:
                success = self._storage_backend.clear()
                if success and self._cache_enabled:
                    self._cache.clear()
                    self._cache_timestamps.clear()
                    self._cache_access_times.clear()

                logger.info("配置已清空")
                return success

            except Exception as e:
                self._stats['errors'] += 1
                logger.error(f"清空配置失败: {e}")
                raise




