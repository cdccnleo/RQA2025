"""
数据缓存类
提供数据缓存功能
"""

from typing import Any, Dict, Optional, Callable
import logging
import pandas as pd
from .cache_manager import CacheManager


class DataCache:

    """数据缓存类"""

    def __init__(self, cache_dir: str = "data_cache"):
        """
        初始化数据缓存

        Args:
            cache_dir: 缓存目录
        """
        from .cache_manager import CacheConfig

        config = CacheConfig(disk_cache_dir=cache_dir)
        self.cache_manager = CacheManager(config)
        self.logger = logging.getLogger(__name__)

    def get_or_compute(self,


                       key: str,
                       compute_func: Callable,
                       *args,
                       **kwargs) -> Any:
        """
        获取缓存数据或计算新数据

        Args:
            key: 缓存键
            compute_func: 计算函数
            *args: 计算函数参数
            **kwargs: 计算函数关键字参数

        Returns:
            数据
        """
        # 尝试从缓存获取
        cached_data = self.get(key)

        if cached_data is not None:
            return cached_data

        # 计算新数据
        data = compute_func(*args, **kwargs)

        # 缓存数据
        self.set(key, data)

        return data

    def get_dataframe(self, key: str) -> Optional[pd.DataFrame]:
        """
        获取DataFrame缓存

        Args:
            key: 缓存键

        Returns:
            DataFrame或None
        """
        try:
            return self.cache_manager.get(key)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to get dataframe cache for %s: %s", key, exc)
            return None

    def set_dataframe(self, key: str, df: pd.DataFrame) -> bool:
        """
        设置DataFrame缓存

        Args:
            key: 缓存键
            df: DataFrame

        Returns:
            是否成功
        """
        try:
            return self.cache_manager.set(key, df)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to set dataframe cache for %s: %s", key, exc)
            return False

    def get_dict(self, key: str) -> Optional[Dict]:
        """
        获取字典缓存

        Args:
            key: 缓存键

        Returns:
            字典或None
        """
        try:
            return self.cache_manager.get(key)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to get dict cache for %s: %s", key, exc)
            return None

    def set_dict(self, key: str, data: Dict) -> bool:
        """
        设置字典缓存

        Args:
            key: 缓存键
            data: 字典数据

        Returns:
            是否成功
        """
        try:
            return self.cache_manager.set(key, data)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to set dict cache for %s: %s", key, exc)
            return False

    def clear(self) -> bool:
        """
        清空缓存

        Returns:
            是否成功
        """
        try:
            return self.cache_manager.clear()
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to clear cache: %s", exc)
            return False

    def exists(self, key: str) -> bool:
        """
        检查缓存是否存在

        Args:
            key: 缓存键

        Returns:
            是否存在
        """
        try:
            return self.cache_manager.exists(key)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to check cache existence for %s: %s", key, exc)
            return False

    def delete(self, key: str) -> bool:
        """
        删除缓存

        Args:
            key: 缓存键

        Returns:
            是否成功
        """
        try:
            return self.cache_manager.delete(key)
        except Exception as exc:  # pragma: no cover - 防御性
            self.logger.warning("Failed to delete cache for %s: %s", key, exc)
            return False

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存数据

        Args:
            key: 缓存键

        Returns:
            缓存数据或None
        """
        try:
            return self.cache_manager.get(key)
        except Exception as exc:
            self.logger.warning("Failed to get cache for %s: %s", key, exc)
            return None

    def set(self, key: str, value: Any) -> bool:
        """
        设置缓存数据

        Args:
            key: 缓存键
            value: 缓存值

        Returns:
            是否成功
        """
        try:
            return self.cache_manager.set(key, value)
        except Exception as exc:
            self.logger.warning("Failed to set cache for %s: %s", key, exc)
            return False
