"""
数据加载器基类定义
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd

from .interfaces import IDataModel

logger = logging.getLogger(__name__)


class LoaderError(Exception):
    """数据加载器异常基类"""
    pass


class ConfigError(LoaderError):
    """配置错误"""
    pass


class DataLoadError(LoaderError):
    """数据加载错误"""
    pass


class BaseDataLoader(ABC):
    """
    数据加载器基类，定义了所有数据加载器必须实现的方法
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器

        Args:
            config: 加载器配置信息
        """
        self.config = config
        self._validate_config()

        # 设置缓存目录
        self.cache_dir = Path(config.get('cache_dir', 'cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 设置重试次数
        self.max_retries = config.get('max_retries', 3)

        # 初始化元数据
        self.metadata = {
            'loader_type': self.__class__.__name__,
            'config_timestamp': datetime.now().isoformat()
        }

    @abstractmethod
    def load(self, start_date: str, end_date: str, frequency: str) -> IDataModel:
        """
        统一的数据加载接口

        Args:
            start_date: 开始日期，格式：YYYY-MM-DD
            end_date: 结束日期，格式：YYYY-MM-DD
            frequency: 数据频率，如 "1d", "1h", "5min"

        Returns:
            IDataModel: 加载的数据模型对象

        Raises:
            DataLoadError: 数据加载失败时抛出
        """
        pass

    def _validate_config(self) -> bool:
        """
        验证配置有效性

        Returns:
            bool: 配置是否有效

        Raises:
            ConfigError: 配置无效时抛出
        """
        required_fields = self.get_required_config_fields()
        for field in required_fields:
            if field not in self.config:
                raise ConfigError(f"Missing required config field: {field}")
        return True

    @abstractmethod
    def get_required_config_fields(self) -> list:
        """
        获取必需的配置字段列表

        Returns:
            list: 必需的配置字段名列表
        """
        return []

    def _get_cache_path(self, key: str) -> Path:
        """
        获取缓存文件路径

        Args:
            key: 缓存键名

        Returns:
            Path: 缓存文件路径
        """
        return self.cache_dir / f"{self.__class__.__name__}_{key}.parquet"

    def _load_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """
        从缓存加载数据

        Args:
            key: 缓存键名

        Returns:
            Optional[pd.DataFrame]: 缓存的数据，如果缓存不存在则返回None
        """
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return None
        return None

    def _save_to_cache(self, key: str, data: pd.DataFrame) -> bool:
        """
        保存数据到缓存

        Args:
            key: 缓存键名
            data: 要缓存的数据

        Returns:
            bool: 是否成功保存到缓存
        """
        try:
            cache_path = self._get_cache_path(key)
            data.to_parquet(cache_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
            return False

    def _generate_cache_key(self, **kwargs) -> str:
        """
        生成缓存键名

        Args:
            **kwargs: 用于生成缓存键的参数

        Returns:
            str: 缓存键名
        """
        # 将所有参数值转换为字符串并排序，确保相同参数生成相同的键名
        sorted_items = sorted(kwargs.items())
        return "_".join(f"{k}_{v}" for k, v in sorted_items)

    def update_metadata(self, **kwargs) -> None:
        """
        更新元数据信息

        Args:
            **kwargs: 要更新的元数据字段
        """
        self.metadata.update(kwargs)
        self.metadata['last_updated'] = datetime.now().isoformat()
