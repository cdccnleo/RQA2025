"""
cache_config_processor 模块

提供 cache_config_processor 相关功能和接口。
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

# 延迟导入以避免循环依赖
# from .multi_level_cache import MultiLevelConfig

logger = logging.getLogger(__name__)


@dataclass
class ProcessedCacheConfig:
    """处理后的缓存配置"""
    raw_config: Dict[str, Any]
    ml_config: Any  # MultiLevelConfig
    levels: Dict[str, Dict[str, Any]]
    fallback_strategy: str
    consistency_check: bool


class CacheConfigProcessor:
    """缓存配置处理器"""

    DEFAULT_FALLBACK_STRATEGY = 'graceful'
    DEFAULT_CONSISTENCY_CHECK = True

    @staticmethod
    def process_config(config: Optional[Union[Dict[str, Any], object]]) -> ProcessedCacheConfig:
        """
        处理缓存配置

        Args:
            config: 原始配置（字典或对象）

        Returns:
            ProcessedCacheConfig: 处理后的配置
        """
        if isinstance(config, dict):
            return CacheConfigProcessor._process_dict_config(config)
        else:
            return CacheConfigProcessor._process_object_config(config)

    @staticmethod
    def _process_dict_config(config: Dict[str, Any]) -> ProcessedCacheConfig:
        """处理字典配置"""
        # 提取levels配置
        levels = dict(config.get('levels', {}))

        # 提取其他配置项
        fallback_strategy = config.get(
            'fallback_strategy', CacheConfigProcessor.DEFAULT_FALLBACK_STRATEGY)
        consistency_check = config.get(
            'consistency_check', CacheConfigProcessor.DEFAULT_CONSISTENCY_CHECK)

        # 创建内部配置对象（简化版）
        ml_config = CacheConfigProcessor._create_ml_config_from_dict(config)

        return ProcessedCacheConfig(
            raw_config=config,
            ml_config=ml_config,
            levels=levels,
            fallback_strategy=fallback_strategy,
            consistency_check=consistency_check
        )

    @staticmethod
    def _process_object_config(config: Optional[object]) -> ProcessedCacheConfig:
        """处理对象配置"""
        # 默认配置
        default_levels = CacheConfigProcessor._create_default_levels()

        if config is None:
            # 创建默认配置对象
            ml_config = CacheConfigProcessor._create_default_ml_config()
        else:
            ml_config = config

        return ProcessedCacheConfig(
            raw_config={},
            ml_config=ml_config,
            levels=default_levels,
            fallback_strategy=CacheConfigProcessor.DEFAULT_FALLBACK_STRATEGY,
            consistency_check=CacheConfigProcessor.DEFAULT_CONSISTENCY_CHECK
        )

    @staticmethod
    def _create_ml_config_from_dict(config: Dict[str, Any]) -> Any:
        """从字典创建MultiLevelConfig"""
        try:
            # 动态导入以避免循环依赖
            from .multi_level_cache import MultiLevelConfig, MemoryConfig, RedisConfig, DiskConfig

            levels = config.get('levels', {})

            # 创建配置对象
            ml_config = MultiLevelConfig()

            # 设置L1配置
            if 'L1' in levels:
                l1_cfg = levels['L1']
                ml_config.l1_config = MemoryConfig(
                    capacity=l1_cfg.get('max_size', 100),
                    ttl=l1_cfg.get('ttl', 60)
                )

            # 设置L2配置
            if 'L2' in levels:
                l2_cfg = levels['L2']
                ml_config.l2_config = RedisConfig(
                    capacity=l2_cfg.get('max_size', 1000),
                    ttl=l2_cfg.get('ttl', 300),
                    host=l2_cfg.get('host', 'localhost'),
                    port=l2_cfg.get('port', 6379)
                )

            # 设置L3配置
            if 'L3' in levels:
                l3_cfg = levels['L3']
                ml_config.l3_config = DiskConfig(
                    capacity=l3_cfg.get('max_size', 10000),
                    ttl=l3_cfg.get('ttl', 3600),
                    cache_dir=l3_cfg.get('cache_dir', './cache')
                )

            return ml_config

        except ImportError:
            # 如果导入失败，返回默认配置
            return CacheConfigProcessor._create_default_ml_config()

    @staticmethod
    def _create_default_ml_config() -> Any:
        """创建默认MultiLevelConfig"""
        try:
            from .multi_level_cache import MultiLevelConfig
            return MultiLevelConfig()
        except ImportError:
            # 返回一个简化的默认配置
            return None

    @staticmethod
    def _create_default_levels() -> Dict[str, Dict[str, Any]]:
        """创建默认levels配置"""
        return {
            'L1': {
                'type': 'memory',
                'max_size': 100,
                'ttl': 60
            }
        }
