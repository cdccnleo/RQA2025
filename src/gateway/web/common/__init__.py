"""
网关Web层通用模块
包含缓存配置等通用组件
"""

from .cache_config import CacheConfig, ARCHITECTURE_STATUS_TTL, DATA_QUALITY_METRICS_TTL

__all__ = ['CacheConfig', 'ARCHITECTURE_STATUS_TTL', 'DATA_QUALITY_METRICS_TTL']

