# src/data/cache/data_cache.py
from typing import Any

from .smart_cache import SmartCacheManager, CachePolicy


class DataCache:
    def __init__(self):
        self.smart_cache = SmartCacheManager()
        self.cache_policies = {
            'stock_daily': CachePolicy(3600, 86400, 600),  # 日线数据
            'level1': CachePolicy(60, 300, 10),  # Level1行情
            'level2': CachePolicy(10, 60, 5),  # Level2行情
            'financial': CachePolicy(86400, 604800, 3600)  # 财务数据
        }

    def get(self, key: str, data_type: str):
        """获取缓存数据"""
        data = self._get_from_cache(key)
        if data:
            self.smart_cache.record_access(key)
        return data

    def set(self, key: str, value: Any, data_type: str):
        """设置缓存数据"""
        policy = self.cache_policies.get(data_type, CachePolicy(3600, 86400, 600))
        ttl = self.smart_cache.get_dynamic_ttl(key, policy)
        self._set_to_cache(key, value, ttl)