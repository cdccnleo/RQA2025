# src/data/china/cache_policy.py
from data.cache.smart_cache import CachePolicy


class ChinaCachePolicy:
    @staticmethod
    def get_policy(data_type: str) -> CachePolicy:
        """获取A股特定缓存策略"""
        policies = {
            'stock_daily': CachePolicy(7200, 172800, 900),  # 日线延长缓存
            'level2': CachePolicy(15, 120, 5),              # Level2较短缓存
            'margin': CachePolicy(3600, 28800, 600),       # 融资融券数据
            'dragon_board': CachePolicy(1800, 14400, 300)  # 龙虎榜数据
        }
        return policies.get(data_type, CachePolicy(3600, 86400, 600))