# src / data / china / cache_policy.py


class CachePolicy:

    """缓存策略类"""

    def __init__(self, ttl: int, max_ttl: int, refresh_interval: int):

        self.ttl = ttl  # 生存时间
        self.max_ttl = max_ttl  # 最大生存时间
        self.refresh_interval = refresh_interval  # 刷新间隔


class ChinaCachePolicy:

    """中国市场数据缓存策略"""

    @staticmethod
    def get_policy(data_type: str):
        """获取指定数据类型的缓存策略

        Args:
            data_type: 数据类型 ('stock_daily', 'level2', 'margin', 'dragon_board')

        Returns:
            CachePolicy: 缓存策略对象
        """
        policies = {
            'stock_daily': CachePolicy(7200, 172800, 900),  # 日线延长缓存
            'level2': CachePolicy(15, 120, 5),              # Level2较短缓存
            'margin': CachePolicy(3600, 28800, 600),       # 融资融券数据
            'dragon_board': CachePolicy(1800, 14400, 300)  # 龙虎榜数据
        }
        return policies.get(data_type, CachePolicy(3600, 86400, 600))
