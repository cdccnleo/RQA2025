"""
缓存工具类

提供缓存相关的工具函数和辅助类
"""

class CacheUtils:
    """缓存工具类"""

    @staticmethod
    def generate_cache_key(*args, **kwargs) -> str:
        """生成缓存键"""
        return "_".join(str(arg) for arg in args)

    @staticmethod
    def is_cacheable(value) -> bool:
        """检查值是否可以缓存"""
        return True

    @staticmethod
    def calculate_hash(data) -> str:
        """计算数据哈希"""
        return str(hash(str(data)))
