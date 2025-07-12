import functools
import hashlib
import pickle
import time
from typing import Callable, Any, Optional
import pandas as pd

class PredictionCache:
    """
    模型预测缓存装饰器

    特性：
    - 基于特征哈希的缓存键
    - 可配置的缓存过期时间
    - 缓存命中率监控
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """生成基于特征数据的缓存键"""
        data = kwargs.get('features') if 'features' in kwargs else args[0]
        if isinstance(data, pd.DataFrame):
            data = data.to_dict()
        return hashlib.md5(pickle.dumps(data)).hexdigest()

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = self._make_key(*args, **kwargs)

            # 检查缓存
            if key in self.cache:
                entry = self.cache[key]
                if time.time() - entry['timestamp'] < self.ttl_seconds:
                    self.hits += 1
                    return entry['prediction']

            # 缓存未命中，调用原始函数
            result = func(*args, **kwargs)
            self.misses += 1

            # 更新缓存
            if len(self.cache) >= self.max_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = {
                'prediction': result,
                'timestamp': time.time()
            }

            return result

        return wrapper

    @property
    def hit_rate(self) -> float:
        """计算缓存命中率"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

def model_cache(max_size: int = 1000, ttl_seconds: int = 3600) -> Callable:
    """
    模型预测缓存装饰器工厂函数

    Args:
        max_size: 最大缓存条目数
        ttl_seconds: 缓存有效期(秒)
    """
    cache = PredictionCache(max_size, ttl_seconds)
    return cache
