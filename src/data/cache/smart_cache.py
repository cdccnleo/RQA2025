# src/data/cache/smart_cache.py
import time
import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class CachePolicy:
    base_ttl: int  # 基础TTL(秒)
    max_ttl: int  # 最大TTL(秒)
    min_ttl: int  # 最小TTL(秒)
    decay_factor: float = 0.9  # 衰减因子


class SmartCacheManager:
    def __init__(self):
        self.access_stats = defaultdict(int)  # 数据访问统计
        self.last_updated = {}  # 最后更新时间

    def get_dynamic_ttl(self, key: str, policy: CachePolicy) -> int:
        """计算动态TTL"""
        now = time.time()
        freq = self._get_access_frequency(key, now)

        # 动态调整公式: TTL = base_ttl * (1 + log(freq + 1))
        dynamic_ttl = int(policy.base_ttl * (1 + math.log1p(freq)))

        # 应用衰减因子(距离上次访问时间越长，TTL越小)
        if key in self.last_updated:
            hours_since_last = (now - self.last_updated[key]) / 3600
            decay = policy.decay_factor ** hours_since_last
            dynamic_ttl = int(dynamic_ttl * decay)

        return min(max(dynamic_ttl, policy.min_ttl), policy.max_ttl)

    def record_access(self, key: str):
        """记录访问"""
        now = time.time()
        self.access_stats[key] += 1
        self.last_updated[key] = now

    def _get_access_frequency(self, key: str, current_time: float) -> float:
        """计算标准化访问频率(次/小时)"""
        if key not in self.access_stats or key not in self.last_updated:
            return 0

        time_elapsed = max(current_time - self.last_updated[key], 1)  # 避免除0
        return self.access_stats[key] / (time_elapsed / 3600)  # 转换为每小时频率