#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的缓存策略模块

实现多种缓存淘汰策略，包括LRU、LFU、FIFO等，并支持策略的动态切换。
"""

import time
from typing import Optional, Dict, Any, Callable
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict

from .cache_manager import CacheEntry, CacheConfig


class CacheStrategy(ABC):
    """缓存策略抽象基类"""

    @abstractmethod
    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        pass

    @abstractmethod
    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        pass

    @abstractmethod
    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass


class LRUStrategy(CacheStrategy):
    """LRU (Least Recently Used) 策略"""

    def __init__(self):
        self._access_order = OrderedDict()  # 记录访问顺序

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        # 更新访问顺序
        self._access_order[key] = time.time()
        # 保持最近访问的在前面
        self._access_order.move_to_end(key, last=False)

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        if key in self._access_order:
            # 更新访问时间和顺序
            self._access_order[key] = time.time()
            self._access_order.move_to_end(key, last=False)

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        if not self._access_order:
            return None

        # 删除最久未使用的
        for key in reversed(list(self._access_order.keys())):
            if key in cache:
                del self._access_order[key]
                return key

        return None

    def get_name(self) -> str:
        """获取策略名称"""
        return "LRU"


class LFUStrategy(CacheStrategy):
    """LFU (Least Frequently Used) 策略"""

    def __init__(self):
        self._frequency = defaultdict(int)  # 记录访问频率
        self._frequency_order = defaultdict(OrderedDict)  # 按频率分组的访问顺序
        self._min_frequency = 1

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        # 新条目频率为1
        self._frequency[key] = 1
        self._frequency_order[1][key] = time.time()
        self._min_frequency = 1

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        if key in self._frequency:
            # 增加频率
            old_freq = self._frequency[key]
            new_freq = old_freq + 1

            # 从旧频率组移除
            if key in self._frequency_order[old_freq]:
                del self._frequency_order[old_freq][key]
                # 如果旧频率组为空，且是最小频率，更新最小频率
                if not self._frequency_order[old_freq] and old_freq == self._min_frequency:
                    self._min_frequency = new_freq

            # 添加到新频率组
            self._frequency[key] = new_freq
            self._frequency_order[new_freq][key] = time.time()

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        # 从最小频率组中删除最旧的
        if self._min_frequency in self._frequency_order:
            for key in self._frequency_order[self._min_frequency]:
                if key in cache:
                    del self._frequency_order[self._min_frequency][key]
                    if key in self._frequency:
                        del self._frequency[key]
                    # 如果最小频率组为空，增加最小频率
                    if not self._frequency_order[self._min_frequency]:
                        self._min_frequency += 1
                    return key

        return None

    def get_name(self) -> str:
        """获取策略名称"""
        return "LFU"


class FIFOStrategy(CacheStrategy):
    """FIFO (First In First Out) 策略"""

    def __init__(self):
        self._insert_order = []  # 记录插入顺序

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        # 添加到插入顺序
        if key not in self._insert_order:
            self._insert_order.append(key)

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        # FIFO策略不关心访问顺序
        pass

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        # 删除最早插入的
        for key in self._insert_order:
            if key in cache:
                self._insert_order.remove(key)
                return key

        return None

    def get_name(self) -> str:
        """获取策略名称"""
        return "FIFO"


class TTLSstrategy(CacheStrategy):
    """TTL (Time To Live) 策略"""

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        # TTL策略主要依赖于过期时间
        pass

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        # TTL策略不关心访问顺序
        pass

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        # 优先淘汰过期的条目
        current_time = time.time()
        for key, entry in cache.items():
            if entry.is_expired():
                return key

        # 如果没有过期条目，淘汰最早创建的
        oldest_key = None
        oldest_time = float('inf')

        for key, entry in cache.items():
            if entry.created_at < oldest_time:
                oldest_time = entry.created_at
                oldest_key = key

        return oldest_key

    def get_name(self) -> str:
        """获取策略名称"""
        return "TTL"


class RandomStrategy(CacheStrategy):
    """随机淘汰策略"""

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        pass

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        pass

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        import random
        if cache:
            return random.choice(list(cache.keys()))
        return None

    def get_name(self) -> str:
        """获取策略名称"""
        return "Random"


class HybridStrategy(CacheStrategy):
    """混合策略"""

    def __init__(self, primary_strategy: CacheStrategy, secondary_strategy: CacheStrategy, 
                 switch_condition: Optional[Callable[[Dict[str, CacheEntry], CacheConfig], bool]] = None):
        """
        初始化混合策略

        Args:
            primary_strategy: 主要策略
            secondary_strategy: 次要策略
            switch_condition: 切换策略的条件函数
        """
        self.primary_strategy = primary_strategy
        self.secondary_strategy = secondary_strategy
        self.switch_condition = switch_condition or (lambda cache, config: len(cache) > config.max_size * 0.8)
        self.current_strategy = primary_strategy

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        # 检查是否需要切换策略
        if self.switch_condition(cache, config):
            self.current_strategy = self.secondary_strategy
        else:
            self.current_strategy = self.primary_strategy

        # 调用当前策略的on_set方法
        self.current_strategy.on_set(cache, key, entry, config)

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        self.current_strategy.on_get(cache, key, entry, config)

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        return self.current_strategy.on_evict(cache, config)

    def get_name(self) -> str:
        """获取策略名称"""
        return f"Hybrid({self.current_strategy.get_name()})"


class AdaptiveStrategy(CacheStrategy):
    """自适应策略"""

    def __init__(self):
        self.strategies = {
            'lru': LRUStrategy(),
            'lfu': LFUStrategy(),
            'fifo': FIFOStrategy(),
            'ttl': TTLSstrategy(),
            'random': RandomStrategy()
        }
        self.current_strategy = self.strategies['lru']
        self.strategy_metrics = defaultdict(lambda: {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'hit_rate': 0.0,
            'last_evaluation': time.time()
        })

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig) -> None:
        """设置缓存时的策略钩子"""
        self.current_strategy.on_set(cache, key, entry, config)

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig) -> None:
        """获取缓存时的策略钩子"""
        # 记录命中率
        if entry:
            self.strategy_metrics[self.current_strategy.get_name()]['hits'] += 1
        else:
            self.strategy_metrics[self.current_strategy.get_name()]['misses'] += 1

        # 计算命中率
        metrics = self.strategy_metrics[self.current_strategy.get_name()]
        total = metrics['hits'] + metrics['misses']
        if total > 0:
            metrics['hit_rate'] = metrics['hits'] / total

        # 每100次访问评估一次策略
        if total % 100 == 0:
            self._evaluate_strategies()

        self.current_strategy.on_get(cache, key, entry, config)

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:
        """需要淘汰时，返回应淘汰的key"""
        key = self.current_strategy.on_evict(cache, config)
        if key:
            self.strategy_metrics[self.current_strategy.get_name()]['evictions'] += 1
        return key

    def _evaluate_strategies(self) -> None:
        """评估并选择最佳策略"""
        best_strategy = self.current_strategy
        best_score = -1

        for name, strategy in self.strategies.items():
            metrics = self.strategy_metrics[name]
            # 计算策略评分（命中率权重最高）
            score = metrics['hit_rate'] * 0.7 + (1 - metrics['evictions'] / 1000) * 0.3
            
            if score > best_score:
                best_score = score
                best_strategy = strategy

        # 如果找到更好的策略，切换
        if best_strategy != self.current_strategy:
            self.current_strategy = best_strategy

    def get_name(self) -> str:
        """获取策略名称"""
        return f"Adaptive({self.current_strategy.get_name()})"


class CacheStrategyFactory:
    """缓存策略工厂"""

    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> CacheStrategy:
        """
        创建缓存策略实例

        Args:
            strategy_name: 策略名称
            **kwargs: 策略参数

        Returns:
            CacheStrategy: 缓存策略实例
        """
        strategy_map = {
            'lru': LRUStrategy,
            'lfu': LFUStrategy,
            'fifo': FIFOStrategy,
            'ttl': TTLSstrategy,
            'random': RandomStrategy,
            'adaptive': AdaptiveStrategy
        }

        if strategy_name in strategy_map:
            return strategy_map[strategy_name](**kwargs)
        elif strategy_name.startswith('hybrid'):
            # 解析混合策略参数
            primary = kwargs.get('primary', 'lru')
            secondary = kwargs.get('secondary', 'lfu')
            primary_strategy = CacheStrategyFactory.create_strategy(primary)
            secondary_strategy = CacheStrategyFactory.create_strategy(secondary)
            return HybridStrategy(primary_strategy, secondary_strategy)
        else:
            raise ValueError(f"Unknown cache strategy: {strategy_name}")


class StrategyManager:
    """策略管理器"""

    def __init__(self):
        self._strategies = {}
        self._default_strategy = 'lru'

    def register_strategy(self, name: str, strategy: CacheStrategy) -> None:
        """注册策略"""
        self._strategies[name] = strategy

    def get_strategy(self, name: str) -> Optional[CacheStrategy]:
        """获取策略"""
        return self._strategies.get(name)

    def get_default_strategy(self) -> CacheStrategy:
        """获取默认策略"""
        return self._strategies.get(self._default_strategy, LRUStrategy())

    def set_default_strategy(self, name: str) -> bool:
        """设置默认策略"""
        if name in self._strategies:
            self._default_strategy = name
            return True
        return False


# 全局策略管理器实例
_strategy_manager = None


def get_strategy_manager() -> StrategyManager:
    """获取全局策略管理器实例"""
    global _strategy_manager
    if _strategy_manager is None:
        _strategy_manager = StrategyManager()
        # 注册默认策略
        _strategy_manager.register_strategy('lru', LRUStrategy())
        _strategy_manager.register_strategy('lfu', LFUStrategy())
        _strategy_manager.register_strategy('fifo', FIFOStrategy())
        _strategy_manager.register_strategy('ttl', TTLSstrategy())
        _strategy_manager.register_strategy('random', RandomStrategy())
        _strategy_manager.register_strategy('adaptive', AdaptiveStrategy())
    return _strategy_manager


def get_cache_strategy(strategy_name: str = 'lru', **kwargs) -> CacheStrategy:
    """获取缓存策略实例"""
    return CacheStrategyFactory.create_strategy(strategy_name, **kwargs)


__all__ = [
    'CacheStrategy',
    'LRUStrategy',
    'LFUStrategy',
    'FIFOStrategy',
    'TTLSstrategy',
    'RandomStrategy',
    'HybridStrategy',
    'AdaptiveStrategy',
    'CacheStrategyFactory',
    'StrategyManager',
    'get_strategy_manager',
    'get_cache_strategy'
]
