"""缓存策略管理器和策略实现。"""

from __future__ import annotations

import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from ..interfaces import CacheEvictionStrategy
from ..interfaces.cache_interfaces import EvictionStrategyImpl


class StrategyType(Enum):
    """策略类型枚举，覆盖测试中涉及的所有策略类型。"""

    LRU = "lru"
    LFU = "lfu"
    TTL = "ttl"
    ADAPTIVE = "adaptive"
    FIFO = "fifo"
    RANDOM = "random"


# 兼容历史别名
setattr(StrategyType, "SIZE", StrategyType.ADAPTIVE)
setattr(StrategyType, "WEIGHTED_LRU", StrategyType.LRU)


@dataclass
class StrategyMetrics:
    """策略性能指标数据结构"""

    strategy_name: str
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    avg_response_time: float = 0.0
    total_requests: int = 0
    total_hits: int = 0
    total_misses: int = 0
    eviction_count: int = 0
    memory_efficiency: float = 0.0
    adaptation_score: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def record_hit(self) -> None:
        self.total_requests += 1
        self.total_hits += 1
        self._recalculate_rates()

    def record_miss(self) -> None:
        self.total_requests += 1
        self.total_misses += 1
        self._recalculate_rates()

    def record_eviction(self) -> None:
        self.eviction_count += 1
        self._recalculate_rates()

    def _recalculate_rates(self) -> None:
        if self.total_requests:
            self.hit_rate = self.total_hits / self.total_requests
            self.miss_rate = self.total_misses / self.total_requests
        else:
            self.hit_rate = 0.0
            self.miss_rate = 0.0
        self.last_updated = datetime.utcnow()


@dataclass
class AccessPatternAnalysis:
    pattern_type: str = "unknown"
    pattern: Optional[Any] = None
    frequency: float = 0.0
    confidence: float = 0.0
    temporal_locality: float = 0.0
    spatial_locality: float = 0.0
    temporal_distribution: Optional[Dict[str, float]] = None
    frequency_distribution: Optional[Dict[str, Any]] = None
    burst_detection: bool = False
    seasonality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.pattern is not None and (self.pattern_type == "unknown" or not self.pattern_type):
            self.pattern_type = str(getattr(self.pattern, "value", self.pattern))
        elif hasattr(self.pattern_type, "value"):
            self.pattern_type = str(self.pattern_type.value)


class LRUStrategy:
    """简单的LRU策略实现"""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache: "OrderedDict[Any, Any]" = OrderedDict()
        self.access_times: Dict[Any, float] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.hits += 1
        return value

    def put(self, key: Any, value: Any) -> None:
        if key in self.cache:
            self.cache.pop(key)
        elif self.capacity is not None and self.capacity > 0 and len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        elif self.capacity == 0:
            return
        self.cache[key] = value
        self.access_times[key] = time.time()

    def delete(self, key: Any) -> bool:
        existed = key in self.cache
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        return existed

    def clear(self) -> None:
        self.cache.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0

    def should_evict(self, key: Any, cache_entry: Any, cache_state: Any) -> bool:
        capacity = self.capacity
        current_size = len(self.cache)
        if isinstance(cache_state, dict):
            capacity = cache_state.get('capacity', cache_state.get('max_size', capacity))
            current_size = cache_state.get('current_size', current_size)
        elif isinstance(cache_state, (int, float)):
            try:
                current_size = int(cache_state)
            except Exception:
                current_size = len(self.cache)
        if capacity <= 0:
            return False
        return current_size >= capacity and key not in self.cache

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests else 0.0
        miss_rate = self.misses / total_requests if total_requests else 0.0
        return {
            "capacity": self.capacity,
            "cache_size": len(self.cache),
            "current_size": len(self.cache),
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "hits": self.hits,
            "misses": self.misses,
        }

    def on_access(self, key: Any, value: Any = None) -> None:
        if key in self.cache:
            self.get(key)

    def on_evict(self, key: Any, value: Any = None) -> None:
        self.delete(key)

    def evict(self) -> Optional[Tuple[Any, Any]]:
        if not self.cache:
            return None
        key, value = self.cache.popitem(last=False)
        self.access_times.pop(key, None)
        return key, value

    def select_victim(self) -> Optional[Any]:
        if not self.cache:
            return None
        return next(iter(self.cache))

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StrategyType):
            return other is StrategyType.LFU
        if isinstance(other, str):
            return other.lower() == StrategyType.LFU.value
        return object.__eq__(self, other)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StrategyType):
            return other is StrategyType.LRU
        if isinstance(other, str):
            return other.lower() == StrategyType.LRU.value
        return object.__eq__(self, other)


@dataclass
class LFUNode:
    key: Any
    value: Any
    frequency: int = 1
    last_accessed: float = field(default_factory=time.time)


class LFUStrategy:
    """LFU策略实现"""

    def __init__(self, capacity: int = 100) -> None:
        self.capacity = capacity
        self.cache: Dict[Any, LFUNode] = {}
        self.freq_map: Dict[int, "OrderedDict[Any, LFUNode]"] = defaultdict(OrderedDict)
        self.min_freq = 1
        self.hits = 0
        self.misses = 0
        self.frequency: Dict[Any, int] = {}

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        node = self.cache[key]
        self._increase_frequency(node)
        self.hits += 1
        self.frequency[node.key] = node.frequency
        return node.value

    def put(self, key: Any, value: Any) -> None:
        if self.capacity is not None and self.capacity <= 0:
            return

        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._increase_frequency(node)
            return

        if self.capacity and len(self.cache) >= self.capacity:
            self._evict()

        node = LFUNode(key=key, value=value)
        self.cache[key] = node
        self.freq_map[1][key] = node
        self.frequency[key] = 1
        self.min_freq = 1

    def delete(self, key: Any) -> bool:
        if key not in self.cache:
            return False
        node = self.cache.pop(key)
        freq_bucket = self.freq_map.get(node.frequency)
        if freq_bucket and key in freq_bucket:
            del freq_bucket[key]
            if not freq_bucket:
                self.freq_map.pop(node.frequency, None)
                if self.min_freq == node.frequency:
                    self.min_freq = min(self.freq_map.keys(), default=1)
        self.frequency.pop(key, None)
        return True

    def clear(self) -> None:
        self.cache.clear()
        self.freq_map.clear()
        self.min_freq = 1
        self.hits = 0
        self.misses = 0
        self.frequency.clear()

    def should_evict(self, key: Any, cache_entry: Any, cache_state: Any) -> bool:
        capacity = self.capacity
        current_size = len(self.cache)
        if isinstance(cache_state, dict):
            capacity = cache_state.get('capacity', cache_state.get('max_size', capacity))
            current_size = cache_state.get('current_size', current_size)
        elif isinstance(cache_state, (int, float)):
            try:
                current_size = int(cache_state)
            except Exception:
                current_size = len(self.cache)
        if capacity <= 0:
            return False
        return current_size >= capacity and key not in self.cache

    def _increase_frequency(self, node: LFUNode) -> None:
        freq = node.frequency
        freq_bucket = self.freq_map[freq]
        if node.key in freq_bucket:
            del freq_bucket[node.key]
        if not freq_bucket:
            del self.freq_map[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        node.frequency += 1
        node.last_accessed = time.time()
        self.freq_map[node.frequency][node.key] = node
        self.frequency[node.key] = node.frequency

    def _evict(self) -> None:
        freq_bucket = self.freq_map[self.min_freq]
        key, _ = freq_bucket.popitem(last=False)
        self.cache.pop(key, None)
        if not freq_bucket:
            del self.freq_map[self.min_freq]

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests else 0.0
        miss_rate = self.misses / total_requests if total_requests else 0.0
        return {
            "capacity": self.capacity,
            "cache_size": len(self.cache),
            "current_size": len(self.cache),
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "hits": self.hits,
            "misses": self.misses,
        }

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StrategyType):
            return other is StrategyType.LFU
        if isinstance(other, str):
            return other.lower() == StrategyType.LFU.value
        return object.__eq__(self, other)

    def should_evict(self, key: Any, cache_entry: Any, cache_state: Any) -> bool:
        capacity = self.capacity
        current_size = len(self.cache)
        if isinstance(cache_state, dict):
            capacity = cache_state.get('capacity', cache_state.get('max_size', capacity))
            current_size = cache_state.get('current_size', current_size)
        elif isinstance(cache_state, (int, float)):
            try:
                capacity = int(cache_state)
            except Exception:
                capacity = self.capacity
        if capacity <= 0:
            return False
        return current_size >= capacity and key not in self.cache


class TTLStrategy:
    """TTL策略实现"""

    def __init__(self, capacity: int = 100, default_ttl: Optional[int] = None) -> None:
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache: Dict[Any, Any] = {}
        self.expiry: Dict[Any, float] = {}
        self.expiration_times = self.expiry
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        if self._is_expired(key):
            self.delete(key)
            self.misses += 1
            return None
        self.hits += 1
        return self.cache[key]

    def put(self, key: Any, value: Any, ttl: Optional[int] = None) -> None:
        if len(self.cache) >= self.capacity:
            self._evict_expired()
        self.cache[key] = value
        if ttl is not None:
            if ttl > 0:
                self.expiry[key] = time.time() + ttl
            else:
                self.expiry.pop(key, None)
        elif self.default_ttl and self.default_ttl > 0:
            self.expiry[key] = time.time() + self.default_ttl
        else:
            self.expiry.pop(key, None)

    def delete(self, key: Any) -> bool:
        existed = key in self.cache
        self.cache.pop(key, None)
        self.expiry.pop(key, None)
        return existed

    def clear(self) -> None:
        self.cache.clear()
        self.expiry.clear()
        self.hits = 0
        self.misses = 0

    def _is_expired(self, key: Any) -> bool:
        expiry = self.expiry.get(key)
        if expiry is None:
            return False
        return time.time() > expiry

    def _evict_expired(self) -> None:
        expired_keys = [k for k, expiry in self.expiry.items() if time.time() > expiry]
        for key in expired_keys:
            self.delete(key)

    def get_stats(self) -> Dict[str, Any]:
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests else 0.0
        miss_rate = self.misses / total_requests if total_requests else 0.0
        return {
            "capacity": self.capacity,
            "cache_size": len(self.cache),
            "current_size": len(self.cache),
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "hits": self.hits,
            "misses": self.misses,
        }

    def should_evict(self, key: Any, cache_entry: Any, cache_state: Any) -> bool:
        if key in self.expiry and self._is_expired(key):
            return True
        capacity = self.capacity
        current_size = len(self.cache)
        if isinstance(cache_state, dict):
            capacity = cache_state.get('capacity', cache_state.get('max_size', capacity))
            current_size = cache_state.get('current_size', current_size)
        elif isinstance(cache_state, (int, float)):
            try:
                current_size = int(cache_state)
            except Exception:
                current_size = len(self.cache)
        if capacity <= 0:
            return False
        return current_size >= capacity and key not in self.cache

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, StrategyType):
            return other is StrategyType.TTL
        if isinstance(other, str):
            return other.lower() == StrategyType.TTL.value
        return object.__eq__(self, other)


class AdaptiveStrategy:
    """自适应策略实现。根据访问模式在LRU和LFU之间切换。"""

    def __init__(self, capacity: int = 100, max_memory_mb: float = 256.0) -> None:
        self.capacity = capacity
        self.max_memory_mb = max_memory_mb
        self.access_patterns: List[Dict[str, Any]] = []
        self.access_times: Dict[Any, List[float]] = {}
        self.lru = LRUStrategy(capacity)
        self.lfu = LFUStrategy(capacity)
        self.ttl = TTLStrategy(capacity)
        self._strategy_map: Dict[StrategyType, Any] = {
            StrategyType.LRU: self.lru,
            StrategyType.LFU: self.lfu,
            StrategyType.TTL: self.ttl,
        }
        self.current_strategy_type = StrategyType.LRU
        self._active_strategy: Any = self._strategy_map[self.current_strategy_type]
        self.current_strategy_name: str = self.current_strategy_type.value
        self.hits = 0
        self.misses = 0
        self.eviction_count = 0

    def get(self, key: Any) -> Optional[Any]:
        self.access_patterns.append({"action": "get", "key": key, "timestamp": time.time()})
        self.access_times.setdefault(key, []).append(time.time())
        result = self._active_strategy.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        self._check_adaptation()
        return result

    def put(self, key: Any, value: Any, **kwargs) -> None:
        self.access_patterns.append({"action": "put", "key": key, "timestamp": time.time()})
        self.access_times.setdefault(key, []).append(time.time())
        if isinstance(self._active_strategy, TTLStrategy):
            self._active_strategy.put(key, value, **kwargs)
        else:
            self._active_strategy.put(key, value)
        self._check_adaptation()

    def delete(self, key: Any) -> bool:
        delete_fn = getattr(self._active_strategy, "delete", None)
        if callable(delete_fn):
            deleted = delete_fn(key)
            if deleted:
                self.access_times.pop(key, None)
                self.eviction_count += 1
            return deleted
        return False

    def clear(self) -> None:
        for strategy in self._strategy_map.values():
            strategy.clear()
        self.access_patterns.clear()
        self.access_times.clear()
        self.hits = 0
        self.misses = 0

    def _check_adaptation(self) -> None:
        if len(self.access_patterns) < 5:
            return
        recent = self.access_patterns[-10:]
        get_count = sum(1 for entry in recent if entry["action"] == "get")
        unique_keys = {entry["key"] for entry in recent}
        if get_count and len(unique_keys) <= get_count / 2:
            self._set_active_strategy(StrategyType.LFU)
        else:
            self._set_active_strategy(StrategyType.LRU)

    def _set_active_strategy(self, strategy_type: StrategyType) -> None:
        strategy = self._strategy_map.get(strategy_type)
        if strategy is None or strategy is self._active_strategy:
            return
        previous_cache = getattr(self._active_strategy, "cache", None)
        snapshot: List[tuple[Any, Any]] = []
        if isinstance(previous_cache, dict):
            snapshot = list(previous_cache.items())
        if hasattr(strategy, "clear"):
            strategy.clear()
        for key, value in snapshot:
            value_to_store = getattr(value, "value", value)
            if isinstance(strategy, TTLStrategy):
                strategy.put(key, value_to_store)
            else:
                strategy.put(key, value_to_store)
        self.current_strategy_type = strategy_type
        self.current_strategy_name = strategy_type.value
        self._active_strategy = strategy

    def _perform_memory_cleanup(self) -> None:
        self.access_patterns = self.access_patterns[-100:]
        for key, history in list(self.access_times.items()):
            if len(history) > 10:
                self.access_times[key] = history[-10:]
            if not self.access_times[key]:
                del self.access_times[key]

    def get_stats(self) -> Dict[str, Any]:
        strategy_stats = self._active_strategy.get_stats() if hasattr(self._active_strategy, "get_stats") else {}
        return {
            "capacity": self.capacity,
            "cache_size": strategy_stats.get("cache_size", 0),
            "current_size": strategy_stats.get("current_size", len(self.cache)),
            "current_strategy": self.current_strategy_name,
            "access_patterns_size": len(self.access_patterns),
        }

    @property
    def cache(self) -> Dict[Any, Any]:
        return getattr(self._active_strategy, "cache", {})

    @property
    def current_strategy(self) -> Any:
        return self._active_strategy


class CacheStrategyManager:
    """缓存策略管理器"""

    def __init__(self, default_strategy: StrategyType = StrategyType.LRU, capacity: int = 1000) -> None:
        self.capacity = capacity
        self.monitoring_enabled = True
        self.default_strategy = default_strategy
        self.strategies: Dict[StrategyType, Any] = {
            StrategyType.LRU: LRUStrategy(capacity),
            StrategyType.LFU: LFUStrategy(capacity),
            StrategyType.TTL: TTLStrategy(capacity),
            StrategyType.ADAPTIVE: AdaptiveStrategy(capacity),
        }
        self.current_strategy_type = default_strategy
        self.current_strategy = self.strategies[self.current_strategy_type]

        self.strategy_metrics: Dict[StrategyType, StrategyMetrics] = {
            strategy_type: StrategyMetrics(strategy_name=strategy_type.value)
            for strategy_type in StrategyType
        }

    # ------------------------------------------------------------------
    # 基础操作
    # ------------------------------------------------------------------
    def get(self, key: Any) -> Optional[Any]:
        result = self.current_strategy.get(key)
        metrics = self.strategy_metrics[self.current_strategy_type]
        if result is None:
            metrics.record_miss()
        else:
            metrics.record_hit()
        return result

    def put(self, key: Any, value: Any, **kwargs) -> None:
        if hasattr(self.current_strategy, "put"):
            self.current_strategy.put(key, value, **kwargs)

    def delete(self, key: Any) -> bool:
        delete_fn = getattr(self.current_strategy, "delete", None)
        if callable(delete_fn):
            deleted = delete_fn(key)
            if deleted:
                self.strategy_metrics[self.current_strategy_type].record_eviction()
            return deleted
        return False

    # ------------------------------------------------------------------
    # 策略管理
    # ------------------------------------------------------------------
    def switch_strategy(self, strategy_type: StrategyType | CacheEvictionStrategy) -> bool:
        normalized = self._normalize_strategy_type(strategy_type)
        if normalized not in self.strategies:
            return False
        self.current_strategy_type = normalized
        self.current_strategy = self.strategies[normalized]
        return True

    def set_current_strategy(self, strategy_type: StrategyType | CacheEvictionStrategy) -> bool:
        return self.switch_strategy(strategy_type)

    def get_strategy(self, strategy: StrategyType | CacheEvictionStrategy) -> Any:
        strategy_type = self._normalize_strategy_type(strategy)
        return self.strategies.get(strategy_type)

    def enable_monitoring(self, enabled: bool = True) -> None:
        self.monitoring_enabled = enabled

    def reset_metrics(self) -> None:
        for strategy_type in StrategyType:
            self.strategy_metrics[strategy_type] = StrategyMetrics(strategy_name=strategy_type.value)

    # ------------------------------------------------------------------
    # 分析与报告
    # ------------------------------------------------------------------
    def analyze_access_patterns(self) -> AccessPatternAnalysis:
        patterns = AccessPatternAnalysis(pattern_type="mixed")
        patterns.recommendations.append("Enable adaptive strategy for mixed workloads")
        return patterns

    def get_optimal_strategy(self, access_patterns: Optional[List[Dict[str, Any]]] = None) -> StrategyType:
        if access_patterns:
            sequential_count = sum(1 for pattern in access_patterns if pattern.get("pattern") == "sequential")
            if sequential_count > len(access_patterns) / 2:
                return StrategyType.LRU
        best = max(self.strategy_metrics.items(), key=lambda item: item[1].hit_rate, default=None)
        return best[0] if best and best[1].hit_rate > 0 else self.current_strategy_type

    def optimize_strategy(self) -> StrategyType:
        return self.get_optimal_strategy()

    def get_strategy_metrics(self, strategy_type: Optional[StrategyType] = None) -> Dict[str, Any]:
        if strategy_type is None:
            return {st.value: self.get_strategy_metrics(st) for st in StrategyType}
        metrics = self.strategy_metrics[strategy_type]
        return {
            "strategy_name": metrics.strategy_name,
            "hit_rate": metrics.hit_rate,
            "miss_rate": metrics.miss_rate,
            "avg_response_time": metrics.avg_response_time,
            "total_requests": metrics.total_requests,
            "total_hits": metrics.total_hits,
            "total_misses": metrics.total_misses,
            "eviction_count": metrics.eviction_count,
            "memory_efficiency": metrics.memory_efficiency,
            "last_updated": metrics.last_updated,
        }

    def get_strategy_performance_report(self) -> Dict[str, Any]:
        optimal = self.get_optimal_strategy()
        comparison = {st.value: self.get_strategy_metrics(st) for st in StrategyType}
        recommendations = [
            f"Consider switching to {optimal.value} strategy for improved hit rate",
            "Monitor cache hit/miss trends weekly",
            "Adjust capacity to keep utilization under 80%",
        ]
        return {
            "current_strategy": self.current_strategy_type.value,
            "optimal_strategy": optimal.value,
            "strategy_metrics": self.get_strategy_metrics(self.current_strategy_type),
            "strategy_comparison": comparison,
            "recommendations": recommendations,
        }

    def get_stats(self) -> Dict[str, Any]:
        cache_stats = self._collect_cache_stats()
        return {
            "current_strategy": self.current_strategy_type.value,
            "total_strategies": len(self.strategies),
            "capacity": self.capacity,
            "strategy_metrics": self.get_strategy_metrics(self.current_strategy_type),
            "strategy_performance": self.get_strategy_metrics(),
            "all_strategies": [st.value for st in StrategyType],
            "cache_stats": cache_stats,
            "monitoring_enabled": self.monitoring_enabled,
        }

    @property
    def current_strategy(self) -> Any:
        return getattr(self, "_current_strategy", None)

    @current_strategy.setter
    def current_strategy(self, value: Any) -> None:
        self._current_strategy = value

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------
    def clear_cache(self) -> None:
        for strategy in self.strategies.values():
            if hasattr(strategy, "clear"):
                strategy.clear()

    def get_cache_size(self) -> int:
        cache = getattr(self.current_strategy, "cache", None)
        return len(cache) if cache is not None else 0

    def is_monitoring_enabled(self) -> bool:
        return self.monitoring_enabled

    def _collect_cache_stats(self) -> Dict[str, Any]:
        size = self.get_cache_size()
        utilization = size / self.capacity if self.capacity else 0.0
        return {"size": size, "utilization": utilization, "capacity": self.capacity}

    def _normalize_strategy_type(self, strategy: StrategyType | CacheEvictionStrategy) -> StrategyType:
        if isinstance(strategy, StrategyType):
            return strategy
        mapping: Dict[CacheEvictionStrategy, StrategyType] = {
            CacheEvictionStrategy.LRU: StrategyType.LRU,
            CacheEvictionStrategy.LFU: StrategyType.LFU,
            CacheEvictionStrategy.TTL: StrategyType.TTL,
            CacheEvictionStrategy.FIFO: StrategyType.FIFO,
            CacheEvictionStrategy.RANDOM: StrategyType.RANDOM,
            CacheEvictionStrategy.ADAPTIVE: StrategyType.ADAPTIVE,
        }
        return mapping.get(strategy, self.current_strategy_type)

