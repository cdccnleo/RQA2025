"""
轻量可测的预测型缓存实现
"""

from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Any, Deque, Dict, Hashable, List, Optional, Tuple


@dataclass
class CacheStats:
    capacity: int
    size: int
    hits: int
    misses: int
    evictions: int


class PredictiveCache:
    """
    最小可用预测缓存：
    - 固定容量 FIFO 淘汰（确定性，便于测试）
    - 简单一阶马尔可夫预测：基于历史序列统计 next-key 频率
    """

    def __init__(self, capacity: int = 64):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity: int = capacity
        self._store: Dict[Hashable, Any] = {}
        self._order: Deque[Hashable] = deque()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        # 预测用转移计数：key -> {next_key: count}
        self._transitions: Dict[Hashable, Dict[Hashable, int]] = defaultdict(lambda: defaultdict(int))
        self._last_key: Optional[Hashable] = None

    # ------------------------ 基本缓存操作 ------------------------ #
    def set(self, key: Hashable, value: Any) -> None:
        if key in self._store:
            self._store[key] = value
            # 维持 FIFO：不调整队列顺序（确定性）
            return

        if len(self._order) >= self.capacity:
            oldest = self._order.popleft()
            self._store.pop(oldest, None)
            self._evictions += 1

        self._order.append(key)
        self._store[key] = value

        # 记录转移（用于预测）
        if self._last_key is not None:
            self._transitions[self._last_key][key] += 1
        self._last_key = key

    def get(self, key: Hashable) -> Optional[Any]:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def delete(self, key: Hashable) -> bool:
        if key in self._store:
            self._store.pop(key, None)
            try:
                self._order.remove(key)
            except ValueError:
                pass
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        self._order.clear()
        self._hits = self._misses = self._evictions = 0
        self._transitions.clear()
        self._last_key = None

    def get_stats(self) -> CacheStats:
        return CacheStats(
            capacity=self.capacity,
            size=len(self._order),
            hits=self._hits,
            misses=self._misses,
            evictions=self._evictions,
        )

    # ------------------------ 预测能力 ------------------------ #
    def predict_next_key(self, current_key: Hashable) -> Optional[Hashable]:
        """
        返回最可能的下一个 key（基于出现频率最大者）；若从未见过返回 None
        """
        next_counts = self._transitions.get(current_key)
        if not next_counts:
            return None
        # 选择计数最高且按键名稳定排序打破并列（确定性）
        candidates: List[Tuple[Hashable, int]] = sorted(next_counts.items(), key=lambda kv: (kv[1], str(kv[0])))
        return candidates[-1][0]

    def top_predictions(self, current_key: Hashable, k: int = 3) -> List[Hashable]:
        """
        返回 top-k 预测 key 列表
        """
        next_counts = self._transitions.get(current_key, {})
        if not next_counts:
            return []
        ordered = sorted(next_counts.items(), key=lambda kv: (kv[1], str(kv[0])), reverse=True)
        return [key for key, _ in ordered[: max(0, k)]]


__all__ = ["PredictiveCache", "CacheStats"]


