"""缓存系统核心接口"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from enum import Enum

class AccessPattern(Enum):
    """访问模式枚举"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

class CacheEvictionStrategy(Enum):
    """缓存淘汰策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    TTL = "ttl"
    ADAPTIVE = "adaptive"

class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"
    COST_AWARE = "cost_aware"
    FIFO = "fifo"
    TTL = "ttl"
    SIZE = "size"

class ICacheComponent(ABC):
    """缓存组件接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass

class CacheInterface(ABC):
    """缓存接口"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存项"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """清空缓存"""
        pass

class CacheEntry:
    """缓存条目"""
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.ttl = ttl
        self.created_at = 0
        self.access_count = 0

class CacheStats:
    """缓存统计信息"""
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.sets = 0
        self.deletes = 0

class CacheManagerInterface(ABC):
    """缓存管理器接口"""
    
    @abstractmethod
    def get_cache(self, name: str) -> Optional[CacheInterface]:
        """获取指定名称的缓存"""
        pass
    
    @abstractmethod
    def create_cache(self, name: str, config: Dict[str, Any]) -> CacheInterface:
        """创建缓存"""
        pass

class CacheFactoryInterface(ABC):
    """缓存工厂接口"""
    
    @abstractmethod
    def create_cache(self, config: Dict[str, Any]) -> CacheInterface:
        """创建缓存实例"""
        pass


# ==================== 驱逐策略实现 ====================

class EvictionStrategyImpl:
    """驱逐策略实现基类
    
    为CacheEvictionStrategy枚举提供具体的策略实现
    """
    
    def __init__(self, strategy: CacheEvictionStrategy):
        """初始化驱逐策略
        
        Args:
            strategy: 策略类型
        """
        self.strategy = strategy
        self._access_times: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = {}
        self._insertion_order: List[str] = []
    
    def should_evict(self, key: str, cache_entry: Any, cache_state: Optional[Dict[str, Any]] = None) -> bool:
        """判断是否应该驱逐某个缓存项
        
        Args:
            key: 缓存键
            cache_entry: 缓存项
            cache_state: 当前缓存状态（大小、容量等）
            
        Returns:
            是否应该驱逐
        """
        if cache_state is None:
            cache_state = {}
        
        current_size = cache_state.get('current_size', 0)
        max_size = cache_state.get('max_size', 1000)
        
        # 如果未超过容量，不需要驱逐
        if current_size < max_size:
            return False
        
        # 根据策略决定是否驱逐
        if self.strategy == CacheEvictionStrategy.LRU:
            return self._should_evict_lru(key, cache_entry, cache_state)
        elif self.strategy == CacheEvictionStrategy.LFU:
            return self._should_evict_lfu(key, cache_entry, cache_state)
        elif self.strategy == CacheEvictionStrategy.FIFO:
            return self._should_evict_fifo(key, cache_entry, cache_state)
        elif self.strategy == CacheEvictionStrategy.TTL:
            return self._should_evict_ttl(key, cache_entry, cache_state)
        elif self.strategy == CacheEvictionStrategy.RANDOM:
            return self._should_evict_random(key, cache_entry, cache_state)
        else:
            # 默认使用LRU策略
            return self._should_evict_lru(key, cache_entry, cache_state)
    
    def _should_evict_lru(self, key: str, entry: Any, state: Dict[str, Any]) -> bool:
        """LRU策略：驱逐最久未使用的"""
        import time
        last_access = self._access_times.get(key, 0)
        current_time = time.time()
        
        # 如果超过5分钟未访问，考虑驱逐
        return (current_time - last_access) > 300
    
    def _should_evict_lfu(self, key: str, entry: Any, state: Dict[str, Any]) -> bool:
        """LFU策略：驱逐使用频率最低的"""
        access_count = self._access_counts.get(key, 0)
        
        # 如果访问次数少于5次，考虑驱逐
        return access_count < 5
    
    def _should_evict_fifo(self, key: str, entry: Any, state: Dict[str, Any]) -> bool:
        """FIFO策略：驱逐最先进入的"""
        if not self._insertion_order:
            return False
        
        # 如果是最先插入的，应该驱逐
        return self._insertion_order[0] == key if self._insertion_order else False
    
    def _should_evict_ttl(self, key: str, entry: Any, state: Dict[str, Any]) -> bool:
        """TTL策略：驱逐已过期的"""
        import time
        
        if hasattr(entry, 'expiry_time'):
            return time.time() > entry.expiry_time
        
        # 如果没有expiry_time，检查是否有ttl字段
        if isinstance(entry, dict) and 'ttl' in entry:
            created_at = entry.get('created_at', 0)
            ttl = entry.get('ttl', 300)
            return time.time() > (created_at + ttl)
        
        return False
    
    def _should_evict_random(self, key: str, entry: Any, state: Dict[str, Any]) -> bool:
        """RANDOM策略：随机驱逐"""
        import random
        # 50%概率驱逐
        return random.random() > 0.5
    
    def record_access(self, key: str) -> None:
        """记录访问
        
        Args:
            key: 访问的键
        """
        import time
        self._access_times[key] = time.time()
        self._access_counts[key] = self._access_counts.get(key, 0) + 1
    
    def record_insertion(self, key: str) -> None:
        """记录插入
        
        Args:
            key: 插入的键
        """
        if key not in self._insertion_order:
            self._insertion_order.append(key)
    
    def remove_key(self, key: str) -> None:
        """移除键的记录
        
        Args:
            key: 要移除的键
        """
        self._access_times.pop(key, None)
        self._access_counts.pop(key, None)
        if key in self._insertion_order:
            self._insertion_order.remove(key)