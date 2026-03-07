from typing import Dict, Optional
from .cache_manager import CacheEntry, CacheConfig, ICacheStrategy


class LFUStrategy(ICacheStrategy):

    """
    最少访问淘汰（LFU）智能缓存策略
    """

    def on_set(self, cache: Dict[str, CacheEntry], key: str, entry: CacheEntry, config: CacheConfig):

        # 可用于预热或优先级调整，这里无需特殊处理
        pass

    def on_get(self, cache: Dict[str, CacheEntry], key: str, entry: Optional[CacheEntry], config: CacheConfig):

        # 访问时自动增加访问计数，已由CacheEntry.access()处理
        pass

    def on_evict(self, cache: Dict[str, CacheEntry], config: CacheConfig) -> Optional[str]:

        # 淘汰访问次数最少的key
        if not cache:
            return None
        min_access = min(entry.access_count for entry in cache.values())
        # 若有多个，优先淘汰最早插入的
        candidates = [k for k, v in cache.items() if v.access_count == min_access]
        if not candidates:
            return None
        # 选最早插入的
        oldest_key = min(candidates, key=lambda k: cache[k].created_at)
        return oldest_key
