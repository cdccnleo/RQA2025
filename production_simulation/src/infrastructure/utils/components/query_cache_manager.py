"""
查询缓存管理器组件

负责管理查询结果的缓存、过期检查和缓存统计。
"""

import hashlib
import threading
import time
from typing import Dict, Optional, Tuple, Any

try:
    from .unified_query import QueryRequest, QueryResult
except ImportError:
    # 如果导入失败，定义基础类型
    from dataclasses import dataclass
    
    @dataclass
    class QueryRequest:
        query_id: str
        query_type: str
        storage_type: str
        params: Dict[str, Any]
    
    @dataclass
    class QueryResult:
        query_id: str
        data: Any
        metadata: Dict[str, Any]


class QueryCacheManager:
    """查询缓存管理器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, cache_enabled: bool = True, cache_ttl: int = 300):
        """
        初始化缓存管理器
        
        Args:
            config: 配置字典（可选），包含max_size、ttl等
            cache_enabled: 是否启用缓存
            cache_ttl: 缓存生存时间（秒）
        """
        self.config = config  # 保存配置
        if config is not None:
            cache_enabled = config.get("enabled", True)
            cache_ttl = config.get("ttl", 300)
            
        self.cache_enabled = cache_enabled
        self.cache_ttl = cache_ttl
        self.query_cache: Dict[str, Tuple[QueryResult, float]] = {}
        self.cache_lock = threading.RLock()
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_cached_result(self, request: QueryRequest) -> Optional[QueryResult]:
        """
        获取缓存的查询结果
        
        Args:
            request: 查询请求
            
        Returns:
            缓存的查询结果，如果未命中则返回None
        """
        if not self.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(request)
        
        with self.cache_lock:
            if cache_key in self.query_cache:
                result, cached_time = self.query_cache[cache_key]
                
                # 检查缓存是否过期
                if time.time() - cached_time < self.cache_ttl:
                    self.cache_hits += 1
                    return result
                else:
                    # 过期缓存，删除
                    del self.query_cache[cache_key]
            
            self.cache_misses += 1
            return None
    
    def cache_result(self, request: QueryRequest, result: QueryResult) -> None:
        """
        缓存查询结果
        
        Args:
            request: 查询请求
            result: 查询结果
        """
        if not self.cache_enabled:
            return
        
        cache_key = self._generate_cache_key(request)
        
        with self.cache_lock:
            self.query_cache[cache_key] = (result, time.time())
    
    def set(self, key: str, value: Any) -> None:
        """
        设置缓存（简化接口）
        
        Args:
            key: 缓存键
            value: 缓存值
        """
        if not self.cache_enabled:
            return
        
        with self.cache_lock:
            self.query_cache[key] = (value, time.time())
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存（简化接口）
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果未命中或过期则返回None
        """
        if not self.cache_enabled:
            return None
        
        with self.cache_lock:
            if key in self.query_cache:
                value, cached_time = self.query_cache[key]
                
                # 检查缓存是否过期
                if time.time() - cached_time < self.cache_ttl:
                    self.cache_hits += 1
                    return value
                else:
                    # 过期缓存，删除
                    del self.query_cache[key]
            
            self.cache_misses += 1
            return None
    
    def clear(self) -> None:
        """
        清空所有缓存
        """
        with self.cache_lock:
            self.query_cache.clear()
    
    def cleanup_expired_cache(self) -> int:
        """
        清理过期的缓存
        
        Returns:
            清理的缓存数量
        """
        current_time = time.time()
        cleaned_count = 0
        
        with self.cache_lock:
            expired_keys = [
                key for key, (_, cached_time) in self.query_cache.items()
                if current_time - cached_time >= self.cache_ttl
            ]
            
            for key in expired_keys:
                del self.query_cache[key]
                cleaned_count += 1
        
        return cleaned_count
    
    def get_cache_hit_rate(self) -> float:
        """
        计算缓存命中率
        
        Returns:
            缓存命中率 (0.0-1.0)
        """
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        with self.cache_lock:
            return {
                'cache_enabled': self.cache_enabled,
                'cache_size': len(self.query_cache),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': self.get_cache_hit_rate(),
                'cache_ttl': self.cache_ttl,
            }
    
    def clear_cache(self) -> None:
        """清空所有缓存"""
        with self.cache_lock:
            self.query_cache.clear()
            self.cache_hits = 0
            self.cache_misses = 0
    
    def _generate_cache_key(self, request: QueryRequest) -> str:
        """
        生成缓存键
        
        Args:
            request: 查询请求
            
        Returns:
            缓存键
        """
        # 使用查询参数生成唯一的缓存键
        key_parts = [
            request.query_type.value if hasattr(request.query_type, 'value') else str(request.query_type),
            request.storage_type.value if hasattr(request.storage_type, 'value') else str(request.storage_type),
            str(request.params)
        ]
        key_string = "_".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()

