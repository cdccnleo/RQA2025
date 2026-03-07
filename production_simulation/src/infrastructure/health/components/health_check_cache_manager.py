"""
健康检查缓存管理器

负责健康检查结果的缓存管理，包括缓存存储、过期检查、清理等功能。
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from src.infrastructure.logging.core.unified_logger import get_unified_logger

logger = get_unified_logger(__name__)

@dataclass
class CacheEntry:
    """缓存条目"""
    key: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    ttl: float = 300.0
    value: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """兼容旧版字段命名并保持数据同步"""
        if self.value is not None and not self.data:
            self.data = dict(self.value)
        elif self.data and self.value is None:
            self.value = dict(self.data)

        if self.key is None and isinstance(self.data, dict):
            self.key = self.data.get('service_name')

    def is_expired(self) -> bool:
        """检查缓存是否已过期"""
        return time.time() - self.timestamp > self.ttl


class HealthCheckCacheManager:
    """
    健康检查缓存管理器
    
    职责：
    - 缓存健康检查结果
    - 管理缓存过期时间
    - 提供缓存清理功能
    - 缓存命中率统计
    """
    
    def __init__(self, default_ttl: float = 300.0, max_entries: int = 1000, *, ttl: Optional[float] = None):
        """
        初始化缓存管理器
        
        Args:
            default_ttl: 默认缓存过期时间(秒)
            max_entries: 最大缓存条目数
        """
        if ttl is not None:
            default_ttl = ttl
        self._default_ttl = default_ttl
        self._max_entries = max_entries
        self._cache: Dict[str, CacheEntry] = {}
        self._hit_count = 0
        self._miss_count = 0
        
    def get_cached_result(self, service_name: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存的健康检查结果
        
        Args:
            service_name: 服务名称
            
        Returns:
            缓存的结果，如果不存在或已过期则返回None
        """
        entry = self._cache.get(service_name)
        
        if entry is None:
            self._miss_count += 1
            return None
            
        if entry.is_expired():
            # 缓存已过期，删除并返回None
            del self._cache[service_name]
            self._miss_count += 1
            logger.debug(f"缓存条目 {service_name} 已过期，已删除")
            return None
            
        self._hit_count += 1
        logger.debug(f"缓存命中: {service_name}")
        return entry.data
    
    def cache_result(self, service_name: str, result: Dict[str, Any], ttl: Optional[float] = None) -> None:
        """
        缓存健康检查结果
        
        Args:
            service_name: 服务名称
            result: 检查结果
            ttl: 缓存过期时间，None则使用默认值
        """
        cache_ttl = ttl if ttl is not None else self._default_ttl
        
        # 检查是否需要清理缓存
        if len(self._cache) >= self._max_entries:
            self._cleanup_expired_entries()
            
        # 如果仍然超出限制，删除最旧的条目
        if len(self._cache) >= self._max_entries:
            self._evict_oldest_entry()
        
        self._cache[service_name] = CacheEntry(
            key=service_name,
            data=result.copy(),
            timestamp=time.time(),
            ttl=cache_ttl
        )
        
        logger.debug(f"缓存结果: {service_name}, TTL: {cache_ttl}s")
    
    def clear_cache(self, service_name: Optional[str] = None) -> bool:
        """
        清除缓存
        
        Args:
            service_name: 特定服务名称，None则清除所有缓存
            
        Returns:
            是否成功清除
        """
        try:
            if service_name is None:
                self._cache.clear()
                logger.info("已清除所有缓存")
            elif service_name in self._cache:
                del self._cache[service_name]
                logger.info(f"已清除服务 {service_name} 的缓存")
            
            return True
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return False
    
    def _cleanup_expired_entries(self) -> None:
        """清理过期的缓存条目"""
        expired_keys = [
            key for key, entry in self._cache.items() 
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self._cache[key]
            
        if expired_keys:
            logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
    
    def _evict_oldest_entry(self) -> None:
        """删除最旧的缓存条目"""
        if not self._cache:
            return
            
        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
        del self._cache[oldest_key]
        logger.debug(f"因缓存容量限制，删除了最旧的条目: {oldest_key}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "cache_entries": len(self._cache),
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": round(hit_rate, 2),
            "max_entries": self._max_entries,
            "default_ttl": self._default_ttl
        }
    
    def is_service_cached(self, service_name: str) -> bool:
        """检查服务是否有有效缓存"""
        entry = self._cache.get(service_name)
        return entry is not None and not entry.is_expired()

