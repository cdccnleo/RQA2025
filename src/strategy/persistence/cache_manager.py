#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理模块

提供统一的缓存管理功能，支持：
1. TTL过期策略
2. LRU淘汰策略
3. 缓存统计监控
4. 缓存预热
5. 缓存穿透保护
6. 线程安全

Author: RQA2025 Development Team
Date: 2026-03-25
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union

from cachetools import TTLCache, LRUCache, cached
from cachetools.keys import hashkey

logger = logging.getLogger(__name__)

T = TypeVar('T')


class CacheStrategy(Enum):
    """缓存策略"""
    TTL = "ttl"           # TTL过期策略
    LRU = "lru"           # LRU淘汰策略
    TTL_LRU = "ttl_lru"   # 组合策略


@dataclass
class CacheConfig:
    """缓存配置"""
    strategy: CacheStrategy = CacheStrategy.TTL
    max_size: int = 1000              # 最大缓存条目数
    ttl_seconds: float = 3600.0       # TTL过期时间（秒）
    cleanup_interval: float = 300.0   # 清理间隔（秒）
    enable_stats: bool = True         # 启用统计
    enable_warmup: bool = False       # 启用预热
    warmup_keys: List[str] = field(default_factory=list)  # 预热键列表


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0                     # 命中次数
    misses: int = 0                   # 未命中次数
    evictions: int = 0                # 淘汰次数
    size: int = 0                     # 当前大小
    max_size: int = 0                 # 最大大小
    hit_rate: float = 0.0             # 命中率
    avg_access_time: float = 0.0      # 平均访问时间
    total_access_time: float = 0.0    # 总访问时间
    access_count: int = 0             # 访问次数
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'size': self.size,
            'max_size': self.max_size,
            'hit_rate': self.hit_rate,
            'avg_access_time': self.avg_access_time,
        }


class CacheManager:
    """
    缓存管理器
    
    统一管理缓存的创建、访问、过期和统计
    
    Attributes:
        config: 缓存配置
        _cache: 底层缓存实例
        _stats: 缓存统计
        _lock: 线程锁
    """
    
    def __init__(self, config: Optional[CacheConfig] = None, cache_name: str = "default"):
        """
        初始化缓存管理器
        
        Args:
            config: 缓存配置
            cache_name: 缓存名称（用于日志和统计）
        """
        self.config = config or CacheConfig()
        self.cache_name = cache_name
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{cache_name}")
        
        # 创建底层缓存
        self._cache = self._create_cache()
        
        # 初始化统计
        self._stats = CacheStats(max_size=self.config.max_size)
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 穿透保护集合（缓存空值）
        self._null_cache: Set[str] = set()
        
        # 启动清理线程
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = False
        if self.config.cleanup_interval > 0:
            self._start_cleanup_thread()
        
        # 执行预热
        if self.config.enable_warmup and self.config.warmup_keys:
            self._warmup()
        
        self.logger.info(f"缓存管理器初始化完成: {cache_name}, 策略: {self.config.strategy.value}")
    
    def _create_cache(self) -> Union[TTLCache, LRUCache]:
        """创建底层缓存实例"""
        if self.config.strategy == CacheStrategy.TTL:
            return TTLCache(
                maxsize=self.config.max_size,
                ttl=self.config.ttl_seconds
            )
        elif self.config.strategy == CacheStrategy.LRU:
            return LRUCache(maxsize=self.config.max_size)
        elif self.config.strategy == CacheStrategy.TTL_LRU:
            # 使用TTLCache，它已经包含LRU语义
            return TTLCache(
                maxsize=self.config.max_size,
                ttl=self.config.ttl_seconds
            )
        else:
            raise ValueError(f"不支持的缓存策略: {self.config.strategy}")
    
    def _start_cleanup_thread(self) -> None:
        """启动清理线程"""
        def cleanup_worker():
            while not self._stop_cleanup:
                time.sleep(self.config.cleanup_interval)
                if not self._stop_cleanup:
                    self._cleanup_expired()
        
        self._cleanup_thread = threading.Thread(
            target=cleanup_worker,
            name=f"CacheCleanup-{self.cache_name}",
            daemon=True
        )
        self._cleanup_thread.start()
        self.logger.debug("清理线程已启动")
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        with self._lock:
            expired_count = 0
            if isinstance(self._cache, TTLCache):
                # TTLCache自动清理过期项
                expired_count = len([k for k in list(self._cache.keys()) 
                                   if k not in self._cache])
            
            # 清理穿透保护集合
            self._null_cache.clear()
            
            if expired_count > 0:
                self.logger.debug(f"清理过期缓存: {expired_count} 项")
    
    def _warmup(self) -> None:
        """缓存预热"""
        self.logger.info(f"开始缓存预热: {len(self.config.warmup_keys)} 个键")
        # 预热逻辑由具体业务实现
        # 这里只记录预热键列表
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，不存在返回None
        """
        start_time = time.time()
        
        with self._lock:
            # 检查穿透保护
            if key in self._null_cache:
                self._update_stats(hit=False, access_time=time.time() - start_time)
                return None
            
            # 获取缓存值
            value = self._cache.get(key)
            
            if value is not None:
                self._update_stats(hit=True, access_time=time.time() - start_time)
                self.logger.debug(f"缓存命中: {key}")
            else:
                self._update_stats(hit=False, access_time=time.time() - start_time)
                self.logger.debug(f"缓存未命中: {key}")
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 自定义TTL（秒），None使用默认
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                # 如果值是None，添加到穿透保护
                if value is None:
                    self._null_cache.add(key)
                    return True
                
                # 从穿透保护中移除
                if key in self._null_cache:
                    self._null_cache.remove(key)
                
                # 设置缓存
                if ttl and isinstance(self._cache, TTLCache):
                    # TTLCache不支持单个键的TTL，需要特殊处理
                    self._cache[key] = value
                else:
                    self._cache[key] = value
                
                self._stats.size = len(self._cache)
                self.logger.debug(f"缓存设置: {key}")
                return True
                
            except Exception as e:
                self.logger.error(f"缓存设置失败: {key}, 错误: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功
        """
        with self._lock:
            try:
                if key in self._cache:
                    del self._cache[key]
                
                if key in self._null_cache:
                    self._null_cache.remove(key)
                
                self._stats.size = len(self._cache)
                self.logger.debug(f"缓存删除: {key}")
                return True
                
            except Exception as e:
                self.logger.error(f"缓存删除失败: {key}, 错误: {e}")
                return False
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._null_cache.clear()
            self._stats = CacheStats(max_size=self.config.max_size)
            self.logger.info("缓存已清空")
    
    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        with self._lock:
            return key in self._cache or key in self._null_cache
    
    def get_stats(self) -> CacheStats:
        """获取缓存统计"""
        with self._lock:
            self._stats.size = len(self._cache)
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                size=self._stats.size,
                max_size=self._stats.max_size,
                hit_rate=self._stats.hit_rate,
                avg_access_time=self._stats.avg_access_time,
            )
    
    def _update_stats(self, hit: bool, access_time: float) -> None:
        """更新统计信息"""
        if not self.config.enable_stats:
            return
        
        if hit:
            self._stats.hits += 1
        else:
            self._stats.misses += 1
        
        total = self._stats.hits + self._stats.misses
        if total > 0:
            self._stats.hit_rate = self._stats.hits / total
        
        self._stats.access_count += 1
        self._stats.total_access_time += access_time
        if self._stats.access_count > 0:
            self._stats.avg_access_time = self._stats.total_access_time / self._stats.access_count
    
    def cache_decorator(
        self,
        key_func: Optional[Callable] = None,
        ttl: Optional[float] = None
    ) -> Callable:
        """
        缓存装饰器
        
        Args:
            key_func: 自定义键生成函数
            ttl: 自定义TTL
            
        Returns:
            装饰器函数
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                if key_func:
                    cache_key = key_func(*args, **kwargs)
                else:
                    cache_key = self._generate_key(func, args, kwargs)
                
                # 尝试从缓存获取
                result = self.get(cache_key)
                if result is not None:
                    return result
                
                # 执行函数
                result = func(*args, **kwargs)
                
                # 缓存结果
                self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        return decorator
    
    def _generate_key(self, func: Callable, args: Tuple, kwargs: Dict) -> str:
        """生成缓存键"""
        key_data = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def stop(self) -> None:
        """停止缓存管理器"""
        self._stop_cleanup = True
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)
        self.logger.info("缓存管理器已停止")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# 全局缓存管理器实例
_cache_managers: Dict[str, CacheManager] = {}


def get_cache_manager(name: str = "default", config: Optional[CacheConfig] = None) -> CacheManager:
    """获取缓存管理器"""
    if name not in _cache_managers:
        _cache_managers[name] = CacheManager(config=config, cache_name=name)
    return _cache_managers[name]


def create_cache_manager(name: str, config: CacheConfig) -> CacheManager:
    """创建新的缓存管理器"""
    if name in _cache_managers:
        raise ValueError(f"缓存管理器已存在: {name}")
    _cache_managers[name] = CacheManager(config=config, cache_name=name)
    return _cache_managers[name]


def remove_cache_manager(name: str) -> None:
    """移除缓存管理器"""
    if name in _cache_managers:
        _cache_managers[name].stop()
        del _cache_managers[name]


def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """获取所有缓存统计"""
    return {name: manager.get_stats().to_dict() 
            for name, manager in _cache_managers.items()}
