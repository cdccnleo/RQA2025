#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层 - 日志器池管理

提供线程安全的日志器池管理，支持日志器的复用和生命周期管理。
优化版本：支持LRU缓存、预热机制和标准化Logger ID生成。
"""

import threading
import time
import re
from collections import OrderedDict
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any

from .unified_logger import UnifiedLogger


class LoggerPoolState(Enum):
    """日志器池状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class LoggerStats:
    """日志器统计信息"""
    logger_id: str
    created_time: float
    last_used_time: float
    use_count: int
    error_count: int
    memory_usage: int

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class LoggerPool:
    """日志器池管理类（优化版：LRU缓存 + 预热机制）"""

    _instance = None
    _lock = threading.Lock()
    
    # 常用Logger列表（基于业务流程驱动架构）
    _PRELOAD_LOGGERS = [
        'data_layer',
        'feature_layer', 
        'trading_layer',
        'risk_layer',
        'ml_layer',
        'infrastructure',
        'gateway',
        'monitoring'
    ]

    def __init__(self, max_size: int = 100, idle_timeout: int = 3600):
        self._loggers: Dict[str, Any] = {}
        self._stats: Dict[str, LoggerStats] = {}
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._lock = threading.RLock()
        self._state = LoggerPoolState.HEALTHY
        self._hit_count = 0      # 命中次数
        self._miss_count = 0     # 未命中次数
        # LRU缓存：使用OrderedDict实现最近最少使用策略
        self._lru_cache = OrderedDict()
        self._min_retain_size = 10  # 最小保留数量，避免清理常用Logger
        self._warmed_up = False  # 预热标志

    @classmethod
    def get_instance(cls) -> "LoggerPool":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _normalize_logger_id(self, logger_id: str) -> str:
        """
        标准化Logger ID生成策略
        
        基于模块路径和层级生成一致的Logger ID，确保同一模块多次请求使用相同ID
        
        Args:
            logger_id: 原始Logger ID
            
        Returns:
            标准化后的Logger ID
        """
        if not logger_id:
            return "default"
        
        # 移除常见的模块路径前缀，保留关键部分
        normalized = logger_id
        
        # 标准化模块路径格式
        # 例如: src.core.integration.adapters.features_adapter -> core.integration.adapters.features
        if normalized.startswith('src.'):
            normalized = normalized[4:]  # 移除 'src.' 前缀
        
        # 提取层级信息（基于业务流程驱动架构）
        # 数据层、特征层、交易层等
        layer_patterns = {
            r'\.data\.|data_layer|data_management': 'data_layer',
            r'\.feature|feature_layer|features': 'feature_layer',
            r'\.trading|trading_layer|trading_': 'trading_layer',
            r'\.risk|risk_layer|risk_': 'risk_layer',
            r'\.ml\.|ml_layer|model_': 'ml_layer',
            r'\.infrastructure|infrastructure': 'infrastructure',
            r'\.gateway|gateway': 'gateway',
            r'\.monitoring|monitoring': 'monitoring',
        }
        
        for pattern, layer_name in layer_patterns.items():
            if re.search(pattern, normalized, re.IGNORECASE):
                # 如果匹配到层级，使用层级名 + 模块名
                module_name = normalized.split('.')[-1] if '.' in normalized else normalized
                return f"{layer_name}.{module_name}"
        
        # 如果没有匹配到层级，使用模块名（去除路径）
        if '.' in normalized:
            return normalized.split('.')[-1]
        
        return normalized

    def get_logger(self, logger_id: str, **kwargs) -> Any:
        """
        获取或创建日志器（优化版：标准化ID + LRU缓存）

        Args:
            logger_id: 日志器ID（将自动标准化）
            **kwargs: 创建参数

        Returns:
            日志器实例
        """
        # 标准化Logger ID
        normalized_id = self._normalize_logger_id(logger_id)
        
        with self._lock:
            # 检查是否已存在（使用标准化ID）
            if normalized_id in self._loggers:
                stats = self._stats[normalized_id]
                stats.last_used_time = time.time()
                stats.use_count += 1
                self._hit_count += 1  # 记录命中
                
                # 更新LRU缓存：移动到末尾（最近使用）
                if normalized_id in self._lru_cache:
                    self._lru_cache.move_to_end(normalized_id)
                else:
                    self._lru_cache[normalized_id] = time.time()
                
                return self._loggers[normalized_id]

            # 检查池大小限制（使用LRU策略）
            if len(self._loggers) >= self._max_size:
                self._evict_lru()

            # 记录未命中（在创建新logger之前）
            self._miss_count += 1

            # 创建新日志器
            logger = self._create_logger(normalized_id, **kwargs)
            if logger:
                self._loggers[normalized_id] = logger
                self._stats[normalized_id] = LoggerStats(
                    logger_id=normalized_id,
                    created_time=time.time(),
                    last_used_time=time.time(),
                    use_count=1,
                    error_count=0,
                    memory_usage=0
                )
                # 添加到LRU缓存
                self._lru_cache[normalized_id] = time.time()

            return logger

    def _create_logger(self, logger_id: str, **kwargs) -> Any:
        """创建日志器实例"""
        try:
            # 这里应该根据实际的日志器实现来创建
            # 暂时返回一个占位符
            return UnifiedLogger(logger_id, **kwargs)
        except Exception:
            # 如果无法创建，返回None
            return None

    def release_logger(self, logger_id: str) -> None:
        """释放日志器"""
        with self._lock:
            if logger_id in self._loggers:
                stats = self._stats.get(logger_id)
                if stats:
                    stats.last_used_time = time.time()

    def remove_logger(self, logger_id: str) -> None:
        """移除日志器"""
        with self._lock:
            if logger_id in self._loggers:
                del self._loggers[logger_id]
                if logger_id in self._stats:
                    del self._stats[logger_id]

    def _evict_oldest(self) -> None:
        """驱逐最旧的日志器（保留方法，向后兼容）"""
        self._evict_lru()

    def _evict_lru(self) -> None:
        """使用LRU策略驱逐日志器"""
        if not self._lru_cache or len(self._loggers) <= self._min_retain_size:
            return

        # 从LRU缓存中移除最久未使用的（第一个）
        # 但保留最小数量的Logger
        while len(self._loggers) > self._min_retain_size and self._lru_cache:
            # 获取最久未使用的Logger ID（第一个）
            oldest_id = next(iter(self._lru_cache))
            
            # 检查是否在预加载列表中，如果是则跳过
            if oldest_id in self._PRELOAD_LOGGERS or any(
                oldest_id.startswith(preload) for preload in self._PRELOAD_LOGGERS
            ):
                # 移动到末尾，避免清理常用Logger
                self._lru_cache.move_to_end(oldest_id)
                # 如果所有Logger都是预加载的，停止清理
                if all(
                    lid in self._PRELOAD_LOGGERS or any(lid.startswith(p) for p in self._PRELOAD_LOGGERS)
                    for lid in self._lru_cache.keys()
                ):
                    break
                continue
            
            # 移除最久未使用的Logger
            self.remove_logger(oldest_id)
            break

    def cleanup_idle_loggers(self) -> int:
        """清理空闲日志器（优化版：保护预加载Logger）"""
        current_time = time.time()
        removed_count = 0

        with self._lock:
            to_remove = []
            for logger_id, stats in self._stats.items():
                # 检查是否空闲且不在预加载列表中
                is_idle = current_time - stats.last_used_time > self._idle_timeout
                is_preloaded = logger_id in self._PRELOAD_LOGGERS or any(
                    logger_id.startswith(preload) for preload in self._PRELOAD_LOGGERS
                )
                
                # 只清理空闲且非预加载的Logger，且确保保留最小数量
                if is_idle and not is_preloaded and len(self._loggers) > self._min_retain_size:
                    to_remove.append(logger_id)

            for logger_id in to_remove:
                self.remove_logger(logger_id)
                removed_count += 1

        return removed_count

    def warmup(self) -> None:
        """
        预热Logger池，预创建常用Logger实例
        
        基于业务流程驱动架构，预创建各业务层的常用Logger
        """
        if self._warmed_up:
            return
        
        with self._lock:
            for logger_id in self._PRELOAD_LOGGERS:
                if logger_id not in self._loggers:
                    try:
                        logger = self._create_logger(logger_id)
                        if logger:
                            self._loggers[logger_id] = logger
                            self._stats[logger_id] = LoggerStats(
                                logger_id=logger_id,
                                created_time=time.time(),
                                last_used_time=time.time(),
                                use_count=0,  # 预热时使用次数为0
                                error_count=0,
                                memory_usage=0
                            )
                            self._lru_cache[logger_id] = time.time()
                    except Exception as e:
                        # 预热失败不影响功能，仅记录
                        import logging
                        logging.debug(f"预热Logger失败: {logger_id}, 错误: {e}")
            
            self._warmed_up = True

    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息（优化版：包含LRU和预热信息）"""
        with self._lock:
            # 计算总请求数和命中率
            total_requests = self._hit_count + self._miss_count
            hit_rate = self._hit_count / total_requests if total_requests > 0 else 0.0
            
            # 计算LRU缓存统计
            lru_size = len(self._lru_cache)
            preloaded_count = sum(
                1 for lid in self._loggers.keys()
                if lid in self._PRELOAD_LOGGERS or any(lid.startswith(p) for p in self._PRELOAD_LOGGERS)
            )
            
            return {
                "pool_size": len(self._loggers),
                "max_size": self._max_size,
                "min_retain_size": self._min_retain_size,
                "total_loggers_created": len(self._stats),
                "created_count": self._miss_count,  # 已创建的logger数量等于未命中次数
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "idle_timeout": self._idle_timeout,
                "state": self._state.value,
                "warmed_up": self._warmed_up,
                "preloaded_count": preloaded_count,
                "lru_cache_size": lru_size,
                "logger_stats": {
                    logger_id: stats.to_dict()
                    for logger_id, stats in self._stats.items()
                }
            }

    def get_pool_status(self) -> Dict[str, Any]:
        """获取池状态"""
        stats = self.get_stats()
        current_size = stats["pool_size"]

        # 计算利用率
        utilization_rate = current_size / self._max_size if self._max_size > 0 else 0

        # 更新状态
        if utilization_rate > 0.9:
            self._state = LoggerPoolState.CRITICAL
        elif utilization_rate > 0.7:
            self._state = LoggerPoolState.WARNING
        else:
            self._state = LoggerPoolState.HEALTHY

        return {
            **stats,
            "utilization_rate": utilization_rate,
            "health_status": self._state.value
        }

    def shutdown(self) -> None:
        """关闭日志器池"""
        with self._lock:
            self._loggers.clear()
            self._stats.clear()
            self._state = LoggerPoolState.FAILED
