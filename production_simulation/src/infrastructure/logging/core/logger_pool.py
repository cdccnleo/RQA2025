
import threading
import time

from .unified_logger import UnifiedLogger
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层 - 日志器池管理

提供线程安全的日志器池管理，支持日志器的复用和生命周期管理。
"""


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
    """日志器池管理类"""

    _instance = None
    _lock = threading.Lock()

    def __init__(self, max_size: int = 100, idle_timeout: int = 3600):
        self._loggers: Dict[str, Any] = {}
        self._stats: Dict[str, LoggerStats] = {}
        self._max_size = max_size
        self._idle_timeout = idle_timeout
        self._lock = threading.RLock()
        self._state = LoggerPoolState.HEALTHY

    @classmethod
    def get_instance(cls) -> "LoggerPool":
        """获取单例实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_logger(self, logger_id: str, **kwargs) -> Any:
        """
        获取或创建日志器

        Args:
            logger_id: 日志器ID
            **kwargs: 创建参数

        Returns:
            日志器实例
        """
        with self._lock:
            # 检查是否已存在
            if logger_id in self._loggers:
                stats = self._stats[logger_id]
                stats.last_used_time = time.time()
                stats.use_count += 1
                return self._loggers[logger_id]

            # 检查池大小限制
            if len(self._loggers) >= self._max_size:
                self._evict_oldest()

            # 创建新日志器
            logger = self._create_logger(logger_id, **kwargs)
            if logger:
                self._loggers[logger_id] = logger
                self._stats[logger_id] = LoggerStats(
                    logger_id=logger_id,
                    created_time=time.time(),
                    last_used_time=time.time(),
                    use_count=1,
                    error_count=0,
                    memory_usage=0
                )

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
        """驱逐最旧的日志器"""
        if not self._stats:
            return

        # 找到最旧的日志器
        oldest_id = min(
            self._stats.keys(),
            key=lambda x: self._stats[x].last_used_time
        )

        # 移除最旧的日志器
        self.remove_logger(oldest_id)

    def cleanup_idle_loggers(self) -> int:
        """清理空闲日志器"""
        current_time = time.time()
        removed_count = 0

        with self._lock:
            to_remove = []
            for logger_id, stats in self._stats.items():
                if current_time - stats.last_used_time > self._idle_timeout:
                    to_remove.append(logger_id)

            for logger_id in to_remove:
                self.remove_logger(logger_id)
                removed_count += 1

        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """获取池统计信息"""
        with self._lock:
            return {
                "pool_size": len(self._loggers),
                "max_size": self._max_size,
                "total_loggers_created": len(self._stats),
                "idle_timeout": self._idle_timeout,
                "state": self._state.value,
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
