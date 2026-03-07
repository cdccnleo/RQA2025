"""
concurrency_controller 模块

提供 concurrency_controller 相关功能和接口。
"""

import logging

import threading
import time

from collections import defaultdict
from datetime import datetime
from src.infrastructure.utils.core.interfaces import IConcurrencyController
from typing import Dict, Any, Optional, Tuple
"""
基础设施?- 日志系统组件

concurrency_controller 模块

日志系统相关的文?
提供日志系统相关的功能实现?
"""

logger = logging.getLogger(__name__)

# 并发控制器常量


class ConcurrencyConstants:
    """并发控制器相关常量"""

    # 默认超时配置 (秒)
    DEFAULT_LOCK_TIMEOUT = 5.0

    # 默认并发度
    DEFAULT_MAX_CONCURRENCY = 1
    DEFAULT_SEMAPHORE_COUNT = 1

    # 统计初始化值
    DEFAULT_ACQUIRE_COUNT = 0
    DEFAULT_RELEASE_COUNT = 0
    DEFAULT_WAIT_TIME_TOTAL = 0.0
    DEFAULT_MAX_WAIT_TIME = 0.0
    DEFAULT_CURRENT_HOLDERS = 0

    # 统计增量值
    STAT_INCREMENT = 1
    TIME_INCREMENT = 1.0

    # 清理配置
    CLEANUP_THRESHOLD = 100  # 统计清理阈值
    STALE_TIMEOUT = 3600  # 过期超时(秒)


class ConcurrencyController(IConcurrencyController):
    """并发控制器实现"""

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._semaphores: Dict[str, threading.Semaphore] = {}
        self._max_concurrency: Dict[str, int] = {}
        self._lock_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "acquire_count": ConcurrencyConstants.DEFAULT_ACQUIRE_COUNT,
                "release_count": ConcurrencyConstants.DEFAULT_RELEASE_COUNT,
                "wait_time_total": ConcurrencyConstants.DEFAULT_WAIT_TIME_TOTAL,
                "max_wait_time": ConcurrencyConstants.DEFAULT_MAX_WAIT_TIME,
                "current_holders": ConcurrencyConstants.DEFAULT_CURRENT_HOLDERS,
                "last_acquire_time": None,
                "last_release_time": None,
            }
        )
        self._global_lock = threading.RLock()

    def acquire_lock(self, resource: str, timeout: float = ConcurrencyConstants.DEFAULT_LOCK_TIMEOUT) -> bool:
        """获取锁"""
        # 初始化锁资源
        lock, semaphore, stats = self._initialize_resource_lock(resource)

        # 记录开始时间
        start_time = time.time()

        try:
            # 尝试获取锁
            return self._try_acquire_lock(resource, lock, semaphore, stats, start_time, timeout)
        except Exception as e:
            logger.error(f"获取资源锁时出错: {resource}, 错误: {e}")
            return False

    def _initialize_resource_lock(self, resource: str) -> Tuple[threading.Lock, threading.Semaphore, dict]:
        """初始化锁资源"""
        with self._global_lock:
            if resource not in self._locks:
                self._locks[resource] = threading.Lock()
                self._semaphores[resource] = threading.Semaphore(
                    ConcurrencyConstants.DEFAULT_SEMAPHORE_COUNT)
                self._max_concurrency[resource] = ConcurrencyConstants.DEFAULT_MAX_CONCURRENCY
                self._lock_stats[resource] = {
                    "acquire_count": ConcurrencyConstants.DEFAULT_ACQUIRE_COUNT,
                    "release_count": ConcurrencyConstants.DEFAULT_RELEASE_COUNT,
                    "wait_time_total": ConcurrencyConstants.DEFAULT_WAIT_TIME_TOTAL,
                    "max_wait_time": ConcurrencyConstants.DEFAULT_MAX_WAIT_TIME,
                    "current_holders": ConcurrencyConstants.DEFAULT_CURRENT_HOLDERS,
                    "last_acquire_time": None,
                    "last_release_time": None,
                }

            return (
                self._locks[resource],
                self._semaphores[resource],
                self._lock_stats[resource]
            )

    def _try_acquire_lock(
        self,
        resource: str,
        lock: threading.Lock,
        semaphore: threading.Semaphore,
        stats: dict,
        start_time: float,
        timeout: float
    ) -> bool:
        """尝试获取锁"""
        # 尝试获取信号量（用于限制并发数）
        acquired = semaphore.acquire(timeout=timeout)
        if not acquired:
            logger.warning(f"获取信号量超时 {resource}")
            return False

        # 使用锁来保护统计信息的更新
        with lock:
            # 更新统计信息
            self._update_lock_stats(stats, start_time)
            logger.debug(f"成功获取资源锁 {resource}, 等待时间: {time.time() - start_time:.3f}s")
        
        return True

    def _update_lock_stats(self, stats: dict, start_time: float):
        """更新锁统计信息"""
        wait_time = time.time() - start_time
        with self._global_lock:
            stats["acquire_count"] += ConcurrencyConstants.STAT_INCREMENT
            stats["wait_time_total"] += wait_time
            stats["max_wait_time"] = max(stats["max_wait_time"], wait_time)
            stats["current_holders"] += ConcurrencyConstants.STAT_INCREMENT
            stats["last_acquire_time"] = datetime.now()

    def release_lock(self, resource: str) -> bool:
        """释放锁"""
        if resource not in self._locks:
            logger.warning(f"尝试释放不存在的资源锁 {resource}")
            return False

        lock = self._locks[resource]
        semaphore = self._semaphores[resource]
        stats = self._lock_stats[resource]

        try:
            # 释放信号量（对应acquire_lock中的semaphore.acquire）
            semaphore.release()

            # 更新统计信息
            with lock:
                with self._global_lock:
                    stats["release_count"] += ConcurrencyConstants.STAT_INCREMENT
                    stats["current_holders"] = max(0, stats["current_holders"] - 1)
                    stats["last_release_time"] = datetime.now()

            logger.debug(f"成功释放资源锁 {resource}")
            return True
        except Exception as e:
            logger.error(f"释放资源锁时出错: {resource}, 错误: {e}")
            return False

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """获取并发统计"""
        stats = {}
        for resource, resource_stats in self._lock_stats.items():
            if resource_stats["acquire_count"] > 0:
                avg_wait_time = resource_stats["wait_time_total"] / resource_stats["acquire_count"]
                utilization_rate = resource_stats["current_holders"] / \
                    self._max_concurrency.get(resource, 1)

            stats[resource] = {
                "acquire_count": resource_stats["acquire_count"],
                "release_count": resource_stats["release_count"],
                "current_holders": resource_stats["current_holders"],
                "max_concurrency": self._max_concurrency.get(resource, 1),
                "avg_wait_time": (avg_wait_time if resource_stats["acquire_count"] > 0 else 0.0),
                "max_wait_time": resource_stats["max_wait_time"],
                "utilization_rate": utilization_rate,
                "last_acquire_time": (
                    resource_stats["last_acquire_time"].isoformat(
                    ) if resource_stats["last_acquire_time"] else None
                ),
                "last_release_time": (
                    resource_stats["last_release_time"].isoformat(
                    ) if resource_stats["last_release_time"] else None
                ),
            }

        return {
            "resources": stats,
            "total_resources": len(self._locks),
            "total_acquires": sum(s["acquire_count"] for s in self._lock_stats.values()),
            "total_releases": sum(s["release_count"] for s in self._lock_stats.values()),
        }

    def set_max_concurrency(self, resource: str, max_concurrent: int) -> None:
        """设置最大并发数"""
        with self._global_lock:
            if resource not in self._locks:
                self._locks[resource] = threading.Lock()
                self._lock_stats[resource] = {
                    "acquire_count": 0,
                    "release_count": 0,
                    "wait_time_total": 0.0,
                    "max_wait_time": 0.0,
                    "current_holders": 0,
                    "last_acquire_time": None,
                    "last_release_time": None,
                }

            self._max_concurrency[resource] = max_concurrent
            self._semaphores[resource] = threading.Semaphore(max_concurrent)

            # 重置当前持有者数量和统计信息
            self._lock_stats[resource]["current_holders"] = 0
            self._lock_stats[resource]["acquire_count"] = 0
            self._lock_stats[resource]["release_count"] = 0

            logger.info(f"设置资源 {resource} 的最大并发数为 {max_concurrent}")

    # 实现抽象接口方法
    def acquire(self, resource: str = "default") -> bool:
        """获取资源锁（接口方法）"""
        return self.acquire_lock(resource)

    def release(self, resource: str = "default") -> bool:
        """释放资源锁（接口方法）"""
        return self.release_lock(resource)

    def get_active_count(self, resource: str = "default") -> int:
        """获取活跃资源数量（接口方法）"""
        if resource not in self._lock_stats:
            return 0
        return self._lock_stats[resource]["current_holders"]

    @property
    def max_concurrent(self) -> int:
        """获取最大并发数（接口属性）"""
        # 返回所有资源的总最大并发数
        return sum(self._max_concurrency.values())

    def get_resource_info(self, resource: str) -> Optional[Dict[str, Any]]:
        """获取资源信息"""
        with self._global_lock:
            if resource not in self._locks:
                return None

            stats = self._lock_stats[resource]
            max_concurrent = self._max_concurrency.get(resource, 1)

            return {
                "resource": resource,
                "max_concurrency": max_concurrent,
                "current_holders": stats["current_holders"],
                "available_slots": max_concurrent - stats["current_holders"],
                "utilization_rate": stats["current_holders"] / max_concurrent,
                "total_acquires": stats["acquire_count"],
                "total_releases": stats["release_count"],
                "avg_wait_time": (
                    stats["wait_time_total"] /
                    stats["acquire_count"] if stats["acquire_count"] > 0 else 0
                ),
                "max_wait_time": stats["max_wait_time"],
            }

    def clear_stats(self) -> None:
        """清理统计信息"""
        with self._global_lock:
            for stats in self._lock_stats.values():
                stats["acquire_count"] = 0
                stats["release_count"] = 0
                stats["wait_time_total"] = 0.0
                stats["max_wait_time"] = 0.0
                stats["current_holders"] = 0
                stats["last_acquire_time"] = None
                stats["last_release_time"] = None

        logger.info("并发统计信息已清空")

    def get_deadlock_detection(self) -> Dict[str, Any]:
        """死锁检测"""
        with self._global_lock:
            deadlock_info = {
                "potential_deadlocks": [],
                "long_holding_locks": [],
                "recommendations": [],
            }

            current_time = datetime.now()

            for resource, stats in self._lock_stats.items():
                # 检查长时间持有的锁
                if stats["last_acquire_time"] and stats["current_holders"] > 0:
                    holding_time = (current_time - stats["last_acquire_time"]).total_seconds()
                    if holding_time > 30:  # 超过30秒
                        deadlock_info["long_holding_locks"].append(
                            {
                                "resource": resource,
                                "holding_time": holding_time,
                                "current_holders": stats["current_holders"],
                            }
                        )

                # 检查高等待时间
                if stats["acquire_count"] > 0:
                    avg_wait_time = stats["wait_time_total"] / stats["acquire_count"]
                    if avg_wait_time > 1.0:  # 平均等待时间超过1秒
                        deadlock_info["recommendations"].append(
                            f"资源 {resource} 平均等待时间较长({avg_wait_time:.3f}s)，建议增加并发数或优化锁粒度"
                        )

            return deadlock_info
