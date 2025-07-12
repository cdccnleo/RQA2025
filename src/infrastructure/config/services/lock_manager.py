import threading
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LockStats:
    acquire_count: int = 0
    hold_time: float = 0.0
    contention_count: int = 0
    last_acquired: Optional[datetime] = None

class LockManager:
    """统一锁管理服务"""

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._stats: Dict[str, LockStats] = {}
        self._global_lock = threading.Lock()
        self._lock_timestamps: Dict[str, Dict[threading.Thread, datetime]] = {}

    def acquire(self, lock_name: str, timeout: float = -1) -> threading.Lock:
        """获取命名锁并返回锁对象
        
        Args:
            lock_name: 锁名称
            timeout: 超时时间(秒)，-1表示无限等待
            
        Returns:
            获取到的锁对象，可直接用于with语句
        """
        if lock_name not in self._locks:
            with self._global_lock:
                if lock_name not in self._locks:
                    self._locks[lock_name] = threading.Lock()
                    self._stats[lock_name] = LockStats()
                    self._lock_timestamps[lock_name] = {}

        lock = self._locks[lock_name]

        # 处理超时参数
        if timeout == -1:
            acquired = lock.acquire(blocking=True)
        else:
            acquired = lock.acquire(timeout=timeout)

        if acquired:
            stats = self._stats[lock_name]
            stats.acquire_count += 1
            stats.last_acquired = datetime.now()

            # 记录获取锁的时间戳
            current_thread = threading.current_thread()
            self._lock_timestamps[lock_name][current_thread] = datetime.now()

            # 如果有其他线程在等待
            if lock.locked():
                stats.contention_count += 1

        return lock if acquired else None

    def release(self, lock_name: str) -> None:
        """释放命名锁"""
        if lock_name in self._locks:
            lock = self._locks[lock_name]
            if lock.locked():
                lock.release()

                # 更新持有时间统计
                current_thread = threading.current_thread()
                if current_thread in self._lock_timestamps[lock_name]:
                    acquire_time = self._lock_timestamps[lock_name][current_thread]
                    hold_time = (datetime.now() - acquire_time).total_seconds()

                    stats = self._stats[lock_name]
                    stats.hold_time += hold_time

                    # 移除时间戳记录
                    del self._lock_timestamps[lock_name][current_thread]
                return True
        return False

    def is_locked(self, lock_name: str) -> bool:
        """检查锁状态"""
        if lock_name in self._locks:
            return self._locks[lock_name].locked()
        return False

    def get_stats(self, lock_name: str) -> Dict:
        """获取锁统计信息"""
        if lock_name in self._stats:
            stats = self._stats[lock_name]
            return {
                "acquire_count": stats.acquire_count,
                "hold_time": stats.hold_time,
                "contention_count": stats.contention_count,
                "last_acquired": stats.last_acquired
            }
        return {}

    def reset_stats(self, lock_name: str) -> None:
        """重置锁统计"""
        if lock_name in self._stats:
            self._stats[lock_name] = LockStats()
