import threading
from typing import Optional, Dict

class LockManager:
    """线程安全的锁管理器"""

    def __init__(self):
        self._locks: Dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def acquire(self, lock_name: str, timeout: Optional[float] = None) -> bool:
        """获取命名锁

        Args:
            lock_name: 锁名称
            timeout: 超时时间(秒)，None表示无限等待

        Returns:
            bool: 是否成功获取锁
        """
        with self._global_lock:
            if lock_name not in self._locks:
                self._locks[lock_name] = threading.Lock()

        lock = self._locks[lock_name]
        if timeout is None:
            return lock.acquire()
        return lock.acquire(timeout=timeout)

    def release(self, lock_name: str) -> None:
        """释放命名锁

        Args:
            lock_name: 锁名称
        """
        with self._global_lock:
            if lock_name in self._locks:
                self._locks[lock_name].release()

    def get_lock_stats(self) -> Dict[str, str]:
        """获取锁状态统计

        Returns:
            Dict: 锁状态字典
        """
        with self._global_lock:
            return {
                name: "locked" if lock.locked() else "unlocked"
                for name, lock in self._locks.items()
            }

def get_default_lock_manager() -> LockManager:
    """获取默认锁管理器单例"""
    if not hasattr(get_default_lock_manager, "_instance"):
        get_default_lock_manager._instance = LockManager()
    return get_default_lock_manager._instance
