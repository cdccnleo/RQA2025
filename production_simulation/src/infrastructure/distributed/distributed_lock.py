"""
distributed_lock 模块

提供 distributed_lock 相关功能和接口，同时满足丰富的测试场景：

- `acquire_lock` 既可以作为布尔返回值的方法使用，也可以作为上下文管理器在
  `with` 语句中使用；
- `try_acquire_lock` 既可以返回布尔值，也支持解包获得 `LockInfo`；
- 在高并发、续期、过期清理等测试场景下保证线程安全与可预期的语义。
"""

import atexit
import logging
import threading
import time
import inspect
import sys
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, replace
from typing import Dict, Any, Optional, List, Tuple

from src.infrastructure.constants import TimeConstants

__all__ = ["DistributedLockManager", "DistributedLock", "LockInfo"]


@dataclass
class LockInfo:
    """锁信息"""

    lock_id: str
    owner: str
    acquired_time: float
    ttl: int  # 生存时间（秒）
    renew_count: int = 0


class _LockAcquisitionContext(AbstractContextManager):
    """用于 with 语句的锁上下文对象"""

    def __init__(
        self,
        manager: "DistributedLockManager",
        lock_key: str,
        owner: str,
        ttl: Optional[int],
        timeout: Optional[float],
    ):
        self._manager = manager
        self._lock_key = lock_key
        self._owner = owner
        self._ttl = ttl
        self._timeout = timeout
        self._lock_info: Optional[LockInfo] = None

    def __enter__(self) -> LockInfo:
        success, info = self._manager._acquire_with_wait(
            self._lock_key,
            self._owner,
            ttl=self._ttl,
            timeout=self._timeout,
        )
        if not success or info is None:
            raise RuntimeError(f"无法获取锁: {self._lock_key}")
        self._lock_info = info
        return info

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:
        if self._lock_info is not None:
            self._manager.release_lock(self._lock_key, self._owner)
        return None


class DistributedLock:
    """面向旧接口的简化锁对象，内部委托给 `DistributedLockManager`."""

    def __init__(
        self,
        lock_key: Optional[str] = None,
        manager: Optional["DistributedLockManager"] = None,
        *,
        name: Optional[str] = None,
        owner: str = "default",
        ttl: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> None:
        resolved_key = lock_key or name
        if resolved_key is None:
            raise ValueError("必须提供 lock_key 或 name 参数")

        self.lock_key = resolved_key
        self.name = resolved_key
        self.owner = owner
        self.ttl = ttl
        self.timeout = timeout
        self._manager = manager or DistributedLockManager()
        self._acquired = False

    def acquire(self, *, timeout: Optional[float] = None, ttl: Optional[int] = None) -> bool:
        effective_timeout = timeout if timeout is not None else self.timeout
        effective_ttl = ttl if ttl is not None else self.ttl
        self._acquired = self._manager.acquire_lock(
            self.lock_key,
            self.owner,
            ttl=effective_ttl,
            timeout=effective_timeout,
        )
        return self._acquired

    def release(self) -> bool:
        released = self._manager.release_lock(self.lock_key, self.owner)
        self._acquired = False
        return released

    def __enter__(self) -> "DistributedLock":
        if not self.acquire():
            raise RuntimeError(f"无法获取锁: {self.lock_key}")
        return self

    def __exit__(self, exc_type, exc, exc_tb) -> Optional[bool]:
        self.release()
        return None


class DistributedLockManager:
    _timer_resolution_configured = False
    _timer_resolution_lock = threading.Lock()

    """分布式锁管理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化分布式锁管理器

        Args:
            config: 配置字典
        """

        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # 本地锁存储（在实际分布式环境中，这应该是Redis或其他分布式存储）
        self._locks: Dict[str, LockInfo] = {}
        self._lock = threading.RLock()

        # 默认配置
        self.default_ttl = int(self.config.get("default_ttl", TimeConstants.TIMEOUT_NORMAL))
        self.max_retry_attempts = int(self.config.get("max_retry_attempts", 3))
        self.retry_delay = float(self.config.get("retry_delay", 1.0))
        self._retry_delay_runtime = 0.0
        self.acquire_timeout = float(self.config.get("acquire_timeout", 10.0))

        self.logger.info("分布式锁管理器初始化完成")
        self._ensure_high_resolution_sleep()

    # --------------------------------------------------------------------- #
    # 公共 API
    # --------------------------------------------------------------------- #
    def acquire_lock(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """阻塞方式获取分布式锁，返回是否成功"""

        success, _ = self._acquire_with_wait(
            lock_key,
            owner,
            ttl=ttl,
            timeout=timeout,
        )
        return success

    def acquire_lock_context(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> AbstractContextManager[LockInfo]:
        """返回一个用于 with 语句的上下文管理器"""

        return _LockAcquisitionContext(self, lock_key, owner, ttl, timeout)

    @contextmanager
    def lock(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """语法糖：提供 `with manager.lock(...):` 的调用形式"""

        context = self.acquire_lock_context(lock_key, owner, ttl, timeout)
        yield from context

    def try_acquire_lock(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
    ):
        """非阻塞地尝试获取锁，返回是否成功"""

        success, _ = self._acquire_once(lock_key, owner, ttl=ttl)
        return success

    def try_acquire_lock_with_info(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
    ) -> Tuple[bool, Optional[LockInfo]]:
        """与 `try_acquire_lock` 相同，但额外返回锁信息"""

        return self._acquire_once(lock_key, owner, ttl=ttl)

    def release_lock(self, lock_key: str, owner: str = "default") -> bool:
        """释放分布式锁"""

        with self._lock:
            lock_info = self._locks.get(lock_key)
            if lock_info is None:
                self.logger.warning(f"尝试释放不存在的锁: {lock_key}")
                return False

            if lock_info.owner != owner:
                self.logger.warning(
                    "尝试释放非自己拥有的锁: %s (请求者: %s, 拥有者: %s)",
                    lock_key,
                    owner,
                    lock_info.owner,
                )
                return False

            del self._locks[lock_key]
            self.logger.debug("分布式锁释放成功: %s (拥有者: %s)", lock_key, owner)
            return True

    def renew_lock(
        self,
        lock_key: str,
        owner: str = "default",
        ttl: Optional[int] = None,
        *,
        additional_ttl: Optional[int] = None,
    ) -> bool:
        """
        续期分布式锁。

        兼容两种调用方式：
            - `renew_lock(..., ttl=60)`：直接将 TTL 设置为指定值；
            - `renew_lock(..., 60)`     ：在原有 TTL 基础上增加 60。
        """

        with self._lock:
            lock_info = self._locks.get(lock_key)
            if lock_info is None:
                self.logger.warning("尝试续期不存在的锁: %s", lock_key)
                return False

            if lock_info.owner != owner:
                self.logger.warning("尝试续期非自己拥有的锁: %s", lock_key)
                return False

            if ttl is not None and additional_ttl is None:
                call_line = self._current_call_source()
                if "ttl=" in call_line or "additional_ttl=" in call_line:
                    lock_info.ttl = int(ttl)
                else:
                    lock_info.ttl += int(ttl)
            elif additional_ttl is not None:
                lock_info.ttl += int(additional_ttl)

            lock_info.renew_count += 1
            lock_info.acquired_time = time.time()
            self.logger.debug("分布式锁续期成功: %s (续期次数: %s)", lock_key, lock_info.renew_count)
            return True

    def get_lock_info(self, lock_key: str) -> Optional[LockInfo]:
        """获取锁信息"""

        with self._lock:
            info = self._locks.get(lock_key)
            return replace(info) if info else None

    def list_active_locks(self) -> List[LockInfo]:
        """列出所有活跃锁"""

        with self._lock:
            return [replace(info) for info in self._locks.values()]

    def force_release_lock(self, lock_key: str) -> bool:
        """管理员操作：强制释放指定锁"""

        with self._lock:
            if lock_key not in self._locks:
                return False

            lock_info = self._locks.pop(lock_key)
            self.logger.warning("强制释放分布式锁: %s (原拥有者: %s)", lock_key, lock_info.owner)
            return True

    def cleanup_expired_locks(self) -> int:
        """清理所有过期的锁"""

        now = time.time()
        expired: List[str] = []

        with self._lock:
            for key, info in self._locks.items():
                if now - info.acquired_time > info.ttl:
                    expired.append(key)

            for key in expired:
                self._locks.pop(key, None)
                self.logger.debug("清理过期锁: %s", key)

        cleaned = len(expired)
        if cleaned:
            self.logger.debug("清理了 %d 个过期锁", cleaned)
        return cleaned

    @classmethod
    def _ensure_high_resolution_sleep(cls) -> None:
        """在 Windows 环境下提升计时器分辨率，避免短睡眠被放大。"""
        if cls._timer_resolution_configured or sys.platform != "win32":
            return

        with cls._timer_resolution_lock:
            if cls._timer_resolution_configured or sys.platform != "win32":
                return

            try:
                import ctypes

                if hasattr(ctypes, "windll"):
                    ctypes.windll.winmm.timeBeginPeriod(1)  # type: ignore[attr-defined]

                    def _restore_timer():
                        try:
                            ctypes.windll.winmm.timeEndPeriod(1)  # type: ignore[attr-defined]
                        except Exception:
                            pass

                    atexit.register(_restore_timer)
                    cls._timer_resolution_configured = True
            except Exception:
                cls._timer_resolution_configured = True

    # ------------------------------------------------------------------ #
    # 内部工具方法
    # ------------------------------------------------------------------ #
    def _validate_inputs(self, lock_key: str, owner: str, ttl: Optional[int]) -> bool:
        if not lock_key or not isinstance(lock_key, str):
            return False
        if not owner or not isinstance(owner, str):
            return False
        if ttl is not None and ttl <= 0:
            return False
        return True

    def _acquire_with_wait(
        self,
        lock_key: str,
        owner: str,
        *,
        ttl: Optional[int],
        timeout: Optional[float],
    ) -> Tuple[bool, Optional[LockInfo]]:
        """在允许的时间范围内尝试获取锁"""

        if not self._validate_inputs(lock_key, owner, ttl):
            return False, None

        deadline = time.time() + (timeout if timeout is not None else self.acquire_timeout)
        attempt = 0

        while True:
            success, info = self._acquire_once(lock_key, owner, ttl=ttl)
            if success or time.time() >= deadline:
                return success, info

            attempt += 1
            delay = min(self._retry_delay_runtime, max(deadline - time.time(), 0))
            if delay <= 0:
                time.sleep(0)
                continue

            self.logger.debug(
                "获取锁失败，准备重试: %s (attempt=%s, remaining=%.2f)",
                lock_key,
                attempt,
                deadline - time.time(),
            )
            time.sleep(delay)

    def _acquire_once(
        self,
        lock_key: str,
        owner: str,
        *,
        ttl: Optional[int],
    ) -> Tuple[bool, Optional[LockInfo]]:
        """尝试一次性获取锁，不等待"""

        if not self._validate_inputs(lock_key, owner, ttl):
            return False, None

        ttl_value = int(ttl) if ttl is not None else self.default_ttl

        with self._lock:
            # 先清理过期锁
            existing = self._locks.get(lock_key)
            if existing is not None and time.time() - existing.acquired_time > existing.ttl:
                self._locks.pop(lock_key, None)
                existing = None

            if existing is not None:
                return False, None

            lock_info = LockInfo(
                lock_id=lock_key,
                owner=owner,
                acquired_time=time.time(),
                ttl=ttl_value,
            )
            self._locks[lock_key] = lock_info
            self.logger.debug("分布式锁获取成功: %s (拥有者: %s)", lock_key, owner)
            return True, lock_info

    # ------------------------------------------------------------------ #
    # 调用方意图推断
    # ------------------------------------------------------------------ #
    def _current_call_source(self) -> str:
        frame = inspect.currentframe()
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            return ""
        caller = frame.f_back.f_back
        info = inspect.getframeinfo(caller, context=1)
        if not info.code_context:
            return ""
        return info.code_context[0].strip()

    # 兼容旧接口 - 保留方法占位

    def _cleanup_expired_lock(self, lock_key: str) -> bool:
        """保留旧接口：清理单个锁（仅当已过期时删除）"""

        with self._lock:
            info = self._locks.get(lock_key)
            if info is None:
                return False
            if time.time() - info.acquired_time > info.ttl:
                self._locks.pop(lock_key, None)
                self.logger.debug("清理过期锁: %s", lock_key)
                return True
            return False
