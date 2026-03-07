"""
数据预热器 - 最小可测实现
"""

import time
import threading
from typing import Callable, Dict, Any, Optional, List


class PreloadTask:
    def __init__(self, name: str, func: Callable[[], Any], interval_seconds: int = 600, enabled: bool = True):
        if not name or not callable(func):
            raise ValueError("invalid preload task")
        self.name = name
        self.func = func
        self.interval_seconds = max(1, int(interval_seconds))
        self.enabled = enabled
        self.last_run_at: Optional[float] = None
        self.last_status: str = "never"
        self.last_error: Optional[str] = None
        self.success_count: int = 0
        self.fail_count: int = 0

    def should_run(self, now: Optional[float] = None) -> bool:
        if not self.enabled:
            return False
        if self.last_run_at is None:
            return True
        now_ts = now if now is not None else time.time()
        return (now_ts - self.last_run_at) >= self.interval_seconds

    def run(self) -> bool:
        try:
            self.func()
            self.last_run_at = time.time()
            self.last_status = "success"
            self.last_error = None
            self.success_count += 1
            return True
        except Exception as e:
            self.last_run_at = time.time()
            self.last_status = "error"
            self.last_error = str(e)
            self.fail_count += 1
            return False


class Preloader:
    """
    - 注册/注销预热任务
    - 幂等执行：尊重任务interval
    - 线程安全的调度开/停
    - 可查询统计
    """

    def __init__(self):
        self._tasks: Dict[str, PreloadTask] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def register_task(self, name: str, func: Callable[[], Any], interval_seconds: int = 600, enabled: bool = True) -> bool:
        with self._lock:
            if name in self._tasks:
                return False
            self._tasks[name] = PreloadTask(name, func, interval_seconds, enabled)
            return True

    def unregister_task(self, name: str) -> bool:
        with self._lock:
            return self._tasks.pop(name, None) is not None

    def enable_task(self, name: str) -> bool:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = True
                return True
            return False

    def disable_task(self, name: str) -> bool:
        with self._lock:
            if name in self._tasks:
                self._tasks[name].enabled = False
                return True
            return False

    def run_once(self) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        with self._lock:
            items = list(self._tasks.items())
        for name, task in items:
            if task.should_run():
                results[name] = task.run()
        return results

    def start(self, poll_interval: int = 1) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                try:
                    self.run_once()
                    # 以短轮询避免长睡眠导致的关闭不及时
                    self._stop.wait(timeout=max(0.1, float(poll_interval)))
                except Exception:
                    # 避免调度器异常退出
                    self._stop.wait(timeout=0.5)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "tasks": {
                    name: {
                        "enabled": task.enabled,
                        "interval_seconds": task.interval_seconds,
                        "last_status": task.last_status,
                        "success_count": task.success_count,
                        "fail_count": task.fail_count,
                        "last_run_at": task.last_run_at,
                    }
                    for name, task in self._tasks.items()
                },
            }

    def list_tasks(self) -> List[str]:
        with self._lock:
            return list(self._tasks.keys())


