
import threading
import time

from ...core.shared_interfaces import ILogger, StandardLogger
from .health_check_executor import HealthCheckExecutor
from .health_check_manager import HealthCheck
from typing import Dict, Any, Optional, Callable
"""
健康检查调度器

职责：处理检查调度和周期执行
"""


class HealthCheckScheduler:
    """
    健康检查调度器

    职责：处理检查调度和周期执行
    """

    def __init__(self, executor: HealthCheckExecutor, logger: Optional[ILogger] = None):
        self.executor = executor
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

        self._running = False
        self._scheduler_thread: Optional[threading.Thread] = None
        self._check_threads: Dict[str, threading.Thread] = {}
        self._last_execution: Dict[str, float] = {}

        self._lock = threading.RLock()

    def start_scheduler(self) -> bool:
        """启动调度器"""
        with self._lock:
            if self._running:
                self.logger.log_warning("调度器已在运行")
                return True

            self._running = True
            self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self._scheduler_thread.start()

            self.logger.log_info("健康检查调度器已启动")
            return True

    def stop_scheduler(self) -> bool:
        """停止调度器"""
        with self._lock:
            if not self._running:
                return True

            self._running = False

            # 等待调度线程结束
            if self._scheduler_thread and self._scheduler_thread.is_alive():
                self._scheduler_thread.join(timeout=5.0)

            # 停止所有检查线程
            for thread in self._check_threads.values():
                if thread.is_alive():
                    thread.join(timeout=2.0)

            self._check_threads.clear()
            self.logger.log_info("健康检查调度器已停止")
            return True

    def schedule_check(self, check: HealthCheck, callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """调度单个检查执行"""
        if not check.enabled:
            return

        def execute_and_callback():
            try:
                result = self.executor.execute_check(check)
                if callback:
                    callback(result)
            except Exception as e:
                self.logger.log_error(f"调度执行检查失败 {check.name}: {e}")

        thread = threading.Thread(target=execute_and_callback, daemon=True)
        thread.start()

        with self._lock:
            self._check_threads[check.name] = thread
            self._last_execution[check.name] = time.time()

    def _scheduler_loop(self):
        """调度器主循环"""
        while self._running:
            try:
                # 检查需要执行的检查项
                current_time = time.time()

                # 这里应该从HealthCheckManager获取检查项并调度执行
                # 由于依赖关系，这里只实现调度逻辑

                time.sleep(10)  # 每10秒检查一次

            except Exception as e:
                self.logger.log_error(f"调度器循环异常: {e}")
                time.sleep(5)

    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        with self._lock:
            return {
                'running': self._running,
                'active_threads': len([t for t in self._check_threads.values() if t.is_alive()]),
                'last_executions': dict(self._last_execution)
            }
