"""
hotreloadmanager 模块

提供 hotreloadmanager 相关功能和接口。
"""

import os

import time

from typing import Dict, Optional, List, Callable
import threading
import logging

"""安全配置相关类"""


class HotReloadManager:
    """热重载管理器"""

    def __init__(self):
        self._watchers: Dict[str, List[Callable]] = {}
        self._last_modified: Dict[str, float] = {}
        self._reload_lock = threading.RLock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._check_interval = 5.0  # 5秒检查一次

    def watch_file(self, file_path: str, callback: Callable):
        """监视配置文件"""
        file_path = os.path.abspath(file_path)

        if file_path not in self._watchers:
            self._watchers[file_path] = []
            self._last_modified[file_path] = self._get_file_mtime(file_path)

        self._watchers[file_path].append(callback)

    def start_monitoring(self):
        """开始监控"""
        if self._monitor_thread is None:
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="ConfigHotReload"
            )
            self._monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self._shutdown_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)

    def _monitor_loop(self):
        """监控循环"""
        while not self._shutdown_event.is_set():
            try:
                self._check_files()
                time.sleep(self._check_interval)
            except Exception as e:
                logger.error(f"配置文件监控错误: {e}")
                time.sleep(10)

    def _check_files(self):
        """检查文件变更"""
        with self._reload_lock:
            for file_path, callbacks in self._watchers.items():
                current_mtime = self._get_file_mtime(file_path)

                if current_mtime != self._last_modified.get(file_path):
                    logger.info(f"配置文件变更检测: {file_path}")
                    self._last_modified[file_path] = current_mtime

                    # 触发回调
                    for callback in callbacks:
                        try:
                            callback(file_path)
                        except Exception as e:
                            logger.error(f"热重载回调执行失败: {e}")

    def _get_file_mtime(self, file_path: str) -> float:
        """获取文件修改时间"""
        try:
            return os.path.getmtime(file_path)
        except OSError:
            return 0.0


logger = logging.getLogger(__name__)




