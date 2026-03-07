#!/usr/bin/env python3
"""
热重载服务

提供配置文件的热重载功能
"""

import logging
import time
import threading
from typing import Dict, Any, Callable, Optional, Set, List
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class HotReloadService:
    """热重载服务

    监控文件变化并触发重载操作
    """

    def __init__(self, watch_paths: Optional[list] = None, reload_interval: float = 1.0):
        """初始化热重载服务

        Args:
            watch_paths: 要监控的文件路径列表
            reload_interval: 监控间隔（秒）
        """
        self.watch_paths = watch_paths or []
        self.reload_interval = reload_interval
        self._callbacks: Dict[str, Set[Callable]] = {}
        self._file_timestamps: Dict[str, float] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

    def add_watch_path(self, path: str):
        """添加监控路径"""
        with self._lock:
            if path not in self.watch_paths:
                self.watch_paths.append(path)

    def remove_watch_path(self, path: str):
        """移除监控路径"""
        with self._lock:
            if path in self.watch_paths:
                self.watch_paths.remove(path)

    def register_callback(self, file_pattern: str, callback: Callable):
        """注册重载回调函数"""
        with self._lock:
            if file_pattern not in self._callbacks:
                self._callbacks[file_pattern] = set()
            self._callbacks[file_pattern].add(callback)

    def unregister_callback(self, file_pattern: str, callback: Callable):
        """取消注册重载回调函数"""
        with self._lock:
            if file_pattern in self._callbacks:
                self._callbacks[file_pattern].discard(callback)

    def start(self):
        """启动热重载监控"""
        if self._running:
            return

        with self._lock:
            self._running = True
            # 初始化文件时间戳
            self._update_timestamps()

            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("热重载服务已启动")

    def stop(self):
        """停止热重载监控"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            logger.info("热重载服务已停止")

    def _update_timestamps(self):
        """更新文件时间戳"""
        for path in self.watch_paths:
            try:
                if os.path.exists(path):
                    self._file_timestamps[path] = os.path.getmtime(path)
            except (OSError, IOError) as e:
                logger.warning(f"无法获取文件时间戳 {path}: {e}")

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                self._check_file_changes()
                time.sleep(self.reload_interval)
            except Exception as e:
                logger.error(f"热重载监控出错: {e}")
                time.sleep(self.reload_interval)

    def _check_file_changes(self) -> List[Dict[str, Any]]:
        """检查文件变化"""
        changed_files = []
        for path in self.watch_paths:
            try:
                if not os.path.exists(path):
                    continue

                current_mtime = os.path.getmtime(path)
                last_mtime = self._file_timestamps.get(path)

                if last_mtime is None:
                    # 新文件
                    self._file_timestamps[path] = current_mtime
                    self._trigger_callbacks(path, "created")
                    changed_files.append({"file_path": path, "event_type": "created"})
                elif current_mtime > last_mtime:
                    # 文件修改
                    self._file_timestamps[path] = current_mtime
                    self._trigger_callbacks(path, "modified")
                    changed_files.append({"file_path": path, "event_type": "modified"})

            except (OSError, IOError) as e:
                logger.warning(f"检查文件变化时出错 {path}: {e}")

        return changed_files

    def _trigger_callbacks(self, file_path: str, event_type: str):
        """触发回调函数"""
        triggered = False

        with self._lock:
            for pattern, callbacks in self._callbacks.items():
                if self._matches_pattern(file_path, pattern):
                    for callback in callbacks:
                        try:
                            callback(file_path, event_type)
                            triggered = True
                        except Exception as e:
                            logger.error(f"执行热重载回调出错: {e}")

        if triggered:
            logger.info(f"文件 {file_path} 触发热重载 ({event_type})")

    def _matches_pattern(self, file_path: str, pattern: str) -> bool:
        """检查文件路径是否匹配模式"""
        # 简单的模式匹配实现
        # 可以根据需要扩展为更复杂的模式匹配

        file_name = os.path.basename(file_path)

        # 支持通配符 * 和 **
        if '*' in pattern:
            import fnmatch
            return fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(file_path, pattern)
        else:
            return pattern in file_path or pattern in file_name

    def reload_now(self, file_path: str):
        """立即重载指定文件"""
        logger.info(f"立即重载文件: {file_path}")
        self._trigger_callbacks(file_path, "reload")

    def get_status(self) -> Dict[str, Any]:
        """获取服务状态"""
        callbacks_count = sum(len(callbacks) for callbacks in self._callbacks.values())
        return {
            "running": self._running,
            "watch_paths": self.watch_paths.copy(),
            "callbacks": callbacks_count,
            "callbacks_count": callbacks_count,
            "monitored_files": len(self._file_timestamps)
        }
