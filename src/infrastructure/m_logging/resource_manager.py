import os
import sys
import time
import threading
import weakref
from typing import List, Dict, Optional, Set, Any
from logging import Handler
from collections import defaultdict
import psutil
import tracemalloc

class ResourceManager:
    """日志资源管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """初始化资源管理器"""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._handlers: Set[Handler] = set()
        self._handler_refs = weakref.WeakSet()
        self._closed = False
        self._pid = os.getpid()
        self._initialized = True

        # 预警配置
        self.warning_thresholds = {
            'cpu': 80,
            'memory': 85,
            'disk': 90,
            'network': 70
        }
        self.warning_callbacks = []
        
        # 内存跟踪
        tracemalloc.start()

    def add_warning_handler(self, callback):
        """添加资源预警回调"""
        self.warning_callbacks.append(callback)

    def register_handler(self, handler: Handler):
        """注册日志处理器"""
        if self._closed:
            raise RuntimeError("ResourceManager已经关闭")

        self._handlers.add(handler)
        self._handler_refs.add(handler)

    def unregister_handler(self, handler: Handler):
        """注销日志处理器"""
        if handler in self._handlers:
            self._handlers.remove(handler)

    def close(self):
        """关闭资源管理器"""
        self._closed = True
        return self.close_all()

    def close_all(self, timeout: float = 5.0) -> bool:
        """安全关闭所有处理器

        Args:
            timeout: 关闭超时时间(秒)

        Returns:
            bool: 是否全部成功关闭
        """
        if self._closed:
            return True

        success = True
        start_time = time.time()

        # 并行关闭处理器
        threads = []
        for handler in list(self._handlers):
            t = threading.Thread(
                target=self._close_handler,
                args=(handler,),
                daemon=True
            )
            t.start()
            threads.append(t)

        # 等待关闭完成
        for t in threads:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time <= 0:
                success = False
                break

            t.join(remaining_time)
            if t.is_alive():
                success = False

        # 验证资源释放
        if success:
            success = self._verify_cleanup()

        self._closed = True
        return success

    def _close_handler(self, handler: Handler):
        """关闭单个处理器"""
        try:
            if hasattr(handler, 'flush'):
                handler.flush()

            if hasattr(handler, 'close'):
                handler.close()

            self.unregister_handler(handler)
        except Exception:
            pass

    def _verify_cleanup(self) -> bool:
        """验证资源是否已释放"""
        # 检查文件描述符
        process = psutil.Process(self._pid)
        open_files = process.open_files()

        # 检查处理器是否已释放
        active_handlers = [h for h in self._handler_refs]

        return len(open_files) == 0 and len(active_handlers) == 0

    def get_resource_usage(self) -> Dict[str, Any]:
        """获取资源使用情况并触发预警"""
        process = psutil.Process(self._pid)
        system = psutil.virtual_memory()

        # 内存使用
        mem_info = process.memory_info()
        mem_rss_pct = mem_info.rss / system.total
        mem_vms_pct = mem_info.vms / system.total

        # 文件描述符
        open_files = len(process.open_files())

        # 处理器状态
        handler_status = {
            'registered': len(self._handlers),
            'active_refs': len([h for h in self._handler_refs])
        }

        # 内存分配跟踪
        snapshot = tracemalloc.take_snapshot()
        stats = snapshot.statistics('lineno')

        # 检查预警条件
        current_usage = {
            'memory_rss': mem_rss_pct,
            'memory_vms': mem_vms_pct,
            'open_files': open_files,
            'handlers': handler_status['registered']
        }

        for metric, value in current_usage.items():
            threshold = self.warning_thresholds.get(metric)
            if threshold and value >= threshold:
                for callback in self.warning_callbacks:
                    try:
                        callback(metric, value, threshold)
                    except Exception:
                        continue

        return {
            'memory_rss': mem_info.rss,
            'memory_rss_pct': mem_rss_pct,
            'memory_vms': mem_info.vms,
            'memory_vms_pct': mem_vms_pct,
            'open_files': open_files,
            'handlers': handler_status,
            'malloc_stats': [
                {
                    'file': stat.traceback[0].filename,
                    'line': stat.traceback[0].lineno,
                    'size': stat.size,
                    'count': stat.count
                }
                for stat in stats[:10]  # 前10个内存分配点
            ]
        }

    def set_cpu_threshold(self, threshold: float):
        """设置CPU使用率阈值"""
        self.warning_thresholds['cpu'] = threshold

    def __del__(self):
        """析构函数确保资源释放"""
        if not self._closed:
            self.close_all()

# 全局函数
def close_all(timeout: float = 5.0) -> bool:
    """关闭所有资源"""
    return ResourceManager().close_all(timeout)

def get_resource_usage() -> Dict[str, Any]:
    """获取资源使用情况"""
    return ResourceManager().get_resource_usage()
